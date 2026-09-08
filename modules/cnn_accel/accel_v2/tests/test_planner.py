"""Tests for `accel_v2.planner`: the local-memory residency policy and
its `DdrTraffic` accounting (`doc/cnn_accel_top_v2_arch.md` section 10's
residency invariants). `test_reference.py` cross-checks that the
*predicted* traffic here matches what `reference.run_reference` actually
moves; this file focuses on the planner's own decisions in isolation."""

from __future__ import annotations

import pytest

from accel_v2 import cases, cases_concat_split, cases_pool_pad, cases_yolo, isa
from accel_v2.model import Activation, Model
from accel_v2.planner import ComputeStep, MoveStep, PlannedProgram, Planner


def _linear_chain(seed: int = 1) -> Model:
    """input -> conv -> conv -> conv -> output: every intermediate has
    exactly one consumer, so a large-enough scratchpad never needs to
    spill anything (each output can free its predecessor immediately)."""
    m = Model(seed=seed)
    x = m.input(8, 8, 4)
    h1 = m.conv2d(x, out_channels=4, kernel=(3, 3), padding=(1, 1, 1, 1), activation=Activation.RELU)
    h2 = m.conv2d(h1, out_channels=4, kernel=(3, 3), padding=(1, 1, 1, 1), activation=Activation.RELU)
    h3 = m.conv2d(h2, out_channels=4, kernel=(3, 3), padding=(1, 1, 1, 1), activation=Activation.RELU)
    m.output(h3)
    return m


def _fanout_chain(seed: int = 7) -> Model:
    """input -> h1 -> {h2a, h2b} -> add -> output: h1 has two consumers,
    so it must stay resident (or be reloaded) across both."""
    m = Model(seed=seed)
    x = m.input(8, 8, 4)
    h1 = m.conv2d(x, out_channels=4, kernel=(3, 3), padding=(1, 1, 1, 1), activation=Activation.RELU)
    h2a = m.conv2d(h1, out_channels=4, kernel=(3, 3), padding=(1, 1, 1, 1), activation=Activation.NONE)
    h2b = m.conv2d(h1, out_channels=4, kernel=(3, 3), padding=(1, 1, 1, 1), activation=Activation.NONE)
    h3 = m.add(h2a, h2b)
    m.output(h3)
    return m




def _straddling_buffers_model(seed: int = 77) -> Model:
    """input -> h1 -> a -> add(a, h1) -> output, with 8x12x8 tensors, i.e.
    768 bytes each. `h1` and `a` are live at the same time (the `add`
    reads both), and 768 does not divide a 1024-byte bank, so a flat
    first-fit allocator puts the second buffer at 768..1536 -- straddling
    the boundary between bank 0 and bank 1."""
    m = Model(seed=seed)
    x = m.input(8, 12, 8, name="x")
    h1 = m.conv2d(x, out_channels=8, kernel=(3, 3), padding=(1, 1, 1, 1), name="h1")
    a = m.conv2d(h1, out_channels=8, kernel=(3, 3), padding=(1, 1, 1, 1), name="a")
    m.output(m.add(a, h1, name="y"))
    return m


def _assert_every_buffer_lies_in_one_bank(planned: PlannedProgram) -> None:
    """The geometric invariant `cnn_accel_tensor_mem` needs and cannot
    check for itself in a value comparison.

    Asserted on the placement directly, on purpose: the emitted program
    and `reference.py` both take their addresses from this same plan, so
    a straddling buffer is *self-consistently* wrong on both sides of any
    value-only DUT-versus-reference check and shows up there as nothing
    at all. Only the geometry gives it away."""
    bank_bytes = planned.bank_bytes
    for name, addr, size in planned.local_placements:
        assert addr // bank_bytes == (addr + size - 1) // bank_bytes, (
            f"local buffer '{name}' at {addr}..{addr + size} straddles the boundary "
            f"between bank {addr // bank_bytes} and bank {(addr + size - 1) // bank_bytes} "
            f"of a {bank_bytes}-byte-per-bank scratchpad; cnn_accel_tensor_mem serves "
            "every transfer from a single bank and would silently clamp (truncate) "
            "every access to this buffer"
        )
        assert addr + size <= planned.tensor_mem_bytes


def test_linear_chain_stays_local_no_spills_with_ample_memory() -> None:
    planned = Planner(tensor_mem_bytes=1 << 16).plan(_linear_chain())
    assert planned.traffic.spill_count == 0
    assert planned.traffic.reload_count == 0
    assert all(isinstance(s, ComputeStep) for s in planned.steps)
    # Only the graph input (read once from DDR) and the graph output
    # (written once to DDR) ever touch DDR for tensor data -- no MoveStep
    # is needed at all in a pure linear chain.
    assert planned.traffic.tensor_load_count == 0
    assert planned.traffic.tensor_store_count == 0


def test_forced_small_memory_produces_explicit_spill_and_reload() -> None:
    """A budget too small to hold `h1` resident across both of its
    consumers forces the planner to spill it (section 10: "an explicit
    spill... and a later reload" rather than silently keeping it live in
    a buffer too small to fit)."""
    planned = Planner(tensor_mem_bytes=1024).plan(_fanout_chain())
    assert planned.traffic.spill_count >= 1
    assert planned.traffic.reload_count >= 1
    move_kinds = [s.kind for s in planned.steps if isinstance(s, MoveStep)]
    assert "spill" in move_kinds
    assert "reload" in move_kinds


def test_too_small_memory_raises() -> None:
    """A single tensor that does not fit in the scratchpad at all (even
    after evicting everything else) is a hard planner error, not a
    silent truncation.

    With a 64-byte scratchpad the failure is specifically "bigger than
    one bank" (a 64-byte scratchpad is one 64-byte bank), which is the
    stricter of the two hard errors: the hardware serves a transfer from
    exactly one bank, so a buffer that exceeds a bank can never be placed
    legally no matter how much of the scratchpad is free. See
    `test_buffer_larger_than_one_bank_is_a_hard_error` for the case where
    the scratchpad as a whole is ample and only the bank is not."""
    with pytest.raises(ValueError, match="more than one 64-byte"):
        Planner(tensor_mem_bytes=64).plan(_linear_chain())


def test_capacity_exhaustion_still_raises_do_not_fit() -> None:
    """The other hard error: the buffer fits in a bank, but the whole
    scratchpad is too small to hold it alongside everything that cannot
    be evicted."""
    with pytest.raises(ValueError, match="do not fit"):
        Planner(tensor_mem_bytes=512, bank_bytes=512).plan(_fanout_chain())


def test_local_buffers_never_straddle_a_bank_boundary() -> None:
    """The regression test for the bug this whole bank-awareness change
    exists for: the planner used to treat the scratchpad as one flat
    range, so a buffer could be placed across a bank boundary and every
    hardware access to it was silently truncated to the addressed bank.

    The two `Planner` calls below differ *only* in `bank_bytes`, and the
    second one -- one bank as wide as the whole scratchpad -- is exactly
    the pre-fix flat allocator. It is here so this test cannot quietly go
    vacuous: it asserts that this geometry really does straddle when
    allocated flat, before asserting that the real, 2-bank planner does
    not."""
    m = _straddling_buffers_model()

    flat = Planner(tensor_mem_bytes=2048, bank_bytes=2048).plan(m)
    assert [(a, s) for _, a, s in flat.local_placements] == [(0, 768), (768, 768)], (
        "this model no longer reproduces the flat-allocator straddle it was "
        "written to reproduce"
    )
    assert any(
        addr // 1024 != (addr + size - 1) // 1024 for _, addr, size in flat.local_placements
    ), "flat allocation of this model must straddle the real 1024-byte bank boundary"

    planned = Planner(tensor_mem_bytes=2048, bank_bytes=1024).plan(_straddling_buffers_model())
    _assert_every_buffer_lies_in_one_bank(planned)
    # Skipped to the next bank rather than bank-aligning everything: the
    # 256 bytes left behind at 768..1024 stay on the free list.
    assert [(a, s) for _, a, s in planned.local_placements] == [(0, 768), (1024, 768)]
    assert planned.traffic.spill_count == 0, (
        "bank-awareness must not turn a program that fits into a spilling one"
    )


def test_skipped_bytes_below_a_bank_boundary_are_still_usable() -> None:
    """The cost of "skip to the next bank" over "bank-align everything":
    the hole left below the boundary is an ordinary free block, handed to
    the next buffer small enough for it. If it were lost, this model
    (768 + 768 + 256 bytes live) would not fit in 2 KiB and would spill."""
    m = Model(seed=78)
    x = m.input(8, 12, 8, name="x")
    sx = m.input(4, 8, 8, name="sx")
    h1 = m.conv2d(x, out_channels=8, kernel=(3, 3), padding=(1, 1, 1, 1), name="h1")
    a = m.conv2d(h1, out_channels=8, kernel=(3, 3), padding=(1, 1, 1, 1), name="a")
    # 4x8x8 = 256 bytes: exactly the hole left below the bank boundary,
    # and allocated while both 768-byte buffers are still live.
    s1 = m.conv2d(sx, out_channels=8, kernel=(1, 1), name="s1")
    m.output(m.add(a, h1, name="y"))
    m.output(m.conv2d(s1, out_channels=8, kernel=(1, 1), name="s2"))

    planned = Planner(tensor_mem_bytes=2048, bank_bytes=1024).plan(m)
    _assert_every_buffer_lies_in_one_bank(planned)
    assert ("s1", 768, 256) in planned.local_placements, (
        f"the 256-byte hole below the bank boundary was not reused: "
        f"{planned.local_placements}"
    )


def test_buffer_larger_than_one_bank_is_a_hard_error() -> None:
    """A buffer bigger than one bank cannot be placed legally at all, no
    matter how much scratchpad is free -- the hardware has no multi-bank
    transfer. That is a hard error with an actionable message, never a
    split across banks."""
    m = _straddling_buffers_model()
    with pytest.raises(ValueError, match="more than one 512-byte"):
        # 8 KiB of scratchpad, ample -- but in 512-byte banks, and the
        # tensors are 768 bytes.
        Planner(tensor_mem_bytes=8192, bank_bytes=512).plan(m)


def test_every_catalogue_case_places_every_buffer_inside_one_bank() -> None:
    """The same geometric invariant over the whole `tb_cnn_accel_top`
    catalogue, so a future case with bigger tensors (or a smaller
    `bank_words`) cannot reintroduce a straddle unnoticed. It would
    otherwise show up only as a data mismatch in GHDL, or -- since the
    program and the reference agree on the wrong address -- not at all."""
    catalogue = (
        cases.all_cases()
        + cases_pool_pad.all_cases()
        + cases_concat_split.all_cases()
        + cases_yolo.all_cases()
    )
    assert catalogue
    for case in catalogue:
        assert case.planned.bank_bytes == case.bank_words * 8, case.name
        assert case.planned.tensor_mem_bytes == case.tensor_mem_bytes, case.name
        _assert_every_buffer_lies_in_one_bank(case.planned)


def test_graph_output_always_lands_in_ddr_never_local() -> None:
    m = _linear_chain()
    planned = Planner().plan(m)
    out_tensor = m.outputs[0]
    compute_steps = [s for s in planned.steps if isinstance(s, ComputeStep)]
    last = compute_steps[-1]
    assert last.op.output is out_tensor
    assert last.output_space == isa.SPACE_DDR
    assert out_tensor.name in planned.tensor_ddr_addr


def test_weight_reuse_charges_zero_extra_weight_bytes() -> None:
    m = Model(seed=3)
    x = m.input(8, 8, 4)
    c1 = m.conv2d(x, out_channels=4, kernel=(3, 3), padding=(1, 1, 1, 1))
    m.output(c1)
    planned_no_reuse = Planner().plan(m)

    m2 = Model(seed=3)
    x2 = m2.input(8, 8, 4)
    c1_2 = m2.conv2d(x2, out_channels=4, kernel=(3, 3), padding=(1, 1, 1, 1))
    c2_2 = m2.conv2d(x2, out_channels=4, kernel=(3, 3), padding=(1, 1, 1, 1), weight_reuse_from=c1_2)
    m2.output(c1_2)
    m2.output(c2_2)
    planned_reuse = Planner().plan(m2)

    # Two independent convs (no reuse) would cost 2x the weight bytes of
    # one; the reused second conv costs the planner nothing extra.
    assert planned_reuse.traffic.weight_bytes == planned_no_reuse.traffic.weight_bytes


def test_local_read_bytes_counted_for_every_operand_fetch() -> None:
    """Regression guard for a planner bug found during development: the
    already-resident fast path in `resolve_input` (and the reload path's
    own subsequent compute-step read) must both charge
    `local_read_bytes` -- `test_reference.py::test_traffic_prediction_
    matches_actual_*` catches this more thoroughly by cross-checking
    against `reference.run_reference`, but this test isolates the
    planner's own arithmetic without needing the executor."""
    planned = Planner(tensor_mem_bytes=1 << 16).plan(_linear_chain())
    # h1 -> h2 and h2 -> h3 are both local reads (each intermediate has
    # exactly one consumer, immediately following, so never spilled).
    assert planned.traffic.local_read_bytes > 0
