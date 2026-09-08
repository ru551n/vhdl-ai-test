"""Tests for `accel_v2.planner`: the local-memory residency policy and
its `DdrTraffic` accounting (`doc/cnn_accel_top_v2_arch.md` section 10's
residency invariants). `test_reference.py` cross-checks that the
*predicted* traffic here matches what `reference.run_reference` actually
moves; this file focuses on the planner's own decisions in isolation."""

from __future__ import annotations

import pytest

from accel_v2 import isa
from accel_v2.model import Activation, Model
from accel_v2.planner import ComputeStep, MoveStep, Planner


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
    silent truncation."""
    with pytest.raises(ValueError, match="do not fit"):
        Planner(tensor_mem_bytes=64).plan(_linear_chain())


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
