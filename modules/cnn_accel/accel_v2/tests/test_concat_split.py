"""Tests for channel `concat`/`split` (`accel_v2.model.Model.concat`/
`split`) and the buffer aliasing `accel_v2.planner` does for them.

The design claim under test is that **neither operation needs any
hardware**. The section-3 activation layout is channel-plane-major
(`byte_offset = ((c_tile * H + y) * W + x) * T + t`, `T = 8`), so a
tensor is `ceil(C/T)` contiguous planes of `H*W*T` bytes:

* a channel concatenation is the operands' planes laid back to back, so
  telling each producer to write at the right plane offset inside one
  buffer *is* the concatenation -- no opcode, no instruction, no byte;
* a channel split at a multiple-of-`T` boundary is a sub-range of the
  parent's planes -- likewise free.

So these tests check three separate things, and the second is the one
that would be quietly wrong if the design were wrong:

1. the *values* are a real channel concatenation/slice
   (`reference.py` versus an independent list-slicing expectation);
2. the *cost* is zero -- byte-for-byte identical `DdrTraffic` to the same
   network without the concat, and no extra step in the program;
3. the aliasing/liveness rules hold: no buffer is recycled while any
   alias of it is still live, and a tensor that would have to be in two
   places at once falls back to an explicit `COPY`.
"""

from __future__ import annotations

import pytest

from accel_v2 import isa
from accel_v2.memimage import MemoryImage
from accel_v2.model import (
    CopyOp,
    Model,
    alias_byte_offset,
    alias_root,
)
from accel_v2.planner import ComputeStep, MoveStep, Planner
from accel_v2.reference import run_reference

_BIG_MEM = 1 << 16


def _plan(model: Model, tensor_mem_bytes: int = _BIG_MEM):
    return Planner(tensor_mem_bytes=tensor_mem_bytes).plan(model)


def _run(model: Model, tensor_mem_bytes: int = _BIG_MEM):
    planned = _plan(model, tensor_mem_bytes)
    return planned, run_reference(planned, MemoryImage())


def _hwc_concat(pixels: int, parts: list[tuple[list[int], int]]) -> list[int]:
    """Independent channel concatenation of LOGICAL HWC lists -- written
    with plain list slicing so it shares no code with the thing under
    test."""
    out: list[int] = []
    for pixel in range(pixels):
        for values, channels in parts:
            out.extend(values[pixel * channels : (pixel + 1) * channels])
    return out


def _hwc_slice(values: list[int], pixels: int, channels: int, first: int, count: int) -> list[int]:
    out: list[int] = []
    for pixel in range(pixels):
        base = pixel * channels + first
        out.extend(values[base : base + count])
    return out


# ---------------------------------------------------------------------------
# 1. Model API: shapes, aliasing structure, auto-naming.
# ---------------------------------------------------------------------------


def test_concat_shape_and_alias_structure() -> None:
    m = Model(seed=1)
    x = m.input(4, 4, 8)
    a = m.conv2d(x, 8, padding=(1, 1, 1, 1))
    b = m.conv2d(x, 16, padding=(1, 1, 1, 1))
    y = m.concat([a, b])

    assert (y.height, y.width, y.channels) == (4, 4, 24)
    assert y.alias_parent is None, "a concat result owns its buffer"
    assert [p.name for p in y.alias_parts] == [a.name, b.name]
    # a occupies plane 0, b planes 1..2 -- the whole point of the design.
    assert (a.alias_parent, a.alias_plane_offset, a.alias_role) == (y, 0, "part")
    assert (b.alias_parent, b.alias_plane_offset, b.alias_role) == (y, 1, "part")
    assert alias_root(a) is y and alias_root(b) is y
    assert alias_byte_offset(a) == 0
    assert alias_byte_offset(b) == 1 * 8 * 4 * 4
    assert y in m.tensors


def test_split_shape_and_alias_structure() -> None:
    m = Model(seed=2)
    x = m.input(4, 4, 8)
    h = m.conv2d(x, 24, padding=(1, 1, 1, 1))
    p0, p1 = m.split(h, [8, 16])

    assert (p0.channels, p1.channels) == (8, 16)
    assert (p0.alias_parent, p0.alias_plane_offset, p0.alias_role) == (h, 0, "slice")
    assert (p1.alias_parent, p1.alias_plane_offset, p1.alias_role) == (h, 1, "slice")
    assert alias_byte_offset(p1) == 1 * 8 * 4 * 4
    assert p0 in m.tensors and p1 in m.tensors


def test_concat_and_split_emit_no_ops() -> None:
    """The design's headline claim, at the graph level: neither builder
    call adds an `Op`, so neither can add a descriptor."""
    m = Model(seed=3)
    x = m.input(4, 4, 8)
    a = m.conv2d(x, 8, padding=(1, 1, 1, 1))
    b = m.conv2d(x, 8, padding=(1, 1, 1, 1))
    before = len(m.ops)
    y = m.concat([a, b])
    m.split(y, [8, 8])
    assert len(m.ops) == before


def test_auto_naming_is_unique_and_prefixed() -> None:
    m = Model(seed=4)
    x = m.input(4, 4, 8)
    a = m.conv2d(x, 8, padding=(1, 1, 1, 1))
    b = m.conv2d(x, 8, padding=(1, 1, 1, 1))
    y = m.concat([a, b])
    parts = m.split(y, [8, 8])
    names = [y.name] + [p.name for p in parts]
    assert y.name.startswith("concat")
    assert all(p.name.startswith("split") for p in parts)
    assert len(set(names)) == len(names)


# ---------------------------------------------------------------------------
# 2. Validation -- actionable messages, not assertions.
# ---------------------------------------------------------------------------


def test_split_rejects_unaligned_boundary() -> None:
    m = Model(seed=5)
    x = m.input(4, 4, 8)
    h = m.conv2d(x, 24, padding=(1, 1, 1, 1))
    with pytest.raises(ValueError) as excinfo:
        m.split(h, [4, 20])
    message = str(excinfo.value)
    assert "channel 4" in message
    assert "multiple of the 8-channel activation plane" in message


def test_split_rejects_sizes_that_do_not_cover_the_tensor() -> None:
    m = Model(seed=6)
    x = m.input(4, 4, 8)
    h = m.conv2d(x, 24, padding=(1, 1, 1, 1))
    with pytest.raises(ValueError, match="sum to 16, but .* has 24 channels"):
        m.split(h, [8, 8])


def test_split_rejects_non_positive_part() -> None:
    m = Model(seed=7)
    x = m.input(4, 4, 8)
    h = m.conv2d(x, 16, padding=(1, 1, 1, 1))
    with pytest.raises(ValueError, match="at least one channel"):
        m.split(h, [16, 0])


def test_concat_rejects_spatial_shape_mismatch() -> None:
    m = Model(seed=8)
    x = m.input(8, 8, 8)
    a = m.conv2d(x, 8, padding=(1, 1, 1, 1))
    b = m.pool_max(x, kernel=(2, 2), stride=(2, 2))
    with pytest.raises(ValueError) as excinfo:
        m.concat([a, b])
    message = str(excinfo.value)
    assert "spatial shape mismatch" in message
    assert "8x8" in message and "4x4" in message


def test_concat_rejects_unaligned_non_final_operand() -> None:
    m = Model(seed=9)
    x = m.input(4, 4, 8)
    a = m.conv2d(x, 4, padding=(1, 1, 1, 1))
    b = m.conv2d(x, 8, padding=(1, 1, 1, 1))
    with pytest.raises(ValueError) as excinfo:
        m.concat([a, b])
    message = str(excinfo.value)
    assert "4 channels" in message
    assert "Only the LAST operand may be unaligned" in message


def test_concat_allows_unaligned_final_operand() -> None:
    """Only the last operand's padding lanes coincide with the result's
    own zero padding, so only it may be unaligned."""
    m = Model(seed=10)
    x = m.input(4, 4, 8)
    a = m.conv2d(x, 8, padding=(1, 1, 1, 1))
    b = m.conv2d(x, 4, padding=(1, 1, 1, 1))
    y = m.concat([a, b])
    assert y.channels == 12


def test_concat_of_one_tensor_is_rejected() -> None:
    m = Model(seed=11)
    x = m.input(4, 4, 8)
    a = m.conv2d(x, 8, padding=(1, 1, 1, 1))
    with pytest.raises(ValueError, match="at least two tensors"):
        m.concat([a])


def test_marking_an_aliased_tensor_a_graph_output_is_rejected() -> None:
    """A graph output must own a DDR `OUTPUTS` allocation, which an alias
    cannot: it lives inside somebody else's buffer."""
    m = Model(seed=12)
    x = m.input(4, 4, 8)
    a = m.conv2d(x, 8, padding=(1, 1, 1, 1))
    b = m.conv2d(x, 8, padding=(1, 1, 1, 1))
    m.concat([a, b])
    with pytest.raises(ValueError) as excinfo:
        m.output(a)
    assert "model.copy" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 3. Numerics -- the values really are a concatenation / a slice.
# ---------------------------------------------------------------------------


def test_two_way_concat_values() -> None:
    m = Model(seed=20)
    x = m.input(5, 5, 8, name="x")
    a = m.conv2d(x, 8, padding=(1, 1, 1, 1), name="a")
    b = m.conv2d(x, 16, padding=(1, 1, 1, 1), name="b")
    y = m.concat([a, b], name="y")
    m.output(m.conv2d(y, 8, padding=(1, 1, 1, 1), name="z"))

    _, result = _run(m)
    expected = _hwc_concat(5 * 5, [(result.tensor_data["a"], 8), (result.tensor_data["b"], 16)])
    assert result.tensor_data["y"] == expected


def test_four_way_concat_values_sppf_shape() -> None:
    """SPPF: one stem plus three successive 5x5 max pools, concatenated."""
    m = Model(seed=21)
    x = m.input(6, 6, 8, name="x")
    stem = m.conv2d(x, 8, padding=(1, 1, 1, 1), name="stem")
    p1 = m.pool_max(stem, kernel=(5, 5), stride=(1, 1), padding=(2, 2, 2, 2), pad_value=-128, name="p1")
    p2 = m.pool_max(p1, kernel=(5, 5), stride=(1, 1), padding=(2, 2, 2, 2), pad_value=-128, name="p2")
    p3 = m.pool_max(p2, kernel=(5, 5), stride=(1, 1), padding=(2, 2, 2, 2), pad_value=-128, name="p3")
    y = m.concat([stem, p1, p2, p3], name="y")
    m.output(m.conv2d(y, 8, padding=(1, 1, 1, 1), name="z"))

    _, result = _run(m)
    assert result.tensor_data["y"] == _hwc_concat(
        6 * 6,
        [(result.tensor_data[n], 8) for n in ("stem", "p1", "p2", "p3")],
    )


def test_split_values_are_channel_slices() -> None:
    m = Model(seed=22)
    x = m.input(5, 5, 8, name="x")
    h = m.conv2d(x, 24, padding=(1, 1, 1, 1), name="h")
    lo, hi = m.split(h, [8, 16], names=["lo", "hi"])
    m.output(m.add(m.conv2d(lo, 8, padding=(1, 1, 1, 1), name="cl"),
                   m.conv2d(hi, 8, padding=(1, 1, 1, 1), name="ch"), name="y"))

    _, result = _run(m)
    h_values = result.tensor_data["h"]
    assert result.tensor_data["lo"] == _hwc_slice(h_values, 5 * 5, 24, 0, 8)
    assert result.tensor_data["hi"] == _hwc_slice(h_values, 5 * 5, 24, 8, 16)


def test_concat_with_unaligned_last_operand_values() -> None:
    m = Model(seed=23)
    x = m.input(4, 4, 8, name="x")
    a = m.conv2d(x, 8, padding=(1, 1, 1, 1), name="a")
    b = m.conv2d(x, 4, padding=(1, 1, 1, 1), name="b")
    y = m.concat([a, b], name="y")
    m.output(m.conv2d(y, 8, padding=(1, 1, 1, 1), name="z"))

    _, result = _run(m)
    assert result.tensor_data["y"] == _hwc_concat(
        4 * 4, [(result.tensor_data["a"], 8), (result.tensor_data["b"], 4)]
    )


def test_nested_concat_values() -> None:
    """`concat([concat([a, b]), c])`: the inner result is itself placed
    inside the outer buffer, so `a` ends up two alias hops from its
    root."""
    m = Model(seed=24)
    x = m.input(4, 4, 8, name="x")
    a = m.conv2d(x, 8, padding=(1, 1, 1, 1), name="a")
    b = m.conv2d(x, 8, padding=(1, 1, 1, 1), name="b")
    c = m.conv2d(x, 8, padding=(1, 1, 1, 1), name="c")
    inner = m.concat([a, b], name="inner")
    outer = m.concat([inner, c], name="outer")
    m.output(m.conv2d(outer, 8, padding=(1, 1, 1, 1), name="z"))

    assert alias_root(a) is outer
    assert alias_byte_offset(a) == 0
    assert alias_byte_offset(b) == 1 * 8 * 4 * 4
    assert alias_byte_offset(c) == 2 * 8 * 4 * 4

    _, result = _run(m)
    assert result.tensor_data["outer"] == _hwc_concat(
        4 * 4, [(result.tensor_data[n], 8) for n in ("a", "b", "c")]
    )


# ---------------------------------------------------------------------------
# 4. The zero-traffic invariant -- the point of the whole design.
# ---------------------------------------------------------------------------


def _split_concat_roundtrip(with_roundtrip: bool, seed: int = 30):
    """The same two-convolution network, with and without a
    `concat(split(h))` inserted between them. The round trip is the
    identity on `h`, so if concat/split really are free the two programs
    must be *indistinguishable*."""
    m = Model(seed=seed)
    x = m.input(6, 6, 8, name="x")
    h = m.conv2d(x, 16, padding=(1, 1, 1, 1), name="h")
    if with_roundtrip:
        lo, hi = m.split(h, [8, 8], names=["lo", "hi"])
        h = m.concat([lo, hi], name="rejoined")
    m.output(m.conv2d(h, 8, padding=(1, 1, 1, 1), name="z"))
    return m


def test_split_concat_roundtrip_costs_exactly_nothing() -> None:
    """`DDR_WR_BYTES`/`DDR_RD_BYTES` -- and every other traffic counter,
    and the instruction count -- are byte-for-byte what the same network
    without the concat costs. If concat ever silently cost a copy, this
    is the test that fails."""
    plain = _plan(_split_concat_roundtrip(False))
    aliased = _plan(_split_concat_roundtrip(True))

    assert aliased.traffic == plain.traffic
    assert aliased.traffic.write_bytes == plain.traffic.write_bytes
    assert aliased.traffic.read_bytes == plain.traffic.read_bytes
    assert len(aliased.steps) == len(plain.steps)
    assert aliased.traffic.tensor_load_count == 0
    assert aliased.traffic.tensor_store_count == 0
    assert not any(isinstance(s, MoveStep) for s in aliased.steps)
    assert not any(isinstance(s.op, CopyOp) for s in aliased.steps if isinstance(s, ComputeStep))


def test_split_concat_roundtrip_is_numerically_the_identity() -> None:
    plain_planned, plain = _run(_split_concat_roundtrip(False))
    alias_planned, aliased = _run(_split_concat_roundtrip(True))
    assert aliased.tensor_data["z"] == plain.tensor_data["z"]
    assert aliased.tensor_data["rejoined"] == plain.tensor_data["h"]
    assert alias_planned.traffic == plain_planned.traffic


def test_concat_adds_no_instruction_and_no_ddr_byte() -> None:
    """A two-way concat between two producers and one consumer: exactly
    three compute steps, one cold load of the graph input, and the only
    DDR write is the graph output."""
    m = Model(seed=31)
    x = m.input(6, 6, 8, name="x")
    a = m.conv2d(x, 8, padding=(1, 1, 1, 1), name="a")
    b = m.conv2d(x, 8, padding=(1, 1, 1, 1), name="b")
    y = m.concat([a, b], name="y")
    z = m.conv2d(y, 8, padding=(1, 1, 1, 1), name="z")
    m.output(z)

    planned, result = _run(m)
    compute = [s for s in planned.steps if isinstance(s, ComputeStep)]
    assert [s.op.name for s in compute] == ["a", "b", "z"]
    assert planned.traffic.write_bytes == z.size_bytes
    assert planned.traffic.tensor_store_count == 0
    assert planned.traffic == result.traffic

    # The two producers wrote adjacent slices of one buffer, in the same
    # space -- that IS the concatenation.
    a_step, b_step = compute[0], compute[1]
    assert a_step.output_space == b_step.output_space == isa.SPACE_LOCAL_TENSOR
    assert b_step.output_addr - a_step.output_addr == a.size_bytes
    assert compute[2].input_addrs[0] == a_step.output_addr


def test_copy_fallback_costs_exactly_the_copies_and_nothing_else() -> None:
    """Quantifies the fallback: forcing two operands through `COPY` adds
    exactly two instructions (2 x 64 descriptor bytes) and two local
    round trips, and not one byte of DDR data traffic."""
    def build(swap: bool) -> Model:
        m = Model(seed=32)
        x = m.input(6, 6, 8, name="x")
        h = m.conv2d(x, 16, padding=(1, 1, 1, 1), name="h")
        lo, hi = m.split(h, [8, 8], names=["lo", "hi"])
        joined = m.concat([hi, lo] if swap else [lo, hi], name="y")
        m.output(m.conv2d(joined, 8, padding=(1, 1, 1, 1), name="z"))
        return m

    free = _plan(build(False))
    copied = _plan(build(True))

    n_copies = sum(1 for s in copied.steps if isinstance(s, ComputeStep) and isinstance(s.op, CopyOp))
    assert n_copies == 2
    assert len(copied.steps) == len(free.steps) + 2
    # Only the extra descriptor fetches differ on DDR; no tensor byte moves.
    assert copied.traffic.read_bytes == free.traffic.read_bytes + 2 * isa.INSTR_WORD_BYTES
    assert copied.traffic.write_bytes == free.traffic.write_bytes
    assert copied.traffic.tensor_load_count == free.traffic.tensor_load_count
    assert copied.traffic.tensor_store_count == 0


# ---------------------------------------------------------------------------
# 5. Planner: aliasing addresses and liveness.
# ---------------------------------------------------------------------------


def _addr_of(planned, tensor_name: str) -> int:
    for step in planned.steps:
        if isinstance(step, ComputeStep) and step.op.output.name == tensor_name:
            return step.output_addr
    raise AssertionError(f"no step produces {tensor_name}")


def test_producers_write_adjacent_slices_of_one_buffer() -> None:
    m = Model(seed=40)
    x = m.input(6, 6, 8, name="x")
    a = m.conv2d(x, 8, padding=(1, 1, 1, 1), name="a")
    b = m.conv2d(x, 16, padding=(1, 1, 1, 1), name="b")
    c = m.conv2d(x, 8, padding=(1, 1, 1, 1), name="c")
    y = m.concat([a, b, c], name="y")
    m.output(m.conv2d(y, 8, padding=(1, 1, 1, 1), name="z"))

    planned = _plan(m)
    base = _addr_of(planned, "a")
    assert _addr_of(planned, "b") == base + a.size_bytes
    assert _addr_of(planned, "c") == base + a.size_bytes + b.size_bytes


def test_split_views_are_sub_ranges_of_the_parent() -> None:
    m = Model(seed=41)
    x = m.input(6, 6, 8, name="x")
    h = m.conv2d(x, 24, padding=(1, 1, 1, 1), name="h")
    lo, mid_hi = m.split(h, [8, 16], names=["lo", "hi"])
    m.output(m.add(m.conv2d(lo, 8, padding=(1, 1, 1, 1), name="cl"),
                   m.conv2d(mid_hi, 8, padding=(1, 1, 1, 1), name="ch"), name="y"))

    planned = _plan(m)
    h_addr = _addr_of(planned, "h")
    steps = {s.op.name: s for s in planned.steps if isinstance(s, ComputeStep)}
    assert steps["cl"].input_addrs[0] == h_addr
    assert steps["ch"].input_addrs[0] == h_addr + lo.size_bytes


def test_concat_buffer_is_not_recycled_while_a_slice_is_still_live() -> None:
    """The liveness rule. `a` is produced first and read last (through
    the concat `y`), with an unrelated tensor allocated in between: if
    the planner freed `a`'s buffer when `a`'s own consumer count hit
    zero, that unrelated tensor would land on top of the concat buffer
    and corrupt it."""
    m = Model(seed=42)
    x = m.input(6, 6, 8, name="x")
    a = m.conv2d(x, 8, padding=(1, 1, 1, 1), name="a")
    b = m.conv2d(x, 8, padding=(1, 1, 1, 1), name="b")
    y = m.concat([a, b], name="y")
    # Two more tensors allocated *after* the concat buffer is only
    # half-written and while it is still live.
    t1 = m.conv2d(x, 8, padding=(1, 1, 1, 1), name="t1")
    t2 = m.conv2d(t1, 8, padding=(1, 1, 1, 1), name="t2")
    z = m.conv2d(y, 8, padding=(1, 1, 1, 1), name="z")
    m.output(m.add(z, t2, name="out"))

    planned = _plan(m)
    y_base = _addr_of(planned, "a")
    y_range = (y_base, y_base + y.size_bytes)
    for name in ("t1", "t2", "z"):
        addr = _addr_of(planned, name)
        size = {"t1": t1, "t2": t2, "z": z}[name].size_bytes
        assert addr + size <= y_range[0] or addr >= y_range[1], (
            f"'{name}' at 0x{addr:x}+{size} overlaps the still-live concat buffer "
            f"'{y.name}' at 0x{y_range[0]:x}..0x{y_range[1]:x}"
        )

    # And the numbers come out right, which is the real proof.
    _, result = _run(m)
    assert result.traffic == planned.traffic


def test_split_parent_is_not_recycled_while_a_view_is_still_live() -> None:
    m = Model(seed=43)
    x = m.input(6, 6, 8, name="x")
    h = m.conv2d(x, 16, padding=(1, 1, 1, 1), name="h")
    lo, hi = m.split(h, [8, 8], names=["lo", "hi"])
    # `h` itself has no consumers at all -- only its two views do. A
    # per-tensor liveness rule would free it immediately.
    first = m.conv2d(lo, 8, padding=(1, 1, 1, 1), name="first")
    second = m.conv2d(hi, 8, padding=(1, 1, 1, 1), name="second")
    m.output(m.add(first, second, name="y"))

    planned = _plan(m)
    h_addr = _addr_of(planned, "h")
    for name, tensor in (("first", first), ("second", second)):
        addr = _addr_of(planned, name)
        assert addr + tensor.size_bytes <= h_addr or addr >= h_addr + h.size_bytes, (
            f"'{name}' overlaps the split parent '{h.name}', whose views are still live"
        )
    _, result = _run(m)
    assert result.traffic == planned.traffic


def test_concat_operand_that_is_also_consumed_elsewhere_stays_free() -> None:
    """A fork: `a` feeds both the concat and an unrelated convolution.
    Reading a tensor from inside a concat buffer is just an address, so
    this needs no copy."""
    m = Model(seed=44)
    x = m.input(6, 6, 8, name="x")
    a = m.conv2d(x, 8, padding=(1, 1, 1, 1), name="a")
    b = m.conv2d(x, 8, padding=(1, 1, 1, 1), name="b")
    y = m.concat([a, b], name="y")
    side = m.conv2d(a, 8, padding=(1, 1, 1, 1), name="side")
    main = m.conv2d(y, 8, padding=(1, 1, 1, 1), name="main")
    m.output(m.add(side, main, name="out"))

    planned, result = _run(m)
    assert not any(isinstance(s, ComputeStep) and isinstance(s.op, CopyOp) for s in planned.steps)
    steps = {s.op.name: s for s in planned.steps if isinstance(s, ComputeStep)}
    assert steps["side"].input_addrs[0] == _addr_of(planned, "a")
    assert steps["main"].input_addrs[0] == _addr_of(planned, "a")
    assert result.traffic == planned.traffic


def test_graph_input_operand_falls_back_to_copy() -> None:
    """A graph input's address is fixed by the DDR `INPUTS` region, so it
    cannot be moved into a concat buffer."""
    m = Model(seed=45)
    x = m.input(6, 6, 8, name="x")
    a = m.conv2d(x, 8, padding=(1, 1, 1, 1), name="a")
    y = m.concat([x, a], name="y")
    m.output(m.conv2d(y, 8, padding=(1, 1, 1, 1), name="z"))

    assert x.alias_parent is None, "a graph input is never re-homed"
    assert len(y.alias_parts) == 2
    copy_part = y.alias_parts[0]
    assert isinstance(copy_part.producer, CopyOp)
    assert copy_part.producer.inputs[0] is x

    planned, result = _run(m)
    assert result.traffic == planned.traffic
    assert result.tensor_data["y"] == _hwc_concat(
        6 * 6, [(result.tensor_data["x"], 8), (result.tensor_data["a"], 8)]
    )


def test_same_tensor_twice_in_one_concat_copies_the_second_occurrence() -> None:
    m = Model(seed=46)
    x = m.input(6, 6, 8, name="x")
    a = m.conv2d(x, 8, padding=(1, 1, 1, 1), name="a")
    y = m.concat([a, a], name="y")
    m.output(m.conv2d(y, 8, padding=(1, 1, 1, 1), name="z"))

    assert y.alias_parts[0] is a
    assert isinstance(y.alias_parts[1].producer, CopyOp)

    planned, result = _run(m)
    a_values = result.tensor_data["a"]
    assert result.tensor_data["y"] == _hwc_concat(6 * 6, [(a_values, 8), (a_values, 8)])
    assert result.traffic == planned.traffic


def test_concat_result_as_graph_output_lands_in_ddr_outputs() -> None:
    m = Model(seed=47)
    x = m.input(6, 6, 8, name="x")
    a = m.conv2d(x, 8, padding=(1, 1, 1, 1), name="a")
    b = m.conv2d(x, 8, padding=(1, 1, 1, 1), name="b")
    y = m.concat([a, b], name="y")
    m.output(y)

    planned, result = _run(m)
    base = planned.tensor_ddr_addr["y"]
    steps = {s.op.name: s for s in planned.steps if isinstance(s, ComputeStep)}
    assert steps["a"].output_space == isa.SPACE_DDR
    assert steps["a"].output_addr == base
    assert steps["b"].output_addr == base + a.size_bytes
    # Exactly the concat's own bytes, written once, by its producers.
    assert planned.traffic.write_bytes == y.size_bytes
    assert planned.traffic.tensor_store_count == 0
    assert result.traffic == planned.traffic


def test_alias_family_spills_and_reloads_as_one_buffer() -> None:
    """Under memory pressure a concat buffer is spilled whole: its slices
    are byte ranges of it and have no separate existence."""
    m = Model(seed=48)
    x = m.input(8, 8, 8, name="x")
    a = m.conv2d(x, 8, padding=(1, 1, 1, 1), name="a")
    b = m.conv2d(x, 8, padding=(1, 1, 1, 1), name="b")
    y = m.concat([a, b], name="y")
    filler = m.conv2d(x, 8, padding=(1, 1, 1, 1), name="filler")
    m.output(m.add(m.conv2d(y, 8, padding=(1, 1, 1, 1), name="z"),
                   m.conv2d(filler, 8, padding=(1, 1, 1, 1), name="f2"), name="out"))

    planned = Planner(tensor_mem_bytes=1792).plan(m)
    spills = [s for s in planned.steps if isinstance(s, MoveStep) and s.kind == "spill"]
    spilled_y = [s for s in spills if s.tensor.name == "y"]
    assert spilled_y, "this budget is meant to force the concat buffer out"
    assert all(s.nbytes == y.size_bytes for s in spilled_y), "a concat buffer must spill whole"
    reloads = [s for s in planned.steps if isinstance(s, MoveStep) and s.kind == "reload"]
    assert any(s.tensor.name == "y" and s.nbytes == y.size_bytes for s in reloads)

    result = run_reference(planned, MemoryImage())
    assert result.traffic == planned.traffic
    ample = run_reference(_plan(m), MemoryImage())
    assert result.tensor_data["out"] == ample.tensor_data["out"]


def test_a_half_written_concat_buffer_is_never_spilled() -> None:
    """The nastiest liveness case: `a` is written into the concat buffer,
    then three unrelated commands run, and only then is `b` written into
    the same buffer. Evicting that buffer in between would spill it,
    `b`'s producer would write to a freshly allocated buffer instead, and
    `a`'s half would be silently lost -- there is no reload on a write
    path to bring it back.

    Swept across a range of budgets: whatever the pressure, the planner
    must never spill a buffer that still has a producer ahead of it, and
    the answer must never change. Under pressure it puts the concat
    buffer in DDR instead (`PlannedProgram.ddr_placements`), where the
    same slice addresses still work because the activation layout is
    identical in both spaces -- it used to refuse outright at the tight
    end of this sweep.
    """

    def build() -> Model:
        m = Model(seed=50)
        x = m.input(8, 8, 8, name="x")
        a = m.conv2d(x, 8, padding=(1, 1, 1, 1), name="a")
        t1 = m.conv2d(x, 8, padding=(1, 1, 1, 1), name="t1")
        t2 = m.conv2d(t1, 8, padding=(1, 1, 1, 1), name="t2")
        b = m.conv2d(t2, 8, padding=(1, 1, 1, 1), name="b")
        y = m.concat([a, b], name="y")
        m.output(m.conv2d(y, 8, padding=(1, 1, 1, 1), name="z"))
        return m

    golden_values = _run(build())[1].tensor_data["z"]

    degraded = 0
    for budget in (8192, 4096, 3072, 2560, 2048, 1792, 1536, 1024):
        planned = Planner(tensor_mem_bytes=budget).plan(build())
        if any(name == "y" for name, _, _ in planned.ddr_placements):
            degraded += 1
        for step in planned.steps:
            if isinstance(step, MoveStep) and step.kind == "spill":
                assert step.tensor.name != "y", (
                    f"budget {budget}: spilled the concat buffer 'y' while its second "
                    "part had not been written yet"
                )
        result = run_reference(planned, MemoryImage())
        assert result.traffic == planned.traffic
        assert result.tensor_data["z"] == golden_values, f"budget {budget} changed the result"
    assert degraded, (
        "budget sweep never got tight enough to push 'y' into DDR -- it proves "
        "nothing about the half-written-concat rule under real pressure"
    )
