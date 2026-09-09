"""Tests for the two planner extensions spatial tiling needs before any
tiling exists: **unit-granular bank confinement** and **DDR-pinned
tensors** (`tiling` design section 2.3/2.4, implementation plan step 3).

Both are relaxations, and a relaxation is exactly the kind of change that
is invisible until it is wrong: `cnn_accel_tensor_mem` *clamps* a request
that runs past its bank rather than refusing it, so a buffer placed at an
illegal address produces a silently truncated transfer whose tail is
stale RAM. Neither `reference.py` (a flat `bytearray`) nor a value-only
DUT comparison can see that, because the program and the reference take
their addresses from the same planner. Every assertion here is therefore
about the *geometry* of a placement, computed independently of the
planner's own reasoning.

`test_tiling_guard.py` is the other half: the relaxation must move no
existing placement at all.
"""

from __future__ import annotations

import random

import pytest

import cnn_accel_model as golden

from accel_v2 import isa
from accel_v2.model import Activation, Model
from accel_v2.planner import _LocalAllocator, DEFAULT_BANK_BYTES, Planner


# ---------------------------------------------------------------------------
# The allocator, in isolation.
# ---------------------------------------------------------------------------


def _units(addr: int, size: int, unit: int) -> list[tuple[int, int]]:
    """The `[lo, hi)` byte ranges the hardware will request against a
    buffer of `size` bytes placed at `addr` whose largest single request
    is `unit` bytes. The last one is short when `unit` does not divide
    `size`. Deliberately re-derived here rather than imported: this is
    the independent statement of what "confined" means, and importing the
    planner's own idea of it would make the test agree with the bug."""
    ranges = []
    offset = 0
    while offset < size:
        ranges.append((addr + offset, addr + min(offset + unit, size)))
        offset += unit
    return ranges


def _confined(addr: int, size: int, unit: int, bank_bytes: int) -> bool:
    return all(lo // bank_bytes == (hi - 1) // bank_bytes for lo, hi in _units(addr, size, unit))


def test_unit_confinement_holds_for_every_placement_of_a_random_workload() -> None:
    """Property: whatever sequence of allocations and frees it is given,
    every address the allocator returns confines every `unit` to one
    bank, and no two live buffers overlap."""
    rng = random.Random(20260909)
    bank_bytes = 512
    for _ in range(200):
        alloc = _LocalAllocator(capacity=8 * bank_bytes, bank_bytes=bank_bytes)
        live: list[tuple[int, int, int]] = []
        for _ in range(30):
            if live and rng.random() < 0.3:
                addr, size, _unit = live.pop(rng.randrange(len(live)))
                alloc.free(addr, size)
                continue
            unit = 8 * rng.randint(1, bank_bytes // 8)
            planes = rng.randint(1, 6)
            size = unit * planes
            addr = alloc.try_alloc(size, unit=unit)
            if addr is None:
                continue
            assert addr % 8 == 0, "allocator broke 8-byte alignment"
            assert 0 <= addr and addr + size <= alloc.capacity
            assert _confined(addr, size, unit, bank_bytes), (
                f"{size} bytes at {addr} in units of {unit} straddle a {bank_bytes}-byte bank"
            )
            for other_addr, other_size, _u in live:
                assert addr + size <= other_addr or other_addr + other_size <= addr, (
                    "allocator handed out overlapping buffers"
                )
            live.append((addr, size, unit))


def test_a_multi_plane_buffer_may_exceed_a_bank_when_its_planes_do_not() -> None:
    """The whole point of the relaxation: a buffer four times a bank is
    placeable when its request unit is a quarter of a bank, and the
    hardware's real constraint (no *request* straddles) still holds."""
    bank_bytes = 1024
    alloc = _LocalAllocator(capacity=4 * bank_bytes, bank_bytes=bank_bytes)
    unit = bank_bytes // 4
    addr = alloc.try_alloc(16 * unit, unit=unit)
    assert addr is not None
    assert _confined(addr, 16 * unit, unit, bank_bytes)
    # ...and the strict rule would have refused it outright.
    strict = _LocalAllocator(capacity=4 * bank_bytes, bank_bytes=bank_bytes)
    assert strict.try_alloc(16 * unit) is None


def test_a_unit_larger_than_a_bank_is_refused_however_small_the_buffer() -> None:
    """`unit > bank_bytes` has no legal address at any occupancy -- the
    hardware would clamp that one request wherever it sat."""
    alloc = _LocalAllocator(capacity=8192, bank_bytes=1024)
    assert alloc.try_alloc(2048, unit=2048) is None
    assert alloc.try_alloc(2048, unit=1024) is not None


def test_a_straddling_unit_slides_the_buffer_not_the_whole_bank() -> None:
    """The placement rule is "bump so the offending unit *starts* on the
    boundary", not "align the buffer to a bank": a two-plane buffer that
    would straddle at its own midpoint moves by half a plane, not by a
    whole bank, and the bytes it skips stay usable.

    Worked by hand: banks of 1024, a 384-byte unit. A 768-byte buffer at
    768 would put unit 0 at [768, 1152) -- across the boundary at 1024 --
    so unit 0 is slid to start at 1024 and the buffer lands at 1024."""
    alloc = _LocalAllocator(capacity=4096, bank_bytes=1024)
    alloc._free = [(768, 4096 - 768)]
    addr = alloc.try_alloc(768, unit=384)
    assert addr == 1024
    # Unit 1 then sits at [1408, 1792), still inside bank 1. Nothing was
    # rounded up to a bank base.
    assert _confined(addr, 768, 384, 1024)


def test_the_skipped_hole_below_a_boundary_is_handed_to_the_next_buffer() -> None:
    """A slid buffer leaves its skipped bytes on the free list, exactly
    as the strict rule's own "skip to the next bank boundary" does."""
    alloc = _LocalAllocator(capacity=4096, bank_bytes=1024)
    alloc._free = [(768, 4096 - 768)]
    assert alloc.try_alloc(768, unit=384) == 1024
    assert alloc.try_alloc(256, unit=256) == 768


def test_unit_equal_to_size_reproduces_the_strict_rule_exactly() -> None:
    """The equivalence the untiled catalogue rests on, checked
    structurally rather than by digest: against a re-implementation of
    the *original* single-bump algorithm, over randomized free lists, the
    unit-granular allocator must return the identical address every
    time."""

    def strict_try_alloc(free, size, bank_bytes, align=8):
        """The algorithm as it stood before unit granularity."""
        for i, (start, length) in enumerate(free):
            aligned_start = start + (-start) % align
            bank_end = (aligned_start // bank_bytes + 1) * bank_bytes
            if aligned_start + size > bank_end:
                aligned_start = bank_end
            pad = aligned_start - start
            if length - pad < size:
                continue
            return aligned_start
        return None

    rng = random.Random(4242)
    bank_bytes = 256
    for _ in range(3000):
        free = []
        cursor = 0
        while cursor < 2048 and rng.random() < 0.8:
            cursor += 8 * rng.randint(0, 20)
            length = 8 * rng.randint(1, 40)
            if cursor + length > 2048:
                break
            free.append((cursor, length))
            cursor += length
        size = 8 * rng.randint(1, bank_bytes // 8)
        alloc = _LocalAllocator(capacity=2048, bank_bytes=bank_bytes)
        alloc._free = list(free)
        assert alloc.try_alloc(size) == strict_try_alloc(free, size, bank_bytes), (
            f"unit-granular allocator diverged from the strict rule for size={size} "
            f"free={free}"
        )


# ---------------------------------------------------------------------------
# The planner's use of the two new `Tensor` fields.
# ---------------------------------------------------------------------------


def _chain(seed: int = 3) -> Model:
    m = Model(seed=seed)
    x = m.input(8, 8, 16, name="x")
    h = m.conv2d(x, 16, kernel=(3, 3), padding=(1, 1, 1, 1), activation=Activation.RELU, name="h")
    y = m.conv2d(h, 16, kernel=(3, 3), padding=(1, 1, 1, 1), activation=Activation.RELU, name="y")
    m.output(y)
    return m


def test_confine_unit_bytes_lets_the_planner_place_a_buffer_bigger_than_a_bank() -> None:
    """A 1024-byte two-plane tensor with 512-byte banks: unplaceable
    under the strict rule (it goes to DDR), placeable when the planner is
    told the largest request against it is one 512-byte plane."""
    strict = Planner(tensor_mem_bytes=4096, bank_bytes=512).plan(_chain())
    assert [name for name, _, _ in strict.ddr_placements] == ["h"]

    model = _chain()
    for t in model.tensors:
        t.confine_unit_bytes = t.plane_bytes
    relaxed = Planner(tensor_mem_bytes=4096, bank_bytes=512).plan(model)
    assert relaxed.ddr_placements == []
    for _, addr, size in relaxed.local_placements:
        assert _confined(addr, size, 512, 512)


def test_pinned_tensors_go_straight_to_ddr_and_are_counted_apart() -> None:
    """`pin_ddr` takes `place_in_ddr` immediately -- no eviction sweep,
    no dependence on pressure -- and its bytes land in `pinned_*_bytes`,
    NOT in `ddr_resident_*_bytes`, so "did anything overflow?" stays a
    separate question from "what did the group boundary cost?"."""
    model = _chain()
    pinned = next(t for t in model.tensors if t.name == "h")
    pinned.pin_ddr = True

    # A scratchpad with room to spare: nothing is under pressure, so a
    # DDR placement here can only be the pin.
    planned = Planner(tensor_mem_bytes=64 * 1024, bank_bytes=8192).plan(model)

    assert planned.ddr_placements == [], "pinning must not look like an overflow fallback"
    assert [name for name, _, _ in planned.pinned_placements] == ["h"]
    assert planned.traffic.ddr_resident_read_bytes == 0
    assert planned.traffic.ddr_resident_write_bytes == 0
    assert planned.traffic.pinned_write_bytes == pinned.size_bytes
    # Read `n_ot` times, not once: 'y' has 16 output channels, so the
    # hardware runs two output-channel passes and re-streams the whole
    # ifmap for each (`planner.ifmap_passes`). Written once -- the output
    # side of a pass writes one *plane*, so the ofmap is written exactly
    # once in total.
    n_ot = -(-16 // golden.PE_ROWS)
    assert n_ot == 2
    assert planned.traffic.pinned_read_bytes == n_ot * pinned.size_bytes

    steps = [s for s in planned.steps if hasattr(s, "op")]
    produce = next(s for s in steps if s.op.output.name == "h")
    consume = next(s for s in steps if s.op.name == "y")
    assert produce.output_space == isa.SPACE_DDR
    assert consume.input_spaces[0] == isa.SPACE_DDR


def test_a_pinned_graph_output_is_allocated_in_the_outputs_region() -> None:
    """A group output that is also a graph output must keep its `OUTPUTS`
    address -- that is the window the testbench exports and the host
    reads. Pinning must not divert it to the `SPILL` arena."""
    from accel_v2.ddrmap import DdrMap

    model = _chain()
    out = model.outputs[0]
    out.pin_ddr = True
    planned = Planner(tensor_mem_bytes=64 * 1024, bank_bytes=8192).plan(model)
    addr = planned.tensor_ddr_addr[out.name]
    ddr_map = planned.ddr_map
    start = DdrMap.OUTPUTS * ddr_map.scale
    assert start <= addr < ddr_map.limit


def test_planner_rejects_a_confinement_unit_larger_than_a_bank_by_falling_back() -> None:
    """`unit > bank_bytes` is unplaceable, and the planner degrades to
    DDR rather than handing out an address the hardware would clamp."""
    model = _chain()
    for t in model.tensors:
        t.confine_unit_bytes = t.size_bytes  # 1024 bytes, banks are 512
    planned = Planner(tensor_mem_bytes=4096, bank_bytes=512).plan(model)
    assert [name for name, _, _ in planned.ddr_placements] == ["h"]


def test_add_planewise_matches_a_plain_add_value_for_value() -> None:
    """The per-plane ADD lowering is an emission detail, not arithmetic:
    splitting along planes and re-concatenating must reproduce the single
    whole-tensor `ADD` exactly."""
    from accel_v2.memimage import MemoryImage
    from accel_v2.reference import run_reference

    def run(planewise: bool) -> list[int]:
        m = Model(seed=11)
        a = m.input(4, 4, 24, name="a")
        b = m.input(4, 4, 24, name="b")
        s = (m.add_planewise if planewise else m.add)(a, b, name="s")
        m.output(m.copy(s, name="out") if planewise else s)
        planned = Planner(tensor_mem_bytes=32 * 1024, bank_bytes=8192).plan(m)
        return run_reference(planned, MemoryImage()).tensor_data["s"]

    assert run(planewise=True) == run(planewise=False)


def test_add_planewise_emits_one_add_per_plane() -> None:
    """...and that it really is one request per plane, which is the whole
    reason it exists (`cnn_accel_elementwise` issues an `ADD` as a single
    whole-tensor request, and a multi-plane strip buffer may span banks)."""
    from accel_v2.model import AddOp

    m = Model(seed=12)
    a = m.input(4, 4, 24, name="a")
    b = m.input(4, 4, 24, name="b")
    m.output(m.copy(m.add_planewise(a, b, name="s"), name="out"))
    adds = [op for op in m.ops if isinstance(op, AddOp)]
    assert len(adds) == 3
    assert all(op.inputs[0].channels == 8 for op in adds)


def test_add_planewise_is_a_plain_add_for_a_single_plane_tensor() -> None:
    """No split/concat churn where there is nothing to split -- which is
    why the untiled catalogue is unaffected by this helper existing."""
    from accel_v2.model import AddOp

    m = Model(seed=13)
    a = m.input(4, 4, 8, name="a")
    b = m.input(4, 4, 8, name="b")
    m.add_planewise(a, b, name="s")
    assert [type(op) for op in m.ops] == [AddOp]


@pytest.mark.parametrize("bad_unit", [0, -8])
def test_a_non_positive_confinement_unit_is_a_loud_error(bad_unit: int) -> None:
    alloc = _LocalAllocator(capacity=1024, bank_bytes=1024)
    with pytest.raises(ValueError, match="confinement unit"):
        alloc.try_alloc(64, unit=bad_unit)
