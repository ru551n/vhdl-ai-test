"""Tests for the planner's DDR-placement fallback: what happens to a
buffer that cannot live in `cnn_accel_tensor_mem` at all.

The motivating shape is YOLOv8's `C2f`. Its concat family is written by
several producers scattered across the block and stays live until the
closing `cv2`, so it is deliberately never an eviction victim
(`planner.evict_one`: spilling a half-written concat buffer would lose
the slices already in it, because nothing reloads on a write path). With
no defragmenter and no eviction available, a `C2f` whose concat family
does not fit had no lowering at all and `Planner.plan` raised.

The fallback is to place such a buffer in DDR instead. That is correct,
not a fudge: arch doc section 3 defines ONE activation layout, used
byte-identically in DDR and in `LOCAL_TENSOR`, and operand space is a
per-operand tag on each instruction. So a concat family in DDR needs no
new mechanism -- each part's producer writes its slice to `base +
plane_offset * H*W*T` with `space_dst = DDR`, and the consumer reads the
assembled tensor with `space_src0 = DDR`. It costs bandwidth and nothing
else.

`tests/test_planner.py` covers the two unplaceability *rules* in
isolation (bigger than one bank; nothing evictable frees enough). This
file covers the behaviour that matters: a real `C2f` degrades smoothly
across the whole range of scratchpad sizes instead of falling off a
cliff, the answer never changes, and no program that already fitted
moves.
"""

from __future__ import annotations

import pytest

from accel_v2 import (
    cases,
    cases_concat_split,
    cases_conv_pad,
    cases_pool_pad,
    cases_yolo,
    isa,
)
from accel_v2.ddrmap import DdrMap
from accel_v2.memimage import MemoryImage
from accel_v2.model import Model, alias_root
from accel_v2.planner import ComputeStep, MoveStep, PlannedProgram, Planner
from accel_v2.reference import run_reference

#: Scratchpad size at and above which `_c2f_model()` fits entirely on
#: chip, measured. Below it the old planner raised `ValueError`; the
#: sweep in `test_c2f_degradation_curve` walks straight through it.
C2F_FITS_LOCALLY_FROM = 3584


def _c2f_model() -> Model:
    """`case_c2f_n1`'s graph, built through the same block builder the
    `tb_cnn_accel_top` catalogue uses, so this file cannot drift from
    what the DUT actually runs."""
    m = Model(seed=403, name="c2f")
    x = m.input(8, 8, 8, name="x")
    stem = m.conv2d(x, 8, kernel=(3, 3), padding=(1, 1, 1, 1), name="stem")
    m.output(cases_yolo._c2f(m, stem, 8, n=1, prefix="c2f"))
    return m


def _values(planned: PlannedProgram) -> list[int]:
    return run_reference(planned, MemoryImage()).tensor_data["c2f_cv2"]


def test_c2f_that_does_not_fit_locally_is_planned_with_its_concat_in_ddr() -> None:
    """The headline case, at a size where the planner used to raise
    `planner: 1536 bytes for 'c2f_cat' do not fit ... even after evicting
    every evictable buffer`.

    The assertion is on the PLACEMENT, not merely on "it did not raise":
    the emitted program and `reference.py` both read their addresses out
    of this same plan, so a wrong placement is self-consistently wrong on
    both sides and a value comparison cannot see it.
    """
    planned = Planner(tensor_mem_bytes=3072).plan(_c2f_model())

    assert [name for name, _, _ in planned.ddr_placements] == ["c2f_cat"]
    assert {name for name, _, _ in planned.local_placements} == {
        "stem",
        "c2f_cv1",
        "c2f_m0_cv1",
        "c2f_m0_cv2",
    }

    cat_addr, cat_size = next(
        (addr, size) for name, addr, size in planned.ddr_placements if name == "c2f_cat"
    )
    assert cat_size == 8 * 8 * 24
    # The concat buffer went to the SPILL arena, not on top of an input
    # or an output region.
    assert DdrMap.SPILL <= cat_addr < DdrMap.OUTPUTS

    # Every producer of a slice writes into the DDR buffer, at the plane
    # offset the alias arithmetic dictates; every consumer reads it from
    # there. Nothing about the layout changed by moving space.
    slice_writes = {}
    for step in planned.steps:
        assert isinstance(step, ComputeStep), "no spill/reload should be needed here"
        if alias_root(step.op.output).name == "c2f_cat":
            assert step.output_space == isa.SPACE_DDR
            slice_writes[step.op.output.name] = step.output_addr
        for t, space, addr in zip(step.op.inputs, step.input_spaces, step.input_addrs):
            if alias_root(t).name == "c2f_cat":
                assert space == isa.SPACE_DDR
                assert cat_addr <= addr < cat_addr + cat_size

    plane_bytes = 8 * 8 * 8
    assert sorted(slice_writes.values()) == [
        cat_addr,
        cat_addr + plane_bytes,
        cat_addr + 2 * plane_bytes,
    ]

    # Placing a buffer in DDR must not also spill anything: the planner
    # recognizes unplaceability BEFORE the eviction loop moves a byte.
    assert planned.traffic.spill_count == 0
    assert planned.traffic.tensor_store_count == 0

    # The cost is reported, and it is exactly one write and one read of
    # the buffer.
    assert planned.traffic.ddr_resident_write_bytes == cat_size
    assert planned.traffic.ddr_resident_read_bytes == cat_size

    # ... and it is a real prediction: running the program moves exactly
    # these bytes.
    result = run_reference(planned, MemoryImage())
    assert result.traffic == planned.traffic
    assert result.tensor_data["c2f_cv2"] == _values(
        Planner(tensor_mem_bytes=1 << 16).plan(_c2f_model())
    )


def test_a_program_that_fits_locally_is_never_moved_to_ddr() -> None:
    """The other half of "deterministic and explainable": above the
    threshold nothing changes at all -- no DDR placement, no DDR traffic
    beyond the input read and the output store."""
    planned = Planner(tensor_mem_bytes=C2F_FITS_LOCALLY_FROM).plan(_c2f_model())
    assert planned.ddr_placements == []
    assert planned.traffic.ddr_resident_read_bytes == 0
    assert planned.traffic.ddr_resident_write_bytes == 0
    assert planned.traffic.write_bytes == 8 * 8 * 8  # the graph output, nothing else


def test_c2f_degradation_curve() -> None:
    """The whole point of the change, stated as a curve.

    Sweep the scratchpad from ample down to absurd. At every size the
    plan must exist, execute to the same answer, and predict its own
    traffic exactly; and the traffic must rise monotonically as the
    scratchpad shrinks -- degradation, not a cliff and not a random
    walk. Every size below `C2F_FITS_LOCALLY_FROM` raised `ValueError`
    before the fallback existed.
    """
    budgets = [8192, 6144, 5120, 4608, 4096, 3584, 3072, 2560, 2048, 1536, 1024, 512, 256, 128]
    golden = _values(Planner(tensor_mem_bytes=1 << 16).plan(_c2f_model()))

    rows = []
    for budget in budgets:
        planned = Planner(tensor_mem_bytes=budget).plan(_c2f_model())
        result = run_reference(planned, MemoryImage())
        assert result.tensor_data["c2f_cv2"] == golden, f"{budget}: answer changed"
        assert result.traffic == planned.traffic, f"{budget}: traffic misprediction"
        rows.append(
            (
                budget,
                planned.traffic.read_bytes,
                planned.traffic.write_bytes,
                len(planned.ddr_placements),
            )
        )

    # Monotone: shrinking the scratchpad never makes the program cheaper,
    # and never un-places a buffer that a larger scratchpad had to push
    # out.
    for (big, rd_big, wr_big, n_big), (small, rd_small, wr_small, n_small) in zip(rows, rows[1:]):
        assert rd_small >= rd_big, f"read traffic fell from {big} to {small} bytes of scratchpad"
        assert wr_small >= wr_big, f"write traffic fell from {big} to {small} bytes of scratchpad"
        assert n_small >= n_big, f"DDR placements fell from {big} to {small} bytes of scratchpad"

    # All-local -> partly-DDR -> mostly-DDR, and the ends are strictly
    # different, so the sweep cannot go vacuous.
    assert rows[0][3] == 0, "the roomiest budget must place nothing in DDR"
    assert rows[-1][3] >= 5, "the tightest budget must place essentially everything in DDR"
    assert rows[-1][1] > rows[0][1], "the tightest budget must cost more reads"
    assert rows[-1][2] > rows[0][2], "the tightest budget must cost more writes"


def test_ddr_placement_is_sticky_and_never_reloaded() -> None:
    """A buffer only lives in DDR because it provably has no local home,
    so it is never reloaded, never spilled, and never half-resident --
    which is what makes a concat family safe there: its later producers
    find it exactly where its earlier ones left it."""
    planned = Planner(tensor_mem_bytes=256).plan(_c2f_model())
    resident = {name for name, _, _ in planned.ddr_placements}
    assert resident
    for step in planned.steps:
        if isinstance(step, MoveStep):
            assert step.tensor.name not in resident, (
                f"'{step.tensor.name}' lives in DDR but got a {step.kind} instruction"
            )


def test_a_real_scale_c2f_plans() -> None:
    """`C2f` at YOLOv8n's actual channel counts, which is the situation
    that motivated all of this: a 20x20x128 concat family is 128,000
    bytes and no plausible scratchpad holds it, so the planner has to be
    able to answer "DDR" or it cannot lower a real network at all.

    Planner-only (no `run_reference`, no simulation): this is a statement
    about lowering, and executing it in Python would cost minutes for
    nothing. `DdrMap(scale=...)` stands in for a real board's DDR; the
    2 MiB simulation map's 256 KiB `SPILL` arena is the next thing that
    overflows at this scale, and that overflow is a property of the
    memory map, not of the strategy.
    """
    m = Model(seed=1, name="c2f_real")
    x = m.input(20, 20, 64, name="x")
    stem = m.conv2d(x, 64, kernel=(3, 3), padding=(1, 1, 1, 1), name="stem")
    m.output(cases_yolo._c2f(m, stem, 64, n=3, prefix="c2f"))

    planned = Planner(ddr_map=DdrMap(scale=64)).plan(m)  # section-4 default scratchpad

    placed = dict((name, size) for name, _, size in planned.ddr_placements)
    assert placed["c2f_cat"] == 20 * 20 * 320  # 2 halves + 3 bottleneck outputs
    assert planned.local_placements == [], (
        "at these sizes nothing fits in a 16 KiB scratchpad; if something does, "
        "this test has stopped testing real-scale behaviour"
    )
    assert planned.traffic.ddr_resident_write_bytes > 0


def test_the_default_ddr_map_is_unchanged_by_scaling_support() -> None:
    """`DdrMap(scale=1)` -- what every `tb_cnn_accel_top` case uses --
    must be the literal section-6 map, byte for byte."""
    plain = DdrMap()
    assert plain.limit == DdrMap.LIMIT
    assert plain.alloc(DdrMap.INPUTS, 16) == DdrMap.INPUTS
    assert plain.alloc(DdrMap.SPILL, 16) == DdrMap.SPILL
    with pytest.raises(ValueError, match="SPILL region overflow"):
        plain.alloc(DdrMap.SPILL, DdrMap.OUTPUTS - DdrMap.SPILL)

    big = DdrMap(scale=4)
    assert big.limit == DdrMap.LIMIT * 4
    assert big.alloc(DdrMap.SPILL, 16) == DdrMap.SPILL * 4
    # 4x the room in every region, same identifiers.
    assert big.alloc(DdrMap.SPILL, (DdrMap.OUTPUTS - DdrMap.SPILL) * 4 - 16) == DdrMap.SPILL * 4 + 16


def test_only_the_one_intended_catalogue_case_uses_the_ddr_fallback() -> None:
    """The guard that the fallback did not silently change where anything
    else lives.

    Every `tb_cnn_accel_top` case was planned entirely on chip before
    this change and must still be, with `case_c2f_concat_in_ddr` -- the
    case added *for* the fallback -- as the single exception. A case
    drifting into DDR placement would still pass its own value check
    (both sides use the plan) while quietly costing bandwidth, so it has
    to be asserted here.
    """
    catalogue = (
        cases.all_cases()
        + cases_pool_pad.all_cases()
        + cases_conv_pad.all_cases()
        + cases_concat_split.all_cases()
        + cases_yolo.all_cases()
    )
    assert catalogue
    using_ddr = {case.name for case in catalogue if case.planned.ddr_placements}
    assert using_ddr == {"c2f_concat_in_ddr"}

    for case in catalogue:
        if case.name in using_ddr:
            continue
        assert case.planned.traffic.ddr_resident_read_bytes == 0, case.name
        assert case.planned.traffic.ddr_resident_write_bytes == 0, case.name
