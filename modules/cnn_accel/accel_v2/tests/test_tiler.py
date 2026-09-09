"""The tiling oracle, plus the independent geometry checks around it
(design section 8 risks R1-R4 and R6, implementation plan step 5).

**The oracle** is the whole point of this file and of doing steps 3-6
before touching the simulator: for every YOLOv8n block builder in the
catalogue, at *every* strip height from 1 to `H_out`, the tiled
program's `reference.py` result must equal the untiled program's, element
for element. The untiled run never sees a strip address, so it is a
genuinely independent statement -- unlike a DUT-versus-reference
comparison, where both sides execute the addresses the planner chose and
a wrong halo is self-consistently wrong on both.

Around it sit the checks the oracle alone cannot make:

* **R1/R3 geometry.** The strip row ranges and halos are re-derived here
  from the closed form in the design document, by code that is not the
  tiler's, and the tiler's own numbers are checked against them. A
  mutation test (`test_the_oracle_catches_a_halo_off_by_one_row`) proves
  the oracle really would fail if the recurrence were wrong by one row.
* **R2 padding.** Padding must appear on a frame edge and nowhere else.
* **R11 partition.** Every `(plane, row)` of every group boundary is
  written by exactly one strip store, checked from the emitted
  addresses.
"""

from __future__ import annotations

import math

import pytest

from accel_v2 import cases_yolo
from accel_v2.memimage import MemoryImage
from accel_v2.model import Activation, Conv2dOp, Model, PoolOp, RowRange, Tensor, UpsampleOp
from accel_v2.planner import Planner
from accel_v2.reference import run_reference
from accel_v2.tiler import (
    TilingError,
    form_groups,
    group_outputs,
    strip_bounds,
    tile,
)
from accel_v2.tiling_checks import check_row_copy_partition, check_units_confined

#: Big enough that nothing in these small graphs is forced into DDR by
#: pressure -- the oracle is about the rewrite, not about the fallback.
#: `test_the_oracle_survives_a_scratchpad_too_small_to_hold_a_strip`
#: covers the opposite.
_MEM = dict(tensor_mem_bytes=256 * 1024, bank_bytes=16 * 1024)


def _run(model: Model, **planner_kwargs) -> dict[str, list[int]]:
    planner = Planner(**{**_MEM, **planner_kwargs})
    planned = planner.plan(model)
    return run_reference(planned, MemoryImage()).tensor_data


def _plan(model: Model, **planner_kwargs):
    return Planner(**{**_MEM, **planner_kwargs}).plan(model)


# ---------------------------------------------------------------------------
# The graphs. One per YOLOv8n block builder, at a size small enough to
# run every S but large enough for the halos to overlap non-trivially.
# ---------------------------------------------------------------------------


def _g_bottleneck(shortcut: bool = True, height: int = 9, channels: int = 8) -> Model:
    m = Model(seed=101)
    x = m.input(height, 6, channels, name="x")
    stem = m.conv2d(x, channels, kernel=(3, 3), padding=(1, 1, 1, 1), name="stem")
    m.output(cases_yolo._bottleneck(m, stem, channels, shortcut=shortcut, prefix="bn"))
    return m


def _g_c2f(n: int = 2, height: int = 9, channels: int = 8, pad_value: int = 0) -> Model:
    m = Model(seed=202)
    x = m.input(height, 6, channels, name="x")
    stem = m.conv2d(x, channels, kernel=(3, 3), padding=(1, 1, 1, 1), pad_value=pad_value, name="stem")
    m.output(cases_yolo._c2f(m, stem, channels, n=n, prefix="c2f"))
    return m


def _g_sppf(height: int = 9, channels: int = 8) -> Model:
    """SPPF's 5x5/stride-1/pad-2 pools with an all-negative tensor: R4's
    regime, the only one in which a wrongly padded tap wins a max."""
    m = Model(seed=303)
    x = m.input(height, 6, channels, name="x")
    stem = m.conv2d(x, channels, kernel=(3, 3), padding=(1, 1, 1, 1), clamp=(-128, -40), name="stem")
    m.output(cases_yolo._sppf(m, stem, channels, prefix="sppf"))
    return m


def _g_neck_upsample(height: int = 5, channels: int = 8) -> Model:
    """UPSAMPLE feeding a concat with a same-resolution skip, then a
    C2f -- the neck merge of section 3.3, and the one construct whose
    output rows must be rounded out to an even boundary."""
    m = Model(seed=404)
    deep = m.input(height, 4, channels, name="deep")
    skip = m.input(height * 2, 8, channels, name="skip")
    up = m.upsample2x(deep, name="up")
    cat = m.concat([up, skip], name="merge")
    m.output(m.conv2d(cat, channels, kernel=(3, 3), padding=(1, 1, 1, 1), name="merge_cv"))
    return m


def _g_stride2(height: int = 9, channels: int = 8) -> Model:
    """A stride-2 3x3/pad-1 conv, which is its own group. R3: strip `s`
    of its output must start at input row `2*o0 - 1`, not at an even
    row. Odd `height` on purpose."""
    m = Model(seed=505)
    x = m.input(height, 6, channels, name="x")
    down = m.conv2d(x, channels, kernel=(3, 3), stride=(2, 2), padding=(1, 1, 1, 1), name="down")
    m.output(m.conv2d(down, channels, kernel=(3, 3), padding=(1, 1, 1, 1), name="after"))
    return m


def _g_detect(height: int = 8, channels: int = 8) -> Model:
    """A Detect scale: two 3x3 -> 3x3 -> 1x1 branches sharing one input,
    both graph outputs. `channels=20` exercises R6 (a final channel tile
    with padding lanes) when the caller asks for it."""
    m = Model(seed=606)
    x = m.input(height, 6, channels, name="x")
    stem = m.conv2d(x, channels, kernel=(3, 3), padding=(1, 1, 1, 1), name="stem")
    for branch in ("box", "cls"):
        h = m.conv2d(stem, channels, kernel=(3, 3), padding=(1, 1, 1, 1), name=f"{branch}_a")
        h = m.conv2d(h, channels, kernel=(3, 3), padding=(1, 1, 1, 1), name=f"{branch}_b")
        m.output(m.conv2d(h, channels, kernel=(1, 1), name=f"{branch}_out"))
    return m


def _g_pool_stride2(height: int = 8, channels: int = 8) -> Model:
    m = Model(seed=707)
    x = m.input(height, 6, channels, name="x")
    stem = m.conv2d(x, channels, kernel=(3, 3), padding=(1, 1, 1, 1), name="stem")
    pooled = m.pool_max(stem, kernel=(2, 2), stride=(2, 2), name="pool")
    m.output(m.conv2d(pooled, channels, kernel=(3, 3), padding=(1, 1, 1, 1), name="after"))
    return m


def _g_act_chain(height: int = 7, channels: int = 8) -> Model:
    """A SiLU LUT in a realistic position -- an elementwise op inside a
    fused chain, which must simply ride along on the strip rows."""
    m = Model(seed=808)
    x = m.input(height, 6, channels, name="x")
    h = m.conv2d(x, channels, kernel=(3, 3), padding=(1, 1, 1, 1), name="cv")
    h = m.act(h, cases_yolo._silu_lut(), name="silu")
    m.output(m.conv2d(h, channels, kernel=(3, 3), padding=(1, 1, 1, 1), name="cv2"))
    return m


def _g_neck_merge_produced_skip(height: int = 4, channels: int = 8) -> Model:
    """The neck merge as the real network has it: the skip is *produced*
    by an earlier group, so the `[up | skip]` concat family straddles a
    group boundary.

    That makes three things happen at once that no single-group case
    reaches: the skip's producing group must store only *its* planes of
    the pinned family; the merge group builds the family locally and
    loads the skip slot straight into it (section 3.3); and the skip's
    other consumer reads the same planes from DDR."""
    m = Model(seed=909)
    x = m.input(height * 2, 8, channels, name="x")
    skip = m.conv2d(x, channels, kernel=(3, 3), padding=(1, 1, 1, 1), name="skip")
    deep = m.conv2d(skip, channels, kernel=(3, 3), stride=(2, 2), padding=(1, 1, 1, 1), name="deep")
    up = m.upsample2x(deep, name="up")
    cat = m.concat([up, skip], name="merge")
    m.output(m.conv2d(cat, channels, kernel=(3, 3), padding=(1, 1, 1, 1), name="merge_cv"))
    return m


def _g_downsample_merge(height: int = 8, channels: int = 8) -> Model:
    """The PAN merge: a stride-2 conv whose output is a *part* of a DDR
    concat family, read by the next group as the assembled tensor. The
    stride-2 conv is its own group, so its store lands at a plane offset
    and a row offset at once."""
    m = Model(seed=1010)
    x = m.input(height, 8, channels, name="x")
    other = m.input(height // 2, 4, channels, name="other")
    stem = m.conv2d(x, channels, kernel=(3, 3), padding=(1, 1, 1, 1), name="stem")
    down = m.conv2d(stem, channels, kernel=(3, 3), stride=(2, 2), padding=(1, 1, 1, 1), name="down")
    tall = m.conv2d(other, channels, kernel=(3, 3), padding=(1, 1, 1, 1), name="tall")
    cat = m.concat([down, tall], name="pan")
    m.output(m.conv2d(cat, channels, kernel=(3, 3), padding=(1, 1, 1, 1), name="pan_cv"))
    # `down` is read a second time, outside the merge group, so its
    # planes really do have to reach DDR.
    m.output(m.conv2d(down, channels, kernel=(1, 1), name="down_again"))
    return m


#: `(name, builder, output height to sweep S over)`. The height a strip
#: is anchored in is the *group output* height, which for these graphs is
#: the graph output's.
_ORACLE_GRAPHS = [
    ("bottleneck_shortcut", lambda: _g_bottleneck(shortcut=True)),
    ("bottleneck_plain", lambda: _g_bottleneck(shortcut=False)),
    ("c2f_n1", lambda: _g_c2f(n=1)),
    ("c2f_n2", lambda: _g_c2f(n=2)),
    ("c2f_padvalue", lambda: _g_c2f(n=1, pad_value=-128)),
    ("c2f_multiplane", lambda: _g_c2f(n=1, channels=16)),
    ("c2f_even_height", lambda: _g_c2f(n=1, height=8)),
    ("sppf", _g_sppf),
    ("sppf_even_height", lambda: _g_sppf(height=8)),
    ("neck_upsample", _g_neck_upsample),
    ("neck_upsample_odd", lambda: _g_neck_upsample(height=3)),
    ("stride2_odd", lambda: _g_stride2(height=9)),
    ("stride2_even", lambda: _g_stride2(height=8)),
    ("pool_stride2", _g_pool_stride2),
    ("detect", _g_detect),
    ("detect_pad_lanes", lambda: _g_detect(channels=20)),
    ("detect_three_channels", lambda: _g_detect(channels=3)),
    ("detect_wide", lambda: _g_detect(channels=12)),
    ("act_chain", _g_act_chain),
    ("neck_merge_produced_skip", _g_neck_merge_produced_skip),
    ("neck_merge_produced_skip_odd", lambda: _g_neck_merge_produced_skip(height=5)),
    ("downsample_merge", _g_downsample_merge),
    ("downsample_merge_multiplane", lambda: _g_downsample_merge(channels=16)),
]


def _max_strips(model: Model) -> int:
    """The largest `S` worth sweeping: the tallest group in the graph."""
    return max(g.height for g in form_groups(model))


# ---------------------------------------------------------------------------
# The oracle.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name, builder", _ORACLE_GRAPHS, ids=[n for n, _ in _ORACLE_GRAPHS])
def test_the_tiled_program_computes_exactly_what_the_untiled_one_does(name, builder) -> None:
    """R1-R4, R6. Every strip height, every graph, bit for bit."""
    reference_values = {t.name: _run(builder())[t.name] for t in builder().outputs}
    heights = _max_strips(builder())

    for strips in range(1, heights + 1):
        tiled = tile(builder(), strips)
        values = _run(tiled.model)
        for output_name, expected in reference_values.items():
            assert values[output_name] == expected, (
                f"{name} at S={strips}: output '{output_name}' differs from the untiled "
                f"result. Strips: "
                f"{[(r.group, r.strip, r.out_rows) for r in tiled.strips]}"
            )


@pytest.mark.parametrize("name, builder", _ORACLE_GRAPHS, ids=[n for n, _ in _ORACLE_GRAPHS])
def test_every_group_boundary_is_written_by_exactly_one_strip(name, builder) -> None:
    """R11, from the emitted addresses rather than from `dst_rows`."""
    for strips in range(1, _max_strips(builder()) + 1):
        tiled = tile(builder(), strips)
        planned = _plan(tiled.model)
        # `tiled.stored` is the set of pinned buffers (and plane slices
        # of pinned concat families) the strips actually write. A pinned
        # family may legitimately have a slot nothing writes to DDR --
        # the group that reads it also produces it -- so asking about the
        # whole family would demand writes that should not exist.
        for destination in tiled.stored:
            check_row_copy_partition(planned, destination)


@pytest.mark.parametrize("name, builder", _ORACLE_GRAPHS, ids=[n for n, _ in _ORACLE_GRAPHS])
def test_every_strip_buffer_is_bank_confined_at_plane_granularity(name, builder) -> None:
    """R5's pytest half: whatever the tiler placed, no single hardware
    request against it crosses a bank."""
    for strips in (1, 2, 3):
        if strips > _max_strips(builder()):
            continue
        check_units_confined(_plan(tile(builder(), strips).model))


# ---------------------------------------------------------------------------
# Geometry, re-derived independently of the tiler.
# ---------------------------------------------------------------------------


def _expected_rows(model: Model, group_ops, out_rows: RowRange, group_outs) -> dict:
    """The row-range recurrence of design section 2.5, written out again
    here in the most literal form possible.

    Deliberately a second implementation: checking the tiler's row ranges
    against the tiler's own recurrence would prove nothing at all. This
    one is a direct transcription of the table in the design document
    plus the two aliasing rules of section 2.5 step 4 -- a `split` slice
    is its parent's rows, and every part of a `concat` is as tall as the
    concat -- with no code shared with `tiler.py`."""
    need: dict[str, RowRange] = {}

    def require(t: Tensor, r0: int, r1: int) -> None:
        lo, hi = max(0, r0), min(t.height, r1)
        if t.name in need:
            lo, hi = min(lo, need[t.name].r0), max(hi, need[t.name].r1)
        if t.name in need and need[t.name] == RowRange(lo, hi):
            return
        need[t.name] = RowRange(lo, hi)
        if t.alias_role == "slice" and t.alias_parent is not None:
            require(t.alias_parent, lo, hi)
        for part in t.alias_parts:
            require(part, lo, hi)

    for root in group_outs:
        require(root, out_rows.r0, out_rows.r1)

    for op in reversed(group_ops):
        if op.output.name not in need:
            require(op.output, out_rows.r0, out_rows.r1)
        rng = need[op.output.name]
        a, b = rng.r0, rng.r1
        if isinstance(op, (Conv2dOp, PoolOp)):
            k, s, p = op.kernel[0], op.stride[0], op.padding[0]
            require(op.inputs[0], a * s - p, (b - 1) * s + k - p)
        elif isinstance(op, UpsampleOp):
            require(op.inputs[0], a // 2, math.ceil(b / 2))
        else:
            for t in op.inputs:
                require(t, a, b)
    return need


def test_the_strip_row_ranges_match_the_closed_form_recurrence() -> None:
    """R1. The tiler's `need` versus the design document's table, for
    every graph and every S."""
    for name, builder in _ORACLE_GRAPHS:
        source = builder()
        for strips in range(1, _max_strips(source) + 1):
            tiled = tile(builder(), strips)
            for group in tiled.groups:
                outs = group_outputs(tiled.source, tiled.groups, group)
                for record in [r for r in tiled.strips if r.group == group.index]:
                    expected = _expected_rows(tiled.source, group.ops, record.out_rows, outs)
                    for tensor_name, rows in expected.items():
                        assert record.rows[tensor_name] == rows, (
                            f"{name} S={strips} group {group.index} strip {record.strip}: "
                            f"'{tensor_name}' materializes {record.rows[tensor_name]}, the "
                            f"closed form says {rows}"
                        )


def test_stride_two_strips_start_on_the_phase_the_formula_says() -> None:
    """R3. For a 3x3/stride-2/pad-1 conv, strip `s`'s input must start at
    row `2*o0 - 1` (and at row 0 only for `o0 == 0`) -- the failure mode
    is every other strip shifted by one input row, which produces a
    perfectly plausible-looking image."""
    for height in (8, 9, 11):
        source = _g_stride2(height=height)
        down_group = next(g for g in form_groups(source) if g.ops[0].name == "down")
        for strips in range(1, down_group.height + 1):
            tiled = tile(_g_stride2(height=height), strips)
            for record in [r for r in tiled.strips if r.group == down_group.index]:
                o0, o1 = record.out_rows.r0, record.out_rows.r1
                assert record.rows["x"] == RowRange(
                    max(0, 2 * o0 - 1), min(height, 2 * (o1 - 1) + 3 - 1)
                ), f"height={height} S={strips} strip {record.strip}"
                pad_top, pad_bottom = record.padding["down"]
                assert pad_top == (1 if o0 == 0 else 0)


def test_padding_appears_on_a_frame_edge_and_nowhere_else() -> None:
    """R2. A padded tap on an interior strip boundary is a wrong value
    that both the DUT and the reference would compute identically."""
    for name, builder in _ORACLE_GRAPHS:
        source = builder()
        for strips in range(1, _max_strips(source) + 1):
            tiled = tile(builder(), strips)
            for record in tiled.strips:
                group = tiled.groups[record.group]
                for op in group.ops:
                    if op.name not in record.padding:
                        continue
                    # Re-derive the pad counts from the closed form: the
                    # padded taps are exactly the rows the receptive
                    # field asks for that do not exist.
                    want = record.rows[op.output.name]
                    k, stride, p = op.kernel[0], op.stride[0], op.padding[0]
                    height = op.inputs[0].height
                    lo = want.r0 * stride - p
                    hi = (want.r1 - 1) * stride + k - p
                    assert record.padding[op.name] == (max(0, -lo), max(0, hi - height)), (
                        f"{name} S={strips}: '{op.name}' on output rows {want} got padding "
                        f"{record.padding[op.name]}, the closed form says "
                        f"{(max(0, -lo), max(0, hi - height))}"
                    )
                    pad_top, pad_bottom = record.padding[op.name]
                    if pad_top:
                        assert lo < 0, (
                            f"{name} S={strips}: '{op.name}' pads the top of an interior "
                            f"strip whose input starts at real row {lo}"
                        )
                    if pad_bottom:
                        assert hi > height, (
                            f"{name} S={strips}: '{op.name}' pads the bottom of an "
                            f"interior strip whose input ends at real row {hi} of {height}"
                        )


def test_a_row_copy_never_splits_a_plane() -> None:
    """R6. Channel-tile padding lanes are zero-filled per beat by the
    compute engines, but a window copy moves raw bytes -- so it must move
    whole plane-rows, never part of one, or the pad lanes of the final
    tile become whatever was there before."""
    from accel_v2.model import RowCopyOp

    for name, builder in _ORACLE_GRAPHS:
        for strips in (1, 2, 3):
            source = builder()
            if strips > _max_strips(source):
                continue
            tiled = tile(builder(), strips)
            for op in tiled.model.ops:
                if not isinstance(op, RowCopyOp):
                    continue
                src, dst = op.inputs[0], op.output
                assert src.plane_count == dst.plane_count
                assert src.width == dst.width
                assert op.src_rows.rows == op.dst_rows.rows


# ---------------------------------------------------------------------------
# The mutation test: does the oracle actually bite?
# ---------------------------------------------------------------------------


def test_the_oracle_catches_a_halo_off_by_one_row(monkeypatch) -> None:
    """Break the recurrence by exactly one halo row and confirm the
    oracle fails. Without this, a green oracle only proves the tests
    ran.

    The mutation shortens a 3x3 conv's input range by one row at the top.
    The tiler's *own* build-time geometry assertion catches most such
    breakages (which is itself the point of having it), so this checks
    for either outcome: a `TilingError` at build time, or -- where the
    geometry still adds up -- a value mismatch at run time."""
    import accel_v2.tiler as tiler_module

    real_input_rows = tiler_module.input_rows

    def broken(op, out_rows):
        ranges = real_input_rows(op, out_rows)
        if isinstance(op, Conv2dOp) and op.kernel[0] == 3 and op.stride[0] == 1:
            return [RowRange(r.r0 + 1, r.r1) for r in ranges]
        return ranges

    monkeypatch.setattr(tiler_module, "input_rows", broken)

    builder = lambda: _g_c2f(n=1)
    expected = {t.name: _run(builder())[t.name] for t in builder().outputs}

    failures = 0
    for strips in range(1, _max_strips(builder()) + 1):
        try:
            values = _run(tile(builder(), strips).model)
        except TilingError:
            failures += 1
            continue
        if any(values[name] != want for name, want in expected.items()):
            failures += 1
    assert failures, (
        "a one-row halo error changed nothing the oracle looks at -- the oracle is not "
        "actually testing the recurrence"
    )


def test_the_oracle_catches_a_window_that_moves_the_wrong_rows(monkeypatch) -> None:
    """The second mutation, and the one that exercises the oracle's
    *value* comparison rather than the tiler's build-time assertion.

    The first mutation (above) breaks the recurrence in a way that makes
    the geometry stop adding up, so the tiler refuses to build it -- good
    news, but it means the value comparison was never reached. This one
    keeps every row *count* correct and moves the wrong rows: every
    extract window slides one row up. Nothing the tiler checks can see
    that (all the heights still match, every descriptor is still the
    right length), and neither could a DUT-versus-reference comparison,
    since both would faithfully execute the same wrong addresses. Only
    the untiled result disagrees.

    `S = 1` is expected to survive it: with one strip no window starts
    above row 0, so there is nothing to slide."""
    real_copy_rows = Model.copy_rows

    def sliding(self, src, rows, *, into=None, at=None, name=None):
        if into is None and rows.r0 > 0:
            rows = RowRange(rows.r0 - 1, rows.r1 - 1)
        return real_copy_rows(self, src, rows, into=into, at=at, name=name)

    builder = lambda: _g_c2f(n=2)
    expected = {t.name: _run(builder())[t.name] for t in builder().outputs}
    monkeypatch.setattr(Model, "copy_rows", sliding)

    caught = 0
    for strips in range(2, _max_strips(builder()) + 1):
        values = _run(tile(builder(), strips).model)
        if any(values[name] != want for name, want in expected.items()):
            caught += 1
    assert caught == _max_strips(builder()) - 1, (
        "sliding every strip window one row up went unnoticed at some strip height -- "
        "the oracle is not comparing what it claims to"
    )


# ---------------------------------------------------------------------------
# Structure of the rewrite.
# ---------------------------------------------------------------------------


def test_groups_break_at_every_stride_two_op_and_at_every_resolution_change() -> None:
    groups = form_groups(_g_stride2())
    assert [[op.name for op in g.ops] for g in groups] == [["down"], ["after"]]

    groups = form_groups(_g_pool_stride2())
    assert [[op.name for op in g.ops] for g in groups] == [["stem"], ["pool"], ["after"]]

    # An UPSAMPLE heads the group that follows it (section 3.3), which
    # falls out of the resolution rule with no special case.
    # (`merge_copy1` is the COPY `Model.concat` inserts because the skip
    # tensor is a graph input and cannot be a concat part in place.)
    groups = form_groups(_g_neck_upsample())
    assert [[op.name for op in g.ops] for g in groups] == [["up", "merge_copy1", "merge_cv"]]

    # A whole C2f fuses -- and so does the stem convolution in front of
    # it, since nothing between them changes resolution: one group of
    # ten ops (stem, cv1, two bottlenecks of cv1/cv2/add, cv2), with the
    # graph input as its only boundary.
    groups = form_groups(_g_c2f(n=2))
    assert len(groups) == 1
    assert [op.name for op in groups[0].ops] == [
        "stem",
        "c2f_cv1",
        "c2f_m0_cv1",
        "c2f_m0_cv2",
        "c2f_m0_add",
        "c2f_m1_cv1",
        "c2f_m1_cv2",
        "c2f_m1_add",
        # the two COPYs `concat` inserts for the `a`/`b` split views,
        # which cannot be concat parts in place
        "c2f_cat_copy0",
        "c2f_cat_copy1",
        "c2f_cv2",
    ]


def test_a_whole_c2f_fuses_into_one_group_with_pinned_boundaries_only() -> None:
    """The shape the design is for: inside a fused C2f nothing goes to
    DDR except the group's own boundaries."""
    tiled = tile(_g_c2f(n=2), 3)
    planned = _plan(tiled.model)
    assert planned.ddr_placements == [], (
        "a fused strip sub-program must be fallback-free; DDR placements: "
        f"{planned.ddr_placements}"
    )
    assert {name for name, _, _ in planned.pinned_placements} <= set(tiled.pinned)


def test_a_strip_conv_reuses_the_original_weight_objects_and_never_reuses_weights() -> None:
    """Every strip computes the same convolution -- literally the same
    list object -- and none of them sets `WEIGHT_REUSE` (R8)."""
    source = _g_c2f(n=1)
    tiled = tile(source, 3)
    originals = {id(op.weight) for op in source.ops if isinstance(op, Conv2dOp)}
    clones = [op for op in tiled.model.ops if isinstance(op, Conv2dOp)]
    assert clones
    for clone in clones:
        assert id(clone.weight) in originals
        assert not clone.weight_reuse


def test_tiling_a_single_strip_still_routes_through_the_pinned_boundaries() -> None:
    """`S = 1` is not a no-op: it is the layer-wise floor of section 7,
    where a group's ifmap is still strip-LOADed into the scratchpad once
    instead of being re-streamed by the output-channel loop."""
    tiled = tile(_g_c2f(n=1), 1)
    assert tiled.strip_counts and all(count == 1 for count in tiled.strip_counts.values())
    assert _run(tiled.model)


def test_strip_bounds_are_ceil_divisions_with_a_short_last_strip() -> None:
    assert strip_bounds(8, 1) == [RowRange(0, 8)]
    assert strip_bounds(8, 2) == [RowRange(0, 4), RowRange(4, 8)]
    assert strip_bounds(8, 3) == [RowRange(0, 3), RowRange(3, 6), RowRange(6, 8)]
    # S=5 over 8 rows is R=2, i.e. four strips, not five -- "ties go to
    # fewer strips".
    assert len(strip_bounds(8, 5)) == 4
    with pytest.raises(TilingError):
        strip_bounds(8, 9)


def test_the_oracle_survives_a_scratchpad_too_small_to_hold_a_strip() -> None:
    """R7. Under enough pressure a strip buffer spills or lands in DDR;
    the values must still be the untiled ones, and the program must still
    be correct -- only slower."""
    builder = lambda: _g_c2f(n=1)
    expected = {t.name: _run(builder())[t.name] for t in builder().outputs}
    tiled = tile(builder(), 3)
    values = _run(tiled.model, tensor_mem_bytes=4096, bank_bytes=2048)
    for name, want in expected.items():
        assert values[name] == want


def test_the_tiled_model_is_deterministic() -> None:
    a = tile(_g_c2f(n=2), 3)
    b = tile(_g_c2f(n=2), 3)
    assert [op.name for op in a.model.ops] == [op.name for op in b.model.ops]
    assert _run(a.model) == _run(b.model)
