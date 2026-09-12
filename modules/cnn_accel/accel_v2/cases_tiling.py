"""`tb_cnn_accel_top` cases for spatial tiling with cross-layer fusion.

A separate catalogue on purpose. Every case here runs a program the
*tiler* produced -- `select` chooses the strip height, `tile` rewrites
the graph, and the planner sees an ordinary (if much larger) `Model` --
so these are the only cases in which a tensor is pinned to DDR, a buffer
is confined at plane granularity, a `RowCopyOp` exists at all, or a
convolution fetches its weights from `space_wgt = LOCAL_TENSOR`. (Two
cases here are deliberately untiled: `ot_restream_from_ddr`, which
isolates the traffic tiling exists to remove, and `unfused_c2f_64ch`,
which is the "before" the headline case is measured against.) The six
untiled catalogues stay exactly as they were, and
`tests/test_tiling_guard.py` keeps proving it by hashing them.

What every case here asserts, beyond the standard element-by-element
comparison and the standard counter checks:

* **The oracle.** The DUT's output is compared against the reference of
  the tiled program (as always), *and* the tiled reference is compared
  against the reference of the **untiled** graph -- a program that has
  never seen a strip address. That second comparison is the only one
  that can see a wrong halo: the DUT and `reference.py` both execute the
  planner's addresses, so a row range that is wrong is wrong
  identically on both sides (memory note `cnn-accel-verification-blind-
  spot`).
* **The geometry, from the emitted addresses.** `tiling_checks`'
  `check_row_copy_partition` (every `(plane, row)` cell of a pinned
  boundary written exactly once), `check_units_confined` (no hardware
  request straddles a bank) and `check_no_unpinned_ddr_placements` (the
  strips really did fit) run on every case.
* **The traffic, exactly.** A fused group's DDR writes are exactly its
  pinned boundary bytes, its `ddr_resident_*` counters are zero, and
  `DDR_RD_BYTES` matches the prediction to the byte -- which is only
  meaningful because the prediction now charges the output-channel loop's
  ifmap re-streaming (`planner.ifmap_passes`).

Sizing. These are integration tests of a *compiler* pass, and simulator
time is linear in MACs: a case is kept to the smallest shape that still
makes its point, and the point is never the arithmetic (every engine is
pinned bit-exactly by its own unit testbench). Channel counts are real
where the case is about channels; spatial sizes are as small as the
strip structure allows.

`recompute_cap` is raised from its 25 % default in most cases here.
That constant is calibrated for a 640x640 network, where a two-row halo
on a 20-row strip is 10 % of the work; on a 16-row tensor the same halo
is most of it, and at the default the rule would rather dissolve the
group than fuse it. Raising the cap does not weaken any claim -- it only
stops the *policy* from vetoing the *mechanism* these cases exist to
exercise.
"""

from __future__ import annotations

import cnn_accel_model as golden

from accel_v2 import isa, yolov8n
from accel_v2.ddrmap import DdrMap
from accel_v2.memimage import MemoryImage
from accel_v2.model import Activation, Conv2dOp, Model
from accel_v2.planner import (
    ComputeStep,
    ConstLoadStep,
    MoveStep,
    Planner,
    RowCopyStep,
    descriptor_count,
    ifmap_passes,
)
from accel_v2.reference import run_reference
from accel_v2.tbcase import (
    CheckFailure,
    TbCase,
    TrafficPolicy,
    build_case,
    case_from_model,
)
from accel_v2.tiler import TiledModel, tile
from accel_v2.tiling_checks import (
    GeometryError,
    check_no_unpinned_ddr_placements,
    check_row_copy_partition,
    check_units_confined,
)
from accel_v2.tiling_select import select

#: Every case here plans against a x4 DDR map -- headroom in the two
#: regions a tiled program grows and an untiled one does not.
#:
#: * `PROGRAM`. One descriptor per activation plane per row copy adds up
#:   fast; the default 60 KiB region holds 960 of them, and a `C2f` at
#:   128 channels cut into four strips emits 281. `_require_program_fits`
#:   asserts the bound per case rather than trusting the headroom.
#: * `WEIGHTS`. A strip clone shares its original's weight image
#:   (`program._weight_allocations`), but a group still holds one image
#:   per convolution, and a single 128-channel 3x3 image is 36 KiB
#:   against a 64 KiB region at scale 1.
#:
#: As shaped today every case here would in fact fit a x1 map; the margin
#: is kept because both numbers are the ones tiling multiplies, and
#: hitting either is a `DdrMap` overflow at build time rather than
#: anything a test could diagnose. It costs simulator memory (the
#: testbench's DDR model is sized from this map, see `TbCase.generics`),
#: so it stays off everywhere else.
_DDR_SCALE = 4


# ---------------------------------------------------------------------------
# Shared machinery
# ---------------------------------------------------------------------------


def _timeout_cycles(model: Model, descriptors: int) -> int:
    """A per-program cycle bound for `g_timeout_cycles`, derived from the
    program rather than guessed.

    The entity default (100 000, set for the untiled catalogue's
    few-thousand-cycle cases) is below what a tiled program needs, and
    the two terms that matter are *not* both about compute. Measured over
    this catalogue, a run costs roughly `3 * MACs / (PE_ROWS * PE_COLS)`
    cycles of array time plus about 600 cycles of fixed FSM overhead per
    descriptor -- and for a strip program of small tensors the second
    term dominates: `stride2_odd_height` is 0.05 MMAC and 21 descriptors,
    and its 14 000 cycles are almost entirely per-command.

    The bound below is 2x the first coefficient and 2.5x the second, plus
    a flat floor. A MACs-only estimate is what let `fused_c2f_128ch` --
    3.4 MMAC in 281 descriptors -- run out of testbench time on a
    perfectly healthy program. `g_watchdog_cycles` is left alone: it is
    `cmd_proc`'s per-*state* bound, reloaded on every state change, so it
    does not scale with program length and is what actually catches a
    hang."""
    array = yolov8n.total_macs(model) // (golden.PE_ROWS * golden.TILE_CHANNELS)
    return max(200_000, 6 * array + 1500 * descriptors)


def _set_timeout(case: TbCase) -> None:
    """Give `case` a `g_timeout_cycles` sized to its own program, unless
    it already carries an explicit override."""
    case.generic_overrides.setdefault(
        "g_timeout_cycles",
        _timeout_cycles(
            case.model,
            sum(descriptor_count(step) for step in case.planned.steps) + 1,
        ),
    )


def _with_extra_check(case: TbCase, extra) -> TbCase:
    """Run `extra(case)` (raising `CheckFailure`/`GeometryError` on
    disagreement) in addition to `TbCase.check_live`'s standard checks,
    exactly as `cases_yolo.py` and `cases_concat_split.py` do it: set as
    the case's `extra_check`, which `check_live` runs first (and knows
    to catch `GeometryError` alongside `CheckFailure`)."""
    case.extra_check = extra
    return case


def _untiled_values(build, *, seed: int, name: str) -> dict[str, list[int]]:
    """Run the **untiled** graph through `reference.py` and return its
    tensor values -- the oracle.

    Planned against a scratchpad large enough that nothing spills and a
    DDR map large enough that nothing overflows, because none of that
    affects the values and all of it costs time. The point is only that
    this program contains no strip, no row copy and no pinned tensor, so
    not one address in it was computed by the code under test."""
    model = Model(seed=seed, name=f"{name}_untiled")
    build(model)
    planned = Planner(
        tensor_mem_bytes=4 << 20, bank_bytes=1 << 20, ddr_map=DdrMap(scale=8)
    ).plan(model)
    return run_reference(planned, MemoryImage()).tensor_data


def _require_oracle(case: TbCase, untiled: dict[str, list[int]]) -> None:
    """Tiled reference == untiled reference, for every graph output.

    Together with `TbCase._check_outputs` (DUT == tiled reference) this
    closes the loop: DUT == tiled == untiled, and only the middle
    equality shares any address with the code under test."""
    for tensor in case.model.outputs:
        want = untiled.get(tensor.name)
        if want is None:
            raise CheckFailure(
                f"the untiled reference produced no value for '{tensor.name}' -- the two "
                "graphs do not have the same outputs, so the oracle is comparing nothing"
            )
        got = case.expected.tensor_data[tensor.name]
        if want != got:
            wrong = sum(1 for a, b in zip(want, got) if a != b)
            index = next(i for i, (a, b) in enumerate(zip(want, got)) if a != b)
            h = index // (tensor.width * tensor.channels)
            rem = index % (tensor.width * tensor.channels)
            raise CheckFailure(
                f"the TILED program does not compute the same '{tensor.name}' as the "
                f"untiled one: {wrong} of {len(want)} elements differ, first at "
                f"h={h} w={rem // tensor.channels} c={rem % tensor.channels} "
                f"(untiled {want[index]}, tiled {got[index]}).\n"
                "  Both were executed by reference.py, so this is a tiling bug -- a halo, "
                "a strip's padding, or a join window -- and not a hardware one."
            )


def _require_geometry(case: TbCase, tiled: TiledModel) -> None:
    """The three reusable geometry assertions of `tiling_checks`, on
    every case.

    `check_row_copy_partition` is asked about `TiledModel.stored` -- the
    buffers (or plane slices of buffers) strip stores actually write --
    and not about the whole pinned family: a pinned concat family can
    have a slot no strip ever stores, because the group that reads it
    also produces it inside its own strips."""
    check_units_confined(case.planned)
    check_no_unpinned_ddr_placements(case.planned)
    for tensor in tiled.stored:
        check_row_copy_partition(case.planned, tensor)


def _require_program_fits(case: TbCase) -> None:
    """R12: the descriptor chain must fit the `PROGRAM` region. `DdrMap`
    raises on overflow, so this can only fail if the region was resized;
    it is asserted anyway because the number it bounds (one descriptor
    per plane per row copy) is the one thing tiling multiplies hardest."""
    descs = sum(descriptor_count(step) for step in case.planned.steps) + 1
    ddr_map = case.planned.ddr_map
    region = DdrMap.WEIGHTS * ddr_map.scale - DdrMap.PROGRAM * ddr_map.scale
    if descs * isa.INSTR_WORD_BYTES > region:
        raise CheckFailure(
            f"{descs} descriptors x {isa.INSTR_WORD_BYTES} bytes = "
            f"{descs * isa.INSTR_WORD_BYTES} does not fit the {region}-byte PROGRAM "
            f"region of a x{ddr_map.scale} DdrMap"
        )


def _require_fused_traffic(case: TbCase, *, expect_pinned_write: int) -> None:
    """A fused group's DDR writes are exactly its pinned boundary bytes.

    This is the invariant the whole design is for, and it is asserted
    against a number derived from the *graph* -- the byte size of the
    tensors the tiling pinned -- not read out of the plan. `DDR_WR_BYTES`
    itself is checked against the prediction (and against the
    testbench's passive AXI monitor) by `TbCase._check_traffic`, so the
    chain runs: bus <- counter <- prediction <- this figure."""
    traffic = case.planned.traffic
    if (traffic.ddr_resident_read_bytes, traffic.ddr_resident_write_bytes) != (0, 0):
        raise CheckFailure(
            "a fused case must pay nothing for the overflow fallback, but the plan "
            f"predicts ddr_resident rd={traffic.ddr_resident_read_bytes} "
            f"wr={traffic.ddr_resident_write_bytes}"
        )
    if traffic.pinned_write_bytes != expect_pinned_write:
        raise CheckFailure(
            f"pinned_write_bytes={traffic.pinned_write_bytes}, expected "
            f"{expect_pinned_write} (the byte size of the tensors this tiling pinned)"
        )
    if traffic.spill_count:
        raise CheckFailure(
            f"{traffic.spill_count} buffer(s) spilled mid-strip; a fused strip must run "
            "entirely on chip, so its only DDR writes are the group-boundary stores"
        )
    if traffic.write_bytes != expect_pinned_write:
        raise CheckFailure(
            f"the program writes {traffic.write_bytes} DDR bytes but its pinned "
            f"boundaries are only {expect_pinned_write} -- something else reached DDR"
        )


def _pinned_bytes(tiled: TiledModel) -> int:
    """Byte size of every tensor the tiling pinned to DDR, from the
    tensors themselves. The closed-form expectation
    `_require_fused_traffic` is checked against."""
    return sum(t.size_bytes for t in tiled.pinned.values() if t.pin_ddr)


def _tiled_case(
    name: str,
    build,
    *,
    seed: int,
    num_banks: int,
    bank_words: int,
    recompute_cap: float = 0.25,
    extra=None,
    traffic: TrafficPolicy | None = None,
    generic_overrides: dict[str, object] | None = None,
    expect_strips: dict[int, int] | None = None,
    expect_groups: int | None = None,
) -> TbCase:
    """Select, tile, plan, and wrap the result in a `TbCase` carrying the
    shared checks plus `extra(case, tiled, selection)`.

    The one ordering that matters: `select` and `tile` must see the **same
    `Model` instance**, because `Selection.groups` holds the very `Op`
    objects the tiler rewrites. Building the graph twice and handing one
    copy's groups to the other's tiler is a `TilingError` about a group
    that "produces nothing anything else reads", which is a confusing way
    to be told the objects do not match."""
    tensor_mem_bytes = num_banks * bank_words * 8
    bank_bytes = bank_words * 8

    model = Model(seed=seed, name=name)
    build(model)
    selection = select(
        model,
        tensor_mem_bytes=tensor_mem_bytes,
        bank_bytes=bank_bytes,
        ddr_scale=_DDR_SCALE,
        recompute_cap=recompute_cap,
    )
    tiled = tile(
        model,
        selection.strips,
        groups=selection.groups,
        resident_groups=selection.resident_groups,
    )
    case = case_from_model(
        name,
        tiled.model,
        num_banks=num_banks,
        bank_words=bank_words,
        ddr_scale=_DDR_SCALE,
        traffic=traffic,
        generic_overrides=generic_overrides,
    )
    _set_timeout(case)
    untiled = _untiled_values(build, seed=seed, name=name)

    def check(case: TbCase) -> None:
        # The plan the rule *said* it was choosing, asserted before
        # anything else: every figure below is a statement about a
        # particular cut, and a case that silently re-tiled itself when
        # the selection rule moved would keep passing while testing
        # something else.
        if expect_groups is not None and len(selection.groups) != expect_groups:
            raise CheckFailure(
                f"the rule cut this graph into {len(selection.groups)} fusion group(s), "
                f"expected {expect_groups}: {[[o.name for o in g.ops] for g in selection.groups]}"
            )
        if expect_strips is not None and selection.strips != expect_strips:
            raise CheckFailure(
                f"the rule chose strips {selection.strips}, expected {expect_strips} "
                f"(fallbacks {selection.fallbacks}, resident {sorted(selection.resident_groups)})"
            )
        _require_oracle(case, untiled)
        _require_geometry(case, tiled)
        _require_program_fits(case)
        if extra is not None:
            extra(case, tiled, selection, untiled)

    return _with_extra_check(case, check)


# ---------------------------------------------------------------------------
# 1. The two accounting cases: what the DUT says the bus did.
# ---------------------------------------------------------------------------


def case_ot_restream_from_ddr() -> TbCase:
    """The output-channel loop really does re-stream the whole ifmap once
    per tile, and the prediction now says so (hardware fact F4).

    Not a tiled case -- it is the *reason* for tiling, isolated. One
    convolution with 32 output channels reads a DDR-resident input; the
    hardware runs `n_ot = 4` passes and kicks the ifmap feeder over the
    entire tensor on each. Before `planner.ifmap_passes` existed the
    prediction charged that input once and `read_bytes` could only be a
    lower bound; here it is asserted to the byte, and separately checked
    to contain four ifmaps' worth and not one.

    A second, 8-output-channel convolution on the same input is the
    control: same shape, same operand, one pass."""

    def build(model: Model) -> None:
        x = model.input(8, 8, 8, name="x")
        wide = model.conv2d(x, 32, kernel=(3, 3), padding=(1, 1, 1, 1), name="wide")
        model.output(model.conv2d(wide, 8, kernel=(1, 1), name="narrow"))

    def check(case: TbCase) -> None:
        step = case.planned.steps[0]
        conv = step.op
        if step.input_spaces[0] != isa.SPACE_DDR:
            raise CheckFailure(
                "this case is only meaningful while 'wide' reads its input straight out "
                f"of DDR, but it names space {step.input_spaces[0]}"
            )
        n_ot = -(-conv.output.channels // golden.PE_ROWS)
        if (n_ot, ifmap_passes(conv)) != (4, 4):
            raise CheckFailure(f"expected 4 output-channel passes, got {n_ot}")
        # The ifmap's share of DDR reads, from the graph: four streams of
        # the 8x8x8 input. Everything else the program reads (weights,
        # bias, scale, descriptors) is charged separately, so this is a
        # lower bound on read_bytes -- and read_bytes itself is asserted
        # exactly against the counter by the standard traffic check.
        want = n_ot * conv.inputs[0].size_bytes
        if case.planned.traffic.read_bytes < want:
            raise CheckFailure(
                f"the plan predicts only {case.planned.traffic.read_bytes} DDR read bytes, "
                f"which is less than the {want} the ifmap alone costs at {n_ot} passes"
            )

    return _with_extra_check(build_case("ot_restream_from_ddr", build, seed=610), check)


def case_resident_weights_strip_pair() -> TbCase:
    """`space_wgt = LOCAL_TENSOR`, on the DUT, for the first time.

    A single 32-output-channel convolution (`n_ot = 4`) cut into two
    strips, with its packed weight/bias/scale images LOADed into the
    scratchpad once. Both strips' convolutions then fetch all four of
    their weight tiles through tensor-memory port `r1` instead of over
    AXI.

    This is the case that can tell a right local weight address from a
    wrong one. `reference.py` consumes a convolution's weights from the
    `Conv2dOp` in their logical form and never builds their packed byte
    image, so its answer does not depend on where those bytes sit; the
    DUT reads the actual scratchpad. A wrong address is therefore a
    mismatch, not an agreement -- which is the opposite of this project's
    usual blind spot.

    What is asserted:

    * the descriptors really name `LOCAL_TENSOR` (the path is exercised,
      not merely available);
    * `DDR_RD_BYTES` exactly, and lower than the same program with DDR
      weights by exactly one image, since with two strips the DDR
      variant reads the image twice;
    * `WEIGHT_LOAD_BYTES` **unchanged** by residency. That is a hardware
      fact worth recording: `cmd_proc`'s `cnt_wgt_bytes_q` increments in
      `st_wgt_req` before the operand-space mux, so the counter measures
      how much the weight buffer was filled, not how much DDR was read.
      The design document's expectation that it would read zero is wrong
      about this register; the claim that residency removes the DDR
      traffic is right, and it is `DDR_RD_BYTES` that shows it.
    """

    def build(model: Model) -> None:
        x = model.input(8, 8, 8, name="x")
        model.output(
            model.conv2d(
                x, 32, kernel=(3, 3), padding=(1, 1, 1, 1), per_channel=True, name="y"
            )
        )

    seed = 611
    model = Model(seed=seed, name="resident_weights_strip_pair")
    build(model)
    tiled = tile(model, 2, resident_groups={0})
    case = case_from_model(
        "resident_weights_strip_pair",
        tiled.model,
        num_banks=4,
        bank_words=1024,
        ddr_scale=_DDR_SCALE,
    )
    _set_timeout(case)
    untiled = _untiled_values(build, seed=seed, name="resident_weights_strip_pair")

    # The same graph, same cut, weights left in DDR -- planned only, as
    # the figure the DUT's own reading is compared against.
    ddr_model = Model(seed=seed, name="resident_weights_strip_pair_ddr")
    build(ddr_model)
    ddr_planned = Planner(
        tensor_mem_bytes=4 * 1024 * 8,
        bank_bytes=1024 * 8,
        ddr_map=DdrMap(scale=_DDR_SCALE),
    ).plan(tile(ddr_model, 2).model)

    def check(case: TbCase) -> None:
        _require_oracle(case, untiled)
        _require_geometry(case, tiled)
        _require_program_fits(case)

        loads = [s for s in case.planned.steps if isinstance(s, ConstLoadStep)]
        if len(loads) != 1:
            raise CheckFailure(
                f"expected exactly one weight-image LOAD shared by both strips, got "
                f"{len(loads)} -- the strips are not sharing an image"
            )
        image_bytes = loads[0].nbytes
        convs = [
            s
            for s in case.planned.steps
            if isinstance(s, ComputeStep) and isinstance(s.op, Conv2dOp)
        ]
        if len(convs) != 2 or not all(
            s.weight_space == isa.SPACE_LOCAL_TENSOR for s in convs
        ):
            raise CheckFailure(
                "both strip convolutions must name space_wgt = LOCAL_TENSOR, got "
                f"{[s.weight_space for s in convs]}"
            )
        for desc in case.program.descs:
            if desc.opcode == isa.OPCODE_CONV2D and desc.space_wgt != isa.SPACE_LOCAL_TENSOR:
                raise CheckFailure("an emitted CONV2D descriptor still names DDR weights")

        # Residency saves exactly one image, less the program fetch of
        # the three extra LOAD descriptors it costs.
        saved = ddr_planned.traffic.read_bytes - case.planned.traffic.read_bytes
        want = image_bytes - len(loads[0].images) * isa.INSTR_WORD_BYTES
        if saved != want:
            raise CheckFailure(
                f"resident weights saved {saved} DDR read bytes, expected {want} "
                f"(one {image_bytes}-byte image, less "
                f"{len(loads[0].images)} extra descriptors)"
            )
        if case.planned.traffic.weight_bytes != ddr_planned.traffic.weight_bytes:
            raise CheckFailure(
                "WEIGHT_LOAD_BYTES must not change with residency: the weight buffer is "
                f"filled the same either way ({case.planned.traffic.weight_bytes} vs "
                f"{ddr_planned.traffic.weight_bytes})"
            )

    return _with_extra_check(case, check)


# ---------------------------------------------------------------------------
# 2. The tiled cases proper.
# ---------------------------------------------------------------------------


def _c2f_build(height, width, channels, *, n=1, shortcut=True):
    def build(model: Model) -> None:
        x = model.input(height, width, channels, name="x")
        model.output(
            yolov8n.c2f(model, x, channels, n=n, shortcut=shortcut, prefix="c2f")
        )

    return build


def _unfused_plan(build, *, seed: int, num_banks: int, bank_words: int):
    """The same graph, same scratchpad, no tiling at all -- the "before"
    the headline case is measured against."""
    model = Model(seed=seed, name="unfused")
    build(model)
    return Planner(
        tensor_mem_bytes=num_banks * bank_words * 8,
        bank_bytes=bank_words * 8,
        ddr_map=DdrMap(scale=_DDR_SCALE),
    ).plan(model)


def case_fused_c2f_64ch() -> TbCase:
    """The headline acceptance case: a whole `C2f` at 64 channels, fused
    into one group and run strip by strip out of a scratchpad that cannot
    hold it any other way.

    16 KiB, as two independent 8 KiB banks. Untiled, this graph's 128-
    channel concat family is 8 KiB on its own and its `cv1` output
    another 8 KiB, so the planner has to leave a buffer in DDR
    (`case_c2f_concat_in_ddr`'s behaviour, at four times the channel
    count) and every intermediate makes a full round trip -- amplified by
    the output-channel loop, which re-streams each of them once per tile.
    Fused, the same 16 KiB holds a three-row strip of the whole block and
    the only DDR traffic left is the group's own boundary: the input it
    reads and the output it writes.

    The case asserts that as a *measurement*, not a prediction: the
    unfused plan of the identical graph is built alongside, and the DUT's
    own `DDR_RD_BYTES`/`DDR_WR_BYTES` -- themselves cross-checked against
    the testbench's passive AXI monitor -- must come in below it. And the
    values must be bit-identical to the untiled reference, which never saw
    a strip address."""
    build = _c2f_build(10, 4, 64)
    unfused = _unfused_plan(build, seed=701, num_banks=3, bank_words=256)

    def extra(case: TbCase, tiled: TiledModel, selection, untiled) -> None:
        if not unfused.ddr_placements:
            raise CheckFailure(
                "this case is only the acceptance case while the UNFUSED graph does not "
                "fit: the planner placed every buffer of it locally, so there is nothing "
                "for fusion to buy here"
            )
        _require_fused_traffic(case, expect_pinned_write=_pinned_bytes(tiled))
        if case.planned.traffic.read_bytes >= unfused.traffic.read_bytes:
            raise CheckFailure(
                f"the fused program reads {case.planned.traffic.read_bytes} DDR bytes "
                f"against the unfused {unfused.traffic.read_bytes} -- fusion bought "
                "nothing on the read side"
            )
        if case.planned.traffic.write_bytes >= unfused.traffic.write_bytes:
            raise CheckFailure(
                f"the fused program writes {case.planned.traffic.write_bytes} DDR bytes "
                f"against the unfused {unfused.traffic.write_bytes}"
            )

    return _tiled_case(
        "fused_c2f_64ch",
        build,
        seed=701,
        # 6 KiB, three 2 KiB banks. The 128-channel concat family is 5 KiB
        # at full height and no single bank holds it, so the untiled plan
        # has to leave it in DDR; a two-row strip of the whole block fits.
        num_banks=3,
        bank_words=256,
        recompute_cap=0.5,
        expect_groups=1,
        expect_strips={0: 2},
        extra=extra,
    )


def case_unfused_c2f_64ch() -> TbCase:
    """`case_fused_c2f_64ch`'s graph, same seed, same 16 KiB scratchpad,
    **not tiled** -- the "before" of the headline comparison, run on the
    DUT so the saving is a difference between two measurements rather
    than between two predictions.

    Both cases assert `DDR_RD_BYTES` and `DDR_WR_BYTES` exactly against
    their own plan, and both cross-check those counters against the
    testbench's passive AXI monitor, so "the fused program moves fewer
    bytes" is observed on the bus at both ends.

    What this program does is what the planner has always done with a
    graph it cannot hold: the 128-channel concat family and `cv1`'s
    output do not fit a bank, so they stay in DDR for their whole
    lifetime and every convolution reads them back over AXI -- once per
    output-channel tile, which at 64 channels is eight times."""
    build = _c2f_build(10, 4, 64)
    seed = 701  # the same seed as the fused case: the same weights, the
    # same input, the same network.
    model = Model(seed=seed, name="unfused_c2f_64ch")
    build(model)
    case = case_from_model(
        "unfused_c2f_64ch",
        model,
        num_banks=3,
        bank_words=256,
        ddr_scale=_DDR_SCALE,
    )
    _set_timeout(case)

    def check(case: TbCase) -> None:
        if not case.planned.ddr_placements:
            raise CheckFailure(
                "the point of this case is that the untiled graph does NOT fit; the "
                "planner placed every buffer locally"
            )
        if case.planned.traffic.ddr_resident_read_bytes == 0:
            raise CheckFailure("nothing was read back out of DDR, so nothing overflowed")

    return _with_extra_check(case, check)


def case_fused_wide_128ch() -> TbCase:
    """Real channel depth: two convolutions with 128 output channels
    (`n_ot = 16` output-channel passes each), concatenated into a
    **256-channel, 32-plane** tensor that the group's closing 1x1 then
    reads -- all inside one fused group, cut in two strips.

    Sixteen passes is the number this design is about: from DDR the
    ifmap is streamed sixteen times (the amplification tiling exists to
    remove), from the scratchpad it is free, and sixteen weight tiles,
    bias rows and scale rows are fetched per convolution. The
    256-channel concat is the widest tensor the catalogue builds: the
    closing convolution's ifmap feeder walks 32 channel tiles per row
    per pass, and the concat is assembled inside the strip -- both of its
    parts are written straight into their plane slots by convolutions of
    the same strip, which is what fusing it means and why the tensor
    never reaches DDR at all.

    **Why this is not a literal `C2f` at 128 channels**, which is what
    the shape list asks for. `cnn_accel_cmd_proc`'s weight serializer
    emits one int8 lane per cycle (`wgt_lanes_left_q <= c_word_bytes` for
    the weight region), so a program costs roughly two cycles per byte of
    `WEIGHT_LOAD_BYTES` -- measured at 1.5x to 3x across this whole
    catalogue, and it dominates everything else including the MACs. A
    `C2f` at 128 channels has two 3x3 convolutions on 64 channels in its
    bottleneck, which is 74 KiB of packed weights per strip, so at two
    strips it is a quarter of a million cycles of *weight fetch alone* --
    over six minutes of simulator time at any spatial size, including
    2x2. The shape here keeps every property that case was for (16
    passes, 32 planes, a 3x3 halo, a pinned boundary, `S = 2`) and pays
    8.4 KiB of weights per strip for it. The C2f *structure* is covered
    at 64 channels by `case_fused_c2f_64ch`, where it is affordable.
    """

    def build(model: Model) -> None:
        x = model.input(8, 4, 16, name="x")
        # A 3x3 on a narrow tensor: cheap in weights, and it is what gives
        # the group a halo at all, so the strips overlap.
        h = model.conv2d(x, 16, kernel=(3, 3), padding=(1, 1, 1, 1), name="stem")
        wide_a = model.conv2d(h, 128, kernel=(1, 1), name="wide_a")
        wide_b = model.conv2d(h, 128, kernel=(1, 1), name="wide_b")
        merged = model.concat([wide_a, wide_b], name="merged")
        model.output(model.conv2d(merged, 8, kernel=(1, 1), name="narrow"))

    def extra(case: TbCase, tiled: TiledModel, selection, untiled) -> None:
        convs = [
            step.op
            for step in case.planned.steps
            if isinstance(step, ComputeStep) and isinstance(step.op, Conv2dOp)
        ]
        passes = {ifmap_passes(op) for op in convs}
        if 16 not in passes:
            raise CheckFailure(
                f"expected a convolution with 16 output-channel passes, got {sorted(passes)}"
            )
        widest = max(op.inputs[0].channels for op in convs)
        if widest != 256:
            raise CheckFailure(
                f"the closing convolution should read a 256-channel concat, got {widest}"
            )
        tiles = max(op.inputs[0].plane_count for op in convs)
        if tiles != 32:
            raise CheckFailure(
                f"the closing convolution should feed 32 channel tiles, got {tiles}"
            )
        _require_fused_traffic(case, expect_pinned_write=_pinned_bytes(tiled))

    return _tiled_case(
        "fused_wide_128ch",
        build,
        seed=702,
        # 8 KiB, two 4 KiB banks: room for a four-row strip of the
        # 32-plane concat (4 KiB) and not for all eight rows of it.
        num_banks=2,
        bank_words=512,
        recompute_cap=3.0,
        expect_groups=1,
        expect_strips={0: 2},
        extra=extra,
    )


def case_sppf_tiled_negative() -> TbCase:
    """SPPF, tiled, on an input that is negative everywhere (R4).

    Three chained 5x5 stride-1 max pools with padding 2 mean a strip's
    input reaches six rows past its output at each end, and the pad value
    is the tensor's zero point, `-128`. On data that is negative
    everywhere -- clamped to `[-128, -40]`, exactly as
    `cases_yolo._sppf` does it -- a padded tap is the only thing that can
    *win* a max, so a strip that pads an interior boundary (or fails to
    pad a real frame edge) produces a different answer. On non-negative
    data the same bug is invisible, because `-128` never wins.

    That makes this the sharpest of the halo checks, and it is decided by
    the comparison against the untiled reference: both would agree with
    each other on the wrong answer if the DUT and `reference.py` were the
    only two opinions."""

    def build(model: Model) -> None:
        # 16 rows, so that a two-strip cut has an interior boundary at
        # all; on an 8-row tensor the rule would (correctly) keep the
        # whole block in one strip and there would be no boundary to pad
        # wrongly.
        x = model.input(16, 8, 16, name="x")
        # Everything negative, and the zero point at the bottom of the
        # range: `pool_max`'s padded taps are -128, so they can only ever
        # be the maximum of a window when the real data is below zero.
        dark = model.conv2d(
            x, 16, kernel=(1, 1), activation=Activation.NONE, clamp=(-128, -40), name="dark"
        )
        model.output(yolov8n.sppf(model, dark, 16, prefix="sppf"))

    def extra(case: TbCase, tiled: TiledModel, selection, untiled) -> None:
        values = untiled["dark"]
        if max(values) > -40 or min(values) < -128:
            raise CheckFailure(
                f"the pooled input must be negative everywhere for the -128 pad value to "
                f"be observable; it spans [{min(values)}, {max(values)}]"
            )
        pooled = [record for record in tiled.strips if any("_p" in n for n in record.padding)]
        if len({record.group for record in pooled}) != 1 or len(pooled) < 2:
            raise CheckFailure(
                f"the three pools must be one group cut into at least two strips; got "
                f"{len(pooled)} strip record(s) over groups "
                f"{sorted({r.group for r in pooled})}"
            )
        # R2, stated directly: a 5x5/p2 pool pads only where its strip
        # touches the real edge of the tensor, and by exactly 2 there.
        for record in pooled:
            for name, (top, bottom) in record.padding.items():
                if "_p" not in name:
                    continue
                want_top = 2 if record.out_rows.r0 == 0 else 0
                if top not in (0, 2) or bottom not in (0, 2):
                    raise CheckFailure(
                        f"pool '{name}' on strip {record.strip} got padding "
                        f"({top}, {bottom}); a 5x5/p2 pool pads by 2 or not at all"
                    )
                if record.out_rows.r0 == 0 and top != want_top:
                    raise CheckFailure(
                        f"pool '{name}' on the first strip must pad the top by 2, got {top}"
                    )
                if record.out_rows.r0 != 0 and top != 0:
                    raise CheckFailure(
                        f"pool '{name}' on interior strip {record.strip} pads the top by "
                        f"{top}; only a strip that touches row 0 may"
                    )
        _require_fused_traffic(case, expect_pinned_write=_pinned_bytes(tiled))

    return _tiled_case(
        "sppf_tiled_negative",
        build,
        seed=703,
        # 4 KiB, two banks: enough for a two-strip cut of this block and
        # not for one strip, so `S = 2` is what the rule picks rather
        # than what the case asks for.
        num_banks=2,
        bank_words=256,
        recompute_cap=3.0,
        expect_groups=1,
        expect_strips={0: 2},
        extra=extra,
    )


def case_neck_upsample_concat() -> TbCase:
    """The FPN/PAN neck merge, fused: `UPSAMPLE` -> `concat` -> `C2f`,
    all one group.

    `UPSAMPLE` is stride-1 for group formation, so it joins the group
    that follows it -- and its row recurrence is the only non-identity,
    non-convolutional one in the table (`[a//2, ceil(b/2))`), which is
    what makes an odd strip boundary interesting. The concat's other part
    is the skip tensor at full resolution, so the strip has to pull a row
    window of it straight into the concat slot: a `RowCopyOp` acting as a
    concat part's producer, which is the mechanism that saves writing and
    re-reading the upsampled tensor at all."""

    def build(model: Model) -> None:
        deep = model.input(4, 4, 16, name="deep")
        skip = model.input(8, 8, 16, name="skip")
        up = model.upsample2x(deep, name="up")
        merged = model.concat([up, skip], name="merged")
        model.output(yolov8n.c2f(model, merged, 32, n=1, shortcut=False, prefix="neck"))

    def extra(case: TbCase, tiled: TiledModel, selection, untiled) -> None:
        if len(selection.groups) != 1:
            raise CheckFailure(
                "the upsample must join the C2f group, but the graph was cut into "
                f"{len(selection.groups)} groups: "
                f"{[[o.name for o in g.ops] for g in selection.groups]}"
            )
        _require_fused_traffic(case, expect_pinned_write=_pinned_bytes(tiled))

    return _tiled_case(
        "neck_upsample_concat",
        build,
        seed=704,
        num_banks=4,
        bank_words=1024,
        recompute_cap=1.0,
        expect_groups=1,
        extra=extra,
    )


def case_stride2_odd_height() -> TbCase:
    """A stride-2 convolution, tiled, on an input with an ODD number of
    rows (R3), and a channel count that is not a whole number of
    activation planes (R6).

    Group formation cuts at every stride > 1, so this convolution is a
    group of its own and its strips are anchored on its own output rows.
    The recurrence then says a strip producing output rows `[a, b)` reads
    input rows `[2a - pad, 2(b-1) + k - pad)`: for every strip but the
    first that starts on an ODD input row, and an implementation that
    quietly rounded to an even one would shift every strip after the
    first by a row -- while still producing exactly the right number of
    output rows, and while the DUT and `reference.py` agreed with each
    other about it.

    13 rows also makes the last strip short in a way that interacts with
    the stride, and 12 channels means the final activation plane has four
    padding lanes that a row copy moves as ordinary bytes."""

    def build(model: Model) -> None:
        x = model.input(13, 8, 12, name="x")
        model.output(
            model.conv2d(
                x, 16, kernel=(3, 3), stride=(2, 2), padding=(1, 1, 1, 1),
                pad_value=-7, name="down",
            )
        )

    def extra(case: TbCase, tiled: TiledModel, selection, untiled) -> None:
        # The row ranges, from the closed form -- not from the tiler.
        # out rows [a, b) of a 3x3/s2/p1 conv read input rows
        # [2a - 1, 2(b-1) + 3 - 1), clipped to the tensor.
        for record in tiled.strips:
            a, b = record.out_rows.r0, record.out_rows.r1
            want_lo, want_hi = max(0, 2 * a - 1), min(13, 2 * (b - 1) + 2)
            got = record.rows["x"]
            if (got.r0, got.r1) != (want_lo, want_hi):
                raise CheckFailure(
                    f"strip {record.strip} of output rows [{a}, {b}) materializes input "
                    f"rows [{got.r0}, {got.r1}); the recurrence says [{want_lo}, {want_hi})"
                )
            if a > 0 and got.r0 % 2 == 0:
                raise CheckFailure(
                    f"strip {record.strip} starts on even input row {got.r0}; every "
                    "interior stride-2 strip starts on an odd one"
                )
            want_pad_top = 1 if a == 0 else 0
            want_pad_bottom = 1 if want_hi == 13 else 0
            if record.padding["down"] != (want_pad_top, want_pad_bottom):
                raise CheckFailure(
                    f"strip {record.strip} got padding {record.padding['down']}, expected "
                    f"({want_pad_top}, {want_pad_bottom}) -- padding belongs only where a "
                    "strip touches the real edge of the tensor"
                )
        _require_fused_traffic(case, expect_pinned_write=_pinned_bytes(tiled))

    return _tiled_case(
        "stride2_odd_height",
        build,
        seed=705,
        # 1 KiB, two banks: small enough that the rule has to cut the
        # seven output rows into four strips, which is what puts an
        # interior boundary at an odd input row.
        num_banks=2,
        bank_words=64,
        recompute_cap=2.0,
        expect_groups=1,
        expect_strips={0: 4},
        extra=extra,
    )


def case_detect_branch_80ch() -> TbCase:
    """A Detect-head branch at its real width: 80 channels, ten
    activation planes, `3x3 -> 1x1` into a graph output.

    80 is the class count of the COCO head and the widest tensor the
    network moves through a row copy. Ten planes means ten descriptors
    for every strip load and ten for every strip store, and it is the
    case where a plane-offset arithmetic error has the most room to hide:
    with one or two planes an off-by-one plane stride lands outside the
    buffer and is caught by `row_copy_cells`' own bounds check, with ten
    it lands in the middle of a neighbouring plane and reads back as
    plausible data.

    (The real branch is `3x3 -> 3x3 -> 1x1`; the middle convolution is
    dropped here because at 80x80 channels it is 3.7 MMAC of simulator
    time that repeats what the first one already proves.)"""

    def build(model: Model) -> None:
        x = model.input(8, 8, 16, name="x")
        h = model.conv2d(x, 80, kernel=(3, 3), padding=(1, 1, 1, 1), name="cv3_0")
        model.output(model.conv2d(h, 80, kernel=(1, 1), name="cv3_1"))

    def extra(case: TbCase, tiled: TiledModel, selection, untiled) -> None:
        planes = {
            len(step.transfers)
            for step in case.planned.steps
            if isinstance(step, RowCopyStep)
        }
        if 10 not in planes:
            raise CheckFailure(
                f"expected a 10-plane row copy for the 80-channel tensor, saw {sorted(planes)}"
            )
        _require_fused_traffic(case, expect_pinned_write=_pinned_bytes(tiled))

    return _tiled_case(
        "detect_branch_80ch",
        build,
        seed=706,
        # 8 KiB, two banks: one strip of this branch is 3 KiB of ten-plane
        # buffers, two of them do not fit, and the rule cuts it in two.
        num_banks=2,
        bank_words=512,
        recompute_cap=1.0,
        expect_groups=1,
        expect_strips={0: 2},
        extra=extra,
    )


def case_eviction_inside_a_strip() -> TbCase:
    """R7: a strip whose live set does not fit, so the planner evicts one
    of its buffers mid-strip and reloads it.

    Deliberately NOT selected by the rule -- `tiling_select` rejects a
    candidate that spills, which is the whole point of the fit test -- so
    the strip height is given directly. What is under test is that the
    eviction machinery still holds inside a strip: a half-written concat
    family is never the victim, a row copy's destination is never evicted
    between two of its planes (the planner sees one op, not
    `plane_count`), and the values come out bit-identical to the untiled
    reference anyway.

    Its traffic is therefore *not* the fused invariant: `DDR_WR_BYTES`
    is the group boundary plus the spills, and the case asserts exactly
    that decomposition rather than pretending the spills are free."""
    build = _c2f_build(8, 4, 32)
    seed = 707
    model = Model(seed=seed, name="eviction_inside_a_strip")
    build(model)
    tiled = tile(model, 2)
    case = case_from_model(
        "eviction_inside_a_strip",
        tiled.model,
        # 2 KiB, two 1 KiB banks: room for most of a strip, not all of
        # it, so the planner has to evict.
        num_banks=2,
        bank_words=128,
        ddr_scale=_DDR_SCALE,
    )
    _set_timeout(case)
    untiled = _untiled_values(build, seed=seed, name="eviction_inside_a_strip")

    def check(case: TbCase) -> None:
        _require_oracle(case, untiled)
        _require_geometry(case, tiled)
        _require_program_fits(case)
        traffic = case.planned.traffic
        if traffic.spill_count == 0:
            raise CheckFailure(
                "this case exists to exercise eviction inside a strip, but nothing was "
                "spilled -- pick a tighter strip height or a smaller scratchpad"
            )
        spill_addrs = {
            step.dst_addr
            for step in case.planned.steps
            if isinstance(step, MoveStep) and step.kind == "spill"
        }
        read_back = any(
            isinstance(step, ComputeStep)
            and any(
                space == isa.SPACE_DDR and addr in spill_addrs
                for space, addr in zip(step.input_spaces, step.input_addrs)
            )
            or (
                isinstance(step, MoveStep)
                and step.kind == "reload"
                and step.src_addr in spill_addrs
            )
            or (isinstance(step, RowCopyStep) and step.src_space == isa.SPACE_DDR
                and any(src in spill_addrs for src, _, _ in step.transfers))
            for step in case.planned.steps
        )
        if not read_back:
            raise CheckFailure(
                "a buffer was spilled and never read again, so the eviction moved bytes "
                "nobody wanted and the case proves nothing about correctness"
            )
        spilled = sum(
            step.nbytes
            for step in case.planned.steps
            if isinstance(step, MoveStep) and step.kind == "spill"
        )
        pinned = _pinned_bytes(tiled)
        if traffic.write_bytes != pinned + spilled:
            raise CheckFailure(
                f"DDR writes are {traffic.write_bytes}, expected {pinned} bytes of group "
                f"boundary plus {spilled} bytes of spill"
            )
        if (traffic.ddr_resident_read_bytes, traffic.ddr_resident_write_bytes) != (0, 0):
            raise CheckFailure(
                "a spill is not a DDR *placement*: the buffer still has a local home, so "
                "the ddr_resident counters must stay zero"
            )

    return _with_extra_check(case, check)


def case_bank_straddling_strip() -> TbCase:
    """R5: a multi-plane strip buffer that really does span a bank
    boundary, at a plane edge.

    The relaxation this design rests on is that the hardware needs "no
    single *request* straddles a bank", not "no buffer straddles a bank":
    `cnn_accel_tensor_mem` serves each request from the bank its address
    decodes to and *clamps* an overrun, so a buffer confined at plane
    granularity is legal exactly as long as no plane crosses a boundary.

    Nothing else in the toolchain can see the difference.
    `reference.py` models the scratchpad as one flat `bytearray` and has
    no banks at all, and the DUT and the reference take their addresses
    from the same plan -- so a plane that DID straddle would be truncated
    on the DUT, filled with stale RAM, and the two would disagree only if
    the stale bytes happened to differ. This case therefore asserts the
    geometry *directly*: at least one local buffer spans a bank boundary
    (or the case is vacuous), and every request against every buffer
    stays inside one (`check_units_confined`), with a plane size chosen
    so that it does not divide the bank. Three pixels wide is a 24-byte
    plane row, so a six-row strip of the 40-channel input is a 144-byte
    plane and a five-row strip of the output a 120-byte one; the banks
    are 512 bytes, which neither divides. The plane grid and the bank
    grid are therefore out of phase everywhere, and a buffer of five
    planes cannot avoid crossing a boundary."""

    def build(model: Model) -> None:
        x = model.input(10, 3, 40, name="x")
        model.output(
            model.conv2d(x, 40, kernel=(3, 3), padding=(1, 1, 1, 1), name="wide")
        )

    def extra(case: TbCase, tiled: TiledModel, selection, untiled) -> None:
        bank = case.planned.bank_bytes
        straddlers = [
            (name, addr, size)
            for name, addr, size in case.planned.local_placements
            if addr // bank != (addr + size - 1) // bank
        ]
        if not straddlers:
            raise CheckFailure(
                f"no local buffer spans a {bank}-byte bank boundary, so this case is not "
                "testing the confinement relaxation at all. Placements: "
                f"{case.planned.local_placements}"
            )
        units = case.planned.local_confine_units
        for name, addr, size in straddlers:
            unit = units.get(name)
            if unit is None or unit >= size:
                raise CheckFailure(
                    f"'{name}' spans a bank but is not plane-confined (unit={unit}, "
                    f"size={size}) -- that is the illegal case, not the relaxed one"
                )
            if bank % unit == 0:
                raise CheckFailure(
                    f"'{name}' has a {unit}-byte plane that divides the {bank}-byte bank, "
                    "so the two grids are in phase and the straddle is trivial"
                )
        _require_fused_traffic(case, expect_pinned_write=_pinned_bytes(tiled))

    return _tiled_case(
        "bank_straddling_strip",
        build,
        seed=708,
        # Eight 512-byte banks. Small banks are what force the straddle:
        # a 5-plane strip buffer is 600-720 bytes, so it cannot fit in one
        # bank at all and the plane confinement is the only thing that
        # makes it placeable.
        num_banks=8,
        bank_words=64,
        recompute_cap=2.0,
        expect_groups=1,
        expect_strips={0: 2},
        extra=extra,
    )


# ---------------------------------------------------------------------------
# Catalogue
# ---------------------------------------------------------------------------



CASE_BUILDERS = (
    case_ot_restream_from_ddr,
    case_resident_weights_strip_pair,
    case_fused_c2f_64ch,
    case_unfused_c2f_64ch,
    case_fused_wide_128ch,
    case_sppf_tiled_negative,
    case_neck_upsample_concat,
    case_stride2_odd_height,
    case_detect_branch_80ch,
    case_eviction_inside_a_strip,
    case_bank_straddling_strip,
)


def all_cases() -> list[TbCase]:
    return [builder() for builder in CASE_BUILDERS]


__all__ = ["CASE_BUILDERS", "TrafficPolicy", "all_cases"]
