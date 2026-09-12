"""`tb_cnn_accel_top` cases for channel CONCAT and SPLIT.

A third catalogue alongside `cases.py` and `cases_pool_pad.py`, registered
by `module_cnn_accel.py` in the same loop and following exactly the same
contract (one function per case, each returning a `TbCase`, each naming
its own seed).

What is under test here is a claim about the *absence* of hardware.
Concat and split are the last two operations YOLOv8n needs (every C2f
block splits and concatenates, SPPF concatenates four tensors, and every
FPN/PAN merge concatenates), and neither of them exists in the ISA --
because neither of them needs to.

The section-3 activation layout is channel-plane-major,

    byte offset = ((c_tile * H + y) * W + x) * T + t,   T = 8

so a tensor is `ceil(C/T)` CONTIGUOUS planes of `H*W*T` bytes. A channel
concatenation is therefore its operands' planes laid back to back: if the
planner gives each producer an output base address inside one buffer, the
concatenation has already happened by the time the last producer retires.
A channel split at a multiple-of-`T` boundary is the mirror image, a
sub-range of the parent's planes. Both are address arithmetic; neither
emits a command.

These cases run that claim on the real DUT. Every one of them checks the
DUT's own DDR writeback against `reference.py` element by element, so a
wrong address shows up as wrong *data*, not merely as a suspicious
counter -- and `case_split_concat_roundtrip` additionally proves the
zero-cost half by showing that the program emitted for a network *with* a
concat is byte-for-byte the program emitted for the same network without
one.
"""

from __future__ import annotations

from accel_v2.model import CopyOp, Model
from accel_v2.planner import ComputeStep, MoveStep
from accel_v2.tbcase import CheckFailure, TbCase, build_case

# Same small shapes as the other catalogues: these are integration tests,
# and every engine's arithmetic is already pinned bit-exactly by its own
# unit-level testbench. 8 channels is exactly one activation plane, which
# is what makes these shapes interesting for aliasing rather than just
# small.
_H = 8
_W = 8
_C = 8

_PAD = (1, 1, 1, 1)


# ---------------------------------------------------------------------------
# Shared assertions about a planned program's shape.
# ---------------------------------------------------------------------------


def _copy_steps(case: TbCase) -> list[ComputeStep]:
    return [
        step
        for step in case.planned.steps
        if isinstance(step, ComputeStep) and isinstance(step.op, CopyOp)
    ]


def _require_no_data_movement(case: TbCase) -> None:
    """The design claim, asserted on the emitted program: the concat/split
    in this case produced no `COPY`, no `LOAD` and no `STORE`."""
    copies = _copy_steps(case)
    if copies:
        raise CheckFailure(
            f"case '{case.name}' was expected to alias its concat/split for free, but "
            f"the planner emitted {len(copies)} COPY instruction(s) "
            f"({', '.join(step.op.output.name for step in copies)}):\n"
            f"{case._program_listing()}"
        )
    # A `cold_load` of a graph input is the residency policy's one
    # legitimate DDR round trip and has nothing to do with the concat; a
    # spill or a reload would mean the alias family did not fit, which
    # these deliberately small cases must never provoke.
    moves = [
        step
        for step in case.planned.steps
        if isinstance(step, MoveStep) and step.kind in ("spill", "reload")
    ]
    if moves:
        raise CheckFailure(
            f"case '{case.name}' expected no spill/reload at all, got "
            f"{[step.kind for step in moves]}:\n{case._program_listing()}"
        )


def _with_extra_check(case: TbCase, extra) -> TbCase:
    """Run `extra(case)` (raising `CheckFailure` on disagreement) in
    addition to `TbCase.check_live`'s standard checks: set as the case's
    `extra_check`, which `check_live` runs first."""
    case.extra_check = extra
    return case


# ---------------------------------------------------------------------------
# 1. Concatenation.
# ---------------------------------------------------------------------------


def case_concat_two_way() -> TbCase:
    """The minimal concat: two convolutions write the two halves of one
    16-channel buffer and a third reads the whole of it.

    If plane addressing is right, `b`'s output base is exactly one
    `H*W*T` plane above `a`'s and the consumer simply starts at `a`'s.
    If it is wrong -- an off-by-one plane, or an interleave instead of a
    concatenation -- the consumer convolves the wrong channels and the
    exported result mismatches element by element.
    """

    def build(model: Model) -> None:
        x = model.input(_H, _W, _C, name="x")
        a = model.conv2d(x, _C, kernel=(3, 3), padding=_PAD, name="a")
        b = model.conv2d(x, _C, kernel=(3, 3), padding=_PAD, name="b")
        y = model.concat([a, b], name="y")
        z = model.conv2d(y, _C, kernel=(1, 1), name="z")
        model.output(z)

    return _with_extra_check(
        build_case("concat_two_way", build, seed=301), _require_no_data_movement
    )


def case_concat_four_way_sppf() -> TbCase:
    """YOLOv8n's SPPF shape: a stem plus three successive 5x5/1/2 max
    pools, all four concatenated into a 32-channel tensor.

    `cases_pool_pad.case_sppf_chain` already runs the pool chain but had
    to substitute an `add` for the concatenation "the ISA does not have".
    This is that block as it is actually written -- and it needs no more
    ISA than the chain did.
    """

    def build(model: Model) -> None:
        x = model.input(_H, _W, _C, name="x")
        stem = model.conv2d(x, _C, kernel=(3, 3), padding=_PAD, clamp=(-128, -40), name="stem")
        p1 = model.pool_max(
            stem, kernel=(5, 5), stride=(1, 1), padding=(2, 2, 2, 2), pad_value=-128, name="p1"
        )
        p2 = model.pool_max(
            p1, kernel=(5, 5), stride=(1, 1), padding=(2, 2, 2, 2), pad_value=-128, name="p2"
        )
        p3 = model.pool_max(
            p2, kernel=(5, 5), stride=(1, 1), padding=(2, 2, 2, 2), pad_value=-128, name="p3"
        )
        y = model.concat([stem, p1, p2, p3], name="y")
        # 1x1 keeps the 32-channel weight image inside WEIGHT_BUFFER_DEPTH;
        # what is under test is the addressing of the four planes, not the
        # kernel.
        z = model.conv2d(y, _C, kernel=(1, 1), name="z")
        model.output(z)

    return _with_extra_check(
        build_case("concat_four_way_sppf", build, seed=302), _require_no_data_movement
    )


def case_concat_is_graph_output() -> TbCase:
    """The concatenation itself is the graph output.

    Its buffer is a DDR `OUTPUTS` allocation rather than a scratchpad
    one, so the two producers write their slices straight to DDR and the
    total write traffic must be exactly the concatenated tensor's size --
    no more (which would mean a copy) and no less (which would mean half
    the result never landed). The exported window is the concat, so the
    element-by-element check covers both halves.
    """

    def build(model: Model) -> None:
        x = model.input(_H, _W, _C, name="x")
        a = model.conv2d(x, _C, kernel=(3, 3), padding=_PAD, name="a")
        b = model.conv2d(x, _C, kernel=(3, 3), padding=_PAD, name="b")
        y = model.concat([a, b], name="y")
        model.output(y)

    def check(case: TbCase) -> None:
        _require_no_data_movement(case)
        y = next(t for t in case.model.tensors if t.name == "y")
        if case.planned.traffic.write_bytes != y.size_bytes:
            raise CheckFailure(
                f"a concat graph output must cost exactly its own "
                f"{y.size_bytes} bytes of DDR write traffic, but the plan predicts "
                f"{case.planned.traffic.write_bytes}"
            )

    return _with_extra_check(build_case("concat_is_graph_output", build, seed=303), check)


def case_concat_operand_fork() -> TbCase:
    """A fork: `a` is both an operand of the concatenation and the input
    of an unrelated convolution.

    Reading a tensor out of the middle of a concat buffer is just an
    address, so this must still cost nothing -- and both readers must
    resolve to the *same* address. It is also the case that most easily
    breaks liveness: `a`'s own consumer count reaches zero long before
    the concat buffer is dead.
    """

    def build(model: Model) -> None:
        x = model.input(_H, _W, _C, name="x")
        a = model.conv2d(x, _C, kernel=(3, 3), padding=_PAD, name="a")
        b = model.conv2d(x, _C, kernel=(3, 3), padding=_PAD, name="b")
        y = model.concat([a, b], name="y")
        side = model.conv2d(a, _C, kernel=(3, 3), padding=_PAD, name="side")
        main = model.conv2d(y, _C, kernel=(1, 1), name="main")
        model.output(model.add(side, main, name="out"))

    def check(case: TbCase) -> None:
        _require_no_data_movement(case)
        steps = {
            step.op.name: step
            for step in case.planned.steps
            if isinstance(step, ComputeStep)
        }
        if steps["side"].input_addrs[0] != steps["main"].input_addrs[0]:
            raise CheckFailure(
                "'a' and the concat buffer must start at the same address, but "
                f"'side' reads 0x{steps['side'].input_addrs[0]:08x} and 'main' reads "
                f"0x{steps['main'].input_addrs[0]:08x}"
            )

    return _with_extra_check(build_case("concat_operand_fork", build, seed=304), check)


# ---------------------------------------------------------------------------
# 2. Splitting.
# ---------------------------------------------------------------------------


def case_split_two_consumers() -> TbCase:
    """C2f's split: one 16-channel tensor, two 8-channel halves, each
    feeding a different convolution.

    The two consumers read the same buffer one plane apart. Getting the
    plane offset wrong swaps the halves, which the element-by-element
    check catches immediately -- the two convolutions have different
    random weights, so the results are not symmetric.
    """

    def build(model: Model) -> None:
        x = model.input(_H, _W, _C, name="x")
        h = model.conv2d(x, 2 * _C, kernel=(3, 3), padding=_PAD, name="h")
        lo, hi = model.split(h, [_C, _C], names=["lo", "hi"])
        left = model.conv2d(lo, _C, kernel=(3, 3), padding=_PAD, name="left")
        right = model.conv2d(hi, _C, kernel=(3, 3), padding=_PAD, name="right")
        model.output(model.add(left, right, name="y"))

    def check(case: TbCase) -> None:
        _require_no_data_movement(case)
        # The DUT-versus-reference comparison alone cannot catch a wrong
        # split offset here: both sides read the tensor through the same
        # planned address, so a planner that pointed both halves at the
        # same plane would be self-consistently wrong. Assert the plane
        # geometry directly.
        steps = {
            step.op.name: step
            for step in case.planned.steps
            if isinstance(step, ComputeStep)
        }
        h = next(t for t in case.model.tensors if t.name == "h")
        plane_bytes = h.plane_bytes
        delta = steps["right"].input_addrs[0] - steps["left"].input_addrs[0]
        if delta != plane_bytes:
            raise CheckFailure(
                f"the two split halves must be exactly one {plane_bytes}-byte activation "
                f"plane apart, but 'left' reads 0x{steps['left'].input_addrs[0]:08x} and "
                f"'right' reads 0x{steps['right'].input_addrs[0]:08x} (delta {delta})"
            )

    return _with_extra_check(build_case("split_two_consumers", build, seed=305), check)


def case_split_concat_roundtrip() -> TbCase:
    """**The zero-traffic proof.** A 16-channel tensor is split in half
    and immediately concatenated back, between two convolutions.

    The round trip is the identity, so a correct implementation must be
    indistinguishable from the same network without it. This case asserts
    exactly that, and in the strongest available form: it builds the
    concat-free twin network from the same seed and requires the *emitted
    program image* -- every descriptor, every weight byte, every seeded
    input byte, and the program entry point -- to be byte-for-byte
    identical, along with the whole `DdrTraffic` prediction.

    Byte-identical programs cannot produce different `DDR_WR_BYTES` or
    `DDR_RD_BYTES` on the DUT, so this pins the invariant the design
    rests on: concat and split move no data. If either ever silently cost
    a copy, the twin's program would gain an instruction and this case
    fails before the simulator is even started.
    """

    def build_with(model: Model) -> None:
        x = model.input(_H, _W, _C, name="x")
        h = model.conv2d(x, 2 * _C, kernel=(3, 3), padding=_PAD, name="h")
        lo, hi = model.split(h, [_C, _C], names=["lo", "hi"])
        rejoined = model.concat([lo, hi], name="rejoined")
        model.output(model.conv2d(rejoined, _C, kernel=(1, 1), name="z"))

    def build_without(model: Model) -> None:
        x = model.input(_H, _W, _C, name="x")
        h = model.conv2d(x, 2 * _C, kernel=(3, 3), padding=_PAD, name="h")
        model.output(model.conv2d(h, _C, kernel=(1, 1), name="z"))

    def check(case: TbCase) -> None:
        _require_no_data_movement(case)
        twin = build_case("split_concat_roundtrip_twin", build_without, seed=306)

        if case.planned.traffic != twin.planned.traffic:
            raise CheckFailure(
                "split+concat changed the predicted DDR traffic, so it is not free:\n"
                f"  with concat   : {case.planned.traffic}\n"
                f"  without concat: {twin.planned.traffic}\n"
                f"{case._program_listing()}"
            )
        if len(case.planned.steps) != len(twin.planned.steps):
            raise CheckFailure(
                f"split+concat added {len(case.planned.steps) - len(twin.planned.steps)} "
                f"instruction(s):\n{case._program_listing()}"
            )
        if case.program.program_addr != twin.program.program_addr:
            raise CheckFailure("the two programs do not even start at the same address")
        if case.program.descs != twin.program.descs:
            for index, (mine, theirs) in enumerate(zip(case.program.descs, twin.program.descs)):
                if mine != theirs:
                    raise CheckFailure(
                        f"descriptor {index} differs between the network with the "
                        f"split+concat and the one without:\n  with   : {mine}\n"
                        f"  without: {theirs}"
                    )
            raise CheckFailure("the two descriptor chains have different lengths")
        if case.program.image.words() != twin.program.image.words():
            raise CheckFailure(
                "the two DDR images differ, so the split+concat is not a pure "
                "re-view of the same bytes"
            )

    return _with_extra_check(build_case("split_concat_roundtrip", build_with, seed=306), check)


# ---------------------------------------------------------------------------
# 3. The fallback: a tensor cannot be in two places at once.
# ---------------------------------------------------------------------------


def case_concat_copy_fallback() -> TbCase:
    """A channel *shuffle*: the two halves of a tensor concatenated back
    in the opposite order.

    This one genuinely moves data -- the result is not any contiguous
    range of the parent -- and both operands are already views of that
    parent, so neither can be re-homed. The planner must notice and fall
    back to two explicit `COPY` instructions rather than emit addresses
    that would alias the halves on top of each other. The DUT then has to
    execute those copies correctly, which the element-by-element check of
    the final result proves.
    """

    def build(model: Model) -> None:
        x = model.input(_H, _W, _C, name="x")
        h = model.conv2d(x, 2 * _C, kernel=(3, 3), padding=_PAD, name="h")
        lo, hi = model.split(h, [_C, _C], names=["lo", "hi"])
        swapped = model.concat([hi, lo], name="swapped")
        model.output(model.conv2d(swapped, _C, kernel=(1, 1), name="z"))

    def check(case: TbCase) -> None:
        copies = _copy_steps(case)
        if len(copies) != 2:
            raise CheckFailure(
                f"a swapped concat needs exactly two COPY instructions, got {len(copies)}:\n"
                f"{case._program_listing()}"
            )
        if case.planned.traffic.tensor_store_count:
            raise CheckFailure(
                "the COPY fallback is local-to-local and must not spill to DDR, but the "
                f"plan predicts {case.planned.traffic.tensor_store_count} STORE(s)"
            )

    return _with_extra_check(build_case("concat_copy_fallback", build, seed=307), check)


def case_concat_graph_input_operand() -> TbCase:
    """Concatenating a graph input with a computed tensor.

    A graph input's address is fixed by the DDR `INPUTS` region, so it is
    the other way a tensor can fail to be re-homed. Exactly one `COPY`
    must appear -- for the input -- and the computed operand must still
    be aliased for free.
    """

    def build(model: Model) -> None:
        x = model.input(_H, _W, _C, name="x")
        a = model.conv2d(x, _C, kernel=(3, 3), padding=_PAD, name="a")
        y = model.concat([x, a], name="y")
        model.output(model.conv2d(y, _C, kernel=(1, 1), name="z"))

    def check(case: TbCase) -> None:
        copies = _copy_steps(case)
        if len(copies) != 1:
            raise CheckFailure(
                f"exactly the graph input should have been copied, got {len(copies)} "
                f"COPY instruction(s):\n{case._program_listing()}"
            )
        if copies[0].op.inputs[0].name != "x":
            raise CheckFailure(
                f"the copied operand should be the graph input 'x', got "
                f"'{copies[0].op.inputs[0].name}'"
            )

    return _with_extra_check(build_case("concat_graph_input_operand", build, seed=308), check)


#: Every case, in the order they are registered as VUnit configs.
CASE_BUILDERS = (
    case_concat_two_way,
    case_concat_four_way_sppf,
    case_concat_is_graph_output,
    case_concat_operand_fork,
    case_split_two_consumers,
    case_split_concat_roundtrip,
    case_concat_copy_fallback,
    case_concat_graph_input_operand,
)


def all_cases() -> list[TbCase]:
    """Build every case. Each call builds fresh objects (each with its own
    `DdrMap`), so cases never share DDR allocation state."""
    return [builder() for builder in CASE_BUILDERS]


__all__ = ["CASE_BUILDERS", "all_cases"]
