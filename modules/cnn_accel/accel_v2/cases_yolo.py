"""`tb_cnn_accel_top` cases shaped like YOLOv8n.

A fourth catalogue alongside `cases.py`, `cases_pool_pad.py` and
`cases_concat_split.py`, registered by `module_cnn_accel.py` in the same
loop and following exactly the same contract (one function per case,
each returning a `TbCase`, each naming its own seed).

The other three catalogues test the accelerator *by feature*: one case
per engine, one per ISA extension, one per aliasing rule. This one tests
it **by topology**. Every construct here is lifted from the block
structure of YOLOv8n -- `Bottleneck`, `C2f`, `SPPF`, a backbone stage,
an FPN/PAN merge, the three-scale head boundary -- because that is the
shape of graph the planner will actually be handed, and because the
interactions those shapes create (a fork feeding both a concat and the
next op, a skip that stays live across a whole stage, a concat whose
operands are themselves split views) are not reachable from any
single-feature case.

Scope, decided up front: the Detect head's softmax / elementwise
multiply / reshape / transpose run on the HOST. These cases model the
network only up to the point where a tensor is written to DDR for the
host to collect, so "the head" here means "three graph outputs".

Everything is checked the way every other top-level case is: the DUT's
own DDR writeback is compared against `reference.py` element by element,
plus the traffic/counter invariants of arch doc section 10. Where a case
is really about *placement* -- a split's plane offset, a concat's slice
addresses -- it additionally asserts the geometry directly. The DUT and
the reference execute the same planned addresses, so a planner address
bug is self-consistently wrong on both sides and invisible to a value
comparison; only an independent statement about the addresses catches
it.

Four things that could not be expressed exactly, stated here rather
than quietly approximated:

* **Scale.** Real YOLOv8n runs 640x640x3 in and 16..256 channels. These
  cases run 8x8x8-ish tensors (a few at 12x12 or 4x4x16 where a
  construct needs the extra room or the simulation time needs the
  saving). The topology is exact; the sizes are not. Nothing here is a
  performance test.
* **SiLU.** Every `Conv` in YOLOv8n is conv-BN-SiLU. The accelerator's
  conv epilogue offers RELU, and SiLU is a separate `ACT` LUT command.
  Putting an `ACT` after *every* convolution would double the command
  count of every case for no extra coverage, so the catalogue uses the
  RELU epilogue by default and `case_silu_through_block` is the case
  that runs a genuine SiLU LUT in a realistic position.
* **A `C2f` under scratchpad pressure does not spill -- it moves to
  DDR.** A concat buffer with producers still ahead of it is
  deliberately never an eviction victim (`planner.evict_one`), and the
  free list is never defragmented, so a `C2f` whose concat family cannot
  be placed contiguously has no spill lowering at all. It used to be a
  hard `ValueError`; it is now a DDR placement (`planner.Planner.plan`'s
  `place_in_ddr`), which is always available because the section-3
  activation layout is byte-identical in DDR and `LOCAL_TENSOR`.
  `case_c2f_concat_in_ddr` is that case, and it is the shape that
  matters for a real network: at YOLOv8n's actual channel counts a
  `C2f` concat family is tens of kilobytes and cannot be on chip at all.
  `case_concat_family_spills` remains the genuinely-spilling neighbour:
  a concat family that is *fully written* before the pressure arrives,
  which is evictable and does spill.
* **Bottleneck kernels.** `ultralytics`' `Bottleneck` is 3x3 then 3x3
  (`Conv(c1, c_, k[0], 1)`, `Conv(c_, c2, k[1], 1, g=g)` with the
  default `k=(3, 3)`), and so is `_bottleneck` here. The 1x1 kernel is
  still exercised everywhere the real network has one -- `C2f`'s own
  `cv1`/`cv2`, `SPPF`'s -- so nothing was lost by making the Bottleneck
  match.
"""

from __future__ import annotations

import math

from accel_v2 import isa
from accel_v2.model import (
    CopyOp,
    Model,
    Tensor,
    alias_byte_offset,
    alias_root,
)
from accel_v2.planner import ComputeStep, MoveStep
from accel_v2.tbcase import CheckFailure, TbCase, TrafficPolicy, build_case

# Same small shapes as every other catalogue. 8 channels is exactly one
# activation plane, which is what makes concat/split offsets interesting
# rather than merely small; 8x8 is wide enough for row tiling and padding
# to do real work.
_H = 8
_W = 8
_C = 8

#: `(top, bottom, left, right)` for a shape-preserving 3x3 convolution.
_PAD = (1, 1, 1, 1)


# ---------------------------------------------------------------------------
# Shared assertions.
# ---------------------------------------------------------------------------


def _with_extra_check(case: TbCase, extra) -> TbCase:
    """Run `extra(case)` (raising `CheckFailure` on disagreement) in
    addition to the standard `post_check`.

    Bound to the instance rather than subclassing `TbCase`, for the same
    reason `cases_concat_split.py` does it: VUnit only ever calls
    `case.post_check`, and this keeps every case in this file a plain
    `build_case` result. Deliberately duplicated rather than imported
    from that file so the two catalogues stay independently editable.
    """
    base = case.post_check

    def post_check(output_path: str) -> bool:
        try:
            extra(case)
        except CheckFailure as exc:
            print(f"\npost_check FAILED for case '{case.name}':\n{exc}\n")
            return False
        return base(output_path)

    case.post_check = post_check  # type: ignore[method-assign]
    return case


def _all_checks(*checks):
    """Compose several `extra` checks into one."""

    def run(case: TbCase) -> None:
        for check in checks:
            check(case)

    return run


def _compute_steps(case: TbCase) -> dict[str, ComputeStep]:
    return {
        step.op.output.name: step
        for step in case.planned.steps
        if isinstance(step, ComputeStep)
    }


def _copy_steps(case: TbCase) -> list[ComputeStep]:
    return [
        step
        for step in case.planned.steps
        if isinstance(step, ComputeStep) and isinstance(step.op, CopyOp)
    ]


def _move_steps(case: TbCase, kind: str) -> list[MoveStep]:
    return [
        step
        for step in case.planned.steps
        if isinstance(step, MoveStep) and step.kind == kind
    ]


def _tensor(case: TbCase, name: str) -> Tensor:
    for t in case.model.tensors:
        if t.name == name:
            return t
    raise CheckFailure(f"case '{case.name}' has no tensor named '{name}'")


def _require_no_spill(case: TbCase) -> None:
    """No compiler-inserted DDR round trip: every intermediate stayed in
    the scratchpad. A `cold_load` of a graph input is the residency
    policy's one legitimate round trip and is allowed."""
    moves = _move_steps(case, "spill") + _move_steps(case, "reload")
    if moves:
        raise CheckFailure(
            f"case '{case.name}' expected no spill/reload, got "
            f"{[step.kind for step in moves]}:\n{case._program_listing()}"
        )


def _require_copy_count(case: TbCase, expected: int) -> None:
    copies = _copy_steps(case)
    if len(copies) != expected:
        names = ", ".join(step.op.output.name for step in copies)
        raise CheckFailure(
            f"case '{case.name}' expected exactly {expected} COPY instruction(s), got "
            f"{len(copies)} ({names or 'none'}):\n{case._program_listing()}"
        )


def _check_concat_geometry(case: TbCase, concat_name: str) -> None:
    """Assert a concat's *placement*, independently of its values.

    The DUT and `reference.py` execute the same planned addresses, so a
    planner that put every operand at the same offset would be
    self-consistently wrong and the element-by-element check would still
    pass. This states the geometry as a separate fact: consecutive
    operands must be exactly `plane_count(operand) * plane_bytes` apart,
    in operand order, and the whole family must be contiguous.
    """
    concat = _tensor(case, concat_name)
    if not concat.alias_parts:
        raise CheckFailure(
            f"'{concat_name}' is not a concat that owns a buffer "
            f"(alias_parts is empty) -- geometry check is meaningless"
        )
    steps = _compute_steps(case)
    plane_bytes = concat.plane_bytes

    addrs: list[tuple[str, int, int]] = []
    for part in concat.alias_parts:
        if part.name not in steps:
            raise CheckFailure(
                f"concat part '{part.name}' has no compute step to read an address "
                f"from:\n{case._program_listing()}"
            )
        addrs.append((part.name, steps[part.name].output_addr, part.plane_count))

    expected_offset = 0
    base = addrs[0][1]
    for name, addr, planes in addrs:
        want = base + expected_offset * plane_bytes
        if addr != want:
            raise CheckFailure(
                f"concat '{concat_name}': operand '{name}' must be written at plane "
                f"{expected_offset} of the concat buffer, i.e. 0x{want:08x}, but the "
                f"planner put it at 0x{addr:08x} "
                f"(delta {addr - base} bytes, one plane is {plane_bytes})\n"
                f"{case._program_listing()}"
            )
        expected_offset += planes

    if expected_offset != concat.plane_count:
        raise CheckFailure(
            f"concat '{concat_name}' covers {expected_offset} planes but the tensor has "
            f"{concat.plane_count}"
        )


def _check_view_geometry(case: TbCase, view_name: str, consumer_name: str) -> None:
    """Assert that `consumer_name` reads `view_name` at exactly the byte
    offset the alias arithmetic says it should, relative to the buffer's
    owning root.

    Same reason as `_check_concat_geometry`: a wrong split offset is
    invisible to a value comparison when both sides use it.
    """
    view = _tensor(case, view_name)
    root = alias_root(view)
    if root is view:
        raise CheckFailure(f"'{view_name}' is not a view of anything")
    steps = _compute_steps(case)
    if consumer_name not in steps:
        raise CheckFailure(f"no compute step produces '{consumer_name}'")
    consumer = steps[consumer_name]
    index = [t.name for t in consumer.op.inputs].index(view_name)
    got = consumer.input_addrs[index]

    if root.name in steps:
        root_addr = steps[root.name].output_addr
    elif root.name in case.planned.tensor_ddr_addr:
        root_addr = case.planned.tensor_ddr_addr[root.name]
    elif root.alias_parts:
        root_addr = steps[root.alias_parts[0].name].output_addr
    else:
        raise CheckFailure(f"cannot determine the address of root '{root.name}'")

    want = root_addr + alias_byte_offset(view)
    if got != want:
        raise CheckFailure(
            f"'{consumer_name}' reads view '{view_name}' at 0x{got:08x}, but the view is "
            f"{alias_byte_offset(view)} bytes into '{root.name}' at 0x{root_addr:08x}, "
            f"i.e. 0x{want:08x}\n{case._program_listing()}"
        )


# ---------------------------------------------------------------------------
# YOLOv8n block builders.
#
# These are the actual block structures, written once and reused, so a
# case body reads as the network diagram it models.
# ---------------------------------------------------------------------------


def _bottleneck(
    model: Model, x: Tensor, channels: int, *, shortcut: bool, prefix: str
) -> Tensor:
    """`ultralytics.nn.modules.block.Bottleneck`.

    `cv1` then `cv2`, plus the residual add when `shortcut` is set. Both
    convolutions are 3x3 with `k//2` padding, exactly as `ultralytics`'
    `Bottleneck` builds them.
    """
    h = model.conv2d(x, channels, kernel=(3, 3), padding=_PAD, name=f"{prefix}_cv1")
    h = model.conv2d(h, channels, kernel=(3, 3), padding=_PAD, name=f"{prefix}_cv2")
    if shortcut:
        h = model.add(h, x, name=f"{prefix}_add")
    return h


def _c2f(
    model: Model,
    x: Tensor,
    channels: int,
    *,
    n: int,
    shortcut: bool = True,
    prefix: str,
) -> Tensor:
    """`ultralytics.nn.modules.block.C2f`, exactly as written there:

        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    So: a 1x1 `cv1` doubling to `2*channels`, a split in half, a chain of
    `n` bottlenecks each fed by the previous one, *every* intermediate
    kept, and a single concat of `2 + n` branches into a 1x1 `cv2`.

    The number of concatenated branches is what varies across the real
    network (n = 1 in nano's backbone stages, n = 1 in the neck), which
    is why the catalogue runs this at more than one `n`.
    """
    cv1 = model.conv2d(x, 2 * channels, kernel=(1, 1), name=f"{prefix}_cv1")
    a, b = model.split(cv1, [channels, channels], names=[f"{prefix}_a", f"{prefix}_b"])
    branches = [a, b]
    tail = b
    for i in range(n):
        tail = _bottleneck(
            model, tail, channels, shortcut=shortcut, prefix=f"{prefix}_m{i}"
        )
        branches.append(tail)
    cat = model.concat(branches, name=f"{prefix}_cat")
    return model.conv2d(cat, channels, kernel=(1, 1), name=f"{prefix}_cv2")


def _sppf(model: Model, x: Tensor, channels: int, *, prefix: str) -> Tensor:
    """`ultralytics.nn.modules.block.SPPF`: a 1x1 `cv1`, the *same* 5x5
    stride-1 pad-2 max pool applied three times in a chain, a concat of
    `[cv1, p1, p2, p3]`, and a 1x1 `cv2`.

    `pad_value=-128` is the int8 quantization zero-point a real quantized
    SPPF pads with; `cases_pool_pad.py` explains why that is only
    *observable* when the tensor is all-negative.
    """
    cv1 = model.conv2d(
        x, channels, kernel=(1, 1), clamp=(-128, -40), name=f"{prefix}_cv1"
    )
    pools = []
    src = cv1
    for i in range(3):
        src = model.pool_max(
            src,
            kernel=(5, 5),
            stride=(1, 1),
            padding=(2, 2, 2, 2),
            pad_value=-128,
            name=f"{prefix}_p{i + 1}",
        )
        pools.append(src)
    cat = model.concat([cv1, *pools], name=f"{prefix}_cat")
    return model.conv2d(cat, channels, kernel=(1, 1), name=f"{prefix}_cv2")


def _silu_lut(scale: float = 1.0 / 16.0) -> list[int]:
    """A genuine SiLU (`x * sigmoid(x)`) table for `model.act`, indexed
    by the raw unsigned activation byte, quantized at `scale`.

    YOLOv8's only activation is SiLU, and on this accelerator SiLU *is*
    the `ACT` LUT -- there is no other way to spell it. Pure Python
    arithmetic, so the table is identical for the reference model and for
    the bytes preloaded into the DUT's DDR image.
    """
    lut = []
    for raw in range(256):
        v = raw - 256 if raw >= 128 else raw
        real = v * scale
        y = real / (1.0 + math.exp(-real))
        q = int(math.floor(y / scale + 0.5))
        lut.append(max(-128, min(127, q)))
    return lut


# ---------------------------------------------------------------------------
# 1. The Bottleneck, with and without its shortcut.
# ---------------------------------------------------------------------------


def case_bottleneck_shortcut() -> TbCase:
    """YOLOv8n `Bottleneck(shortcut=True)` -- the unit inside every
    backbone `C2f`.

    `stem -> cv1 -> cv2 -> add(stem)`. The point is the residual: `stem`
    is produced, then read again two commands later, and must still be
    the untouched bytes in the scratchpad (invariants R3/R4). The stem
    convolution is there so the shortcut operand is a *local* tensor,
    which is what it is in the real network -- a graph input would take
    the DDR path instead and test nothing about residency.
    """

    def build(model: Model) -> None:
        x = model.input(_H, _W, _C, name="x")
        stem = model.conv2d(x, _C, kernel=(3, 3), padding=_PAD, name="stem")
        y = _bottleneck(model, stem, _C, shortcut=True, prefix="bn")
        model.output(y)

    return _with_extra_check(
        build_case("bottleneck_shortcut", build, seed=401),
        _all_checks(_require_no_spill, lambda case: _require_copy_count(case, 0)),
    )


def case_bottleneck_no_shortcut() -> TbCase:
    """YOLOv8n `Bottleneck(shortcut=False)` -- what the neck's `C2f`
    blocks use, because their input and output channel counts differ.

    Structurally it is just a two-convolution chain, so the invariant it
    carries is the strong one: with no residual keeping anything alive,
    every intermediate is dead the instant it is consumed and `DDR_WR_BYTES`
    must be exactly the final output. It is the control for
    `case_bottleneck_shortcut`: if that one fails and this one passes,
    the bug is in liveness, not in the convolutions.
    """

    def build(model: Model) -> None:
        x = model.input(_H, _W, _C, name="x")
        stem = model.conv2d(x, _C, kernel=(3, 3), padding=_PAD, name="stem")
        y = _bottleneck(model, stem, _C, shortcut=False, prefix="bn")
        model.output(y)

    return _with_extra_check(
        build_case("bottleneck_no_shortcut", build, seed=402),
        _all_checks(_require_no_spill, lambda case: _require_copy_count(case, 0)),
    )


# ---------------------------------------------------------------------------
# 2. C2f -- the block YOLOv8 is built out of.
# ---------------------------------------------------------------------------


def _c2f_checks(prefix: str, n: int):
    """A `C2f`'s planner contract.

    The two split halves `a` and `b` are *views* of `cv1`'s buffer, so
    they cannot also be slices of the concat buffer -- a tensor cannot be
    in two places at once -- and the planner must fall back to an
    explicit `COPY` for each. Every bottleneck output, by contrast, is a
    fresh op output and must be placed into its concat slice for free.
    So a `C2f` costs exactly two `COPY` instructions, whatever `n` is.
    That is a real cost of the real block and it is asserted here rather
    than hidden: if it ever changes, a case should fail and be updated
    deliberately.
    """

    def check(case: TbCase) -> None:
        _require_no_spill(case)
        _require_copy_count(case, 2)
        _check_concat_geometry(case, f"{prefix}_cat")
        # `b` feeds the first bottleneck as a view of cv1's buffer; the
        # value check cannot tell a wrong plane offset from a right one.
        _check_view_geometry(case, f"{prefix}_b", f"{prefix}_m0_cv1")

    return check


def case_c2f_n1() -> TbCase:
    """A genuine `C2f(n=1)` -- YOLOv8n's backbone and neck blocks are all
    `n = 1` at the nano scale.

    Three concatenated branches (`a`, `b`, `m0`), so the concat buffer is
    24 channels = 3 activation planes, and `b` is simultaneously a concat
    operand and the input of the bottleneck chain. That double role is
    the single most awkward thing in the whole network for a liveness
    analysis, and it is here in its real form.
    """

    def build(model: Model) -> None:
        x = model.input(_H, _W, _C, name="x")
        stem = model.conv2d(x, _C, kernel=(3, 3), padding=_PAD, name="stem")
        model.output(_c2f(model, stem, _C, n=1, prefix="c2f"))

    return _with_extra_check(build_case("c2f_n1", build, seed=403), _c2f_checks("c2f", 1))


def case_c2f_n2() -> TbCase:
    """`C2f(n=2)` -- the same block at YOLOv8s/m depth, where the
    bottlenecks form a chain and *four* branches are concatenated.

    What changes with `n` is the number of concat operands and the length
    of the dependency chain feeding the last one, so this is the case
    that catches an off-by-one in the plane walk (a bug that a two- or
    three-operand concat can easily be blind to) and a liveness rule that
    frees a concat buffer as soon as its *last* part is written rather
    than when its consumer has run.
    """

    def build(model: Model) -> None:
        x = model.input(_H, _W, _C, name="x")
        stem = model.conv2d(x, _C, kernel=(3, 3), padding=_PAD, name="stem")
        model.output(_c2f(model, stem, _C, n=2, prefix="c2f"))

    return _with_extra_check(build_case("c2f_n2", build, seed=404), _c2f_checks("c2f", 2))


def case_c2f_multi_plane_split() -> TbCase:
    """`C2f` at 16 channels, on a 4x4 tensor to hold the simulation time
    down: `cv1` produces 32, the split boundary is at channel 16, and
    every branch is *two* activation planes.

    Every other split in this repository's tests happens at exactly one
    plane, where "plane offset 1" and "one plane of bytes" are the same
    number and a planner that confused a plane index with a plane count
    would pass anyway. Here they differ, so the geometry assertion is
    load-bearing: `b` must start 2 planes into `cv1`, and the concat's
    three operands must be 0, 2 and 4 planes into its buffer (n=1 gives
    3 branches of 2 planes each).
    """

    def build(model: Model) -> None:
        x = model.input(4, 4, 16, name="x")
        stem = model.conv2d(x, 16, kernel=(1, 1), name="stem")
        model.output(_c2f(model, stem, 16, n=1, prefix="c2f"))

    def check(case: TbCase) -> None:
        _c2f_checks("c2f", 1)(case)
        b = _tensor(case, "c2f_b")
        if b.alias_plane_offset != 2:
            raise CheckFailure(
                f"the second half of a 32-channel split must start at plane 2, "
                f"got {b.alias_plane_offset}"
            )

    return _with_extra_check(build_case("c2f_multi_plane_split", build, seed=405), check)


# ---------------------------------------------------------------------------
# 3. SPPF -- the block the 5x5 pooling work was done for.
# ---------------------------------------------------------------------------


def case_sppf_block() -> TbCase:
    """YOLOv8n's `SPPF` in full: 1x1 `cv1`, three chained 5x5/1/2 max
    pools, a four-way concat and a 1x1 `cv2`.

    `cases_pool_pad.case_sppf_chain` runs the pool chain and
    `cases_concat_split.case_concat_four_way_sppf` runs the four-way
    concat; neither is the block. This is, including the detail that
    matters most to the planner: `cv1` is both the first concat operand
    *and* the input of the pool chain, so its buffer is written as slice
    0 of the concat and then read three commands later, while the pools
    write slices 1..3 of the very buffer they are reading a neighbouring
    slice of. Nothing may be copied.
    """

    def build(model: Model) -> None:
        x = model.input(_H, _W, _C, name="x")
        model.output(_sppf(model, x, _C, prefix="sppf"))

    return _with_extra_check(
        build_case("sppf_block", build, seed=406),
        _all_checks(
            _require_no_spill,
            lambda case: _require_copy_count(case, 0),
            lambda case: _check_concat_geometry(case, "sppf_cat"),
        ),
    )


# ---------------------------------------------------------------------------
# 4. A backbone stage, and the two neck merges.
# ---------------------------------------------------------------------------


def case_backbone_stage() -> TbCase:
    """One YOLOv8n backbone stage: a stride-2 3x3 `Conv` downsample
    followed by a `C2f`.

    The downsample is what makes this more than "a C2f with a conv in
    front": the stage's tensors change shape mid-graph (12x12 in, 6x6
    out), so every buffer the C2f allocates is a different size from the
    one feeding it, and the local allocator has to reuse a freed 12x12
    slot for 8x8 buffers. That is the fragmentation path.
    """

    def build(model: Model) -> None:
        x = model.input(12, 12, _C, name="x")
        down = model.conv2d(
            x, _C, kernel=(3, 3), stride=(2, 2), padding=_PAD, name="down"
        )
        model.output(_c2f(model, down, _C, n=1, prefix="c2f"))

    return _with_extra_check(
        build_case("backbone_stage", build, seed=407), _c2f_checks("c2f", 1)
    )


def case_neck_merge_upsample() -> TbCase:
    """The FPN (top-down) merge: upsample the deep tensor, concatenate it
    with the shallower backbone skip, run a `C2f`.

    The interesting part is not the upsample, it is the *skip*: `shallow`
    is produced first, then the deep branch runs several commands, and
    only then is `shallow` consumed -- by a concat, so it must have been
    written into the right slice of a buffer that did not exist yet when
    it was produced. Both concat operands are op outputs, so it must
    still cost nothing.
    """

    def build(model: Model) -> None:
        x = model.input(_H, _W, _C, name="x")
        shallow = model.conv2d(x, _C, kernel=(3, 3), padding=_PAD, name="shallow")
        deep = model.conv2d(
            shallow, _C, kernel=(3, 3), stride=(2, 2), padding=_PAD, name="deep"
        )
        deep2 = model.conv2d(deep, _C, kernel=(3, 3), padding=_PAD, name="deep2")
        up = model.upsample2x(deep2, name="up")
        merged = model.concat([up, shallow], name="merged")
        model.output(_c2f(model, merged, _C, n=1, shortcut=False, prefix="c2f"))

    return _with_extra_check(
        build_case("neck_merge_upsample", build, seed=408),
        _all_checks(
            _require_no_spill,
            lambda case: _check_concat_geometry(case, "merged"),
            # Two COPYs from the C2f's own split halves, and no others:
            # the merge itself must be free.
            lambda case: _require_copy_count(case, 2),
        ),
    )


def case_neck_merge_downsample() -> TbCase:
    """The PAN (bottom-up) merge, the other half of the neck: a stride-2
    3x3 `Conv` on the high-resolution branch, concatenated with the
    deeper tensor, then a `C2f`.

    Mirror image of `case_neck_merge_upsample`, and it exercises the one
    thing that one cannot: a concat whose two operands were produced at
    *different resolutions* and only agree in shape because of the
    stride-2 convolution in between. A geometry error in the strided
    conv's output shape would show up here as a concat the model refuses
    to build, which is exactly the early failure wanted.
    """

    def build(model: Model) -> None:
        x = model.input(12, 12, _C, name="x")
        high = model.conv2d(x, _C, kernel=(3, 3), padding=_PAD, name="high")
        deep = model.conv2d(
            high, _C, kernel=(3, 3), stride=(2, 2), padding=_PAD, name="deep"
        )
        down = model.conv2d(
            high, _C, kernel=(3, 3), stride=(2, 2), padding=_PAD, name="down"
        )
        merged = model.concat([down, deep], name="merged")
        model.output(_c2f(model, merged, _C, n=1, shortcut=False, prefix="c2f"))

    return _with_extra_check(
        build_case("neck_merge_downsample", build, seed=409),
        _all_checks(
            lambda case: _check_concat_geometry(case, "merged"),
            lambda case: _require_copy_count(case, 2),
        ),
    )


# ---------------------------------------------------------------------------
# 5. The head boundary: three graph outputs.
# ---------------------------------------------------------------------------


def case_multi_scale_head() -> TbCase:
    """The Detect-head boundary: one backbone forking into three scales,
    each ending in its own graph output written to DDR.

    Everything downstream of these three tensors -- the DFL softmax, the
    elementwise multiply, the reshapes and transposes -- runs on the
    HOST, by decision, so this is where the accelerator's program ends.
    What it tests is the multiple-graph-output path: three separate
    `OUTPUTS` allocations, three closing stores, and a `DDR_WR_BYTES`
    that must equal exactly the sum of the three (the default
    `write_bytes_exact` policy), with the export window spanning all
    three and every one compared element by element.

    The three scales have genuinely different resolutions (12x12, 6x6,
    3x3), as in the real network, so a harness that assumed one output
    size would fail here rather than silently checking the same tensor
    three times.
    """

    def build(model: Model) -> None:
        x = model.input(12, 12, _C, name="x")
        p3 = model.conv2d(x, _C, kernel=(3, 3), padding=_PAD, name="p3")
        p4 = model.conv2d(p3, _C, kernel=(3, 3), stride=(2, 2), padding=_PAD, name="p4")
        p5 = model.conv2d(p4, _C, kernel=(3, 3), stride=(2, 2), padding=_PAD, name="p5")
        model.output(model.conv2d(p3, _C, kernel=(1, 1), name="head_p3"))
        model.output(model.conv2d(p4, _C, kernel=(1, 1), name="head_p4"))
        model.output(model.conv2d(p5, _C, kernel=(1, 1), name="head_p5"))

    def check(case: TbCase) -> None:
        want = sum(t.size_bytes for t in case.model.outputs)
        if case.planned.traffic.write_bytes != want:
            raise CheckFailure(
                f"three graph outputs must cost exactly {want} bytes of DDR write "
                f"traffic, but the plan predicts {case.planned.traffic.write_bytes}"
            )
        sizes = {t.size_bytes for t in case.model.outputs}
        if len(sizes) != 3:
            raise CheckFailure(
                f"the three head outputs are meant to be three different sizes, got {sizes}"
            )

    return _with_extra_check(build_case("multi_scale_head", build, seed=410), check)


# ---------------------------------------------------------------------------
# 6. The whole thing, small.
# ---------------------------------------------------------------------------


def case_yolo_e2e() -> TbCase:
    """A YOLOv8n-shaped network end to end: stem, two backbone stages,
    SPPF, an FPN merge, three head outputs.

    Tiny tensors (12x12 in, 8 channels throughout), real topology. This
    is the only case in the catalogue that runs every construct in one
    program, and it is the one that catches interaction bugs the
    per-block cases cannot: an allocator that fragments across stages, a
    concat buffer freed one command too early because a *later* block
    reused the slot, a weight buffer not re-filled after the pool chain.

    Explicitly NOT modelled, by decision: everything after the three
    outputs (the Detect head's softmax, elementwise multiply, reshape and
    transpose) runs on the host.
    """

    def build(model: Model) -> None:
        x = model.input(12, 12, _C, name="x")
        stem = model.conv2d(
            x, _C, kernel=(3, 3), stride=(2, 2), padding=_PAD, name="stem"
        )
        # Backbone stage 1 at 6x6, stage 2 at 3x3.
        s1 = _c2f(model, stem, _C, n=1, prefix="s1")
        down = model.conv2d(
            s1, _C, kernel=(3, 3), stride=(2, 2), padding=_PAD, name="down"
        )
        s2 = _c2f(model, down, _C, n=1, prefix="s2")
        deep = _sppf(model, s2, _C, prefix="sppf")

        # FPN merge back up to 8x8 against the stage-1 skip.
        up = model.upsample2x(deep, name="up")
        merged = model.concat([up, s1], name="merged")
        neck = _c2f(model, merged, _C, n=1, shortcut=False, prefix="neck")

        model.output(model.conv2d(neck, _C, kernel=(1, 1), name="head_p3"))
        model.output(model.conv2d(deep, _C, kernel=(1, 1), name="head_p4"))
        model.output(model.conv2d(deep, _C, kernel=(3, 3), stride=(2, 2), name="head_p5"))

    def check(case: TbCase) -> None:
        _check_concat_geometry(case, "merged")
        _check_concat_geometry(case, "sppf_cat")

    return _with_extra_check(build_case("yolo_e2e", build, seed=411), check)


# ---------------------------------------------------------------------------
# 7. Edge cases the real topology creates.
# ---------------------------------------------------------------------------


def case_fork_four_consumers() -> TbCase:
    """One tensor feeding four different ops.

    YOLOv8n forks a backbone tensor into a head branch, a neck skip and
    the next stage; four consumers is that, plus one. A fork is where a
    reference-counted liveness analysis goes wrong most cheaply: the
    buffer must survive until the *fourth* read, not the first, and every
    reader must resolve to the same address. Both are asserted -- the
    address equality directly, because four consumers reading the same
    wrong address would agree with the reference model perfectly.
    """

    def build(model: Model) -> None:
        x = model.input(_H, _W, _C, name="x")
        hub = model.conv2d(x, _C, kernel=(3, 3), padding=_PAD, name="hub")
        a = model.conv2d(hub, _C, kernel=(1, 1), name="a")
        b = model.conv2d(hub, _C, kernel=(3, 3), padding=_PAD, name="b")
        c = model.pool_max(hub, kernel=(3, 3), stride=(1, 1), padding=_PAD, name="c")
        d = model.act(hub, _silu_lut(), name="d")
        model.output(model.add(model.add(a, b, name="ab"), model.add(c, d, name="cd"), name="y"))

    def check(case: TbCase) -> None:
        _require_no_spill(case)
        steps = _compute_steps(case)
        addrs = {name: steps[name].input_addrs[0] for name in ("a", "b", "c", "d")}
        if len(set(addrs.values())) != 1:
            raise CheckFailure(
                f"all four consumers must read 'hub' at the same address, got {addrs}"
            )
        if steps["hub"].output_addr != next(iter(addrs.values())):
            raise CheckFailure(
                f"'hub' is written at 0x{steps['hub'].output_addr:08x} but read at "
                f"0x{next(iter(addrs.values())):08x}"
            )

    return _with_extra_check(build_case("fork_four_consumers", build, seed=412), check)


def case_diamond_unequal_depth() -> TbCase:
    """A diamond whose two arms are of very different depth: one arm is a
    single 1x1 convolution, the other is a five-command chain, and they
    rejoin in an `add`.

    Every FPN merge in YOLOv8n is a diamond of this shape. The short arm
    is produced early and then sits untouched while the long arm runs,
    which is precisely the liveness window that a "free the oldest
    buffer" policy gets wrong. This case fits in the scratchpad -- see
    `case_long_skip_forces_spill` for the same shape that does not.
    """

    def build(model: Model) -> None:
        x = model.input(_H, _W, _C, name="x")
        root = model.conv2d(x, _C, kernel=(3, 3), padding=_PAD, name="root")
        short = model.conv2d(root, _C, kernel=(1, 1), name="short")
        long = root
        for i in range(5):
            long = model.conv2d(long, _C, kernel=(3, 3), padding=_PAD, name=f"long{i}")
        model.output(model.add(short, long, name="y"))

    return _with_extra_check(
        build_case("diamond_unequal_depth", build, seed=413), _require_no_spill
    )


def case_long_skip_forces_spill() -> TbCase:
    """The same diamond with a scratchpad too small to hold the long-lived
    skip: the planner must spill it and reload it, and this case asserts
    that it actually did.

    A backbone skip in the real network is live across an entire stage,
    which at real channel counts is far more than any scratchpad holds.
    That case must still be bit-exact through the DDR round trip
    (invariant R5) -- and it must be a round trip the *program* contains,
    not a hidden one, which is what comparing `TENSOR_LOAD_COUNT` and
    `TENSOR_STORE_COUNT` against the planner's own explicit counts
    proves.

    The geometry: 1024 bytes of scratchpad is exactly two 8x8x8 tensors,
    while the skip stays live across four intermediates. The assertion
    that a spill happened is what stops this quietly degrading into an
    ordinary local case if the allocator is ever made cleverer.
    """

    def build(model: Model) -> None:
        x = model.input(_H, _W, _C, name="x")
        root = model.conv2d(x, _C, kernel=(3, 3), padding=_PAD, name="root")
        skip = model.conv2d(root, _C, kernel=(1, 1), name="skip")
        h = root
        for i in range(4):
            h = model.conv2d(h, _C, kernel=(3, 3), padding=_PAD, name=f"h{i}")
        model.output(model.add(skip, h, name="y"))

    def check(case: TbCase) -> None:
        spills = _move_steps(case, "spill")
        reloads = _move_steps(case, "reload")
        if not spills or not reloads:
            raise CheckFailure(
                f"this case exists to exercise the spill path, but the plan contains "
                f"{len(spills)} spill(s) and {len(reloads)} reload(s):\n"
                f"{case._program_listing()}"
            )

    return _with_extra_check(
        build_case(
            "long_skip_forces_spill", build, seed=414, num_banks=1, bank_words=128
        ),
        check,
    )


def case_deep_local_chain() -> TbCase:
    """An eight-command chain of alternating 3x3 and 1x1 convolutions,
    every intermediate resident.

    A YOLOv8n backbone is, stripped of its skips, a long chain like this,
    and the rev-2 claim is that a chain costs *no* DDR writes until the
    output. `TrafficPolicy.write_bytes_exact` (on by default) enforces it
    against both the DUT's counter and the testbench's independent AXI
    W-channel monitor; this case additionally asserts the plan itself
    contains no `STORE` at all, so a failure separates "the planner
    decided to spill" from "the hardware wrote DDR when it should not
    have".
    """

    def build(model: Model) -> None:
        t = model.input(_H, _W, _C, name="x")
        for i in range(8):
            kernel = (3, 3) if i % 2 == 0 else (1, 1)
            padding = _PAD if i % 2 == 0 else (0, 0, 0, 0)
            t = model.conv2d(t, _C, kernel=kernel, padding=padding, name=f"h{i}")
        model.output(t)

    def check(case: TbCase) -> None:
        _require_no_spill(case)
        if case.planned.traffic.tensor_store_count != 0:
            raise CheckFailure(
                f"a fully local chain must contain no STORE instruction, got "
                f"{case.planned.traffic.tensor_store_count}:\n{case._program_listing()}"
            )
        y = case.model.outputs[0]
        if case.planned.traffic.write_bytes != y.size_bytes:
            raise CheckFailure(
                f"the only DDR write must be the {y.size_bytes}-byte output, but the plan "
                f"predicts {case.planned.traffic.write_bytes}"
            )

    return _with_extra_check(build_case("deep_local_chain", build, seed=415), check)


def case_concat_operand_forked_elsewhere() -> TbCase:
    """A concat operand that is *also* the input of an unrelated branch,
    in its YOLOv8n form: a backbone tensor that goes both into a neck
    concat and on into the next stage.

    The operand's bytes have to live inside the concat buffer, and the
    other consumer has to read them there -- at the concat's slice
    address, not at some address of its own. Asserted directly: the
    unrelated consumer's operand address must equal the concat slice's,
    which a value comparison alone cannot establish.
    """

    def build(model: Model) -> None:
        x = model.input(_H, _W, _C, name="x")
        shared = model.conv2d(x, _C, kernel=(3, 3), padding=_PAD, name="shared")
        other = model.conv2d(x, _C, kernel=(1, 1), name="other")
        cat = model.concat([shared, other], name="cat")
        branch = model.conv2d(shared, _C, kernel=(3, 3), padding=_PAD, name="branch")
        main = model.conv2d(cat, _C, kernel=(1, 1), name="main")
        model.output(model.add(branch, main, name="y"))

    def check(case: TbCase) -> None:
        _require_no_spill(case)
        _require_copy_count(case, 0)
        _check_concat_geometry(case, "cat")
        steps = _compute_steps(case)
        if steps["branch"].input_addrs[0] != steps["shared"].output_addr:
            raise CheckFailure(
                f"'branch' must read 'shared' where it was written "
                f"(0x{steps['shared'].output_addr:08x}), but reads "
                f"0x{steps['branch'].input_addrs[0]:08x}"
            )
        if steps["main"].input_addrs[0] != steps["shared"].output_addr:
            raise CheckFailure(
                "the concat buffer must start at its first operand's address, but "
                f"'main' reads 0x{steps['main'].input_addrs[0]:08x} while 'shared' was "
                f"written at 0x{steps['shared'].output_addr:08x}"
            )

    return _with_extra_check(
        build_case("concat_operand_forked_elsewhere", build, seed=416), check
    )


def case_nested_concat() -> TbCase:
    """A concat of a concat.

    YOLOv8n's PAN builds merges out of tensors that are themselves the
    outputs of merged blocks; nesting the aliasing one level is the
    structural stress that creates. The inner concat owns no producer of
    its own -- it *is* its parts -- so making it an operand of the outer
    one requires the alias chain to be walked transitively
    (`alias_root`/`alias_plane_offset_total`) rather than one level deep.
    A one-level implementation would place the inner concat's parts
    relative to the inner buffer and lose the outer offset entirely,
    which the geometry assertion below catches directly: `inner`'s two
    parts must be at planes 0 and 1 *of the outer buffer*, and `c` at
    plane 2.
    """

    def build(model: Model) -> None:
        x = model.input(_H, _W, _C, name="x")
        a = model.conv2d(x, _C, kernel=(3, 3), padding=_PAD, name="a")
        b = model.conv2d(x, _C, kernel=(1, 1), name="b")
        c = model.conv2d(x, _C, kernel=(3, 3), padding=_PAD, name="c")
        inner = model.concat([a, b], name="inner")
        outer = model.concat([inner, c], name="outer")
        model.output(model.conv2d(outer, _C, kernel=(1, 1), name="y"))

    def check(case: TbCase) -> None:
        _require_no_spill(case)
        _require_copy_count(case, 0)
        steps = _compute_steps(case)
        plane = _tensor(case, "outer").plane_bytes
        base = steps["a"].output_addr
        for name, planes in (("a", 0), ("b", 1), ("c", 2)):
            want = base + planes * plane
            if steps[name].output_addr != want:
                raise CheckFailure(
                    f"nested concat: '{name}' must land at plane {planes} of the outer "
                    f"buffer (0x{want:08x}), got 0x{steps[name].output_addr:08x}\n"
                    f"{case._program_listing()}"
                )
        if steps["y"].input_addrs[0] != base:
            raise CheckFailure(
                f"the outer concat must start at plane 0 (0x{base:08x}), but 'y' reads "
                f"0x{steps['y'].input_addrs[0]:08x}"
            )

    return _with_extra_check(build_case("nested_concat", build, seed=417), check)


def case_silu_through_block() -> TbCase:
    """A real SiLU (`x * sigmoid(x)`) LUT in a realistic position: after
    a convolution, feeding a concat, with a second SiLU branch alongside.

    SiLU is YOLOv8's only activation and on this accelerator it is the
    `ACT` LUT and nothing else, so this is the shape every `Conv` in the
    real network has. Placing the `ACT` output *directly into a concat
    slice* is the part worth testing: `ACT` writes through the
    elementwise engine rather than the conv epilogue, and the concat
    aliasing must apply to it identically.

    The LUT is generated in pure Python (`_silu_lut`) at a fixed scale of
    1/16, so the same 256 bytes are preloaded into the DUT's DDR image
    and used by the reference model -- the table is data, not a claim.
    """

    def build(model: Model) -> None:
        lut = _silu_lut()
        x = model.input(_H, _W, _C, name="x")
        a = model.conv2d(x, _C, kernel=(3, 3), padding=_PAD, name="a")
        sa = model.act(a, lut, name="sa")
        b = model.conv2d(x, _C, kernel=(1, 1), name="b")
        sb = model.act(b, lut, name="sb")
        cat = model.concat([sa, sb], name="cat")
        model.output(model.conv2d(cat, _C, kernel=(1, 1), name="y"))

    return _with_extra_check(
        build_case("silu_through_block", build, seed=418),
        _all_checks(
            _require_no_spill,
            lambda case: _require_copy_count(case, 0),
            lambda case: _check_concat_geometry(case, "cat"),
        ),
    )


def case_c2f_output_is_graph_output() -> TbCase:
    """A `C2f` whose `cv2` writes straight to DDR as a graph output --
    the last block before the head.

    The block's whole aliasing family (a concat buffer with three slices,
    two of them copies of split views) has to be built in the scratchpad
    while the *final* op's destination is a DDR `OUTPUTS` allocation.
    That mixes the two output spaces inside one block, which no
    per-feature case does, and the exact-write-bytes policy then pins
    that the concat buffer never touched DDR: total writes must be one
    output tensor.
    """

    def build(model: Model) -> None:
        x = model.input(_H, _W, _C, name="x")
        model.output(_c2f(model, x, _C, n=1, prefix="c2f"))

    def check(case: TbCase) -> None:
        _c2f_checks("c2f", 1)(case)
        y = case.model.outputs[0]
        if case.planned.traffic.write_bytes != y.size_bytes:
            raise CheckFailure(
                f"a C2f ending in a graph output must write exactly {y.size_bytes} DDR "
                f"bytes, but the plan predicts {case.planned.traffic.write_bytes}"
            )

    return _with_extra_check(
        build_case("c2f_output_is_graph_output", build, seed=419), check
    )


def case_two_concats_share_operand() -> TbCase:
    """One tensor concatenated into *two different* merges.

    A PAN variant: a backbone tensor that is merged into both the
    top-down and the bottom-up path. A tensor cannot be in two places at
    once, so exactly one of the two concats can alias it for free and the
    other must fall back to a `COPY` -- and it must be the *second* one,
    since the first has already claimed the bytes. This is the sharpest
    test of the "can this operand be re-homed?" rule, because unlike
    `cases_concat_split.case_concat_copy_fallback` (where the operand is
    a split view) or `case_concat_graph_input_operand` (where it is a
    graph input), here the operand is an ordinary op output that *would*
    have been eligible had it not already been spoken for.

    The copy's placement is asserted rather than inferred: `cat2`'s first
    slice must be the copy's output, and the shared operand's own address
    must still be inside `cat1`.
    """

    def build(model: Model) -> None:
        x = model.input(_H, _W, _C, name="x")
        shared = model.conv2d(x, _C, kernel=(3, 3), padding=_PAD, name="shared")
        b = model.conv2d(x, _C, kernel=(1, 1), name="b")
        c = model.conv2d(x, _C, kernel=(1, 1), name="c")
        cat1 = model.concat([shared, b], name="cat1")
        cat2 = model.concat([shared, c], name="cat2")
        y1 = model.conv2d(cat1, _C, kernel=(1, 1), name="y1")
        y2 = model.conv2d(cat2, _C, kernel=(1, 1), name="y2")
        model.output(model.add(y1, y2, name="y"))

    def check(case: TbCase) -> None:
        _require_no_spill(case)
        _require_copy_count(case, 1)
        copy = _copy_steps(case)[0]
        if copy.op.inputs[0].name != "shared":
            raise CheckFailure(
                f"the copied operand must be the already-placed 'shared', got "
                f"'{copy.op.inputs[0].name}'"
            )
        _check_concat_geometry(case, "cat1")
        _check_concat_geometry(case, "cat2")
        steps = _compute_steps(case)
        if steps["y1"].input_addrs[0] != steps["shared"].output_addr:
            raise CheckFailure(
                "'shared' must be slice 0 of 'cat1', but 'y1' reads "
                f"0x{steps['y1'].input_addrs[0]:08x} while 'shared' was written at "
                f"0x{steps['shared'].output_addr:08x}"
            )
        if steps["y2"].input_addrs[0] != copy.output_addr:
            raise CheckFailure(
                "'cat2' must start at its copied first slice, but 'y2' reads "
                f"0x{steps['y2'].input_addrs[0]:08x} while the copy landed at "
                f"0x{copy.output_addr:08x}"
            )

    return _with_extra_check(build_case("two_concats_share_operand", build, seed=421), check)


def case_concat_family_spills() -> TbCase:
    """A whole concat *family* spilled to DDR and reloaded, with its
    slices still addressable afterwards.

    The planner spills and reloads an alias family as one buffer, because
    its slices are byte ranges of it and have nothing finer to evict. Two
    rules meet here and they are easy to get wrong together: a buffer
    with producers still ahead of it is never a victim (it would lose the
    slices already written into it), but once every part *has* been
    written the family becomes an ordinary evictable buffer -- and after
    the reload its consumer must still find slice 0 at the buffer's base.

    The geometry is a three-way merge (1536 bytes, three activation
    planes) whose consumer sits at the far end of a four-command chain,
    in a 2048-byte scratchpad. That is the real situation for a YOLOv8n
    neck skip: live across a whole stage, far too big to keep resident.
    Bit-exactness through the round trip is invariant R5, and it is only
    meaningful if the spill really happened, which this asserts.

    `bank_words` must be a power of two (`cnn_accel_tensor_mem` asserts
    it at elaboration), which is why the scratchpad is one 256-word bank
    rather than the 192 words the pressure would otherwise be tuned to.
    """

    def build(model: Model) -> None:
        x = model.input(_H, _W, _C, name="x")
        a = model.conv2d(x, _C, kernel=(3, 3), padding=_PAD, name="a")
        b = model.conv2d(a, _C, kernel=(1, 1), name="b")
        c = model.conv2d(b, _C, kernel=(1, 1), name="c")
        cat = model.concat([a, b, c], name="cat")
        t = c
        for i in range(4):
            t = model.conv2d(t, _C, kernel=(3, 3), padding=_PAD, name=f"t{i}")
        main = model.conv2d(cat, _C, kernel=(1, 1), name="main")
        model.output(model.add(main, t, name="y"))

    def check(case: TbCase) -> None:
        _require_copy_count(case, 0)
        spills = _move_steps(case, "spill")
        if not any(step.tensor.name == "cat" for step in spills):
            raise CheckFailure(
                "this case exists to spill the concat family itself, but the plan spills "
                f"{[step.tensor.name for step in spills] or 'nothing'}:\n"
                f"{case._program_listing()}"
            )
        reloads = _move_steps(case, "reload")
        if not any(step.tensor.name == "cat" for step in reloads):
            raise CheckFailure(
                f"'cat' was spilled but never reloaded:\n{case._program_listing()}"
            )
        # After the reload, 'main' must read the family at its new base --
        # slice 0, i.e. offset 0. Both sides use the same planned address,
        # so state it as its own fact.
        reload = next(step for step in reloads if step.tensor.name == "cat")
        main = _compute_steps(case)["main"]
        if main.input_addrs[0] != reload.dst_addr:
            raise CheckFailure(
                f"'main' must read the reloaded concat buffer at its new base "
                f"0x{reload.dst_addr:08x}, but reads 0x{main.input_addrs[0]:08x}"
            )

    return _with_extra_check(
        build_case(
            "concat_family_spills", build, seed=422, num_banks=1, bank_words=256
        ),
        check,
    )


def case_two_c2f_blocks_back_to_back() -> TbCase:
    """Two `C2f` blocks in a row, which is what a backbone stage boundary
    looks like once the downsample is stripped out.

    The second block's allocations land in the holes the first one's
    concat family left behind, so this is the allocator-fragmentation
    case: the first `C2f`'s 24-channel concat buffer is freed, then a
    second 24-channel buffer plus five 8-channel ones have to be placed
    into the resulting free list. It is also the case that would catch a
    concat buffer whose family was never fully freed -- the second block
    would then fail to fit and spill, which the no-spill assertion turns
    into a failure instead of a silent slowdown.
    """

    def build(model: Model) -> None:
        x = model.input(_H, _W, _C, name="x")
        stem = model.conv2d(x, _C, kernel=(3, 3), padding=_PAD, name="stem")
        first = _c2f(model, stem, _C, n=1, prefix="c2fa")
        model.output(_c2f(model, first, _C, n=1, shortcut=False, prefix="c2fb"))

    def check(case: TbCase) -> None:
        _require_no_spill(case)
        _require_copy_count(case, 4)  # two per C2f
        _check_concat_geometry(case, "c2fa_cat")
        _check_concat_geometry(case, "c2fb_cat")

    return _with_extra_check(
        build_case("two_c2f_blocks_back_to_back", build, seed=420), check
    )


def case_c2f_concat_in_ddr() -> TbCase:
    """The case the catalogue could not express until the planner learned
    to leave an unplaceable buffer in DDR: a `C2f` whose concat family
    does not fit in the scratchpad.

    The geometry is chosen so that exactly one thing is unplaceable and
    nothing else changes. The scratchpad is 4 KiB -- ample for this graph
    -- but it is four INDEPENDENT 1 KiB banks, and `cnn_accel_tensor_mem`
    serves every transfer from the single bank the address decodes to.
    The 24-channel concat buffer is 8*8*24 = 1536 bytes, larger than a
    bank, so no legal local address for it exists at any occupancy. Every
    other buffer (512 or 1024 bytes) still fits, so the block runs
    exactly as `case_c2f_n1` does except that its concat family lives in
    DDR: the two `COPY`s and the bottleneck's `ADD` write their slices
    straight to DDR at the right plane offsets, and `cv2` reads the
    assembled 24-channel tensor back from there.

    That is not a contrived corner. It is what *every* real `C2f` looks
    like: at YOLOv8n's channel counts the concat family is tens of
    kilobytes and no plausible scratchpad holds it. This case runs that
    lowering end to end at a size a simulator can afford.

    What it asserts beyond the usual element-by-element comparison:

    * the *placement* directly -- which buffers are local and which are
      in DDR -- because the DUT and `reference.py` take their addresses
      from the same plan and would agree on a wrong one;
    * the concat slice geometry, exactly as the local `C2f` cases do,
      which is the claim "DDR and LOCAL_TENSOR have the same layout"
      cashed out as an assertion;
    * `DDR_WR_BYTES` exactly -- now the concat family (1536 bytes) plus
      the graph output (512) -- cross-checked against the testbench's own
      passive AXI monitor, so "it went to DDR" is observed on the bus and
      not merely predicted;
    * no spill: an unplaceable buffer is recognized before the eviction
      loop moves anything, so the fallback must not cost a single extra
      `STORE` beyond the traffic of the buffer itself.
    """

    def build(model: Model) -> None:
        x = model.input(_H, _W, _C, name="x")
        stem = model.conv2d(x, _C, kernel=(3, 3), padding=_PAD, name="stem")
        model.output(_c2f(model, stem, _C, n=1, prefix="c2f"))

    def check(case: TbCase) -> None:
        _require_no_spill(case)
        _require_copy_count(case, 2)
        _check_concat_geometry(case, "c2f_cat")

        placed_in_ddr = {name for name, _, _ in case.planned.ddr_placements}
        if placed_in_ddr != {"c2f_cat"}:
            raise CheckFailure(
                "expected exactly the concat family 'c2f_cat' to be left in DDR, got "
                f"{sorted(placed_in_ddr) or 'nothing'} -- this case is only meaningful "
                "while the concat buffer is the one buffer that does not fit\n"
                f"{case._program_listing()}"
            )
        local = {name for name, _, _ in case.planned.local_placements}
        if local != {"stem", "c2f_cv1", "c2f_m0_cv1", "c2f_m0_cv2"}:
            raise CheckFailure(
                "the rest of the block must still be resident; local buffers are "
                f"{sorted(local)}\n{case._program_listing()}"
            )

        # Every instruction that touches the concat family must name DDR
        # on the side that touches it, and stay inside the buffer.
        cat_addr = case.planned.tensor_ddr_addr["c2f_cat"]
        cat_size = _tensor(case, "c2f_cat").size_bytes
        for index, step in enumerate(case.planned.steps):
            if not isinstance(step, ComputeStep):
                continue
            if alias_root(step.op.output).name != "c2f_cat":
                continue
            if step.output_space != isa.SPACE_DDR:
                raise CheckFailure(
                    f"step {index} writes a slice of the DDR-resident concat buffer "
                    f"but names space {step.output_space}\n{case._program_listing()}"
                )
            if not cat_addr <= step.output_addr < cat_addr + cat_size:
                raise CheckFailure(
                    f"step {index} writes outside the concat buffer "
                    f"[0x{cat_addr:08x}, 0x{cat_addr + cat_size:08x})\n"
                    f"{case._program_listing()}"
                )

        predicted = case.planned.traffic
        if (predicted.ddr_resident_write_bytes, predicted.ddr_resident_read_bytes) != (
            cat_size,
            cat_size,
        ):
            raise CheckFailure(
                "the DDR placement should cost exactly one write and one read of the "
                f"{cat_size}-byte concat buffer, but the plan predicts "
                f"wr={predicted.ddr_resident_write_bytes} "
                f"rd={predicted.ddr_resident_read_bytes}"
            )

    return _with_extra_check(
        build_case(
            "c2f_concat_in_ddr",
            build,
            seed=421,
            # 4 KiB of scratchpad, but as four independent 1 KiB banks --
            # ample for every buffer except the 1536-byte concat family,
            # which no single bank can hold.
            num_banks=4,
            bank_words=128,
        ),
        check,
    )


# ---------------------------------------------------------------------------
# Catalogue
# ---------------------------------------------------------------------------

#: Every case, in the order they are registered as VUnit configs.
CASE_BUILDERS = (
    case_bottleneck_shortcut,
    case_bottleneck_no_shortcut,
    case_c2f_n1,
    case_c2f_n2,
    case_c2f_multi_plane_split,
    case_sppf_block,
    case_backbone_stage,
    case_neck_merge_upsample,
    case_neck_merge_downsample,
    case_multi_scale_head,
    case_yolo_e2e,
    case_fork_four_consumers,
    case_diamond_unequal_depth,
    case_long_skip_forces_spill,
    case_deep_local_chain,
    case_concat_operand_forked_elsewhere,
    case_nested_concat,
    case_silu_through_block,
    case_c2f_output_is_graph_output,
    case_two_concats_share_operand,
    case_concat_family_spills,
    case_two_c2f_blocks_back_to_back,
    case_c2f_concat_in_ddr,
)


def all_cases() -> list[TbCase]:
    """Build every case. Each call builds fresh objects (each with its own
    `DdrMap`), so cases never share DDR allocation state."""
    return [builder() for builder in CASE_BUILDERS]


__all__ = ["CASE_BUILDERS", "TrafficPolicy", "all_cases"]
