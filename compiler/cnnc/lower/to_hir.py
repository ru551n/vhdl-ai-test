"""GIR -> HIR lowering with capability checks (doc/tosa_compiler_plan.md
§2.2, §4, §6, §13 M6).

`to_hir` picks the target unit that executes each compute op, checks every
capability the unit advertises (dtypes, kernel/stride/dilation support,
rescale/clamp epilogue limits, `Unit.constraints`), and emits a
`stage='mapped'` `HirModule`: buffers for every input/const/output/
intermediate tensor and one `HirOp` per compute op, with params named
exactly like `cnn_accel_model.LayerDesc` fields. Two op kinds lower today:

* `fused_conv` -> a `conv_layer` on the unit advertising `conv2d`,
  reading input/weight/bias (+ a SCALE_TABLE const when the rescale is
  per-channel);
* `pool` (mode `max`) -> a `max_pool` on the unit advertising
  `max_pool2d`, reading only its input -- pooling needs no constants at
  all.
* `add` -> an `add` on the unit advertising `add` (ISA v2.0's
  `OPCODE_ADD`), reading two activation buffers and no constants.
* `table` -> an `act` on the unit advertising `table` (ISA v2.0's
  `OPCODE_ACT`), reading its input plus a 256-byte ACT_LUT const.
* `upsample` -> an `upsample` on the unit advertising `upsample` (ISA
  v2.0's `OPCODE_UPSAMPLE`), reading only its input.
* `concat`/`slice` -> NO op at all, only buffer views: a channel range of
  an S6 plane-tiled activation is a contiguous range of its buffer, so a
  channel concat is its operands' producers writing into one buffer at
  different plane offsets, and a channel slice is a window into a buffer
  somebody else wrote (see `_lower_concat`/`_lower_slice`).

Any GIR op that survived `fuse` unfused (a standalone `conv2d`/`rescale`/
`clamp`) has no accelerator instruction and is rejected here, not silently
dropped. No pass mutates its input; scheduling (`HirOp.seq`) and memory
planning (`Buffer.addr`) are later stages (M7).

One lowering is not a pure per-op mapping: a convolution whose TOSA
`in_zp` is non-zero needs `pad_value = in_zp` on its descriptor AND a
`bias' = bias - in_zp * sum(w)` fold in the *constant* it reads, so the
bias image packed for that tensor depends on the op that consumes it (see
`_zero_point_bias_delta` and `_fold_bias_for_zero_point`).
"""

from __future__ import annotations

import dataclasses

from cnnc.errors import CapabilityError
from cnnc.gir.ir import (
    PLANE_MAJOR,
    AddAttrs,
    ConcatAttrs,
    DepthToSpaceAttrs,
    FusedConvAttrs,
    Graph,
    Op,
    PoolAttrs,
    SliceAttrs,
    Tensor,
    UpsampleAttrs,
)
from cnnc.hir.ir import Buffer, HirModule, HirOp
from cnnc.hir.verify import verify_hir
from cnnc.lower.layout import (
    activation_bytes,
    pack_act_lut,
    pack_bias_tiled,
    pack_scale_table,
    pack_weights_tiled,
)
from cnnc.target.constraints import check
from cnnc.target.contract import RescaleCaps, Target, Unit

_STAGE = "to_hir"

# ISA v1.1 (HW milestone H1) added `output_offset` and CLAMP_EN/`clamp_min`/
# `clamp_max` (doc/tosa_compiler_plan.md §5 extension 1); M11 lowers TOSA
# `rescale.out_zp` and any `clamp` onto them. Epilogue encoding rule:
#
#   isa_version >= 1.1: `output_offset = out_zp`, `clamp_en = 1`,
#       `[clamp_min, clamp_max]` = the fused clamp's bounds if present,
#       else `[-128, 127]` (the int8 saturate TOSA rescale does anyway),
#       `relu_en = 0`. So the MVP fixture's ReLU is `CLAMP_EN,[0,127]` --
#       bit-identical in behaviour to the v1.0 `RELU_EN` encoding
#       (`cnn_accel_model.bias_requantize_relu`), differing only in the
#       flags byte and W13.
#   isa_version 1.0 (legacy): `relu_en = 1` for clamp `[0,127]`, `0` for
#       `[-128,127]`/none; any other clamp and any `out_zp != 0` is a
#       CapabilityError (no field to carry it).
#
# TOSA `rescale` computes `clamp_i8(s + out_zp)` and a following `clamp`
# narrows within int8, so HW `clamp(s + output_offset, lo, hi)` with
# `[lo, hi]` inside `[-128, 127]` is the exact composition (gir.interp is
# the reference for the TOSA side; see tests/test_fixtures_m11.py).
_W13_ISA_VERSION = "1.1"
_INT8_RANGE = (-128, 127)
# ISA v1.2 (HW milestone H2) added PER_CHANNEL_EN/`scale_addr`: a per-
# output-channel (multiplier, shift) table in DDR (doc/tosa_compiler_plan.md
# §5 extension 2, M12). A `fused_conv` whose rescale is `per_channel` lowers
# to a `conv_layer` that additionally reads one `Buffer(role=const,
# layout=SCALE_TABLE)` -- `pack_scale_table`'s `[OT][pe_rows]` image of
# `(multiplier, shift - implicit_shift)`, the same byte image
# `cnn_accel_model.pack_scale_table_for_hw` defines -- and carries
# `per_channel_en=True`; the backend points W14 `scale_addr` at that buffer.
# The descriptor's scalar `requant_scale`/`requant_shift` are ignored by
# the HW while the flag is set and are emitted as 0. Only a unit whose
# `epilogue.rescale.per_channel` is true (discovered from `PER_CHANNEL_EN`
# in `cnn_accel_constants.FLAGS`, i.e. ISA >= 1.2) gets here; older ISAs
# raise `CapabilityError` in `_check_capabilities`.
_SCALE_TABLE_ISA_VERSION = "1.2"

_ENV_FIELDS = (
    "in_width",
    "in_height",
    "in_channels",
    "out_channels",
    "kernel_h",
    "kernel_w",
    "stride_h",
    "stride_w",
    "pad_top",
    "pad_bottom",
    "pad_left",
    "pad_right",
)

_UNFUSED_REASON = {
    "conv2d": "conv2d without a fused rescale/clamp epilogue has no standalone accelerator instruction",
    "rescale": "rescale without a preceding fused conv2d has no standalone accelerator instruction",
    "clamp": "clamp range is not admissible for fusion (see epilogue.clamp_ranges) and has no standalone accelerator instruction",
}


def select_unit(target: Target, kind: str) -> Unit:
    """The first `target.units` entry advertising `kind` in `Unit.ops`."""
    for unit in target.units:
        if kind in unit.ops:
            return unit
    raise CapabilityError(f"no unit implements {kind!r}", stage=_STAGE, constraint=kind)


def layer_env(params: dict) -> dict:
    """The subset of a `conv_layer` `HirOp.params` that `Unit.constraints`
    expressions may reference (doc/tosa_compiler_plan.md §4)."""
    return {name: params[name] for name in _ENV_FIELDS}


def _dtype_bytes(dtype: str) -> int:
    return 4 if dtype == "i32" else 1


def _isa_at_least(version: str, minimum: str) -> bool:
    def parts(v: str) -> tuple[int, ...]:
        return tuple(int(x) for x in v.split("."))

    return parts(version) >= parts(minimum)


def _rounding_gate(target: Target) -> Unit:
    conv_unit = select_unit(target, "conv2d")
    rounding = conv_unit.epilogue.rescale.rounding
    if rounding != "half_up":
        raise CapabilityError(
            f"target {target.name} rounds {rounding}; TOSA rescale requires half_up",
            stage=_STAGE,
            unit=conv_unit.name,
            constraint="rounding",
        )
    return conv_unit


def _uses_w13_epilogue(unit: Unit) -> bool:
    """True iff `unit` is programmed with the ISA v1.1 epilogue encoding
    (`output_offset` + `CLAMP_EN`/`clamp_min`/`clamp_max`, see module
    comment); False selects the legacy `RELU_EN` encoding."""
    return _isa_at_least(unit.isa_version, _W13_ISA_VERSION)


def _clamp_params(op: Op, attrs: FusedConvAttrs, unit: Unit) -> dict:
    """The epilogue clamp/ReLU part of a `conv_layer`'s params (module
    comment for the rule). Also enforces `epilogue.clamp_ranges`: a fused
    clamp outside the advertised ranges is a lowering bug (FusePass only
    folds admissible clamps) or a hand-built graph, and is rejected."""
    clamp = attrs.clamp
    bounds = _INT8_RANGE if clamp is None else (clamp.min, clamp.max)
    lo, hi = bounds
    if not (_INT8_RANGE[0] <= lo <= hi <= _INT8_RANGE[1]):
        raise CapabilityError(
            f"clamp {bounds} is not a non-empty sub-range of int8 {_INT8_RANGE}",
            op_id=op.id, stage=_STAGE, unit=unit.name, constraint="clamp",
        )
    clamp_ranges = unit.epilogue.clamp_ranges
    if clamp_ranges != "any" and clamp is not None and bounds not in clamp_ranges:
        raise CapabilityError(
            f"clamp {bounds} not in epilogue.clamp_ranges {clamp_ranges}",
            op_id=op.id, stage=_STAGE, unit=unit.name, constraint="clamp",
        )

    if _uses_w13_epilogue(unit):
        return {"relu_en": False, "clamp_en": True, "clamp_min": lo, "clamp_max": hi}

    # Legacy ISA v1.0 encoding: only ReLU and plain int8 saturate exist.
    if bounds == (0, 127):
        return {"relu_en": True}
    if bounds == _INT8_RANGE:
        return {"relu_en": False}
    raise CapabilityError(
        f"general clamp {bounds} requires isa_version >= {_W13_ISA_VERSION} (CLAMP_EN/clamp_min/clamp_max); "
        f"unit {unit.name!r} is isa_version {unit.isa_version}",
        op_id=op.id, stage=_STAGE, unit=unit.name, constraint="clamp",
    )


def _output_offset_params(op: Op, attrs: FusedConvAttrs, unit: Unit) -> dict:
    """`output_offset` (= TOSA `rescale.out_zp`) for ISA >= 1.1; on a v1.0
    unit only `out_zp == 0` is representable."""
    out_zp = attrs.rescale.out_zp
    if _uses_w13_epilogue(unit):
        return {"output_offset": int(out_zp)}
    if out_zp != 0:
        raise CapabilityError(
            f"rescale out_zp {out_zp} != 0 requires isa_version >= {_W13_ISA_VERSION} (output_offset); "
            f"unit {unit.name!r} is isa_version {unit.isa_version}",
            op_id=op.id, stage=_STAGE, unit=unit.name, constraint="rescale.out_zp",
        )
    return {}


def _check_capabilities(op: Op, x: Tensor, w: Tensor, b: Tensor, y: Tensor, unit: Unit) -> None:
    attrs: FusedConvAttrs = op.attrs
    conv, rescale = attrs.conv, attrs.rescale
    caps: RescaleCaps = unit.epilogue.rescale

    if x.shape[0] != unit.batch:
        raise CapabilityError(
            f"batch {x.shape[0]} != unit batch {unit.batch}", op_id=op.id, stage=_STAGE, unit=unit.name, constraint="batch"
        )

    for role, actual in (("input", x.dtype), ("weight", w.dtype), ("bias", b.dtype), ("output", y.dtype)):
        want = unit.dtypes.get(role)
        if want is not None and actual != want:
            raise CapabilityError(
                f"{role} dtype {actual!r} != required {want!r}",
                op_id=op.id, stage=_STAGE, unit=unit.name, constraint=f"dtypes.{role}",
            )
    want_acc = unit.dtypes.get("accumulator")
    if want_acc is not None and conv.acc_dtype != want_acc:
        raise CapabilityError(
            f"accumulator dtype {conv.acc_dtype!r} != required {want_acc!r}",
            op_id=op.id, stage=_STAGE, unit=unit.name, constraint="dtypes.accumulator",
        )

    if conv.dilation not in unit.dilation:
        raise CapabilityError(
            f"dilation {conv.dilation} not in {unit.dilation}", op_id=op.id, stage=_STAGE, unit=unit.name, constraint="dilation"
        )

    oc, kh, kw, _ic = w.shape
    if unit.kernels != "any" and (kh, kw) not in unit.kernels:
        raise CapabilityError(
            f"kernel {(kh, kw)} not in {unit.kernels}", op_id=op.id, stage=_STAGE, unit=unit.name, constraint="kernels"
        )

    if unit.strides != "any" and conv.stride not in unit.strides:
        raise CapabilityError(
            f"stride {conv.stride} not in {unit.strides}", op_id=op.id, stage=_STAGE, unit=unit.name, constraint="strides"
        )

    if conv.w_zp != 0:
        # Folding a weight zero-point would need `w_zp * sum(x)` over each
        # window, i.e. a term that depends on the INPUT and so cannot be
        # baked into a constant at compile time (unlike `in_zp`, see
        # `_zero_point_bias_delta`). Symmetric int8 weight quantization --
        # what TOSA emits for conv2d -- always has `w_zp == 0`.
        raise CapabilityError(
            f"weight zero point {conv.w_zp} != 0 cannot be folded: the correction term "
            "w_zp * sum(x) over each window depends on the input, not just on constants",
            op_id=op.id, stage=_STAGE, unit=unit.name, constraint="conv.w_zp",
        )
    if conv.in_zp != 0:
        pad_en = any(p > 0 for p in conv.pad)
        if pad_en and not _isa_at_least(unit.isa_version, _PAD_VALUE_ISA_VERSION):
            raise CapabilityError(
                f"conv input zero point {conv.in_zp} != 0 with padding needs the ISA "
                f"v{_PAD_VALUE_ISA_VERSION} 'pad_value' field (TOSA pads with in_zp; a hardware that "
                f"pads with literal 0 adds sum(w) * in_zp to every border output); unit {unit.name!r} "
                f"is isa_version {unit.isa_version}",
                op_id=op.id, stage=_STAGE, unit=unit.name, constraint="conv.in_zp",
            )
        lo, hi = _INT8_RANGE
        if not (lo <= conv.in_zp <= hi):
            raise CapabilityError(
                f"conv in_zp {conv.in_zp} outside the signed pad_value byte range {_INT8_RANGE}",
                op_id=op.id, stage=_STAGE, unit=unit.name, constraint="conv.in_zp",
            )

    if rescale.in_zp != 0:
        raise CapabilityError(
            f"rescale in_zp {rescale.in_zp} != 0", op_id=op.id, stage=_STAGE, unit=unit.name, constraint="rescale.in_zp"
        )

    if rescale.out_zp != 0 and not caps.output_zp:
        raise CapabilityError(
            f"rescale out_zp {rescale.out_zp} != 0 not supported by target",
            op_id=op.id, stage=_STAGE, unit=unit.name, constraint="rescale.out_zp",
        )
    if not (_INT8_RANGE[0] <= rescale.out_zp <= _INT8_RANGE[1]):
        raise CapabilityError(
            f"rescale out_zp {rescale.out_zp} outside int8 {_INT8_RANGE}",
            op_id=op.id, stage=_STAGE, unit=unit.name, constraint="rescale.out_zp",
        )

    if rescale.per_channel and not caps.per_channel:
        raise CapabilityError(
            "per-channel rescale not supported by target",
            op_id=op.id, stage=_STAGE, unit=unit.name, constraint="rescale.per_channel",
        )
    if rescale.per_channel and not _isa_at_least(unit.isa_version, _SCALE_TABLE_ISA_VERSION):
        raise CapabilityError(
            f"per-channel rescale needs the ISA v{_SCALE_TABLE_ISA_VERSION} scale table "
            f"(PER_CHANNEL_EN/scale_addr); unit is ISA v{unit.isa_version}",
            op_id=op.id, stage=_STAGE, unit=unit.name, constraint="rescale.per_channel",
        )
    if rescale.per_channel and len(rescale.multiplier) != oc:
        raise CapabilityError(
            f"per-channel rescale has {len(rescale.multiplier)} multipliers for {oc} output channels",
            op_id=op.id, stage=_STAGE, unit=unit.name, constraint="rescale.per_channel",
        )

    if rescale.rounding != "SINGLE_ROUND":
        raise CapabilityError(
            f"rounding {rescale.rounding} != SINGLE_ROUND", op_id=op.id, stage=_STAGE, unit=unit.name, constraint="rescale.rounding"
        )

    for shift in rescale.shift:
        if not (caps.shift_min <= shift <= caps.shift_max):
            raise CapabilityError(
                f"shift {shift} outside [{caps.shift_min}, {caps.shift_max}]",
                op_id=op.id, stage=_STAGE, unit=unit.name, constraint="rescale.shift",
            )

    mult_max = 2 ** (caps.multiplier_bits - 1) - 1
    for multiplier in rescale.multiplier:
        if not (0 <= multiplier <= mult_max):
            raise CapabilityError(
                f"multiplier {multiplier} outside [0, {mult_max}]",
                op_id=op.id, stage=_STAGE, unit=unit.name, constraint="rescale.multiplier",
            )
    del oc


def _zero_point_bias_delta(w: Tensor, oc: int) -> tuple[int, ...]:
    """`-in_zp * sum(w[o])` per output channel, without the `in_zp` factor:
    returns `sum_{kh,kw,ic} w[o, kh, kw, ic]` for each `o`.

    Why the fold exists at all. TOSA computes

        acc[o] = sum_window (x_pad - in_zp) * w      (x_pad padded with in_zp)

    so a padded tap contributes exactly nothing. The accelerator computes

        hw[o] = sum_window x_pad * w                 (x_pad padded with pad_value)

    With `pad_value = in_zp` the two see the SAME padded input, and

        acc[o] = hw[o] - in_zp * sum_window w

    where the sum runs over the whole kernel -- padded taps included --
    and is therefore the same at every output position, not just in the
    interior. That position-independence is what makes this a compile-time
    constant fold rather than a per-output correction, and it holds only
    because `pad_value == in_zp`. Setting one without the other is
    silently wrong: `pad_value` alone leaves the `-in_zp * sum(w)` term
    unsubtracted, and the fold alone mis-corrects every border output."""
    values = w.values
    if values is None or len(values) != w.numel:
        # Without the actual weights there is no fold, and returning zeros
        # would silently disable the correction rather than fail.
        raise CapabilityError(
            f"weight tensor %{w.id} has no compile-time values, so bias cannot be folded for a "
            "non-zero input zero-point",
            stage=_STAGE, constraint="conv.in_zp",
        )
    per_channel = w.numel // oc
    return tuple(sum(values[o * per_channel : (o + 1) * per_channel]) for o in range(oc))


def _fold_bias_for_zero_point(
    op: Op, bias: Tensor, weight: Tensor, in_zp: int
) -> tuple[int, ...]:
    """`bias'[o] = bias[o] - in_zp * sum(w[o])`, range-checked against the
    int32 bias field the hardware actually reads."""
    oc = weight.shape[0]
    sums = _zero_point_bias_delta(weight, oc)
    values = bias.values or (0,) * oc
    folded = tuple(int(values[o]) - in_zp * sums[o] for o in range(oc))
    lo, hi = -(2**31), 2**31 - 1
    for o, value in enumerate(folded):
        if not (lo <= value <= hi):
            raise CapabilityError(
                f"zero-point-folded bias[{o}] = {values[o]} - {in_zp} * {sums[o]} = {value} "
                f"does not fit int32 [{lo}, {hi}]",
                op_id=op.id, stage=_STAGE, constraint="conv.in_zp",
            )
    return folded


def _accumulator_note(op: Op, kh: int, kw: int, in_channels: int, bias_values: tuple[int, ...]) -> str | None:
    max_bias = max((abs(v) for v in bias_values), default=0)
    worst = kh * kw * in_channels * 128 * 127 + max_bias
    if worst > 2**31 - 1:
        return f"op {op.id}: worst-case accumulation {worst} may exceed int32 (contract D3)"
    return None


def _scale_table_buffer(op: Op, y_id: str, rescale, caps: RescaleCaps, unit: Unit, *, space_name: str, align: int) -> Buffer:
    """The ISA v1.2 per-channel `(multiplier, shift)` table as a const HIR
    buffer, `%<y>.scale`, padded to whole `internal_tiling.cout` tiles."""
    shifts = tuple(int(s) - caps.implicit_shift for s in rescale.shift)
    data = pack_scale_table(tuple(int(m) for m in rescale.multiplier), shifts, unit.internal_tiling.cout)
    return Buffer(
        id=f"%{y_id}.scale",
        space=space_name,
        size_bytes=len(data),
        align=align,
        role="const",
        layout="SCALE_TABLE",
        shape=(len(rescale.multiplier), 2),
        dtype="i32",
        data=data,
        gir_tensor=None,
    )


def _lower_fused_conv(
    op: Op, graph: Graph, unit: Unit, *, space_name: str, align: int, activation_layout: str, plane_channels: int
) -> tuple[HirOp, Buffer, Buffer | None, str | None]:
    x_id, w_id, b_id = op.inputs
    y_id = op.outputs[0]
    x, w, b, y = graph.tensor(x_id), graph.tensor(w_id), graph.tensor(b_id), graph.tensor(y_id)

    _check_capabilities(op, x, w, b, y, unit)

    attrs: FusedConvAttrs = op.attrs
    conv, rescale = attrs.conv, attrs.rescale
    caps = unit.epilogue.rescale
    oc, kh, kw, _ic = w.shape
    pad_t, pad_b, pad_l, pad_r = conv.pad
    stride_h, stride_w = conv.stride

    params = {
        "in_width": x.shape[2],
        "in_height": x.shape[1],
        "in_channels": x.shape[3],
        "out_channels": oc,
        "kernel_h": kh,
        "kernel_w": kw,
        "stride_h": stride_h,
        "stride_w": stride_w,
        "pad_top": pad_t,
        "pad_bottom": pad_b,
        "pad_left": pad_l,
        "pad_right": pad_r,
    }

    _check_constraints(op, unit, layer_env(params))

    params["bias_en"] = True
    params["requant_en"] = True
    params.update(_clamp_params(op, attrs, unit))
    params.update(_output_offset_params(op, attrs, unit))
    params["pad_en"] = any(p > 0 for p in (pad_t, pad_b, pad_l, pad_r))
    # ISA v2.1: pad with the input's zero-point, exactly as TOSA does. The
    # matching `bias' = bias - in_zp * sum(w)` fold happens where the bias
    # constant is packed (`_zero_point_bias_delta` / `to_hir`); the two are
    # a pair and neither is correct alone.
    if conv.in_zp != 0:
        params["pad_value"] = int(conv.in_zp)
    reads = [f"%{x_id}", f"%{w_id}", f"%{b_id}"]
    scale_buf: Buffer | None = None
    if rescale.per_channel:
        # The scalar W-fields are dead while PER_CHANNEL_EN is set; the
        # table (4th read, -> W14 `scale_addr` in the backend) carries the
        # per-channel pairs.
        params["requant_scale"] = 0
        params["requant_shift"] = 0
        params["per_channel_en"] = True
        scale_buf = _scale_table_buffer(op, y_id, rescale, caps, unit, space_name=space_name, align=align)
        reads.append(scale_buf.id)
    else:
        params["requant_scale"] = int(rescale.multiplier[0])
        params["requant_shift"] = int(rescale.shift[0]) - caps.implicit_shift

    note = _accumulator_note(op, kh, kw, x.shape[3], b.values or ())

    hir_op = HirOp(
        id=y_id,  # placeholder; renumbered by the caller once all ops are known
        unit=unit.name,
        kind="conv_layer",
        params=params,
        reads=tuple(reads),
        writes=(f"%{y_id}",),
        deps=(),
        gir_op=op.id,
    )

    y_buf = Buffer(
        id=f"%{y_id}",
        space=space_name,
        size_bytes=activation_bytes(y.shape, plane_channels, _dtype_bytes(y.dtype)),
        align=align,
        role="output" if y_id in graph.outputs else "intermediate",
        layout=activation_layout,
        shape=y.shape,
        dtype=y.dtype,
        gir_tensor=y_id,
    )
    return hir_op, y_buf, scale_buf, note


_POOL_ENV_FIELDS = (
    "in_width",
    "in_height",
    "in_channels",
    "pool_kernel_h",
    "pool_kernel_w",
    "pool_stride_h",
    "pool_stride_w",
    "pad_top",
    "pad_bottom",
    "pad_left",
    "pad_right",
)

# ISA v2.1 added the opcode-agnostic W10 `pad_value` byte. Max pooling
# NEEDS it: TOSA pads MAX_POOL2D with the type minimum, and an ISA that
# can only pad with a literal 0 computes a different (wrong) result at
# every border output whose window is mostly negative -- which, for an
# int8 activation tensor, is most of them. So a padded max pool is
# rejected rather than approximated on an older ISA.
_PAD_VALUE_ISA_VERSION = "2.1"


def _check_constraints(op: Op, unit: Unit, env: dict) -> None:
    for constraint in unit.constraints:
        violation = check(constraint, env)
        if violation is not None:
            raise CapabilityError(
                f"constraint {constraint.expr!r} violated: actual {violation.actual}, limit "
                f"{constraint.value if constraint.value is not None else constraint.by}",
                op_id=op.id, stage=_STAGE, unit=unit.name, constraint=constraint.expr,
            )


def _lower_pool(
    op: Op, graph: Graph, unit: Unit, *, space_name: str, align: int,
    activation_layout: str, plane_channels: int,
) -> tuple[HirOp, Buffer]:
    """A GIR `pool` -> one `max_pool` `HirOp` + its output buffer.

    Pooling needs no constants at all (no weights, no bias, no scale
    table), so unlike `_lower_fused_conv` this reads exactly one buffer
    and adds nothing to the constant pool."""
    x_id = op.inputs[0]
    y_id = op.outputs[0]
    x, y = graph.tensor(x_id), graph.tensor(y_id)
    attrs: PoolAttrs = op.attrs

    if attrs.mode != "max":
        raise CapabilityError(
            f"pool mode {attrs.mode!r} has no lowering (only 'max'; see "
            "frontend.tosa_import._import_avg_pool2d for why average pooling is refused)",
            op_id=op.id, stage=_STAGE, unit=unit.name, constraint="pool.mode",
        )
    if x.shape[0] != unit.batch:
        raise CapabilityError(
            f"batch {x.shape[0]} != unit batch {unit.batch}",
            op_id=op.id, stage=_STAGE, unit=unit.name, constraint="batch",
        )
    for role, actual in (("input", x.dtype), ("output", y.dtype)):
        want = unit.dtypes.get(role)
        if want is not None and actual != want:
            raise CapabilityError(
                f"{role} dtype {actual!r} != required {want!r}",
                op_id=op.id, stage=_STAGE, unit=unit.name, constraint=f"dtypes.{role}",
            )
    if unit.kernels != "any" and attrs.kernel not in unit.kernels:
        raise CapabilityError(
            f"pool kernel {attrs.kernel} not in {unit.kernels}",
            op_id=op.id, stage=_STAGE, unit=unit.name, constraint="kernels",
        )
    if unit.strides != "any" and attrs.stride not in unit.strides:
        raise CapabilityError(
            f"pool stride {attrs.stride} not in {unit.strides}",
            op_id=op.id, stage=_STAGE, unit=unit.name, constraint="strides",
        )

    pad_t, pad_b, pad_l, pad_r = attrs.pad
    pad_en = any(p > 0 for p in attrs.pad)
    if pad_en and not _isa_at_least(unit.isa_version, _PAD_VALUE_ISA_VERSION):
        raise CapabilityError(
            f"padded max pooling needs the ISA v{_PAD_VALUE_ISA_VERSION} 'pad_value' field (TOSA pads "
            f"MAX_POOL2D with the int8 minimum {attrs.pad_value}, not 0); unit {unit.name!r} is "
            f"isa_version {unit.isa_version}",
            op_id=op.id, stage=_STAGE, unit=unit.name, constraint="pad_value",
        )

    params = {
        "in_width": x.shape[2],
        "in_height": x.shape[1],
        "in_channels": x.shape[3],
        "pool_kernel_h": attrs.kernel[0],
        "pool_kernel_w": attrs.kernel[1],
        "pool_stride_h": attrs.stride[0],
        "pool_stride_w": attrs.stride[1],
        "pad_top": pad_t,
        "pad_bottom": pad_b,
        "pad_left": pad_l,
        "pad_right": pad_r,
    }
    _check_constraints(op, unit, {name: params[name] for name in _POOL_ENV_FIELDS})
    params["pad_en"] = pad_en
    # Written unconditionally, not only when padding is on: `pad_value` is
    # a descriptor field, and leaving it stale/implicit is exactly the
    # class of bug that made padded pooling wrong before ISA v2.1.
    params["pad_value"] = int(attrs.pad_value)

    hir_op = HirOp(
        id=y_id,  # placeholder; renumbered by the caller once all ops are known
        unit=unit.name,
        kind="max_pool",
        params=params,
        reads=(f"%{x_id}",),
        writes=(f"%{y_id}",),
        deps=(),
        gir_op=op.id,
    )
    y_buf = Buffer(
        id=f"%{y_id}",
        space=space_name,
        size_bytes=activation_bytes(y.shape, plane_channels, _dtype_bytes(y.dtype)),
        align=align,
        role="output" if y_id in graph.outputs else "intermediate",
        layout=activation_layout,
        shape=y.shape,
        dtype=y.dtype,
        gir_tensor=y_id,
    )
    return hir_op, y_buf


# ISA v2.0 added the elementwise/resample family (`ADD` and friends). It
# is a separate rung from the v2.1 `pad_value` one because it is a
# separate thing: v2.0 added opcodes, v2.1 added a descriptor byte.
_ELEMENTWISE_ISA_VERSION = "2.0"

_ADD_ENV_FIELDS = ("in_width", "in_height", "in_channels")


def _lower_add(
    op: Op, graph: Graph, unit: Unit, *, space_name: str, align: int,
    activation_layout: str, plane_channels: int,
) -> tuple[HirOp, Buffer]:
    """A GIR `add` -> one `add` `HirOp` + its output buffer.

    Reads TWO activation buffers and writes one; no constants at all (the
    single `(requant_scale, requant_shift)` pair rides in the descriptor's
    own W9 fields, and the second operand's ADDRESS rides in W15
    `xfer_bytes` -- the backend's job, see `emit._build_add_descriptor`).
    """
    a_id, b_id = op.inputs
    y_id = op.outputs[0]
    a, b, y = graph.tensor(a_id), graph.tensor(b_id), graph.tensor(y_id)
    attrs: AddAttrs = op.attrs
    caps: RescaleCaps = unit.epilogue.rescale

    if not _isa_at_least(unit.isa_version, _ELEMENTWISE_ISA_VERSION):
        raise CapabilityError(
            f"elementwise add needs the ISA v{_ELEMENTWISE_ISA_VERSION} ADD opcode; unit "
            f"{unit.name!r} is isa_version {unit.isa_version}",
            op_id=op.id, stage=_STAGE, unit=unit.name, constraint="add",
        )
    if a.shape[0] != unit.batch:
        raise CapabilityError(
            f"batch {a.shape[0]} != unit batch {unit.batch}",
            op_id=op.id, stage=_STAGE, unit=unit.name, constraint="batch",
        )
    for role, actual in (("input", a.dtype), ("input", b.dtype), ("output", y.dtype)):
        want = unit.dtypes.get(role)
        if want is not None and actual != want:
            raise CapabilityError(
                f"{role} dtype {actual!r} != required {want!r}",
                op_id=op.id, stage=_STAGE, unit=unit.name, constraint=f"dtypes.{role}",
            )
    if a.shape != b.shape or y.shape != a.shape:
        raise CapabilityError(
            f"add shapes {a.shape}/{b.shape}/{y.shape} must be identical (no broadcasting)",
            op_id=op.id, stage=_STAGE, unit=unit.name, constraint="shape",
        )
    if not (caps.shift_min <= attrs.shift <= caps.shift_max):
        raise CapabilityError(
            f"add shift {attrs.shift} outside [{caps.shift_min}, {caps.shift_max}]",
            op_id=op.id, stage=_STAGE, unit=unit.name, constraint="rescale.shift",
        )
    mult_max = 2 ** (caps.multiplier_bits - 1) - 1
    if not (0 <= attrs.multiplier <= mult_max):
        raise CapabilityError(
            f"add multiplier {attrs.multiplier} outside [0, {mult_max}]",
            op_id=op.id, stage=_STAGE, unit=unit.name, constraint="rescale.multiplier",
        )

    params = {
        "in_width": a.shape[2],
        "in_height": a.shape[1],
        "in_channels": a.shape[3],
    }
    _check_constraints(op, unit, {name: params[name] for name in _ADD_ENV_FIELDS})
    params["requant_en"] = True
    params["requant_scale"] = int(attrs.multiplier)
    params["requant_shift"] = int(attrs.shift) - caps.implicit_shift

    hir_op = HirOp(
        id=y_id,  # placeholder; renumbered by the caller once all ops are known
        unit=unit.name,
        kind="add",
        params=params,
        reads=(f"%{a_id}", f"%{b_id}"),
        writes=(f"%{y_id}",),
        deps=(),
        gir_op=op.id,
    )
    y_buf = Buffer(
        id=f"%{y_id}",
        space=space_name,
        size_bytes=activation_bytes(y.shape, plane_channels, _dtype_bytes(y.dtype)),
        align=align,
        role="output" if y_id in graph.outputs else "intermediate",
        layout=activation_layout,
        shape=y.shape,
        dtype=y.dtype,
        gir_tensor=y_id,
    )
    return hir_op, y_buf


def _lower_table(
    op: Op, graph: Graph, unit: Unit, *, space_name: str, align: int,
    activation_layout: str, plane_channels: int,
) -> tuple[HirOp, Buffer, Buffer]:
    """A GIR `table` -> one `act` `HirOp`, its output buffer, AND the
    256-byte ACT_LUT const buffer it reads.

    The LUT buffer is built here rather than in `to_hir`'s generic const
    loop because its byte image is not the tensor's own values: TOSA and
    the hardware index the same 256 answers differently, and
    `lower.layout.pack_act_lut` is the rotation between them.
    """
    x_id, table_id = op.inputs
    y_id = op.outputs[0]
    x, table, y = graph.tensor(x_id), graph.tensor(table_id), graph.tensor(y_id)

    if not _isa_at_least(unit.isa_version, _ELEMENTWISE_ISA_VERSION):
        raise CapabilityError(
            f"table activation needs the ISA v{_ELEMENTWISE_ISA_VERSION} ACT opcode; unit "
            f"{unit.name!r} is isa_version {unit.isa_version}",
            op_id=op.id, stage=_STAGE, unit=unit.name, constraint="table",
        )
    if x.shape[0] != unit.batch:
        raise CapabilityError(
            f"batch {x.shape[0]} != unit batch {unit.batch}",
            op_id=op.id, stage=_STAGE, unit=unit.name, constraint="batch",
        )
    for role, actual in (("input", x.dtype), ("output", y.dtype)):
        want = unit.dtypes.get(role)
        if want is not None and actual != want:
            raise CapabilityError(
                f"{role} dtype {actual!r} != required {want!r}",
                op_id=op.id, stage=_STAGE, unit=unit.name, constraint=f"dtypes.{role}",
            )

    params = {
        "in_width": x.shape[2],
        "in_height": x.shape[1],
        "in_channels": x.shape[3],
    }
    _check_constraints(op, unit, {name: params[name] for name in _ADD_ENV_FIELDS})
    params["act_lut_en"] = True

    lut_data = pack_act_lut(table.values)
    lut_buf = Buffer(
        id=f"%{table_id}",
        space=space_name,
        size_bytes=len(lut_data),
        align=align,
        role="const",
        layout="ACT_LUT",
        shape=table.shape,
        dtype=table.dtype,
        data=lut_data,
        gir_tensor=table_id,
    )
    hir_op = HirOp(
        id=y_id,  # placeholder; renumbered by the caller once all ops are known
        unit=unit.name,
        kind="act",
        params=params,
        reads=(f"%{x_id}", lut_buf.id),
        writes=(f"%{y_id}",),
        deps=(),
        gir_op=op.id,
    )
    y_buf = Buffer(
        id=f"%{y_id}",
        space=space_name,
        size_bytes=activation_bytes(y.shape, plane_channels, _dtype_bytes(y.dtype)),
        align=align,
        role="output" if y_id in graph.outputs else "intermediate",
        layout=activation_layout,
        shape=y.shape,
        dtype=y.dtype,
        gir_tensor=y_id,
    )
    return hir_op, y_buf, lut_buf


#: The only replication factor `OPCODE_UPSAMPLE` implements. Read from
#: the golden model rather than restated, so a hardware that ever grows a
#: second factor does not leave a stale 2 here -- see
#: `_hardware_upsample_factor`.
_UPSAMPLE_FACTOR_CONSTRAINT = "upsample.factor"


def _lower_upsample(
    op: Op, graph: Graph, unit: Unit, *, space_name: str, align: int,
    activation_layout: str, plane_channels: int, hw_factor: int,
) -> tuple[HirOp, Buffer]:
    """A GIR `upsample` -> one `upsample` `HirOp` + its output buffer.

    The descriptor carries only the INPUT geometry: the output is
    `hw_factor` times larger in both spatial dimensions by definition of
    the opcode, and there is no field to say otherwise. So any other
    factor is refused here rather than encoded into an instruction that
    would quietly do 2x."""
    x_id = op.inputs[0]
    y_id = op.outputs[0]
    x, y = graph.tensor(x_id), graph.tensor(y_id)
    attrs: UpsampleAttrs = op.attrs

    if not _isa_at_least(unit.isa_version, _ELEMENTWISE_ISA_VERSION):
        raise CapabilityError(
            f"nearest-neighbour upsample needs the ISA v{_ELEMENTWISE_ISA_VERSION} UPSAMPLE opcode; "
            f"unit {unit.name!r} is isa_version {unit.isa_version}",
            op_id=op.id, stage=_STAGE, unit=unit.name, constraint="upsample",
        )
    if attrs.factor != hw_factor:
        raise CapabilityError(
            f"upsample factor {attrs.factor} != {hw_factor}: OPCODE_UPSAMPLE replicates by exactly "
            f"{hw_factor} in both spatial dimensions and has no field for any other factor "
            "(doc/cnn_accel_top_v2_arch.md section 12 limitation 5)",
            op_id=op.id, stage=_STAGE, unit=unit.name, constraint=_UPSAMPLE_FACTOR_CONSTRAINT,
        )
    if x.shape[0] != unit.batch:
        raise CapabilityError(
            f"batch {x.shape[0]} != unit batch {unit.batch}",
            op_id=op.id, stage=_STAGE, unit=unit.name, constraint="batch",
        )
    for role, actual in (("input", x.dtype), ("output", y.dtype)):
        want = unit.dtypes.get(role)
        if want is not None and actual != want:
            raise CapabilityError(
                f"{role} dtype {actual!r} != required {want!r}",
                op_id=op.id, stage=_STAGE, unit=unit.name, constraint=f"dtypes.{role}",
            )

    params = {
        "in_width": x.shape[2],
        "in_height": x.shape[1],
        "in_channels": x.shape[3],
    }
    # The unit's geometry bounds are ISA field widths, and it is the INPUT
    # geometry that goes in those fields -- but the OUTPUT is what has to
    # fit in memory and in any downstream op, so check both.
    _check_constraints(op, unit, {name: params[name] for name in _ADD_ENV_FIELDS})
    _check_constraints(
        op, unit,
        {"in_width": y.shape[2], "in_height": y.shape[1], "in_channels": y.shape[3]},
    )

    hir_op = HirOp(
        id=y_id,  # placeholder; renumbered by the caller once all ops are known
        unit=unit.name,
        kind="upsample",
        params=params,
        reads=(f"%{x_id}",),
        writes=(f"%{y_id}",),
        deps=(),
        gir_op=op.id,
    )
    y_buf = Buffer(
        id=f"%{y_id}",
        space=space_name,
        size_bytes=activation_bytes(y.shape, plane_channels, _dtype_bytes(y.dtype)),
        align=align,
        role="output" if y_id in graph.outputs else "intermediate",
        layout=activation_layout,
        shape=y.shape,
        dtype=y.dtype,
        gir_tensor=y_id,
    )
    return hir_op, y_buf


#: The first ISA version with `OPCODE_DEPTH_TO_SPACE` and its W10
#: `dts_factor` byte.
_DEPTH_TO_SPACE_ISA_VERSION = "2.2"

_DEPTH_TO_SPACE_FACTOR_CONSTRAINT = "depth_to_space.factor"
_DEPTH_TO_SPACE_ORDER_CONSTRAINT = "depth_to_space.channel_order"
_DEPTH_TO_SPACE_TILE_CONSTRAINT = "depth_to_space.out_channels"


def _lower_depth_to_space(
    op: Op, graph: Graph, unit: Unit, *, space_name: str, align: int,
    activation_layout: str, plane_channels: int, hw_factor: int,
) -> tuple[HirOp, Buffer]:
    """A GIR `depth_to_space` -> one `depth_to_space` `HirOp` + its output
    buffer.

    Unlike `_lower_upsample`, the descriptor needs BOTH channel counts:
    this is the only opcode in the elementwise family whose output
    channel count differs from its input's, so `out_channels` cannot be
    implied by the geometry the way UPSAMPLE's is (and the hardware's
    defensive geometry re-check reads it -- see the DEPTH_TO_SPACE block
    in `cnn_accel_elementwise.vhd`).

    Three things are refused here rather than encoded into an instruction
    that would quietly compute something else:

      * a factor the hardware does not implement (`dts_factor` IS an ISA
        field, so this is a hardware-capability check, not an encoding
        one);
      * a CHANNEL-major grouping -- the hardware only does plane-major,
        and `passes.depth_to_space_channels` is what turns the other one
        into it. Reaching here still channel-major means that pass could
        not find a permutable producer, and the honest answer is to say
        so, not to emit the plane-major opcode against channel-major data;
      * an `out_channels` that is not a whole number of activation-plane
        channel tiles. The engine moves whole `T`-byte beats and has no
        hardware to gather byte lanes out of one, so a sub-tile
        `out_channels` (the luma-only `1` and the RGB `3` of a real
        super-resolution model, notably) is `ERR_BAD_GEOMETRY` on the
        device; padding the producing convolution's output channels up to
        a multiple of `T` is a graph-level change, so it is named as such
        rather than attempted behind the caller's back.
    """
    x_id = op.inputs[0]
    y_id = op.outputs[0]
    x, y = graph.tensor(x_id), graph.tensor(y_id)
    attrs: DepthToSpaceAttrs = op.attrs

    if not _isa_at_least(unit.isa_version, _DEPTH_TO_SPACE_ISA_VERSION):
        raise CapabilityError(
            f"depth-to-space needs the ISA v{_DEPTH_TO_SPACE_ISA_VERSION} DEPTH_TO_SPACE opcode; "
            f"unit {unit.name!r} is isa_version {unit.isa_version}",
            op_id=op.id, stage=_STAGE, unit=unit.name, constraint="depth_to_space",
        )
    if attrs.factor != hw_factor:
        raise CapabilityError(
            f"depth_to_space factor {attrs.factor} != {hw_factor}: OPCODE_DEPTH_TO_SPACE implements "
            f"exactly factor {hw_factor} in v1 hardware (cnn_accel_cmd_proc rejects any other "
            "dts_factor with ERR_BAD_GEOMETRY)",
            op_id=op.id, stage=_STAGE, unit=unit.name, constraint=_DEPTH_TO_SPACE_FACTOR_CONSTRAINT,
        )
    if attrs.channel_order != PLANE_MAJOR:
        raise CapabilityError(
            f"depth_to_space channel_order {attrs.channel_order!r}: OPCODE_DEPTH_TO_SPACE groups "
            f"input channels PLANE-major (cin = (dy*r + dx)*out_channels + c) and has no other "
            "mode. A channel-major (nn.PixelShuffle) grouping is turned into the plane-major one "
            "by permuting the producing convolution's output-channel rows at compile time "
            "(passes.depth_to_space_channels); reaching the lowering still channel-major means no "
            "such producer was found, so the permutation has to be applied to the graph before it "
            "gets here",
            op_id=op.id, stage=_STAGE, unit=unit.name, constraint=_DEPTH_TO_SPACE_ORDER_CONSTRAINT,
        )
    out_channels = y.shape[3]
    if out_channels % plane_channels:
        raise CapabilityError(
            f"depth_to_space out_channels {out_channels} is not a multiple of the activation "
            f"plane width {plane_channels}: the engine moves whole {plane_channels}-byte channel "
            "tiles and cannot gather byte lanes out of one, so the hardware refuses this geometry "
            "with ERR_BAD_GEOMETRY. Pad the producing convolution's output channels up to "
            f"{-(-out_channels // plane_channels) * plane_channels} "
            f"(in_channels {x.shape[3]} -> "
            f"{-(-out_channels // plane_channels) * plane_channels * attrs.factor ** 2}) "
            "and drop the padding channels on the host",
            op_id=op.id, stage=_STAGE, unit=unit.name, constraint=_DEPTH_TO_SPACE_TILE_CONSTRAINT,
        )
    if x.shape[0] != unit.batch:
        raise CapabilityError(
            f"batch {x.shape[0]} != unit batch {unit.batch}",
            op_id=op.id, stage=_STAGE, unit=unit.name, constraint="batch",
        )
    for role, actual in (("input", x.dtype), ("output", y.dtype)):
        want = unit.dtypes.get(role)
        if want is not None and actual != want:
            raise CapabilityError(
                f"{role} dtype {actual!r} != required {want!r}",
                op_id=op.id, stage=_STAGE, unit=unit.name, constraint=f"dtypes.{role}",
            )

    params = {
        "in_width": x.shape[2],
        "in_height": x.shape[1],
        "in_channels": x.shape[3],
        "out_channels": out_channels,
        "dts_factor": attrs.factor,
    }
    # Input geometry goes in the ISA fields; the OUTPUT is what has to fit
    # in memory and in any downstream op. Both are checked, exactly as in
    # `_lower_upsample` -- and here the two differ in the channel axis as
    # well as the spatial ones.
    _check_constraints(op, unit, {name: params[name] for name in _ADD_ENV_FIELDS})
    _check_constraints(
        op, unit,
        {"in_width": y.shape[2], "in_height": y.shape[1], "in_channels": y.shape[3]},
    )

    hir_op = HirOp(
        id=y_id,  # placeholder; renumbered by the caller once all ops are known
        unit=unit.name,
        kind="depth_to_space",
        params=params,
        reads=(f"%{x_id}",),
        writes=(f"%{y_id}",),
        deps=(),
        gir_op=op.id,
    )
    y_buf = Buffer(
        id=f"%{y_id}",
        space=space_name,
        size_bytes=activation_bytes(y.shape, plane_channels, _dtype_bytes(y.dtype)),
        align=align,
        role="output" if y_id in graph.outputs else "intermediate",
        layout=activation_layout,
        shape=y.shape,
        dtype=y.dtype,
        gir_tensor=y_id,
    )
    return hir_op, y_buf


# ---------------------------------------------------------------------------
# Buffer views: `concat` and `slice` (doc/tosa_compiler_plan.md §7).
#
# Neither emits an instruction. Activations live in DDR as decision-S6
# channel planes, `[C/T][H][W][T]`, so a channel range of a tensor is a
# CONTIGUOUS range of whole planes -- which means:
#
#   * a channel `slice` is a window into the buffer its input already
#     lives in (`alias_kind="view"`), and
#   * a channel `concat` is its operands' producers being redirected to
#     write into one buffer at different plane offsets
#     (`alias_kind="part"`), i.e. producer-directed placement, exactly
#     what `accel_v2.model.Model.concat` does with
#     `Tensor.alias_parent`/`alias_plane_offset`.
#
# Two limits follow from the plane granularity itself, and both are
# enforced below rather than assumed:
#
#   * the axis must be the CHANNEL axis. GIR is NHWC, so that is axis 3.
#     (The real YOLOv8n artifact's concats say `axis = 1` because it is
#     NCHW; a graph that reaches this compiler has already been converted,
#     and if it has not, its convolutions would not have imported either.)
#   * only the LAST operand of a concat may have a channel count that is
#     not a whole number of planes. A non-final operand with, say, 12
#     channels occupies 2 planes, of which 4 lanes are padding -- and
#     those padding lanes are exactly where the next operand's first
#     channels have to be. Its producer would overwrite them.
# ---------------------------------------------------------------------------

_CHANNEL_AXIS = 3


def _alias_buffer(buf: Buffer, parent_id: str, plane_offset: int, kind: str) -> Buffer:
    return dataclasses.replace(
        buf, alias_parent=parent_id, alias_plane_offset=plane_offset, alias_kind=kind
    )


def _check_channel_axis(op: Op, axis: int, rank: int, what: str) -> None:
    if axis == _CHANNEL_AXIS and rank == 4:
        return
    raise CapabilityError(
        f"{what} on axis {axis} of a rank-{rank} tensor has no zero-cost lowering: activations are "
        f"stored as channel planes [C/{{T}}][H][W][T] (decision S6), so only the channel axis "
        f"(axis {_CHANNEL_AXIS} of an NHWC tensor) is a contiguous range of a buffer. Any other "
        "axis would interleave every plane and needs a real copy, which this compiler does not "
        "emit",
        op_id=op.id, stage=_STAGE, constraint=f"{what}.axis",
    )


def _lower_slice(
    op: Op, graph: Graph, buffers: dict, *, space_name: str, align: int,
    activation_layout: str, plane_channels: int,
) -> Buffer:
    """A GIR channel `slice` -> a read-only `view` Buffer over its input's
    buffer. No `HirOp` at all."""
    x_id = op.inputs[0]
    y_id = op.outputs[0]
    x, y = graph.tensor(x_id), graph.tensor(y_id)
    attrs: SliceAttrs = op.attrs

    _check_channel_axis(op, _CHANNEL_AXIS, len(x.shape), "slice")
    for axis in range(3):
        if attrs.start[axis] != 0 or attrs.size[axis] != x.shape[axis]:
            raise CapabilityError(
                f"slice takes [{attrs.start[axis]}, {attrs.start[axis] + attrs.size[axis]}) of axis "
                f"{axis} (extent {x.shape[axis]}): only the channel axis may be sliced, every other "
                "axis must be taken whole",
                op_id=op.id, stage=_STAGE, constraint="slice.axis",
            )

    start_c, size_c = attrs.start[_CHANNEL_AXIS], attrs.size[_CHANNEL_AXIS]
    if start_c % plane_channels != 0:
        raise CapabilityError(
            f"slice starts at channel {start_c}, which is not a multiple of the {plane_channels}-"
            f"channel activation plane: a view can only begin on a plane boundary, since a plane is "
            "the smallest thing that is contiguous in DDR",
            op_id=op.id, stage=_STAGE, constraint="slice.start",
        )

    parent_id = f"%{x_id}"
    parent = buffers.get(parent_id)
    if parent is None:
        raise CapabilityError(
            f"slice input %{x_id} has no buffer to view into", op_id=op.id, stage=_STAGE, constraint="slice"
        )
    if parent.layout != activation_layout:
        raise CapabilityError(
            f"slice input %{x_id} has layout {parent.layout!r}, not the {activation_layout!r} "
            "channel-plane activation layout a view is defined over",
            op_id=op.id, stage=_STAGE, constraint="slice",
        )

    return Buffer(
        id=f"%{y_id}",
        space=space_name,
        size_bytes=activation_bytes(y.shape, plane_channels, _dtype_bytes(y.dtype)),
        align=align,
        role="output" if y_id in graph.outputs else "intermediate",
        layout=activation_layout,
        shape=y.shape,
        dtype=y.dtype,
        gir_tensor=y_id,
        alias_parent=parent_id,
        alias_plane_offset=start_c // plane_channels,
        alias_kind="view",
    )


def _storage_sources(buffer_id: str, buffers: dict) -> list:
    """`HirModule.storage_dependencies` over the plain dict `to_hir` is
    still building (no `HirModule` exists yet at that point).

    Only the ancestor direction is needed here: a concat's parts are
    aliased when the concat is lowered, which is always after the ops that
    read them, so nothing lowered so far can be reading a concat result.
    """
    sources = [buffer_id]
    current = buffer_id
    while True:
        buf = buffers.get(current)
        if buf is None or buf.alias_parent is None or buf.alias_parent in sources:
            return sources
        sources.append(buf.alias_parent)
        current = buf.alias_parent


def _concat_groups(op: Op, graph: Graph, buffers: dict, plane_channels: int) -> list:
    """Collapse a concat's operand list into the buffers that will
    actually be placed, as `[(buffer id, channels, description)]`.

    Usually one operand is one buffer. The exception is the shape that
    makes YOLOv8n's C2f block work: several operands that are already
    `slice` VIEWS of one tensor. Their addresses are not theirs to choose
    -- they follow their parent -- so what has to move into the concat
    result is the PARENT, once, covering all of them. That is legal
    exactly when the run of operands are consecutive views tiling the
    whole parent in order, which is checked here; anything else would
    require the parent's pieces to sit at addresses that are not a fixed
    distance apart, and is refused.
    """
    def planes(channels: int) -> int:
        return -(-channels // plane_channels)

    groups: list = []
    seen: set[str] = set()
    index = 0
    while index < len(op.inputs):
        part_id = op.inputs[index]
        buffer_id = f"%{part_id}"
        buf = buffers.get(buffer_id)
        if buf is None:
            raise CapabilityError(
                f"concat operand %{part_id} has no buffer", op_id=op.id, stage=_STAGE, constraint="concat"
            )

        if buf.alias_kind != "view":
            groups.append((buffer_id, graph.tensor(part_id).shape[_CHANNEL_AXIS], f"%{part_id}"))
            index += 1
        else:
            root_id = buf.alias_parent
            root = buffers[root_id]
            members: list[str] = []
            offset = 0
            cursor = index
            while cursor < len(op.inputs):
                candidate = buffers.get(f"%{op.inputs[cursor]}")
                if candidate is None or candidate.alias_kind != "view" or candidate.alias_parent != root_id:
                    break
                if candidate.alias_plane_offset != offset:
                    break
                offset += planes(graph.tensor(op.inputs[cursor]).shape[_CHANNEL_AXIS])
                members.append(op.inputs[cursor])
                cursor += 1
            root_channels = root.shape[_CHANNEL_AXIS]
            if offset != planes(root_channels):
                raise CapabilityError(
                    f"concat operand %{part_id} is a view of {root_id!r}, and the operands around it "
                    f"do not cover {root_id!r} completely and in order (they cover {offset} of its "
                    f"{planes(root_channels)} channel planes). A view cannot be moved on its own -- "
                    "its address is its parent's plus a fixed offset -- so the whole parent has to "
                    "go into the concat result as one piece, which is only possible when its views "
                    "are concatenated back-to-back in their original order",
                    op_id=op.id, stage=_STAGE, constraint="concat.operand",
                )
            groups.append((root_id, root_channels, f"{root_id} (via views %" + ", %".join(members) + ")"))
            index = cursor

        if groups[-1][0] in seen:
            raise CapabilityError(
                f"concat places {groups[-1][0]!r} twice; a tensor cannot occupy two channel ranges "
                "of one result",
                op_id=op.id, stage=_STAGE, constraint="concat.operand",
            )
        seen.add(groups[-1][0])

    return groups


def _lower_concat(
    op: Op, graph: Graph, buffers: dict, *, space_name: str, align: int,
    activation_layout: str, plane_channels: int,
) -> tuple[Buffer, dict]:
    """A GIR channel `concat` -> the result Buffer plus a rewritten
    (aliased) Buffer for every operand. No `HirOp` at all: the operands'
    own producers write the result between them.

    Returns `(result_buffer, {operand buffer id: aliased Buffer})`."""
    y_id = op.outputs[0]
    y = graph.tensor(y_id)
    attrs: ConcatAttrs = op.attrs

    _check_channel_axis(op, attrs.axis, len(y.shape), "concat")

    y_buf = Buffer(
        id=f"%{y_id}",
        space=space_name,
        size_bytes=activation_bytes(y.shape, plane_channels, _dtype_bytes(y.dtype)),
        align=align,
        role="output" if y_id in graph.outputs else "intermediate",
        layout=activation_layout,
        shape=y.shape,
        dtype=y.dtype,
        gir_tensor=y_id,
    )

    groups = _concat_groups(op, graph, buffers, plane_channels)

    rewritten: dict = {}
    channel = 0
    last = len(groups) - 1
    for index, (buffer_id, group_channels, what) in enumerate(groups):
        part_buf = buffers[buffer_id]
        if part_buf.role != "intermediate":
            # An input/const/output operand already lives somewhere else,
            # and redirecting its producer is not an option (it has none,
            # or its address is pinned). Moving it would need a COPY
            # instruction, which this compiler does not schedule.
            raise CapabilityError(
                f"concat operand {what} is a {part_buf.role!r} buffer, which cannot be placed "
                "inside the concat result: only a tensor produced by another op in this graph can "
                "be redirected to write there. Materialising it would need a COPY instruction, "
                "which this compiler does not emit yet",
                op_id=op.id, stage=_STAGE, constraint="concat.operand",
            )
        if part_buf.alias_parent is not None:
            raise CapabilityError(
                f"concat operand {what} is already a view of {part_buf.alias_parent!r}; a tensor "
                "cannot be placed inside two buffers at once",
                op_id=op.id, stage=_STAGE, constraint="concat.operand",
            )
        if part_buf.layout != activation_layout:
            raise CapabilityError(
                f"concat operand {what} has layout {part_buf.layout!r}, not the "
                f"{activation_layout!r} channel-plane activation layout",
                op_id=op.id, stage=_STAGE, constraint="concat.operand",
            )
        if index != last and group_channels % plane_channels != 0:
            raise CapabilityError(
                f"concat operand {index} ({what}) has {group_channels} channels, which is not a "
                f"multiple of the {plane_channels}-channel activation plane, and it is not the last "
                f"operand. Its last plane's {(-group_channels) % plane_channels} padding lanes sit "
                f"exactly where operand {index + 1}'s first channels must go, so its producer would "
                "overwrite them. Only the final operand may have a partial plane",
                op_id=op.id, stage=_STAGE, constraint="concat.operand",
            )

        rewritten[buffer_id] = _alias_buffer(part_buf, y_buf.id, channel // plane_channels, "part")
        channel += group_channels

    if channel != y.shape[_CHANNEL_AXIS]:
        raise CapabilityError(
            f"concat operands cover {channel} channels, result has {y.shape[_CHANNEL_AXIS]}",
            op_id=op.id, stage=_STAGE, constraint="concat",
        )
    return y_buf, rewritten


def to_hir(graph: Graph, target: Target) -> HirModule:
    conv_unit = _rounding_gate(target)

    if len(target.memory.spaces) != 1:
        raise AssertionError(f"to_hir MVP requires exactly one memory space, target has {sorted(target.memory.spaces)}")
    space_name, space = next(iter(target.memory.spaces.items()))
    activation_layout = target.memory.activation_layout
    plane_channels = target.memory.activation_plane_channels
    tiling = conv_unit.internal_tiling

    for op in graph.ops:
        if op.kind not in (
            "const", "fused_conv", "pool", "add", "table", "upsample", "depth_to_space",
            "concat", "slice",
        ):
            reason = _UNFUSED_REASON.get(op.kind, "the accelerator has no addressable instruction for this op kind")
            raise CapabilityError(
                f"no unit implements {op.kind} standalone; not fused because {reason}",
                op_id=op.id, stage=_STAGE, constraint=op.kind,
            )

    # Ops in GIR order: a graph may interleave convolutions and pools, and
    # the HIR op order (which `schedule` then only re-orders within the
    # dependency constraints) has to follow the source order, not group by
    # kind.
    compute_ops = [
        op for op in graph.ops
        if op.kind in (
            "fused_conv", "pool", "add", "table", "upsample", "depth_to_space", "concat", "slice"
        )
    ]
    pool_unit = select_unit(target, "max_pool2d") if any(op.kind == "pool" for op in compute_ops) else None
    add_unit = select_unit(target, "add") if any(op.kind == "add" for op in compute_ops) else None
    table_unit = select_unit(target, "table") if any(op.kind == "table" for op in compute_ops) else None
    upsample_unit = select_unit(target, "upsample") if any(op.kind == "upsample" for op in compute_ops) else None
    dts_unit = (
        select_unit(target, "depth_to_space")
        if any(op.kind == "depth_to_space" for op in compute_ops)
        else None
    )

    weight_of: dict[str, str] = {}
    # ACT LUT operands are consts too, but their byte image is not their
    # own values (`pack_act_lut` rotates the index order), and `_lower_table`
    # builds their buffer itself. Excluded from the generic const loop
    # below so they are packed exactly once, by the code that knows how.
    act_lut_tensors = {op.inputs[1] for op in graph.ops if op.kind == "table"}
    # bias tensor id -> the zero-point-folded values that must be packed
    # instead of the tensor's own (see `_fold_bias_for_zero_point`).
    folded_bias: dict[str, tuple[int, ...]] = {}
    folded_bias_owner: dict[str, str] = {}
    for op in (o for o in compute_ops if o.kind == "fused_conv"):
        _x_id, w_id, b_id = op.inputs
        weight_of[w_id] = "TILED_OHWI"
        weight_of[b_id] = "I32_TILED"
        in_zp = op.attrs.conv.in_zp
        if in_zp == 0:
            continue
        folded = _fold_bias_for_zero_point(op, graph.tensor(b_id), graph.tensor(w_id), in_zp)
        previous = folded_bias.get(b_id)
        if previous is not None and previous != folded:
            # A bias constant shared (CSE'd) by two convolutions whose
            # folds differ would need two different byte images at one
            # address. Duplicating the constant is a graph rewrite, not
            # something the memory planner can do behind the scenes, so
            # say which two ops disagree rather than picking one.
            raise CapabilityError(
                f"bias constant %{b_id} is shared with op {folded_bias_owner[b_id]!r}, but the two "
                "need different zero-point-folded bias images; duplicate the constant before lowering",
                op_id=op.id, stage=_STAGE, constraint="conv.in_zp",
            )
        folded_bias[b_id] = folded
        folded_bias_owner[b_id] = op.id

    buffers: dict[str, Buffer] = {}
    for tid in graph.inputs:
        t = graph.tensor(tid)
        buffers[f"%{tid}"] = Buffer(
            id=f"%{tid}", space=space_name,
            size_bytes=activation_bytes(t.shape, plane_channels, _dtype_bytes(t.dtype)), align=space.align,
            role="input", layout=activation_layout, shape=t.shape, dtype=t.dtype, gir_tensor=tid,
        )

    for op in graph.ops:
        if op.kind != "const":
            continue
        tid = op.outputs[0]
        if tid in act_lut_tensors:
            continue
        t = graph.tensor(tid)
        layout = weight_of.get(tid, "TILED_OHWI" if len(t.shape) == 4 else "I32_TILED")
        if layout == "TILED_OHWI":
            data = pack_weights_tiled(t.values or (), t.shape, tiling.cin, tiling.cout)
        else:
            data = pack_bias_tiled(folded_bias.get(tid, t.values or ()), tiling.cout)
        buffers[f"%{tid}"] = Buffer(
            id=f"%{tid}", space=space_name, size_bytes=len(data), align=space.align,
            role="const", layout=layout, shape=t.shape, dtype=t.dtype, data=data, gir_tensor=tid,
        )

    ops: list[HirOp] = []
    notes: list[str] = []
    writer_of: dict[str, str] = {}
    idx = 0
    for op in compute_ops:
        # `concat`/`slice` produce no instruction at all -- they are the
        # buffer-view lowering (see `_lower_concat`/`_lower_slice`), so
        # they consume no op index and add nothing to `ops`. They still
        # have to run HERE, in GIR order, because a concat REWRITES the
        # buffers its operands' producers already claimed.
        if op.kind == "slice":
            view = _lower_slice(
                op, graph, buffers, space_name=space_name, align=space.align,
                activation_layout=activation_layout, plane_channels=plane_channels,
            )
            buffers[view.id] = view
            continue
        if op.kind == "concat":
            y_buf, rewritten = _lower_concat(
                op, graph, buffers, space_name=space_name, align=space.align,
                activation_layout=activation_layout, plane_channels=plane_channels,
            )
            buffers[y_buf.id] = y_buf
            buffers.update(rewritten)
            continue

        if op.kind == "pool":
            hir_op, y_buf = _lower_pool(
                op, graph, pool_unit, space_name=space_name, align=space.align,
                activation_layout=activation_layout, plane_channels=plane_channels,
            )
            scale_buf, note = None, None
        elif op.kind == "add":
            hir_op, y_buf = _lower_add(
                op, graph, add_unit, space_name=space_name, align=space.align,
                activation_layout=activation_layout, plane_channels=plane_channels,
            )
            scale_buf, note = None, None
        elif op.kind == "upsample":
            hir_op, y_buf = _lower_upsample(
                op, graph, upsample_unit, space_name=space_name, align=space.align,
                activation_layout=activation_layout, plane_channels=plane_channels,
                hw_factor=upsample_unit.upsample_factor,
            )
            scale_buf, note = None, None
        elif op.kind == "depth_to_space":
            hir_op, y_buf = _lower_depth_to_space(
                op, graph, dts_unit, space_name=space_name, align=space.align,
                activation_layout=activation_layout, plane_channels=plane_channels,
                hw_factor=dts_unit.depth_to_space_factor,
            )
            scale_buf, note = None, None
        elif op.kind == "table":
            # `scale_buf` is the generic "this op also brought a const
            # buffer with it" slot; for `table` that const is the ACT LUT.
            hir_op, y_buf, scale_buf = _lower_table(
                op, graph, table_unit, space_name=space_name, align=space.align,
                activation_layout=activation_layout, plane_channels=plane_channels,
            )
            note = None
        else:
            hir_op, y_buf, scale_buf, note = _lower_fused_conv(
                op, graph, conv_unit, space_name=space_name, align=space.align,
                activation_layout=activation_layout, plane_channels=plane_channels,
            )
        op_id = f"#{idx}"
        idx += 1
        # Reads resolve through aliases: an op consuming a `tosa.slice`
        # view depends on whoever wrote the buffer that view looks into.
        deps = set()
        for bid in hir_op.reads:
            for source in _storage_sources(bid, buffers):
                if source in writer_of:
                    deps.add(writer_of[source])
        hir_op = hir_op.replace(id=op_id, deps=tuple(sorted(deps)))
        ops.append(hir_op)
        if scale_buf is not None:
            buffers[scale_buf.id] = scale_buf
        buffers[y_buf.id] = y_buf
        writer_of[y_buf.id] = op_id
        if note is not None:
            notes.append(note)

    entry_inputs = tuple(f"%{tid}" for tid in graph.inputs)
    entry_outputs = tuple(f"%{tid}" for tid in graph.outputs)

    module = HirModule(
        target_name=target.name,
        ops=tuple(ops),
        buffers=buffers,
        entry_inputs=entry_inputs,
        entry_outputs=entry_outputs,
        stage="mapped",
        notes=tuple(notes),
    )
    verify_hir(module, target)
    return module
