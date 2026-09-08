"""GIR -> HIR lowering with capability checks (doc/tosa_compiler_plan.md
§2.2, §4, §6, §13 M6).

`to_hir` picks the target unit that executes each `fused_conv`, checks
every capability the unit advertises (dtypes, kernel/stride/dilation
support, rescale/clamp epilogue limits, `Unit.constraints`), and emits a
`stage='mapped'` `HirModule`: buffers for every input/const/output/
intermediate tensor and one `conv_layer` `HirOp` per `fused_conv`, with
params named exactly like `cnn_accel_model.LayerDesc` fields. Any GIR op
that survived `fuse` unfused (a standalone `conv2d`/`rescale`/`clamp`) has
no accelerator instruction and is rejected here, not silently dropped.
No pass mutates its input; scheduling (`HirOp.seq`) and memory planning
(`Buffer.addr`) are later stages (M7).
"""

from __future__ import annotations

from cnnc.errors import CapabilityError
from cnnc.gir.ir import FusedConvAttrs, Graph, Op, Tensor
from cnnc.hir.ir import Buffer, HirModule, HirOp
from cnnc.hir.verify import verify_hir
from cnnc.lower.layout import (
    activation_bytes,
    pack_bias_tiled,
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
# output-channel (multiplier, shift) table in DDR. Emitting that table as a
# constant buffer and pointing `scale_addr` at it is compiler milestone M12
# (after M11, doc/tosa_compiler_plan.md); until then a
# per-channel rescale the target CAN take is rejected here, at the fused op,
# rather than silently lowered with only channel 0's pair.
_H2_PER_CHANNEL_LOWERING_IMPLEMENTED = False

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

    if conv.in_zp != 0 or conv.w_zp != 0:
        raise CapabilityError(
            "zero points not supported: padding is literal 0 in HW",
            op_id=op.id, stage=_STAGE, unit=unit.name, constraint="conv.zero_point",
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
    if rescale.per_channel and not _H2_PER_CHANNEL_LOWERING_IMPLEMENTED:
        raise CapabilityError(
            "per-channel rescale: scale table (PER_CHANNEL_EN/scale_addr) lowering not implemented yet (M12)",
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


def _accumulator_note(op: Op, kh: int, kw: int, in_channels: int, bias_values: tuple[int, ...]) -> str | None:
    max_bias = max((abs(v) for v in bias_values), default=0)
    worst = kh * kw * in_channels * 128 * 127 + max_bias
    if worst > 2**31 - 1:
        return f"op {op.id}: worst-case accumulation {worst} may exceed int32 (contract D3)"
    return None


def _lower_fused_conv(
    op: Op, graph: Graph, unit: Unit, *, space_name: str, align: int, activation_layout: str, plane_channels: int
) -> tuple[HirOp, Buffer, str | None]:
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

    env = layer_env(params)
    for constraint in unit.constraints:
        violation = check(constraint, env)
        if violation is not None:
            raise CapabilityError(
                f"constraint {constraint.expr!r} violated: actual {violation.actual}, limit "
                f"{constraint.value if constraint.value is not None else constraint.by}",
                op_id=op.id, stage=_STAGE, unit=unit.name, constraint=constraint.expr,
            )

    params["bias_en"] = True
    params["requant_en"] = True
    params.update(_clamp_params(op, attrs, unit))
    params.update(_output_offset_params(op, attrs, unit))
    params["pad_en"] = any(p > 0 for p in (pad_t, pad_b, pad_l, pad_r))
    params["requant_scale"] = int(rescale.multiplier[0])
    params["requant_shift"] = int(rescale.shift[0]) - caps.implicit_shift

    note = _accumulator_note(op, kh, kw, x.shape[3], b.values or ())

    hir_op = HirOp(
        id=y_id,  # placeholder; renumbered by the caller once all ops are known
        unit=unit.name,
        kind="conv_layer",
        params=params,
        reads=(f"%{x_id}", f"%{w_id}", f"%{b_id}"),
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
    return hir_op, y_buf, note


def to_hir(graph: Graph, target: Target) -> HirModule:
    conv_unit = _rounding_gate(target)

    if len(target.memory.spaces) != 1:
        raise AssertionError(f"to_hir MVP requires exactly one memory space, target has {sorted(target.memory.spaces)}")
    space_name, space = next(iter(target.memory.spaces.items()))
    activation_layout = target.memory.activation_layout
    plane_channels = target.memory.activation_plane_channels
    tiling = conv_unit.internal_tiling

    for op in graph.ops:
        if op.kind not in ("const", "fused_conv"):
            reason = _UNFUSED_REASON.get(op.kind, "the accelerator has no addressable instruction for this op kind")
            raise CapabilityError(
                f"no unit implements {op.kind} standalone; not fused because {reason}",
                op_id=op.id, stage=_STAGE, constraint=op.kind,
            )

    fused_ops = [op for op in graph.ops if op.kind == "fused_conv"]

    weight_of: dict[str, str] = {}
    for op in fused_ops:
        _x_id, w_id, b_id = op.inputs
        weight_of[w_id] = "TILED_OHWI"
        weight_of[b_id] = "I32_TILED"

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
        t = graph.tensor(tid)
        layout = weight_of.get(tid, "TILED_OHWI" if len(t.shape) == 4 else "I32_TILED")
        if layout == "TILED_OHWI":
            data = pack_weights_tiled(t.values or (), t.shape, tiling.cin, tiling.cout)
        else:
            data = pack_bias_tiled(t.values or (), tiling.cout)
        buffers[f"%{tid}"] = Buffer(
            id=f"%{tid}", space=space_name, size_bytes=len(data), align=space.align,
            role="const", layout=layout, shape=t.shape, dtype=t.dtype, data=data, gir_tensor=tid,
        )

    ops: list[HirOp] = []
    notes: list[str] = []
    writer_of: dict[str, str] = {}
    for idx, op in enumerate(fused_ops):
        hir_op, y_buf, note = _lower_fused_conv(
            op, graph, conv_unit, space_name=space_name, align=space.align, activation_layout=activation_layout,
            plane_channels=plane_channels,
        )
        op_id = f"#{idx}"
        hir_op = hir_op.replace(id=op_id, deps=tuple(sorted({writer_of[bid] for bid in hir_op.reads if bid in writer_of})))
        ops.append(hir_op)
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
