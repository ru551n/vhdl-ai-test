"""GIR verifier (doc/tosa_compiler_plan.md §11, "imported GIR" row).

Every violation raises `VerifyError(op_id=..., stage="verify")` naming the
offending op and the violated rule.
"""

from __future__ import annotations

from cnnc.errors import VerifyError
from cnnc.gir.ir import (
    CHANNEL_ORDERS,
    DTYPES,
    AddAttrs,
    ClampAttrs,
    ConcatAttrs,
    ConvAttrs,
    DepthToSpaceAttrs,
    FusedConvAttrs,
    Graph,
    Op,
    PoolAttrs,
    RescaleParams,
    SliceAttrs,
    Tensor,
    UpsampleAttrs,
    conv2d_output_shape,
    depth_to_space_output_shape,
    dtype_range,
    pool2d_output_shape,
    upsample_output_shape,
)

# TOSA v1.0 RESCALE: multiplier is a signed int32; MVP additionally
# requires non-negative (negative multipliers are never produced by the
# quantization schemes this compiler targets, and the HW's requant_scale
# is stored as a plain unsigned magnitude with the sign folded elsewhere).
_MULT_MAX = 2**31

# TOSA v1.0 spec, RESCALE: legal `shift` range when `scale32=true` is
# [2, 62] (an 8-bit encoded field reserving the extremes).
_SHIFT_MIN, _SHIFT_MAX = 2, 62


def _fail(op_id: str | None, message: str) -> None:
    raise VerifyError(message, op_id=op_id, stage="verify")


def verify(graph: Graph) -> None:
    _verify_ssa(graph)
    for tid, t in graph.tensors.items():
        producer = graph.producer(tid)
        op_id = producer.id if producer is not None else f"%{tid}"
        if t.dtype not in DTYPES:
            _fail(op_id, f"tensor %{tid} has unsupported dtype {t.dtype!r}, expected one of {DTYPES}")
    # N == 1 is only meaningful for NHWC activation tensors; OHWI weights
    # and 1-D biases legitimately have a non-1 leading dimension. Graph
    # inputs are the root of every activation's batch dimension in M2's
    # op set (conv2d checks/propagates it via `conv2d_output_shape`;
    # rescale/clamp require identical in/out shape), so checking here is
    # sufficient without misfiring on weight/bias tensors.
    for tid in graph.inputs:
        t = graph.tensors.get(tid)
        if t is not None and len(t.shape) == 4 and t.shape[0] != 1:
            _fail(f"%{tid}", f"graph input %{tid} has batch size {t.shape[0]} != 1 (GIR requires N == 1)")
    for op in graph.ops:
        if op.kind == "const":
            _verify_const(graph, op)
        elif op.kind == "conv2d":
            _verify_conv2d(graph, op)
        elif op.kind == "rescale":
            _verify_rescale(graph, op)
        elif op.kind == "clamp":
            _verify_clamp(graph, op)
        elif op.kind == "pool":
            _verify_pool(graph, op)
        elif op.kind == "add":
            _verify_add(graph, op)
        elif op.kind == "table":
            _verify_table(graph, op)
        elif op.kind == "upsample":
            _verify_upsample(graph, op)
        elif op.kind == "depth_to_space":
            _verify_depth_to_space(graph, op)
        elif op.kind == "concat":
            _verify_concat(graph, op)
        elif op.kind == "slice":
            _verify_slice(graph, op)
        elif op.kind == "fused_conv":
            _verify_fused_conv(graph, op)
        else:
            _fail(op.id, f"unknown op kind {op.kind!r}")
    for tid in graph.outputs:
        if tid not in graph.tensors:
            _fail(f"%{tid}", "graph output tensor is not defined")


def _verify_ssa(graph: Graph) -> None:
    for tid in graph.inputs:
        if tid not in graph.tensors:
            _fail(f"%{tid}", "graph input has no Tensor entry")
    defined: set[str] = set(graph.inputs)
    for op in graph.ops:
        for inp in op.inputs:
            if inp not in defined:
                _fail(op.id, f"use of undefined tensor %{inp}")
        for out in op.outputs:
            if out in defined:
                _fail(op.id, f"tensor %{out} produced more than once")
            if out not in graph.tensors:
                _fail(op.id, f"output %{out} has no Tensor entry")
            defined.add(out)


def _verify_const(graph: Graph, op: Op) -> None:
    t = graph.tensors[op.outputs[0]]
    if t.values is None:
        _fail(op.id, "const op has no values")
        return
    if len(t.values) != t.numel:
        _fail(op.id, f"const value count {len(t.values)} != numel {t.numel}")
    lo, hi = dtype_range(t.dtype)
    for v in t.values:
        if not (lo <= v <= hi):
            _fail(op.id, f"const value {v} out of range for {t.dtype} [{lo}, {hi}]")


def _verify_conv_shapes(op_id: str, x: Tensor, w: Tensor, b: Tensor, out: Tensor, attrs: ConvAttrs) -> None:
    if len(x.shape) != 4:
        _fail(op_id, f"conv input rank {len(x.shape)} != 4")
    if x.shape[0] != 1:
        _fail(op_id, f"conv input batch {x.shape[0]} != 1")
    if len(w.shape) != 4:
        _fail(op_id, f"conv weight rank {len(w.shape)} != 4")
    if len(b.shape) != 1:
        _fail(op_id, f"conv bias rank {len(b.shape)} != 1")
    oc = w.shape[0]
    if b.shape[0] != oc:
        _fail(op_id, f"conv bias length {b.shape[0]} != out_channels {oc}")
    if w.shape[3] != x.shape[3]:
        _fail(op_id, f"conv weight IC {w.shape[3]} != input C {x.shape[3]}")
    if w.dtype != "i8":
        _fail(op_id, f"conv weight dtype {w.dtype!r} must be i8")
    if b.dtype != "i32":
        _fail(op_id, f"conv bias dtype {b.dtype!r} must be i32")
    if any(p < 0 for p in attrs.pad):
        _fail(op_id, f"conv pad {attrs.pad} has a negative element")
    if any(s < 1 for s in attrs.stride):
        _fail(op_id, f"conv stride {attrs.stride} must be >= 1")
    if any(d < 1 for d in attrs.dilation):
        _fail(op_id, f"conv dilation {attrs.dilation} must be >= 1")
    # TOSA/IREE additionally require the stride to evenly divide the
    # padded input extent minus the dilated kernel span, i.e. the
    # `conv2d_output_shape` division below must be exact; a non-exact
    # division is silently floored into a smaller-than-expected output
    # that IREE's conv lowering rejects at codegen time, so catch it here.
    kh, kw = w.shape[1], w.shape[2]
    in_h, in_w = x.shape[1], x.shape[2]
    pad_t, pad_b, pad_l, pad_r = attrs.pad
    stride_h, stride_w = attrs.stride
    dil_h, dil_w = attrs.dilation
    rem_h = (in_h - 1 + pad_t + pad_b - dil_h * (kh - 1)) % stride_h
    if rem_h != 0:
        _fail(
            op_id,
            f"conv stride_h {stride_h} does not evenly divide padded H extent "
            f"(in_h={in_h}, pad=({pad_t},{pad_b}), k={kh}, dilation={dil_h}): remainder {rem_h}",
        )
    rem_w = (in_w - 1 + pad_l + pad_r - dil_w * (kw - 1)) % stride_w
    if rem_w != 0:
        _fail(
            op_id,
            f"conv stride_w {stride_w} does not evenly divide padded W extent "
            f"(in_w={in_w}, pad=({pad_l},{pad_r}), k={kw}, dilation={dil_w}): remainder {rem_w}",
        )
    if attrs.acc_dtype != "i32":
        _fail(op_id, f"conv acc_dtype {attrs.acc_dtype!r} must be i32")
    expected = conv2d_output_shape(x.shape, w.shape, attrs)
    if out.shape != expected:
        _fail(op_id, f"conv output shape {out.shape} != computed {expected}")
    if out.dtype != attrs.acc_dtype:
        _fail(op_id, f"conv output dtype {out.dtype} != acc_dtype {attrs.acc_dtype}")
    in_lo, in_hi = dtype_range(x.dtype)
    if not (in_lo <= attrs.in_zp <= in_hi):
        _fail(op_id, f"conv in_zp {attrs.in_zp} out of range for {x.dtype}")
    w_lo, w_hi = dtype_range(w.dtype)
    if not (w_lo <= attrs.w_zp <= w_hi):
        _fail(op_id, f"conv w_zp {attrs.w_zp} out of range for {w.dtype}")


def _verify_conv2d(graph: Graph, op: Op) -> None:
    x, w, b = (graph.tensors[i] for i in op.inputs)
    out = graph.tensors[op.outputs[0]]
    _verify_conv_shapes(op.id, x, w, b, out, op.attrs)


def _verify_rescale_params(op_id: str, x: Tensor, out_shape: tuple[int, ...], out_dtype: str, attrs: RescaleParams) -> None:
    if x.dtype not in ("i32", "i8"):
        _fail(op_id, f"rescale input dtype {x.dtype} must be i32 or i8")
    if out_dtype not in DTYPES:
        _fail(op_id, f"rescale output dtype {out_dtype} not in {DTYPES}")
    if out_shape != x.shape:
        _fail(op_id, f"rescale output shape {out_shape} != input shape {x.shape}")
    c = x.shape[-1] if x.shape else 1
    if attrs.per_channel:
        if len(attrs.multiplier) != c or len(attrs.shift) != c:
            _fail(op_id, f"rescale per_channel=true requires multiplier/shift length {c}")
    else:
        if len(attrs.multiplier) != 1 or len(attrs.shift) != 1:
            _fail(op_id, "rescale per_channel=false requires multiplier/shift length 1")
    if attrs.scale32:
        for m in attrs.multiplier:
            if not (0 <= m < _MULT_MAX):
                _fail(op_id, f"rescale multiplier {m} out of range [0, {_MULT_MAX})")
    for s in attrs.shift:
        if not (_SHIFT_MIN <= s <= _SHIFT_MAX):
            _fail(op_id, f"rescale shift {s} out of range [{_SHIFT_MIN}, {_SHIFT_MAX}]")
    in_lo, in_hi = dtype_range(x.dtype)
    if not (in_lo <= attrs.in_zp <= in_hi):
        _fail(op_id, f"rescale in_zp {attrs.in_zp} out of range for {x.dtype}")
    if x.dtype == "i32" and attrs.in_zp != 0:
        _fail(op_id, "rescale in_zp must be 0 for i32 input (TOSA spec)")
    out_lo, out_hi = dtype_range(out_dtype)
    if not (out_lo <= attrs.out_zp <= out_hi):
        _fail(op_id, f"rescale out_zp {attrs.out_zp} out of range for {out_dtype}")
    if out_dtype == "i32" and attrs.out_zp != 0:
        _fail(op_id, "rescale out_zp must be 0 for i32 output (TOSA spec)")


def _verify_rescale(graph: Graph, op: Op) -> None:
    x = graph.tensors[op.inputs[0]]
    out = graph.tensors[op.outputs[0]]
    _verify_rescale_params(op.id, x, out.shape, out.dtype, op.attrs)


def _verify_clamp(graph: Graph, op: Op) -> None:
    x = graph.tensors[op.inputs[0]]
    out = graph.tensors[op.outputs[0]]
    attrs: ClampAttrs = op.attrs
    if out.dtype != x.dtype:
        _fail(op.id, f"clamp output dtype {out.dtype} != input dtype {x.dtype}")
    if out.shape != x.shape:
        _fail(op.id, f"clamp output shape {out.shape} != input shape {x.shape}")
    if attrs.min > attrs.max:
        _fail(op.id, f"clamp min {attrs.min} > max {attrs.max}")
    lo, hi = dtype_range(x.dtype)
    if not (lo <= attrs.min <= hi):
        _fail(op.id, f"clamp min {attrs.min} out of range for {x.dtype}")
    if not (lo <= attrs.max <= hi):
        _fail(op.id, f"clamp max {attrs.max} out of range for {x.dtype}")


_POOL_MODES = ("max", "avg")


def _verify_pool(graph: Graph, op: Op) -> None:
    x = graph.tensors[op.inputs[0]]
    out = graph.tensors[op.outputs[0]]
    attrs: PoolAttrs = op.attrs
    if attrs.mode not in _POOL_MODES:
        _fail(op.id, f"pool mode {attrs.mode!r} not in {_POOL_MODES}")
    if len(x.shape) != 4:
        _fail(op.id, f"pool input rank {len(x.shape)} != 4")
    if x.shape[0] != 1:
        _fail(op.id, f"pool input batch {x.shape[0]} != 1")
    if out.dtype != x.dtype:
        _fail(op.id, f"pool output dtype {out.dtype} != input dtype {x.dtype}")
    if any(p < 0 for p in attrs.pad):
        _fail(op.id, f"pool pad {attrs.pad} has a negative element")
    if any(k < 1 for k in attrs.kernel):
        _fail(op.id, f"pool kernel {attrs.kernel} must be >= 1")
    if any(s < 1 for s in attrs.stride):
        _fail(op.id, f"pool stride {attrs.stride} must be >= 1")
    lo, hi = dtype_range(x.dtype)
    if not (lo <= attrs.pad_value <= hi):
        _fail(op.id, f"pool pad_value {attrs.pad_value} out of range for {x.dtype}")
    # TOSA MAX_POOL2D pads with the type minimum so a padded tap can never
    # win the max; anything else silently corrupts every border output.
    if attrs.mode == "max" and attrs.pad_value != lo:
        _fail(op.id, f"max pool pad_value {attrs.pad_value} != {x.dtype} minimum {lo} (TOSA MAX_POOL2D padding)")
    expected = pool2d_output_shape(x.shape, attrs)
    if out.shape != expected:
        _fail(op.id, f"pool output shape {out.shape} != computed {expected}")


def _verify_add(graph: Graph, op: Op) -> None:
    a, b = (graph.tensors[i] for i in op.inputs)
    out = graph.tensors[op.outputs[0]]
    attrs: AddAttrs = op.attrs
    for name, t in (("lhs", a), ("rhs", b), ("output", out)):
        if t.dtype != "i8":
            _fail(op.id, f"add {name} dtype {t.dtype!r} must be i8")
    if a.shape != b.shape:
        _fail(op.id, f"add operand shapes {a.shape} != {b.shape} (no broadcasting)")
    if out.shape != a.shape:
        _fail(op.id, f"add output shape {out.shape} != operand shape {a.shape}")
    # Same bounds `_verify_rescale_params` puts on a standalone rescale:
    # the per-operand rescale IS a TOSA rescale, folded into the op.
    if not (0 <= attrs.multiplier < _MULT_MAX):
        _fail(op.id, f"add multiplier {attrs.multiplier} out of range [0, {_MULT_MAX})")
    if not (_SHIFT_MIN <= attrs.shift <= _SHIFT_MAX):
        _fail(op.id, f"add shift {attrs.shift} out of range [{_SHIFT_MIN}, {_SHIFT_MAX}]")
    # Scale > 1 is refused, not because the accelerator cannot encode it,
    # but because `passes.fuse`'s equivalence argument for folding the two
    # per-operand rescales in only holds while the rescale cannot leave
    # int8 (see `_add_rescale_foldable`). Keeping the bound here means a
    # hand-built graph cannot sneak past that argument either.
    if attrs.multiplier > (1 << attrs.shift):
        _fail(
            op.id,
            f"add rescale multiplier {attrs.multiplier} > 2**shift {1 << attrs.shift} (scale > 1): "
            "a per-operand rescale that can leave int8 is not equivalent to the accelerator's "
            "sum-then-saturate ADD",
        )


#: TOSA `TABLE` on an int8 input takes exactly `2**8` entries, indexed by
#: `value - type_min`.
_TABLE_ENTRIES = 256


def _verify_table(graph: Graph, op: Op) -> None:
    x, table = (graph.tensors[i] for i in op.inputs)
    out = graph.tensors[op.outputs[0]]
    for name, t in (("input", x), ("table", table), ("output", out)):
        if t.dtype != "i8":
            _fail(op.id, f"table {name} dtype {t.dtype!r} must be i8")
    if table.shape != (_TABLE_ENTRIES,):
        _fail(op.id, f"table operand shape {table.shape} != ({_TABLE_ENTRIES},)")
    if table.values is None:
        # Without compile-time values there is no LUT image to pack, and a
        # runtime-computed table has no instruction (the accelerator loads
        # its LUT from a constant DDR address).
        _fail(op.id, "table operand has no compile-time values")
    if out.shape != x.shape:
        _fail(op.id, f"table output shape {out.shape} != input shape {x.shape}")


def _verify_upsample(graph: Graph, op: Op) -> None:
    x = graph.tensors[op.inputs[0]]
    out = graph.tensors[op.outputs[0]]
    attrs: UpsampleAttrs = op.attrs
    if len(x.shape) != 4:
        _fail(op.id, f"upsample input rank {len(x.shape)} != 4")
    if x.shape[0] != 1:
        _fail(op.id, f"upsample input batch {x.shape[0]} != 1")
    if out.dtype != x.dtype:
        _fail(op.id, f"upsample output dtype {out.dtype} != input dtype {x.dtype}")
    if attrs.factor < 1:
        _fail(op.id, f"upsample factor {attrs.factor} must be >= 1")
    expected = upsample_output_shape(x.shape, attrs.factor)
    if out.shape != expected:
        _fail(op.id, f"upsample output shape {out.shape} != computed {expected}")


def _verify_depth_to_space(graph: Graph, op: Op) -> None:
    x = graph.tensors[op.inputs[0]]
    out = graph.tensors[op.outputs[0]]
    attrs: DepthToSpaceAttrs = op.attrs
    if len(x.shape) != 4:
        _fail(op.id, f"depth_to_space input rank {len(x.shape)} != 4")
    if x.shape[0] != 1:
        _fail(op.id, f"depth_to_space input batch {x.shape[0]} != 1")
    if out.dtype != x.dtype:
        _fail(op.id, f"depth_to_space output dtype {out.dtype} != input dtype {x.dtype}")
    if attrs.factor < 2:
        _fail(op.id, f"depth_to_space factor {attrs.factor} must be >= 2")
    if attrs.channel_order not in CHANNEL_ORDERS:
        _fail(op.id, f"depth_to_space channel_order {attrs.channel_order!r} not in {CHANNEL_ORDERS}")
    if x.shape[3] % (attrs.factor * attrs.factor):
        # Checked here rather than left to `depth_to_space_output_shape`
        # so the diagnostic is a GIR verify failure naming the op, not a
        # ValueError from a shape helper.
        _fail(
            op.id,
            f"depth_to_space input channels {x.shape[3]} is not divisible by factor**2 = "
            f"{attrs.factor}**2",
        )
    expected = depth_to_space_output_shape(x.shape, attrs.factor)
    if out.shape != expected:
        _fail(op.id, f"depth_to_space output shape {out.shape} != computed {expected}")


def _verify_concat(graph: Graph, op: Op) -> None:
    parts = [graph.tensors[i] for i in op.inputs]
    out = graph.tensors[op.outputs[0]]
    attrs: ConcatAttrs = op.attrs
    if not parts:
        _fail(op.id, "concat has no operands")
        return
    rank = len(out.shape)
    if not (0 <= attrs.axis < rank):
        _fail(op.id, f"concat axis {attrs.axis} outside rank {rank}")
        return
    total = 0
    for part in parts:
        if len(part.shape) != rank:
            _fail(op.id, f"concat operand %{part.id} rank {len(part.shape)} != output rank {rank}")
            return
        if part.dtype != out.dtype:
            _fail(op.id, f"concat operand %{part.id} dtype {part.dtype} != output dtype {out.dtype}")
        for axis, (p, o) in enumerate(zip(part.shape, out.shape)):
            if axis != attrs.axis and p != o:
                _fail(
                    op.id,
                    f"concat operand %{part.id} shape {part.shape} differs from output {out.shape} "
                    f"on axis {axis}, which is not the concat axis {attrs.axis}",
                )
                return
        total += part.shape[attrs.axis]
    if total != out.shape[attrs.axis]:
        _fail(
            op.id,
            f"concat operands sum to {total} on axis {attrs.axis}, output has {out.shape[attrs.axis]}",
        )


def _verify_slice(graph: Graph, op: Op) -> None:
    x = graph.tensors[op.inputs[0]]
    out = graph.tensors[op.outputs[0]]
    attrs: SliceAttrs = op.attrs
    rank = len(x.shape)
    if len(attrs.start) != rank or len(attrs.size) != rank:
        _fail(op.id, f"slice start/size must have {rank} elements, got {attrs.start}/{attrs.size}")
        return
    if out.dtype != x.dtype:
        _fail(op.id, f"slice output dtype {out.dtype} != input dtype {x.dtype}")
    for axis, (start, size, extent) in enumerate(zip(attrs.start, attrs.size, x.shape)):
        if start < 0 or size < 1 or start + size > extent:
            _fail(
                op.id,
                f"slice axis {axis}: [{start}, {start + size}) is not inside [0, {extent})",
            )
            return
    if out.shape != tuple(attrs.size):
        _fail(op.id, f"slice output shape {out.shape} != size {tuple(attrs.size)}")


def _verify_fused_conv(graph: Graph, op: Op) -> None:
    x, w, b = (graph.tensors[i] for i in op.inputs)
    out = graph.tensors[op.outputs[0]]
    attrs: FusedConvAttrs = op.attrs
    conv_shape = conv2d_output_shape(x.shape, w.shape, attrs.conv)
    conv_out = Tensor(id=f"{out.id}$conv", shape=conv_shape, dtype=attrs.conv.acc_dtype)
    _verify_conv_shapes(op.id, x, w, b, conv_out, attrs.conv)
    _verify_rescale_params(op.id, conv_out, conv_shape, "i8", attrs.rescale)
    if out.dtype != "i8":
        _fail(op.id, f"fused_conv output dtype {out.dtype} must be i8")
    if out.shape != conv_shape:
        _fail(op.id, f"fused_conv output shape {out.shape} != conv shape {conv_shape}")
    if attrs.clamp is not None:
        if attrs.clamp.min > attrs.clamp.max:
            _fail(op.id, f"fused_conv clamp min {attrs.clamp.min} > max {attrs.clamp.max}")
        lo, hi = dtype_range("i8")
        if not (lo <= attrs.clamp.min <= hi) or not (lo <= attrs.clamp.max <= hi):
            _fail(op.id, f"fused_conv clamp [{attrs.clamp.min}, {attrs.clamp.max}] out of range for i8")
