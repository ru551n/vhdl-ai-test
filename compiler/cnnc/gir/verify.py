"""GIR verifier (doc/tosa_compiler_plan.md §11, "imported GIR" row).

Every violation raises `VerifyError(op_id=..., stage="verify")` naming the
offending op and the violated rule.
"""

from __future__ import annotations

from cnnc.errors import VerifyError
from cnnc.gir.ir import (
    DTYPES,
    ClampAttrs,
    ConvAttrs,
    FusedConvAttrs,
    Graph,
    Op,
    RescaleParams,
    Tensor,
    conv2d_output_shape,
    dtype_range,
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
