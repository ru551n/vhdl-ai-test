"""GIR interpreter: TOSA-semantics reference execution (doc/tosa_compiler_plan.md
§6, §10 item 1). `numpy` is used here (and in `cnnc.testing`) only -- every
other module stays pure Python. Arrays are worked on as `int64` internally;
each op's result is range-checked against its declared GIR dtype and only
cast to the narrower numpy dtype (`int8`/`int32`) when handed back to the
caller, so intermediate values never silently wrap.

`apply_scale_32` is the scalar (pure Python int) TOSA spec formula; the same
math, vectorised over numpy `int64` arrays with per-channel broadcasting, is
`rescale_array`. Both must agree bit-for-bit (see `test_interp.py`).
"""

from __future__ import annotations

import numpy as np

from cnnc.errors import CompilerError
from cnnc.gir.ir import ClampAttrs, ConvAttrs, FusedConvAttrs, Graph, RescaleParams, dtype_range

_NP_DTYPE: dict[str, type] = {"i8": np.int8, "i32": np.int32}


class AccumulatorOverflow(CompilerError):
    """A conv2d accumulator (post-bias) left the `int32` range. TOSA/§6
    treats this as an invalid program, not silent wraparound."""


def apply_scale_32(value: int, multiplier: int, shift: int, rounding: str) -> int:
    """TOSA `apply_scale_32` (spec; doc/tosa_compiler_plan.md §6), scalar
    reference. `value` is already zero-point-shifted (`x - in_zp`).

        r = 1 << (shift - 1)                         [+/- 2**30 if DOUBLE_ROUND and shift > 31]
        s = (value * multiplier + r) >> shift         (int64 arithmetic/floor shift)

    `INFERENCE` is treated as `SINGLE_ROUND` (the spec leaves the choice to
    the implementation; IREE's choice is SINGLE_ROUND).
    """
    assert 0 <= multiplier < 2**31
    prod = value * multiplier
    assert abs(prod) < 2**63
    r = 1 << (shift - 1)
    if rounding == "DOUBLE_ROUND" and shift > 31:
        r += (1 << 30) if value >= 0 else -(1 << 30)
    return (prod + r) >> shift


def rescale_array(x: np.ndarray, params: RescaleParams, out_dtype: str) -> np.ndarray:
    """Vectorised `numpy.int64` equivalent of `apply_scale_32` applied
    elementwise (+ `out_zp` + clamp to `out_dtype`), broadcasting
    per-channel `multiplier`/`shift` along the last (channel) axis."""
    v = x.astype(np.int64) - np.int64(params.in_zp)
    if params.per_channel:
        shape = (1,) * (v.ndim - 1) + (-1,)
        mult = np.asarray(params.multiplier, dtype=np.int64).reshape(shape)
        shift = np.asarray(params.shift, dtype=np.int64).reshape(shape)
    else:
        mult = np.int64(params.multiplier[0])
        shift = np.int64(params.shift[0])
    assert np.all((mult >= 0) & (mult < 2**31))
    prod = v * mult
    assert np.all(np.abs(prod) < 2**63)
    r = np.int64(1) << (shift - 1)
    if params.rounding == "DOUBLE_ROUND":
        corr = np.where(v >= 0, np.int64(1) << 30, -(np.int64(1) << 30))
        r = r + np.where(shift > 31, corr, np.int64(0))
    s = (prod + r) >> shift
    y = s + np.int64(params.out_zp)
    lo, hi = dtype_range(out_dtype)
    return np.clip(y, lo, hi)


def _conv2d_int64(x: np.ndarray, w: np.ndarray, bias: np.ndarray, attrs: ConvAttrs, op_id: str) -> np.ndarray:
    """TOSA conv2d (doc/tosa_compiler_plan.md §6):

        acc[n,oy,ox,oc] = sum_{ky,kx,ic} (x_pad[..] - in_zp)(w[oc,ky,kx,ic] - w_zp) + bias[oc]

    `x_pad` is `x` padded with `in_zp` so `(pad - in_zp) == 0` -- padded
    taps contribute nothing, matching the TOSA spec exactly (not HW's
    literal-zero padding). Bias is added inside conv2d per TOSA.
    """
    x64 = x.astype(np.int64)
    n, in_h, in_w, cin = x64.shape
    oc, kh, kw, w_cin = w.shape
    assert w_cin == cin
    pad_t, pad_b, pad_l, pad_r = attrs.pad
    sh, sw = attrs.stride
    dh, dw = attrs.dilation
    out_h = (in_h + pad_t + pad_b - dh * (kh - 1) - 1) // sh + 1
    out_w = (in_w + pad_l + pad_r - dw * (kw - 1) - 1) // sw + 1

    x_pad = np.pad(x64, ((0, 0), (pad_t, pad_b), (pad_l, pad_r), (0, 0)), constant_values=attrs.in_zp)
    x_shifted = x_pad - np.int64(attrs.in_zp)
    w_shifted = w.astype(np.int64) - np.int64(attrs.w_zp)

    acc = np.zeros((n, out_h, out_w, oc), dtype=np.int64)
    for ky in range(kh):
        for kx in range(kw):
            row0, col0 = ky * dh, kx * dw
            patch = x_shifted[:, row0 : row0 + sh * (out_h - 1) + 1 : sh, col0 : col0 + sw * (out_w - 1) + 1 : sw, :]
            acc += np.tensordot(patch, w_shifted[:, ky, kx, :], axes=([3], [1]))
    acc += bias.astype(np.int64)

    lo, hi = dtype_range(attrs.acc_dtype)
    if np.any((acc < lo) | (acc > hi)):
        raise AccumulatorOverflow(f"conv2d accumulator outside {attrs.acc_dtype} range [{lo}, {hi}]", op_id=op_id)
    return acc


def _clamp(x: np.ndarray, attrs: ClampAttrs) -> np.ndarray:
    return np.clip(x.astype(np.int64), attrs.min, attrs.max)


def evaluate_all(graph: Graph, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Execute every op in `graph` and return every tensor, keyed by tensor
    id (no `%`), cast to its declared GIR dtype's numpy type."""
    values: dict[str, np.ndarray] = {}

    for tid in graph.inputs:
        tensor = graph.tensor(tid)
        arr = np.asarray(inputs[tid])
        assert arr.shape == tensor.shape, f"input %{tid}: shape {arr.shape} != declared {tensor.shape}"
        assert arr.dtype == _NP_DTYPE[tensor.dtype], f"input %{tid}: dtype {arr.dtype} != declared {tensor.dtype}"
        values[tid] = arr.astype(np.int64)

    for op in graph.ops:
        if op.kind == "const":
            tensor = graph.tensor(op.outputs[0])
            values[op.outputs[0]] = np.asarray(tensor.values, dtype=np.int64).reshape(tensor.shape)

    for op in graph.ops:
        if op.kind == "const":
            continue
        out_tensor = graph.tensor(op.outputs[0])

        if op.kind == "conv2d":
            x, w, b = (values[i] for i in op.inputs)
            result = _conv2d_int64(x, w, b, op.attrs, op.id)
        elif op.kind == "rescale":
            result = rescale_array(values[op.inputs[0]], op.attrs, out_tensor.dtype)
        elif op.kind == "clamp":
            result = _clamp(values[op.inputs[0]], op.attrs)
        elif op.kind == "fused_conv":
            fused: FusedConvAttrs = op.attrs
            x, w, b = (values[i] for i in op.inputs)
            acc = _conv2d_int64(x, w, b, fused.conv, op.id)
            # verify.py pins the unfused rescale's output dtype to "i8" for
            # fused_conv; fusion is literally the composition of the three ops.
            result = rescale_array(acc, fused.rescale, "i8")
            if fused.clamp is not None:
                result = _clamp(result, fused.clamp)
        else:
            raise ValueError(f"interp: unsupported op kind {op.kind!r}")

        lo, hi = dtype_range(out_tensor.dtype)
        assert result.shape == out_tensor.shape, f"{op.id}: result shape {result.shape} != declared {out_tensor.shape}"
        assert np.all((result >= lo) & (result <= hi)), f"{op.id}: result outside {out_tensor.dtype} range [{lo}, {hi}]"
        values[op.outputs[0]] = result.astype(np.int64)

    return {tid: arr.astype(_NP_DTYPE[graph.tensor(tid).dtype]) for tid, arr in values.items()}


def run(graph: Graph, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Execute `graph` and return only the graph outputs, keyed by tensor id."""
    all_values = evaluate_all(graph, inputs)
    return {tid: all_values[tid] for tid in graph.outputs}
