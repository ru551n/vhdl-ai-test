"""Deterministic GIR text printer (`print_gir`, golden-tested) and a JSON
dump for `--dump-after-all` (`to_json`; constant values are hashed rather
than embedded so dumps stay small and diffable).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json

from cnnc.gir.ir import (
    AddAttrs,
    Attrs,
    ConvAttrs,
    DepthToSpaceAttrs,
    FusedConvAttrs,
    Graph,
    Op,
    PoolAttrs,
    RescaleParams,
    Tensor,
)


def _shape_dtype(shape: tuple[int, ...], dtype: str) -> str:
    return "x".join(str(d) for d in shape) + "x" + dtype


def _fmt_list(values: tuple[int, ...]) -> str:
    return "[" + ",".join(str(v) for v in values) + "]"


def _bit(v: bool) -> str:
    return "1" if v else "0"


def _const_stats(t: Tensor) -> str:
    values = t.values or ()
    n = len(values)
    if n == 0:
        return "(0 values)"
    if all(v == values[0] for v in values):
        return f"({n} values, splat {values[0]})"
    return f"({n} values, min {min(values)} max {max(values)})"


def _conv_attrs_str(a: ConvAttrs, *, with_acc: bool) -> str:
    parts = [
        f"pad={_fmt_list(a.pad)}",
        f"stride={_fmt_list(a.stride)}",
        f"dil={_fmt_list(a.dilation)}",
        f"in_zp={a.in_zp}",
        f"w_zp={a.w_zp}",
    ]
    if with_acc:
        parts.append(f"acc={a.acc_dtype}")
    return " ".join(parts)


def _rescale_attrs_str(r: RescaleParams, *, with_scale32: bool) -> str:
    parts = [
        f"mult={_fmt_list(r.multiplier)}",
        f"shift={_fmt_list(r.shift)}",
        f"per_channel={_bit(r.per_channel)}",
        f"in_zp={r.in_zp}",
        f"out_zp={r.out_zp}",
        f"round={r.rounding}",
    ]
    if with_scale32:
        parts.append(f"scale32={_bit(r.scale32)}")
    return " ".join(parts)


def _format_op(graph: Graph, op: Op) -> str:
    out_t = graph.tensor(op.outputs[0])
    shape_str = _shape_dtype(out_t.shape, out_t.dtype)
    if op.kind == "const":
        return f"{op.id} = const : {shape_str} {_const_stats(out_t)}"
    operands = ", ".join(f"%{i}" for i in op.inputs)
    if op.kind == "conv2d":
        return f"{op.id} = conv2d {operands} {{{_conv_attrs_str(op.attrs, with_acc=True)}}} : {shape_str}"
    if op.kind == "rescale":
        return f"{op.id} = rescale {operands} {{{_rescale_attrs_str(op.attrs, with_scale32=True)}}} : {shape_str}"
    if op.kind == "clamp":
        return f"{op.id} = clamp {operands} {{min={op.attrs.min} max={op.attrs.max}}} : {shape_str}"
    if op.kind == "pool":
        a: PoolAttrs = op.attrs
        return (
            f"{op.id} = pool_{a.mode} {operands} "
            f"{{kernel={_fmt_list(a.kernel)} stride={_fmt_list(a.stride)} pad={_fmt_list(a.pad)} "
            f"pad_value={a.pad_value}}} : {shape_str}"
        )
    if op.kind == "add":
        a: AddAttrs = op.attrs
        return f"{op.id} = add {operands} {{mult={a.multiplier} shift={a.shift}}} : {shape_str}"
    if op.kind == "table":
        return f"{op.id} = table {operands} : {shape_str}"
    if op.kind == "upsample":
        return f"{op.id} = upsample {operands} {{factor={op.attrs.factor}}} : {shape_str}"
    if op.kind == "depth_to_space":
        dts: DepthToSpaceAttrs = op.attrs
        return (
            f"{op.id} = depth_to_space {operands} "
            f"{{factor={dts.factor} channel_order={dts.channel_order}}} : {shape_str}"
        )
    if op.kind == "concat":
        return f"{op.id} = concat {operands} {{axis={op.attrs.axis}}} : {shape_str}"
    if op.kind == "slice":
        return (
            f"{op.id} = slice {operands} "
            f"{{start={_fmt_list(op.attrs.start)} size={_fmt_list(op.attrs.size)}}} : {shape_str}"
        )
    if op.kind == "fused_conv":
        a: FusedConvAttrs = op.attrs
        conv_part = _conv_attrs_str(a.conv, with_acc=False)
        rescale_part = _rescale_attrs_str(a.rescale, with_scale32=False)
        clamp_part = "none" if a.clamp is None else f"[{a.clamp.min},{a.clamp.max}]"
        return (
            f"{op.id} = fused_conv {operands} "
            f"{{conv: {conv_part}; rescale: {rescale_part}; clamp: {clamp_part}}} : {shape_str}"
        )
    raise ValueError(f"unknown op kind {op.kind!r}")


def print_gir(graph: Graph) -> str:
    args = ", ".join(
        f"%{tid}: {_shape_dtype(graph.tensor(tid).shape, graph.tensor(tid).dtype)}" for tid in graph.inputs
    )
    out_strs = [_shape_dtype(graph.tensor(tid).shape, graph.tensor(tid).dtype) for tid in graph.outputs]
    result = out_strs[0] if len(out_strs) == 1 else "(" + ", ".join(out_strs) + ")"
    lines = [f"gir.func @{graph.name}({args}) -> {result} {{"]
    for op in graph.ops:
        lines.append("  " + _format_op(graph, op))
    lines.append("  return " + ", ".join(f"%{tid}" for tid in graph.outputs))
    lines.append("}")
    return "\n".join(lines) + "\n"


def _tensor_json(t: Tensor) -> dict:
    d: dict = {"shape": list(t.shape), "dtype": t.dtype}
    if t.logical_shape is not None:
        d["logical_shape"] = list(t.logical_shape)
    if t.values is not None:
        d["numel"] = len(t.values)
        d["values_sha256"] = hashlib.sha256(
            json.dumps(list(t.values), separators=(",", ":")).encode()
        ).hexdigest()
    return d


def _attrs_json(attrs: Attrs) -> dict | None:
    if attrs is None:
        return None
    return dataclasses.asdict(attrs)


def to_json(graph: Graph) -> dict:
    return {
        "name": graph.name,
        "inputs": list(graph.inputs),
        "outputs": list(graph.outputs),
        "tensors": {tid: _tensor_json(graph.tensors[tid]) for tid in sorted(graph.tensors)},
        "ops": [
            {
                "id": op.id,
                "kind": op.kind,
                "inputs": list(op.inputs),
                "outputs": list(op.outputs),
                "attrs": _attrs_json(op.attrs),
            }
            for op in graph.ops
        ],
    }
