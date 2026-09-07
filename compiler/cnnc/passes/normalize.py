"""`NormalizePass` (M4, doc/tosa_compiler_plan.md §11 "normalized GIR"):
remove identity clamps and drop dead ops. Semantics-preserving: `interp`
gives byte-identical results before and after.

Identity clamp removal is target-aware (M11 preview): a clamp whose
`[min, max]` exactly covers its dtype's full range is a no-op and, on a
target whose every unit accepts an unrestricted clamp (`epilogue.clamp_ranges
== "any"`), keeping it costs nothing and stays traceable, so it is *kept*.
Otherwise (including `ctx.target is None`, i.e. no capability information)
it is removed. With the current `cnn_accel` target (`clamp_ranges =
[[-128,127],[0,127]]`, not `"any"`) identity clamps are always removed.

One aliasing edge case is deliberately conservative: if the clamp's input
is itself a graph input *and* its output is a graph output, removing the
clamp would make the same tensor id appear as both -- rewiring is skipped
and the (semantically still-identity) clamp is kept, rather than aliasing
declared inputs and outputs.
"""

from __future__ import annotations

import dataclasses

from cnnc.gir.ir import Graph, Op, dtype_range


def _identity_removal_enabled(ctx) -> bool:
    if ctx.target is None:
        return True
    return not any(unit.epilogue.clamp_ranges == "any" for unit in ctx.target.units)


def _is_identity_clamp(graph: Graph, op: Op) -> bool:
    if op.kind != "clamp":
        return False
    lo, hi = dtype_range(graph.tensor(op.inputs[0]).dtype)
    return op.attrs.min == lo and op.attrs.max == hi


def _remove_identity_clamps(graph: Graph, ctx) -> Graph:
    if not _identity_removal_enabled(ctx):
        return graph
    rename: dict[str, str] = {}
    kept_ops: list[Op] = []
    for op in graph.ops:
        if _is_identity_clamp(graph, op):
            in_id, out_id = op.inputs[0], op.outputs[0]
            if in_id in graph.inputs and out_id in graph.outputs:
                kept_ops.append(op)  # aliasing edge case (see module docstring): keep
                continue
            rename[out_id] = in_id
            continue
        kept_ops.append(op)
    if not rename:
        return graph

    def resolve(tid: str) -> str:
        seen: set[str] = set()
        while tid in rename and tid not in seen:
            seen.add(tid)
            tid = rename[tid]
        return tid

    new_ops = tuple(dataclasses.replace(op, inputs=tuple(resolve(i) for i in op.inputs)) for op in kept_ops)
    new_outputs = tuple(resolve(t) for t in graph.outputs)
    new_tensors = {tid: t for tid, t in graph.tensors.items() if tid not in rename}
    return graph.replace(ops=new_ops, outputs=new_outputs, tensors=new_tensors)


def _drop_dead_ops(graph: Graph) -> Graph:
    ops = graph.ops
    while True:
        used = set(graph.outputs)
        for op in ops:
            used.update(op.inputs)
        live_ops = tuple(op for op in ops if any(o in used for o in op.outputs))
        if live_ops == ops:
            break
        ops = live_ops
    if ops == graph.ops:
        return graph
    kept_ids = set(graph.inputs) | set(graph.outputs)
    for op in ops:
        kept_ids.update(op.inputs)
        kept_ids.update(op.outputs)
    new_tensors = {tid: t for tid, t in graph.tensors.items() if tid in kept_ids}
    return graph.replace(ops=ops, tensors=new_tensors)


class NormalizePass:
    name = "normalize"

    def run(self, graph: Graph, ctx) -> Graph:
        graph = _remove_identity_clamps(graph, ctx)
        graph = _drop_dead_ops(graph)
        return graph
