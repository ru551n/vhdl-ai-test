"""`FusePass` (M5, doc/tosa_compiler_plan.md §9, §11 "fused GIR"): rewrite
`conv2d -> rescale [-> clamp]` chains into one `fused_conv` when the
target's unit epilogue admits it. Fusion only *re-groups* three ops into
one; no quantization parameter is recomputed (the HW's implicit `15`
shift is absorbed later, in `to_hir`), so `gir.interp.run` -- which
evaluates `fused_conv` as the literal composition of the three original
ops -- gives byte-identical results before and after this pass.

Pattern and admissibility (conv `in_zp`/`w_zp` are *not* fusion blockers:
they are a `to_hir` capability rejection, not a re-grouping decision):

* `conv2d` output must have exactly one user, a `rescale`, and must not
  itself be a graph output (it is about to be deleted).
* the `rescale` must be admissible: `in_zp == 0` (always required for i32
  input anyway), `per_channel` only if `caps.per_channel`, `out_zp != 0`
  only if `caps.output_zp`, `rounding == "SINGLE_ROUND"` (run
  `legalize_rescale` first), every `shift` in `[caps.shift_min,
  caps.shift_max]`.
* if the `rescale` output has exactly one user, a `clamp`, is not itself a
  graph output, and that clamp's `[min, max]` is admissible per
  `epilogue.clamp_ranges` (`"any"`, or exact membership in the list), the
  clamp is folded in too; otherwise `conv2d + rescale` are fused alone and
  the clamp (if any) is left standalone for `to_hir` to reject.
"""

from __future__ import annotations

import dataclasses

from cnnc.gir.ir import AddAttrs, ClampAttrs, FusedConvAttrs, Graph, Op, RescaleParams
from cnnc.target.contract import Epilogue, Target


def _epilogue(target: Target) -> Epilogue:
    for unit in target.units:
        if "conv2d" in unit.ops:
            return unit.epilogue
    raise ValueError(f"target {target.name!r} has no unit exposing a conv2d epilogue")


def _rescale_admissible(params: RescaleParams, epilogue: Epilogue) -> bool:
    caps = epilogue.rescale
    if params.in_zp != 0:
        return False
    if params.per_channel and not caps.per_channel:
        return False
    if params.out_zp != 0 and not caps.output_zp:
        return False
    if params.rounding != "SINGLE_ROUND":
        return False
    if any(s < caps.shift_min or s > caps.shift_max for s in params.shift):
        return False
    return True


def _clamp_admissible(attrs: ClampAttrs, clamp_ranges) -> bool:
    if clamp_ranges == "any":
        return True
    return (attrs.min, attrs.max) in clamp_ranges


def _add_rescale_foldable(rescale_op: Op, graph: Graph, epilogue: Epilogue) -> bool:
    """Can this `rescale` be folded into the `add` that consumes it?

    The accelerator's ADD computes `sat_i8(scale(a) + scale(b))` -- one
    saturation, after the sum. TOSA's `rescale(i8 -> i8)` clamps each
    operand to int8 *before* the add, so the fold is only exact while
    that per-operand clamp cannot fire. It cannot when the scale is <= 1
    (`multiplier <= 2**shift`): every int8 input then maps into int8, and
    the two expressions coincide for every input. A scale > 1 is left
    unfused (and `to_hir` then rejects the standalone rescale by name)
    rather than fused into a silently different program.

    The input dtype check is load-bearing and not a formality: a
    convolution's OWN epilogue rescale (i32 -> i8) would otherwise match
    every other condition here, and folding it into a following add would
    strip the convolution of its epilogue -- leaving a standalone
    `conv2d` that `to_hir` then rejects. Only an i8 -> i8 rescale is an
    operand rescale. That check is also what makes this fold independent
    of the order it runs in relative to the conv fusion below.

    The rest are shape constraints of the single `(requant_scale,
    requant_shift)` pair the descriptor carries: per-tensor only, and no
    zero points, since ADD has neither an input nor an output offset."""
    params: RescaleParams = rescale_op.attrs
    if graph.tensors[rescale_op.inputs[0]].dtype != "i8":
        return False
    if graph.tensors[rescale_op.outputs[0]].dtype != "i8":
        return False
    if params.per_channel or len(params.multiplier) != 1 or len(params.shift) != 1:
        return False
    if params.in_zp != 0 or params.out_zp != 0:
        return False
    if params.rounding != "SINGLE_ROUND":
        return False
    if params.input_unsigned or params.output_unsigned or not params.scale32:
        return False
    caps = epilogue.rescale
    shift = params.shift[0]
    if not (caps.shift_min <= shift <= caps.shift_max):
        return False
    return params.multiplier[0] <= (1 << shift)


def _fold_add_rescales(graph: Graph, epilogue: Epilogue) -> Graph:
    """`rescale(a, m, s) , rescale(b, m, s) -> add` => one `add` carrying
    `(m, s)` in its `AddAttrs`.

    This is the quantized residual-shortcut idiom, and it is the only
    shape the hardware can execute: `OPCODE_ADD` holds ONE
    `(requant_scale, requant_shift)` pair for both operands, so the two
    rescales must be identical to fold at all. Both must feed only this
    add (and not be graph outputs), since they are about to be deleted.

    Like the conv fusion above, this only re-groups ops -- no quantization
    parameter is recomputed -- and `_add_rescale_foldable` bounds it to
    the cases where the regrouping is value-preserving, so `gir.interp`
    gives identical results before and after (pinned by
    `tests/test_add.py`)."""
    skip_ids: set[str] = set()
    removed_tensors: set[str] = set()
    new_ops: list[Op] = []
    changed = False

    for op in graph.ops:
        if op.kind != "add":
            continue
        producers = [graph.producer(tid) for tid in op.inputs]
        if any(p is None or p.kind != "rescale" for p in producers):
            continue
        lhs, rhs = producers
        if lhs is rhs:
            # One rescale feeding both operands: it has two users, so the
            # "sole user" rule below would reject it anyway, and folding
            # it would delete a tensor the add still reads twice.
            continue
        if not all(_add_rescale_foldable(p, graph, epilogue) for p in producers):
            continue
        if lhs.attrs.multiplier != rhs.attrs.multiplier or lhs.attrs.shift != rhs.attrs.shift:
            continue
        if any(
            p.outputs[0] in graph.outputs or len(graph.users(p.outputs[0])) != 1 for p in producers
        ):
            continue
        skip_ids.update(p.id for p in producers)
        removed_tensors.update(p.outputs[0] for p in producers)
        changed = True

    if not changed:
        return graph

    for op in graph.ops:
        if op.id in skip_ids:
            continue
        if op.kind != "add" or not any(graph.producer(tid) is not None and graph.producer(tid).id in skip_ids for tid in op.inputs):
            new_ops.append(op)
            continue
        lhs, rhs = (graph.producer(tid) for tid in op.inputs)
        new_ops.append(
            Op(
                id=op.id,
                kind="add",
                inputs=(lhs.inputs[0], rhs.inputs[0]),
                outputs=op.outputs,
                attrs=AddAttrs(multiplier=int(lhs.attrs.multiplier[0]), shift=int(lhs.attrs.shift[0])),
            )
        )

    new_tensors = {tid: t for tid, t in graph.tensors.items() if tid not in removed_tensors}
    return graph.replace(ops=tuple(new_ops), tensors=new_tensors)


class FusePass:
    name = "fuse"

    def run(self, graph: Graph, ctx) -> Graph:
        if ctx.target is None:
            return graph
        epilogue = _epilogue(ctx.target)
        graph = _fold_add_rescales(graph, epilogue)
        if not epilogue.bias:
            return graph

        skip_ids: set[str] = set()
        removed_tensors: set[str] = set()
        new_ops: list[Op] = []
        changed = False

        for op in graph.ops:
            if op.id in skip_ids:
                continue
            if op.kind != "conv2d":
                new_ops.append(op)
                continue

            conv_out = op.outputs[0]
            conv_users = graph.users(conv_out)
            if conv_out in graph.outputs or len(conv_users) != 1 or conv_users[0].kind != "rescale":
                new_ops.append(op)
                continue
            rescale_op = conv_users[0]
            if not _rescale_admissible(rescale_op.attrs, epilogue):
                new_ops.append(op)
                continue

            rescale_out = rescale_op.outputs[0]
            rescale_users = graph.users(rescale_out)
            clamp_op = None
            if (
                rescale_out not in graph.outputs
                and len(rescale_users) == 1
                and rescale_users[0].kind == "clamp"
                and _clamp_admissible(rescale_users[0].attrs, epilogue.clamp_ranges)
            ):
                clamp_op = rescale_users[0]

            out_id = clamp_op.outputs[0] if clamp_op is not None else rescale_out
            fused_attrs = FusedConvAttrs(
                conv=op.attrs,
                rescale=rescale_op.attrs,
                clamp=clamp_op.attrs if clamp_op is not None else None,
            )
            new_ops.append(Op(id=f"%{out_id}", kind="fused_conv", inputs=op.inputs, outputs=(out_id,), attrs=fused_attrs))
            skip_ids.add(rescale_op.id)
            removed_tensors.add(conv_out)
            if clamp_op is not None:
                skip_ids.add(clamp_op.id)
                removed_tensors.add(rescale_out)
            changed = True

        if not changed:
            return graph
        new_tensors = {tid: t for tid, t in graph.tensors.items() if tid not in removed_tensors}
        return graph.replace(ops=tuple(new_ops), tensors=new_tensors)
