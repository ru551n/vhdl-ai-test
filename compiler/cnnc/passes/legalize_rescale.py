"""`LegalizeRescalePass` (M4, doc/tosa_compiler_plan.md §6, §11 "legalized
GIR"): rewrite every `rescale` (and, defensively, an already-fused
`fused_conv.rescale`) so its `shift` and `rounding` are directly
expressible by the target's rescale capability, without changing the
computed value.

Exactness of the `shift < shift_min` rewrite
---------------------------------------------
TOSA's scalar formula (`gir.interp.apply_scale_32`) is a floor shift of an
exact numerator::

    s = (v*m + 2**(shift-1)) >> shift

For any `k >= 0`, rewriting `m' = m << k`, `shift' = shift + k`::

    v*m' + 2**(shift'-1) = v*m*2**k + 2**(shift-1)*2**k = (v*m + 2**(shift-1)) * 2**k

so the rewritten numerator is the *original* numerator multiplied by
`2**k`, and its shift amount grows by the same `k`. `(x * 2**k) >> (n+k)
== x >> n` bit-for-bit for any integer `x` (positive or negative) and any
`n, k >= 0` -- multiplying by `2**k` before an arithmetic right-shift by
`n+k` is exact, it is not an approximation. Hence::

    apply_scale_32(v, m, shift) == apply_scale_32(v, m << k, shift + k)

for *every* integer `v`, provided `m << k` still fits the target's signed
multiplier width (checked below: `RescaleCaps.multiplier_bits`, signed ->
max value `2**(bits-1) - 1`). This is the same identity for `DOUBLE_ROUND`
(the extra +/-`2**30` term is independent of `k`) but `DOUBLE_ROUND` is
rejected outright below since no `cnn_accel` epilogue can express its
value-sign-dependent correction (doc/tosa_compiler_plan.md §6).

Why the legalized shift never needs a wider verifier range
------------------------------------------------------------
`gir/verify.py` enforces the TOSA `scale32=true` range `shift in [2, 62]`
unconditionally (do not weaken it here). Legalization only ever *raises*
an under-`shift_min` shift up towards `shift_min` (`cnn_accel`: `15`):
`k = shift_min - shift <= shift_min - 2 = 13` (since the input shift is
already `>= 2` per the TOSA range), so the legalized shift is
`shift + k = shift_min < 28`, always inside `[2, 62]` for any
`shift_min <= 49`. No test in this milestone needs a shift outside that
range post-legalization.
"""

from __future__ import annotations

import dataclasses

from cnnc.errors import LegalizeError
from cnnc.gir.ir import FusedConvAttrs, Graph, RescaleParams
from cnnc.target.contract import RescaleCaps, Target


def _rescale_caps(target: Target) -> RescaleCaps:
    for unit in target.units:
        if "conv2d" in unit.ops:
            return unit.epilogue.rescale
    raise LegalizeError(f"target {target.name!r} has no unit exposing a rescale capability")


def _legalize_params(op_id: str, params: RescaleParams, caps: RescaleCaps, notes: list[str]) -> RescaleParams:
    if params.rounding == "DOUBLE_ROUND":
        raise LegalizeError("DOUBLE_ROUND rounding is not expressible on this target", op_id=op_id)
    rounding = params.rounding
    if rounding == "INFERENCE":
        rounding = "SINGLE_ROUND"
        notes.append(f"{op_id}: INFERENCE rounding treated as SINGLE_ROUND")

    mult_max = (1 << (caps.multiplier_bits - 1)) if caps.multiplier_signed else (1 << caps.multiplier_bits)
    new_mult: list[int] = []
    new_shift: list[int] = []
    for m, s in zip(params.multiplier, params.shift):
        if s < caps.shift_min:
            k = caps.shift_min - s
            m2 = m << k
            if m2 >= mult_max:
                raise LegalizeError(
                    f"rescale multiplier {m} << {k} = {m2} overflows the target's "
                    f"{caps.multiplier_bits}-bit {'signed' if caps.multiplier_signed else 'unsigned'} "
                    f"multiplier (max {mult_max - 1}) after shift<{caps.shift_min} legalization",
                    op_id=op_id,
                )
            m, s = m2, s + k
        if s > caps.shift_max:
            raise LegalizeError(f"rescale shift {s} exceeds target shift_max {caps.shift_max}", op_id=op_id)
        new_mult.append(m)
        new_shift.append(s)

    if rounding == params.rounding and new_mult == list(params.multiplier) and new_shift == list(params.shift):
        return params
    return dataclasses.replace(params, multiplier=tuple(new_mult), shift=tuple(new_shift), rounding=rounding)


class LegalizeRescalePass:
    name = "legalize_rescale"

    def run(self, graph: Graph, ctx) -> Graph:
        if ctx.target is None:
            return graph
        caps = _rescale_caps(ctx.target)
        changed = False
        new_ops = []
        for op in graph.ops:
            if op.kind == "rescale":
                new_attrs = _legalize_params(op.id, op.attrs, caps, ctx.notes)
                if new_attrs is not op.attrs:
                    changed = True
                    op = dataclasses.replace(op, attrs=new_attrs)
            elif op.kind == "fused_conv":
                fused: FusedConvAttrs = op.attrs
                new_rescale = _legalize_params(op.id, fused.rescale, caps, ctx.notes)
                if new_rescale is not fused.rescale:
                    changed = True
                    op = dataclasses.replace(op, attrs=dataclasses.replace(fused, rescale=new_rescale))
            new_ops.append(op)
        if not changed:
            return graph
        return graph.replace(ops=tuple(new_ops))
