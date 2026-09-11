"""`PermuteDepthToSpaceChannelsPass`: turn a CHANNEL-major pixel shuffle
into the PLANE-major one the hardware implements, for free.

`OPCODE_DEPTH_TO_SPACE` groups its input channels plane-major --
`cin = (dy*r + dx)*out_channels + c`, i.e. `r**2` back-to-back contiguous
`out_channels`-wide planes -- because under the `[C/T][H][W][T]` DDR
layout that makes every `(dy, dx)` plane a whole run of channel TILES, so
the engine only ever moves whole `T`-byte beats. PyTorch's
`nn.PixelShuffle`, and therefore almost every real exported graph, means
the other grouping: `cin = c*r**2 + dy*r + dx`.

The two differ only in WHICH INPUT CHANNEL feeds which output
sub-position. An input channel of the shuffle is an output channel of the
convolution that produced it, and which output channel a convolution
produces is decided entirely by which row of its weight tensor (and the
matching row of its bias, and of any per-channel requant table) is used.
So the conversion is a compile-time relabelling of constant rows: zero
runtime cost, no extra instruction, no data movement. That is what this
pass does.

    conv2d -> [rescale] -> [clamp] -> depth_to_space{channel_major}
    ==>
    conv2d' -> [rescale'] -> [clamp] -> depth_to_space{plane_major}

Everything between the convolution and the shuffle has to be
order-independent in the channel axis for the permutation to commute
through it, which is exactly what the ops allowed in that window are:

  * `clamp` (ReLU -- the only thing between conv and shuffle in ESPCN) is
    elementwise and carries no per-channel state at all, so it commutes
    trivially and is left untouched;
  * `rescale` is per-element too; a PER-TENSOR one carries no channel
    state either, and a PER-CHANNEL one has exactly one
    `(multiplier, shift)` per channel, which is permuted alongside the
    weight rows.

Conservative by construction. The rewrite is applied only when every
tensor in that window has a single consumer (permuting a value that
something else also reads would change what that something else sees) and
the weight/bias constants are not shared with another op (one constant
cannot hold two different byte images at one address). When any of that
does not hold, the op is left channel-major and `lower.to_hir` refuses it
by name -- an honest "this graph needs the permutation and it could not
be applied here", rather than a silently wrong instruction.

Semantics-preserving: `gir.interp` gives byte-identical results before and
after, which is the property `tests/test_depth_to_space.py` asserts
directly rather than inferring from the shapes.
"""

from __future__ import annotations

import dataclasses

from cnnc.gir.ir import (
    CHANNEL_MAJOR,
    PLANE_MAJOR,
    DepthToSpaceAttrs,
    Graph,
    Op,
    Tensor,
)

#: GIR op kinds that may sit between the producing convolution and the
#: shuffle. See the module docstring for why each one commutes.
_TRANSPARENT_KINDS = ("clamp", "rescale")
_CONV_KINDS = ("conv2d", "fused_conv")


def plane_major_source_rows(factor: int, out_channels: int) -> tuple[int, ...]:
    """`src[j] = i`: the CHANNEL-major input channel `i` whose data must
    end up in PLANE-major input channel `j`.

        j = (dy*factor + dx)*out_channels + c      (plane-major)
        i = c*factor**2 + dy*factor + dx           (channel-major)

    Read as a weight-row permutation: the new row `j` of the producing
    convolution is the old row `i`.
    """
    src = [0] * (factor * factor * out_channels)
    for plane in range(factor * factor):
        for c in range(out_channels):
            src[plane * out_channels + c] = c * factor * factor + plane
    return tuple(src)


def _permute_rows(values: tuple[int, ...], src: tuple[int, ...], row_len: int) -> tuple[int, ...]:
    """Reorder `values`, viewed as `len(src)` rows of `row_len` elements,
    so that new row `j` is old row `src[j]`."""
    out: list[int] = []
    for j in src:
        out.extend(values[j * row_len : (j + 1) * row_len])
    return tuple(out)


def _consumer_count(graph: Graph) -> dict[str, int]:
    counts: dict[str, int] = {}
    for op in graph.ops:
        for tid in op.inputs:
            counts[tid] = counts.get(tid, 0) + 1
    for tid in graph.outputs:
        counts[tid] = counts.get(tid, 0) + 1
    return counts


class _NotApplicable(Exception):
    """This `depth_to_space` cannot be converted here; leave it alone and
    let `lower.to_hir` produce the diagnostic."""


def _find_producer_chain(
    graph: Graph, dts: Op, counts: dict[str, int]
) -> tuple[Op, list[Op]]:
    """`(conv op, [transparent ops between it and `dts`])`, or raise
    `_NotApplicable`."""
    tid = dts.inputs[0]
    transparent: list[Op] = []
    while True:
        if counts.get(tid, 0) != 1:
            # Someone else reads this value in its channel-major order.
            raise _NotApplicable
        producer = graph.producer(tid)
        if producer is None:
            raise _NotApplicable  # a graph input, or nothing at all
        if producer.kind in _TRANSPARENT_KINDS:
            transparent.append(producer)
            tid = producer.inputs[0]
            continue
        if producer.kind in _CONV_KINDS:
            return producer, transparent
        raise _NotApplicable


def _permuted_conv_consts(
    graph: Graph, conv: Op, src: tuple[int, ...], counts: dict[str, int]
) -> dict[str, Tensor]:
    """The permuted weight and bias tensors for `conv`."""
    if len(conv.inputs) != 3:
        raise _NotApplicable
    _x_id, w_id, b_id = conv.inputs
    w, b = graph.tensor(w_id), graph.tensor(b_id)
    if w.values is None or b.values is None:
        raise _NotApplicable  # not compile-time constant: nothing to relabel
    if len(w.shape) != 4 or w.shape[0] != len(src) or b.shape != (len(src),):
        raise _NotApplicable
    if counts.get(w_id, 0) != 1 or counts.get(b_id, 0) != 1:
        # Shared (CSE'd) with another convolution, which needs the
        # unpermuted image at the same address.
        raise _NotApplicable
    row_len = w.shape[1] * w.shape[2] * w.shape[3]
    return {
        w_id: dataclasses.replace(w, values=_permute_rows(w.values, src, row_len)),
        b_id: dataclasses.replace(b, values=_permute_rows(b.values, src, 1)),
    }


def _permuted_rescale(attrs, src: tuple[int, ...]):
    """A per-channel `RescaleParams` with its rows permuted; a per-tensor
    one unchanged (it has no channel axis to permute)."""
    if not attrs.per_channel:
        return attrs
    if len(attrs.multiplier) != len(src) or len(attrs.shift) != len(src):
        raise _NotApplicable
    return dataclasses.replace(
        attrs,
        multiplier=tuple(attrs.multiplier[j] for j in src),
        shift=tuple(attrs.shift[j] for j in src),
    )


def _rewrite_one(graph: Graph, dts: Op, counts: dict[str, int]) -> Graph:
    attrs: DepthToSpaceAttrs = dts.attrs
    out_channels = graph.tensor(dts.outputs[0]).shape[3]
    src = plane_major_source_rows(attrs.factor, out_channels)
    if len(src) != graph.tensor(dts.inputs[0]).shape[3]:
        raise _NotApplicable

    conv, transparent = _find_producer_chain(graph, dts, counts)
    new_tensors = _permuted_conv_consts(graph, conv, src, counts)

    replacements: dict[str, Op] = {}
    if conv.kind == "fused_conv":
        replacements[conv.id] = dataclasses.replace(
            conv, attrs=dataclasses.replace(conv.attrs, rescale=_permuted_rescale(conv.attrs.rescale, src))
        )
    for op in transparent:
        if op.kind == "rescale":
            replacements[op.id] = dataclasses.replace(op, attrs=_permuted_rescale(op.attrs, src))
    replacements[dts.id] = dataclasses.replace(
        dts, attrs=dataclasses.replace(attrs, channel_order=PLANE_MAJOR)
    )

    ops = tuple(replacements.get(op.id, op) for op in graph.ops)
    tensors = dict(graph.tensors)
    tensors.update(new_tensors)
    return graph.replace(ops=ops, tensors=tensors)


class PermuteDepthToSpaceChannelsPass:
    name = "depth_to_space_channels"

    def run(self, graph: Graph, ctx) -> Graph:
        # Snapshot the op ids up front: `_rewrite_one` returns a new graph
        # each time, and a later shuffle must still be looked at even if
        # an earlier one had to be declined.
        targets = [
            op.id
            for op in graph.ops
            if op.kind == "depth_to_space" and op.attrs.channel_order == CHANNEL_MAJOR
        ]
        for op_id in targets:
            current = next((op for op in graph.ops if op.id == op_id), None)
            if current is None or current.attrs.channel_order != CHANNEL_MAJOR:
                continue
            try:
                graph = _rewrite_one(graph, current, _consumer_count(graph))
            except _NotApplicable:
                # Leave it channel-major; `lower.to_hir` is what says why
                # it cannot be lowered. Noted so the manifest records that
                # the compiler looked at this and declined, rather than
                # never having considered it.
                ctx.notes.append(
                    f"depth_to_space {op_id}: channel-major (nn.PixelShuffle) grouping could not "
                    "be permuted into the hardware's plane-major order -- no single-consumer "
                    "convolution with unshared weight/bias constants produces its input"
                )
        return graph
