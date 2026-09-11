"""`PadDepthToSpaceChannelsPass`: widen a pixel shuffle's output channel
count up to a whole activation-plane channel tile, for free.

`OPCODE_DEPTH_TO_SPACE` moves whole `T`-byte channel TILES -- the
elementwise engine has no gather/scatter for individual byte lanes within
a tile -- so the instruction's `out_channels` descriptor field has to be a
whole multiple of `T` (`cnn_accel_cmd_proc` rejects anything else with
`ERR_BAD_GEOMETRY`). A real super-resolution model's tail does not oblige:
ESPCN's is `out_channels = 1` (luma) or `3` (RGB), which makes the
`factor**2` input planes `factor**2` byte LANES of one tile.

That is a genuine hardware constraint, but it is NOT a reason to refuse
the graph, because of a convention this compiler already applies to every
activation tensor it has (`lower.layout.activation_bytes`): a tensor whose
true channel count `C` is not a multiple of `T` is STORED padded to
`ceil(C/T)*T` lanes, and `lower.layout.unpack_activation_planes` drops
those padding lanes on the way back out. Buffer SIZE is padded; logical
SHAPE is not. YOLOv8n's first layer (`C = 3`) already relies on this.

So the fix needs no new hardware and no new convention -- only that the
shuffle actually PRODUCE the padded tile rather than a sub-tile:

    conv2d{f**2 * C rows} -> [rescale] -> [clamp] -> depth_to_space{C}
    ==>
    conv2d{f**2 * P rows} -> [rescale] -> [clamp] -> depth_to_space{P}

with `P = ceil(C/T)*T` and every added output channel a DUMMY: its weight
row is all zeros and its bias entry is zero, so it contributes nothing and
computes nothing anyone reads. It is the same zero padding
`layout.pack_weights_tiled`/`pack_bias_for_hw` already write into the
lanes past `out_channels` of a weight tile -- here it is just made
explicit in the graph so the SHUFFLE's descriptor can name it.

Note what does NOT change: `activation_bytes((1, H, W, C), T)` and
`activation_bytes((1, H, W, P), T)` are the SAME NUMBER (both are
`ceil(C/T)*T*H*W`), because `C` and `P` occupy the same number of whole
planes by construction. Widening the shuffle's output tensor therefore
moves no byte and costs no DDR; it only makes the GIR shape agree with the
`out_channels` the instruction is about to be given, which is the thing
that must not be left disagreeing.

The TRUE channel count is not lost: the widened tensor carries it as
`Tensor.logical_shape`, which `lower.to_hir` copies onto the output
`Buffer` and the backend records in the program manifest, so a caller
reading the result back still gets its `C` channels and not `P`.

Where this runs, and why it is its own pass
-------------------------------------------
Before `PermuteDepthToSpaceChannelsPass`, and separate from it, although
that pass touches the very same weight/bias/scale rows:

  * padding applies to a PLANE-major shuffle too, and the permutation pass
    by construction only looks at channel-major ones. Folding padding in
    would mean running a pass named "convert the grouping" on graphs whose
    grouping is already right;
  * the permutation's row map, `plane_major_source_rows(factor,
    out_channels)`, is a function of the FINAL channel count. Padding
    first means the permutation pass needs no change at all and cannot
    see a half-padded graph;
  * the two have different preconditions and different failure modes, and
    keeping them apart keeps each one's postcondition assertable on its
    own.

Conservative in exactly the same way as the permutation pass, for the same
reasons: a single-consumer chain back to a convolution whose weight and
bias constants are not shared with anything else. When that does not hold
(most obviously a shuffle reading a graph INPUT -- padding it would change
the graph's own input shape, which is not this compiler's to change), the
op is left alone and `lower.to_hir` refuses it by name.
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
    depth_to_space_output_shape,
)

from .depth_to_space_channels import (
    _NotApplicable,
    _consumer_count,
    _find_producer_chain,
)


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def padded_source_rows(
    factor: int, out_channels: int, padded_out_channels: int, channel_order: str
) -> tuple[int | None, ...]:
    """`src[j] = i or None`: which row of the producing convolution's
    `factor**2 * out_channels`-row weight tensor becomes row `j` of its
    widened `factor**2 * padded_out_channels`-row one. `None` means "a
    dummy row": zeros, contributing nothing.

    Which rows move depends on how the shuffle groups its input channels,
    so both orders are spelled out from their own index formula:

        plane-major   cin = (dy*factor + dx)*out_channels + c
        channel-major cin = c*factor**2 + dy*factor + dx

    Plane-major interleaves -- every plane grows from `out_channels` to
    `padded_out_channels` wide, so the real rows are scattered. Channel-
    major is a pure APPEND: the extra channels `c >= out_channels` sit
    entirely after the existing ones.
    """
    planes = factor * factor
    src: list[int | None] = [None] * (planes * padded_out_channels)
    if channel_order == PLANE_MAJOR:
        for plane in range(planes):
            for c in range(out_channels):
                src[plane * padded_out_channels + c] = plane * out_channels + c
    elif channel_order == CHANNEL_MAJOR:
        for c in range(out_channels):
            for plane in range(planes):
                src[c * planes + plane] = c * planes + plane
    else:  # pragma: no cover - DepthToSpaceAttrs.__post_init__ rejects these
        raise ValueError(f"unknown channel_order {channel_order!r}")
    return tuple(src)


def _widen_rows(
    values: tuple[int, ...], src: tuple[int | None, ...], row_len: int
) -> tuple[int, ...]:
    """`values` viewed as rows of `row_len`, re-emitted as `len(src)` rows
    where row `j` is old row `src[j]`, or a zero row when `src[j]` is
    `None`."""
    out: list[int] = []
    zero = [0] * row_len
    for i in src:
        out.extend(zero if i is None else values[i * row_len : (i + 1) * row_len])
    return tuple(out)


def _widened_conv_consts(
    graph: Graph, conv: Op, src: tuple[int | None, ...], old_rows: int, counts: dict[str, int]
) -> dict[str, Tensor]:
    """The zero-extended weight and bias tensors for `conv`."""
    if len(conv.inputs) != 3:
        raise _NotApplicable
    _x_id, w_id, b_id = conv.inputs
    w, b = graph.tensor(w_id), graph.tensor(b_id)
    if w.values is None or b.values is None:
        raise _NotApplicable  # not compile-time constant: nothing to extend
    if len(w.shape) != 4 or w.shape[0] != old_rows or b.shape != (old_rows,):
        raise _NotApplicable
    if counts.get(w_id, 0) != 1 or counts.get(b_id, 0) != 1:
        # Shared (CSE'd) with another convolution, which needs the
        # unwidened image at the same address.
        raise _NotApplicable
    row_len = w.shape[1] * w.shape[2] * w.shape[3]
    new_rows = len(src)
    return {
        w_id: dataclasses.replace(
            w, shape=(new_rows,) + tuple(w.shape[1:]), values=_widen_rows(w.values, src, row_len)
        ),
        b_id: dataclasses.replace(
            b, shape=(new_rows,), values=_widen_rows(b.values, src, 1)
        ),
    }


def _widened_rescale(attrs, src: tuple[int | None, ...], old_rows: int):
    """A per-channel `RescaleParams` zero-extended to `len(src)` rows; a
    per-tensor one unchanged (it has no channel axis).

    A dummy channel gets `multiplier = 0` -- which sends every value to
    the output zero point, i.e. contributes nothing -- and KEEPS a real
    channel's `shift`, because `shift` has a legal TOSA range ([2, 62] for
    scale32) that `gir.verify` enforces and 0 is not in it. Zeroing the
    multiplier is what makes the channel dummy; the shift is then
    irrelevant arithmetic on a zero.
    """
    if not attrs.per_channel:
        return attrs
    if len(attrs.multiplier) != old_rows or len(attrs.shift) != old_rows:
        raise _NotApplicable
    fill_shift = attrs.shift[0]
    return dataclasses.replace(
        attrs,
        multiplier=tuple(0 if i is None else attrs.multiplier[i] for i in src),
        shift=tuple(fill_shift if i is None else attrs.shift[i] for i in src),
    )


def _widen_channels(t: Tensor, channels: int) -> Tensor:
    """`t` with its NHWC channel axis set to `channels`; no `logical_shape`
    (an intermediate value nobody outside the graph reads)."""
    return dataclasses.replace(t, shape=tuple(t.shape[:3]) + (channels,))


def _rewrite_one(graph: Graph, dts: Op, plane_channels: int, counts: dict[str, int]) -> Graph:
    attrs: DepthToSpaceAttrs = dts.attrs
    x_id, y_id = dts.inputs[0], dts.outputs[0]
    x, y = graph.tensor(x_id), graph.tensor(y_id)
    out_channels = y.shape[3]
    padded = _ceil_div(out_channels, plane_channels) * plane_channels
    planes = attrs.factor * attrs.factor
    if x.shape[3] != planes * out_channels:
        raise _NotApplicable

    src = padded_source_rows(attrs.factor, out_channels, padded, attrs.channel_order)
    conv, transparent = _find_producer_chain(graph, dts, counts)
    new_tensors = _widened_conv_consts(graph, conv, src, planes * out_channels, counts)

    replacements: dict[str, Op] = {}
    if conv.kind == "fused_conv":
        replacements[conv.id] = dataclasses.replace(
            conv,
            attrs=dataclasses.replace(
                conv.attrs, rescale=_widened_rescale(conv.attrs.rescale, src, planes * out_channels)
            ),
        )
    for op in transparent:
        if op.kind == "rescale":
            replacements[op.id] = dataclasses.replace(
                op, attrs=_widened_rescale(op.attrs, src, planes * out_channels)
            )

    # Every activation between the convolution's output and the shuffle's
    # input now carries `planes * padded` channels.
    tensors = dict(graph.tensors)
    tensors.update(new_tensors)
    for op in (conv, *transparent):
        for tid in op.outputs:
            tensors[tid] = _widen_channels(tensors[tid], planes * padded)
    new_y_shape = depth_to_space_output_shape(tensors[x_id].shape, attrs.factor)
    assert new_y_shape[3] == padded
    # The shuffle's output is the one tensor whose TRUE shape a caller
    # outside the graph may still want: it can be the graph's own output.
    tensors[y_id] = dataclasses.replace(
        y, shape=new_y_shape, logical_shape=y.logical_shape or y.shape
    )

    ops = tuple(replacements.get(op.id, op) for op in graph.ops)
    return graph.replace(ops=ops, tensors=tensors)


class PadDepthToSpaceChannelsPass:
    name = "depth_to_space_pad"

    def run(self, graph: Graph, ctx) -> Graph:
        if ctx.target is None:
            return graph
        plane_channels = ctx.target.memory.activation_plane_channels
        targets = [
            op.id
            for op in graph.ops
            if op.kind == "depth_to_space"
            and graph.tensor(op.outputs[0]).shape[3] % plane_channels
        ]
        for op_id in targets:
            current = next((op for op in graph.ops if op.id == op_id), None)
            if current is None:
                continue
            out_channels = graph.tensor(current.outputs[0]).shape[3]
            try:
                graph = _rewrite_one(graph, current, plane_channels, _consumer_count(graph))
            except _NotApplicable:
                # Left as-is; `lower.to_hir` is what says why it cannot be
                # lowered. Noted so the manifest records that the compiler
                # looked at this and declined.
                ctx.notes.append(
                    f"depth_to_space {op_id}: out_channels {out_channels} is not a whole "
                    f"{plane_channels}-lane channel tile and could not be padded up to one -- no "
                    "single-consumer convolution with unshared weight/bias constants produces its "
                    "input"
                )
            else:
                padded = _ceil_div(out_channels, plane_channels) * plane_channels
                ctx.notes.append(
                    f"depth_to_space {op_id}: out_channels padded {out_channels} -> {padded} "
                    f"(a whole {plane_channels}-lane activation channel tile, which is the only "
                    "granularity OPCODE_DEPTH_TO_SPACE moves); the producing convolution gained "
                    f"{(padded - out_channels) * current.attrs.factor ** 2} zero output channels "
                    f"and the stored output keeps its true {out_channels} channels in its first "
                    "lanes"
                )
        return graph
