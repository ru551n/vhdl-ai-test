"""Graph IR (GIR): the WHAT (doc/tosa_compiler_plan.md §2.1).

Single-function, SSA graph, NHWC activations with `N == 1`, OHWI weights
(`[OC, KH, KW, IC]`, identical to TOSA's and `cnn_accel`'s order -- no
transpose needed at lowering), integer dtypes only. Every node is a frozen
dataclass; passes return new `Graph`s rather than mutating existing ones.
"""

from __future__ import annotations

import dataclasses
from types import MappingProxyType
from typing import Union

DTYPES = ("i8", "i32")

_DTYPE_RANGE = {
    "i8": (-128, 127),
    "i32": (-(2**31), 2**31 - 1),
}


def dtype_range(dtype: str) -> tuple[int, int]:
    """Inclusive `(min, max)` representable range for a GIR dtype."""
    lo_hi = _DTYPE_RANGE.get(dtype)
    if lo_hi is None:
        raise KeyError(f"unknown GIR dtype {dtype!r}")
    return lo_hi


@dataclasses.dataclass(frozen=True)
class Tensor:
    id: str
    shape: tuple[int, ...]
    dtype: str
    values: tuple[int, ...] | None = None  # row-major constant data; None if not compile-time constant
    #: The shape the SOURCE graph declared for this tensor, when a pass has
    #: had to widen `shape` to suit the hardware's storage granularity.
    #: `None` (the normal case) means `shape` is both -- nothing was
    #: widened. Today the only producer is
    #: `passes.depth_to_space_pad.PadDepthToSpaceChannelsPass`, which pads a
    #: pixel shuffle's output channel count up to a whole activation channel
    #: tile because `OPCODE_DEPTH_TO_SPACE` moves nothing smaller; the extra
    #: channels are dummies computed from zero weights.
    #:
    #: `shape` stays the authority on what the hardware WRITES (and so on
    #: what the instruction's descriptor says and how many bytes the buffer
    #: holds); `logical_shape` is the authority on what the tensor's VALUE
    #: is. `lower.layout.unpack_activation_planes` is the boundary between
    #: the two, exactly as it already is for the padding lanes of any
    #: ordinary tensor whose channel count is not a multiple of the plane
    #: width -- a C=3 RGB input, say. Carried to the program manifest via
    #: `hir.ir.Buffer.logical_shape` so a caller reading a result back gets
    #: its real channels rather than the padding.
    logical_shape: tuple[int, ...] | None = None

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n


@dataclasses.dataclass(frozen=True)
class ConvAttrs:
    pad: tuple[int, int, int, int]  # (top, bottom, left, right), TOSA order
    stride: tuple[int, int]  # (h, w)
    dilation: tuple[int, int]  # (h, w)
    in_zp: int
    w_zp: int
    acc_dtype: str


@dataclasses.dataclass(frozen=True)
class RescaleParams:
    multiplier: tuple[int, ...]  # length 1 (per-tensor) or C (per-channel)
    shift: tuple[int, ...]  # same length as multiplier
    per_channel: bool
    in_zp: int
    out_zp: int
    rounding: str  # SINGLE_ROUND | DOUBLE_ROUND | INFERENCE
    scale32: bool
    input_unsigned: bool
    output_unsigned: bool


@dataclasses.dataclass(frozen=True)
class ClampAttrs:
    min: int
    max: int


@dataclasses.dataclass(frozen=True)
class PoolAttrs:
    """TOSA `MAX_POOL2D`/`AVG_POOL2D` geometry.

    `pad_value` is the value a padded tap takes and is NOT free: TOSA
    defines MAX_POOL2D's padding as the *minimum representable value* of
    the element type (so a padded tap can never win a max), which is what
    the accelerator's ISA v2.1 `pad_value` field carries. It is stored
    explicitly rather than re-derived at each stage so `interp` and the
    emitted descriptor provably agree on one number.
    """

    mode: str  # "max" | "avg"
    kernel: tuple[int, int]  # (h, w)
    stride: tuple[int, int]  # (h, w)
    pad: tuple[int, int, int, int]  # (top, bottom, left, right), TOSA order
    pad_value: int


@dataclasses.dataclass(frozen=True)
class AddAttrs:
    """Two-input elementwise add of int8 tensors, with ONE shared rescale
    applied to each operand before the sum:

        out = sat_i8(apply_scale_32(a, multiplier, shift)
                     + apply_scale_32(b, multiplier, shift))

    Written in TOSA primitives (`interp.apply_scale_32`), but shaped by
    the accelerator's `OPCODE_ADD`, which carries exactly one
    `(requant_scale, requant_shift)` pair for both operands and rounds
    each operand *before* the sum. Two roundings, not one: that is
    observable, so the GIR op says so rather than leaving it to the
    backend.

    `multiplier = 1 << 15, shift = 15` is the identity rescale
    (`apply_scale_32(v, 2**15, 15) == v` exactly) and is what a bare
    `tosa.add` on two int8 tensors imports as; a quantized residual add
    reaches non-identity values through `passes.fuse`, which folds the
    two per-operand `tosa.rescale`s in.

    Saturation, not wraparound, is the final step -- what the hardware
    does. TOSA leaves an int8 ADD whose exact sum is unrepresentable
    undefined, so the two agree wherever TOSA is defined.
    """

    multiplier: int
    shift: int


@dataclasses.dataclass(frozen=True)
class UpsampleAttrs:
    """Nearest-neighbour spatial upsample of an NHWC tensor:

        out[0, y, x, c] = in[0, y // factor, x // factor, c]

    A pure index permutation -- no arithmetic, no rounding, nothing to
    requantize -- which is why it has no scale of any kind.

    TOSA has no such op. What a TOSA producer emits for a nearest-2x
    upsample is a `reshape -> tile -> reshape -> tile -> reshape` chain
    over a rank-5 intermediate, and `frontend.tosa_import` recognises
    that chain by *evaluating* it rather than by matching its shapes (see
    `_match_upsample_chains`). This op is what the chain collapses to,
    and it exists because the accelerator has a single instruction for
    it (`OPCODE_UPSAMPLE`).
    """

    factor: int


#: `DepthToSpaceAttrs.channel_order`: which input channel produces which
#: output sub-position. The two orders are the same *shape* function and
#: different *index* functions, so they cannot be told apart from the
#: tensor shapes -- only by evaluating the mapping, which is exactly what
#: `frontend.tosa_import._match_depth_to_space_chains` does.
PLANE_MAJOR = "plane_major"
CHANNEL_MAJOR = "channel_major"
CHANNEL_ORDERS = (PLANE_MAJOR, CHANNEL_MAJOR)


@dataclasses.dataclass(frozen=True)
class DepthToSpaceAttrs:
    """Sub-pixel convolution's pixel-shuffle step on an NHWC tensor:
    `factor**2` input channels are traded for a `factor`x larger frame in
    both spatial dimensions, so `in_channels = factor**2 * out_channels`.

    Like `UpsampleAttrs` this is a pure index permutation -- no
    arithmetic, nothing to requantize -- and like it, TOSA has no op for
    it: an exporter spells it as `reshape -> transpose -> reshape` over a
    rank-6 intermediate, which `frontend.tosa_import` recognises by
    *evaluating* the chain's index mapping.

    `channel_order` is the part that has no analogue in `upsample`, and
    it is the whole reason this attribute is not just a factor:

      * `plane_major` (`cin = (dy*factor + dx)*out_channels + c`) is what
        `OPCODE_DEPTH_TO_SPACE` implements -- `factor**2` back-to-back
        contiguous `out_channels`-wide planes, so the engine only ever
        moves whole channel tiles (`cnn_accel_model.depth_to_space`).
      * `channel_major` (`cin = c*factor**2 + dy*factor + dx`) is what
        PyTorch's `nn.PixelShuffle` means, and therefore what a real
        exported graph almost always contains. The hardware cannot do it;
        `passes.depth_to_space_channels.PermuteDepthToSpaceChannelsPass`
        rewrites it into the plane-major one by permuting the *producing
        convolution's* output-channel rows at compile time, which is free
        (it only relabels which weight row makes which channel).

    Carrying the order explicitly, rather than normalising it away in the
    frontend, is what lets that rewrite be a separate, verifiable,
    semantics-preserving pass -- and lets `lower.to_hir` refuse a
    channel-major op by name if the rewrite could not be applied, instead
    of silently emitting an instruction that computes the other one.
    """

    factor: int
    channel_order: str = PLANE_MAJOR

    def __post_init__(self) -> None:
        if self.channel_order not in CHANNEL_ORDERS:
            raise ValueError(
                f"DepthToSpaceAttrs.channel_order {self.channel_order!r} not in {CHANNEL_ORDERS}"
            )


@dataclasses.dataclass(frozen=True)
class ConcatAttrs:
    """TOSA `CONCAT` along one axis.

    Only the channel axis is ever lowered (see
    `lower.to_hir._lower_concat`): activations live in DDR as
    `[C/T][H][W][T]` channel planes (decision S6), so concatenating along
    C is contiguous plane ranges of one buffer and costs nothing, while
    concatenating along H or W would interleave every plane and cost a
    full copy. The axis is carried here rather than assumed so the GIR
    stays a faithful record of the TOSA it came from.
    """

    axis: int


@dataclasses.dataclass(frozen=True)
class SliceAttrs:
    """TOSA `SLICE`: `out = in[start : start + size]` per axis."""

    start: tuple[int, ...]
    size: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class FusedConvAttrs:
    """Composition of conv2d + rescale + optional clamp (M5). Defined now
    so the `Op.attrs` union is stable; nothing produces `fused_conv` yet."""

    conv: ConvAttrs
    rescale: RescaleParams
    clamp: ClampAttrs | None


Attrs = Union[
    ConvAttrs, RescaleParams, ClampAttrs, PoolAttrs, AddAttrs, UpsampleAttrs,
    DepthToSpaceAttrs, ConcatAttrs, SliceAttrs, FusedConvAttrs, None,
]


@dataclasses.dataclass(frozen=True)
class Op:
    id: str  # f"%{outputs[0]}", i.e. the MLIR SSA name of its first output
    kind: str  # const | conv2d | rescale | clamp | pool | add | table | upsample | depth_to_space | concat | slice | fused_conv
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    attrs: Attrs = None


@dataclasses.dataclass(frozen=True)
class Graph:
    name: str
    tensors: MappingProxyType  # tensor id (no '%') -> Tensor
    ops: tuple[Op, ...]
    inputs: tuple[str, ...]  # tensor ids (no '%'), graph-level inputs
    outputs: tuple[str, ...]  # tensor ids (no '%'), in `func.return` order

    def __post_init__(self) -> None:
        if not isinstance(self.tensors, MappingProxyType):
            object.__setattr__(self, "tensors", MappingProxyType(dict(self.tensors)))

    def tensor(self, tensor_id: str) -> Tensor:
        return self.tensors[tensor_id]

    def producer(self, tensor_id: str) -> Op | None:
        for op in self.ops:
            if tensor_id in op.outputs:
                return op
        return None

    def users(self, tensor_id: str) -> tuple[Op, ...]:
        return tuple(op for op in self.ops if tensor_id in op.inputs)

    def replace(self, **kwargs) -> "Graph":
        return dataclasses.replace(self, **kwargs)


def conv2d_output_shape(
    in_shape: tuple[int, int, int, int],
    w_shape: tuple[int, int, int, int],
    attrs: ConvAttrs,
) -> tuple[int, int, int, int]:
    """TOSA conv2d output shape, NHWC input `[N, H, W, C]` / OHWI weight
    `[OC, KH, KW, IC]`:

        out_dim = (in_dim + pad_lo + pad_hi - dil * (k - 1) - 1) // stride + 1

    applied independently to H (pad top/bottom) and W (pad left/right).
    """
    n, in_h, in_w, _ = in_shape
    oc, kh, kw, _ = w_shape
    pad_t, pad_b, pad_l, pad_r = attrs.pad
    stride_h, stride_w = attrs.stride
    dil_h, dil_w = attrs.dilation
    out_h = (in_h + pad_t + pad_b - dil_h * (kh - 1) - 1) // stride_h + 1
    out_w = (in_w + pad_l + pad_r - dil_w * (kw - 1) - 1) // stride_w + 1
    return (n, out_h, out_w, oc)


def upsample_output_shape(
    in_shape: tuple[int, int, int, int], factor: int
) -> tuple[int, int, int, int]:
    """Nearest-neighbour upsample output shape, NHWC `[N, H, W, C]`: both
    spatial dimensions scaled by `factor`, batch and channels untouched."""
    n, h, w, c = in_shape
    return (n, h * factor, w * factor, c)


def depth_to_space_output_shape(
    in_shape: tuple[int, int, int, int], factor: int
) -> tuple[int, int, int, int]:
    """Depth-to-space output shape, NHWC `[N, H, W, C]`: both spatial
    dimensions scaled by `factor`, channels divided by `factor**2`. The
    exact inverse trade of `upsample_output_shape`, which is why the
    channel count must divide evenly -- a caller that has not checked
    gets a `ValueError` rather than a silently floored channel count."""
    n, h, w, c = in_shape
    if factor < 1:
        raise ValueError(f"depth_to_space factor {factor} must be >= 1")
    if c % (factor * factor):
        raise ValueError(
            f"depth_to_space in_channels {c} is not divisible by factor**2 = {factor}**2"
        )
    return (n, h * factor, w * factor, c // (factor * factor))


def pool2d_output_shape(
    in_shape: tuple[int, int, int, int],
    attrs: PoolAttrs,
) -> tuple[int, int, int, int]:
    """TOSA MAX_POOL2D/AVG_POOL2D output shape, NHWC `[N, H, W, C]`:

        out_dim = (in_dim + pad_lo + pad_hi - k) // stride + 1

    i.e. `conv2d_output_shape` with dilation 1 and a channel count that is
    carried through instead of coming from a weight tensor (pooling is
    channel-preserving).
    """
    n, in_h, in_w, c = in_shape
    kh, kw = attrs.kernel
    pad_t, pad_b, pad_l, pad_r = attrs.pad
    stride_h, stride_w = attrs.stride
    out_h = (in_h + pad_t + pad_b - kh) // stride_h + 1
    out_w = (in_w + pad_l + pad_r - kw) // stride_w + 1
    return (n, out_h, out_w, c)
