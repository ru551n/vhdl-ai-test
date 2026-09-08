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
class FusedConvAttrs:
    """Composition of conv2d + rescale + optional clamp (M5). Defined now
    so the `Op.attrs` union is stable; nothing produces `fused_conv` yet."""

    conv: ConvAttrs
    rescale: RescaleParams
    clamp: ClampAttrs | None


Attrs = Union[ConvAttrs, RescaleParams, ClampAttrs, PoolAttrs, FusedConvAttrs, None]


@dataclasses.dataclass(frozen=True)
class Op:
    id: str  # f"%{outputs[0]}", i.e. the MLIR SSA name of its first output
    kind: str  # const | conv2d | rescale | clamp | pool | fused_conv
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
