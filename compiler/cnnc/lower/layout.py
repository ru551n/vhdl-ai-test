"""DDR byte images of the accelerator's tensors (doc/cnn_accel_arch.md
decisions S6, D10, D11).

Every function here is a pure function of the logical tensor plus the
tiling parameters the target discovered (`Memory.activation_plane_channels`,
`Unit.internal_tiling.cin/cout`); nothing is hard-coded to one hardware
configuration. The golden model (`cnn_accel_model.pack_activation_planes`,
`pack_weights_for_hw`, `pack_bias_for_hw`) defines the same images
independently; `tests/test_layout.py` pins the two against each other.

- Activations (S6): channel-tiled planes `[ceil(C/T)][H][W][T]`, the
  channels of the last plane beyond `C` zero-padded, so a buffer is always
  `ceil(C/T) * T * H * W` bytes.
- Weights (D10/D11): tile-major `[OT][IT][kh][kw][pe_rows][tile_channels]`
  with `oc = ot*pe_rows + r`, `ic = it*tile_channels + c`; lanes past
  `out_channels`/`in_channels` are written as 0, never omitted.
- Bias: `[OT][pe_rows]` int32 little-endian, zero-padded like the weights'
  output-channel tiles.
- ACT LUT (ISA v2.0, `OPCODE_ACT`): 256 int8 bytes, the TOSA `TABLE`
  operand rotated by 128 into the hardware's raw-byte index order; see
  `pack_act_lut`.
- Scale table (ISA v1.2 / H2, doc/tosa_compiler_plan.md §5 extension 2):
  `[OT][pe_rows]` entries of `SCALE_TABLE_ENTRY_BYTES` (8) bytes each,
  `multiplier int32 LE | shift uint8 | 3 zero bytes`, padded with all-zero
  entries to whole `pe_rows` tiles exactly like the bias (the weight buffer
  fills its `scale_buffer` lanes in the same tile-load phase as the bias).
  `shift` here is the *descriptor* shift (TOSA shift minus the target's
  `implicit_shift`), the same semantics as W-field `requant_shift`.
"""

from __future__ import annotations

import struct

import numpy as np


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def activation_bytes(shape: tuple[int, ...], plane_channels: int, elem_bytes: int = 1) -> int:
    """Size of an `N x H x W x C` (N == 1) activation stored as S6 planes."""
    _n, h, w, c = shape
    return _ceil_div(c, plane_channels) * plane_channels * h * w * elem_bytes


def pack_activation_planes(arr: np.ndarray, plane_channels: int) -> bytes:
    """NHWC (N == 1) array -> S6 plane image bytes."""
    n, h, w, c = arr.shape
    if n != 1:
        raise ValueError(f"activation batch must be 1, got shape {arr.shape}")
    planes = _ceil_div(c, plane_channels)
    padded = np.zeros((h, w, planes * plane_channels), dtype=arr.dtype)
    padded[:, :, :c] = arr[0]
    # [H][W][P][T] -> [P][H][W][T]
    tiled = padded.reshape(h, w, planes, plane_channels).transpose(2, 0, 1, 3)
    return np.ascontiguousarray(tiled).tobytes(order="C")


def unpack_activation_planes(raw: bytes, shape: tuple[int, ...], dtype, plane_channels: int) -> np.ndarray:
    """Exact inverse of `pack_activation_planes`; drops the padding lanes."""
    n, h, w, c = shape
    if n != 1:
        raise ValueError(f"activation batch must be 1, got shape {shape}")
    planes = _ceil_div(c, plane_channels)
    tiled = np.frombuffer(raw, dtype=dtype).reshape(planes, h, w, plane_channels)
    padded = tiled.transpose(1, 2, 0, 3).reshape(h, w, planes * plane_channels)
    return np.ascontiguousarray(padded[:, :, :c]).reshape(1, h, w, c)


def packed_weight_bytes(shape: tuple[int, ...], tile_channels: int, pe_rows: int) -> int:
    oc, kh, kw, ic = shape
    return _ceil_div(oc, pe_rows) * _ceil_div(ic, tile_channels) * kh * kw * tile_channels * pe_rows


def pack_weights_tiled(values: tuple[int, ...], shape: tuple[int, ...], tile_channels: int, pe_rows: int) -> bytes:
    """Logical OHWI int8 weights -> D10 tile-major image (see module doc)."""
    oc, kh, kw, ic = shape
    if len(values) != oc * kh * kw * ic:
        raise ValueError(f"weight value count {len(values)} != prod{shape}")
    out = bytearray()
    for ot in range(_ceil_div(oc, pe_rows)):
        for it in range(_ceil_div(ic, tile_channels)):
            for kr in range(kh):
                for kc in range(kw):
                    for r in range(pe_rows):
                        o = ot * pe_rows + r
                        for c in range(tile_channels):
                            i = it * tile_channels + c
                            v = values[((o * kh + kr) * kw + kc) * ic + i] if o < oc and i < ic else 0
                            out.append(v & 0xFF)
    return bytes(out)


def packed_bias_bytes(out_channels: int, pe_rows: int) -> int:
    return _ceil_div(out_channels, pe_rows) * pe_rows * 4


def pack_bias_tiled(values: tuple[int, ...], pe_rows: int) -> bytes:
    """Per-output-channel int32 bias -> `[OT][pe_rows]` little-endian image."""
    oc = len(values)
    padded = list(values) + [0] * (_ceil_div(oc, pe_rows) * pe_rows - oc)
    return struct.pack(f"<{len(padded)}i", *padded)


ACT_LUT_ENTRIES = 256


def pack_act_lut(values: tuple[int, ...]) -> bytes:
    """A TOSA `TABLE` operand -> the accelerator's 256-byte ACT LUT image.

    The two tables hold the same 256 answers in a DIFFERENT ORDER, and
    that rotation is the whole content of this function:

    * TOSA indexes an int8 TABLE by `value - type_min`, i.e. entry `i`
      answers input `i - 128`; the table runs -128, -127, ... 127.
    * the hardware indexes by the raw byte, `lut[v & 0xFF]`, i.e. entry
      `i` answers input `i` for `i < 128` and `i - 256` above it; the
      table runs 0, 1, ... 127, -128, ... -1
      (`cnn_accel_model.act_lut`).

    So `hw[i] = tosa[(i + 128) % 256]`. Getting this backwards is not a
    crash, it is a silently wrong activation on every value, which is why
    it lives in one named function with one test rather than inline at the
    lowering site."""
    if len(values) != ACT_LUT_ENTRIES:
        raise ValueError(f"ACT LUT must have exactly {ACT_LUT_ENTRIES} entries, got {len(values)}")
    return bytes(int(values[(i + 128) % ACT_LUT_ENTRIES]) & 0xFF for i in range(ACT_LUT_ENTRIES))


SCALE_TABLE_ENTRY_BYTES = 8
_SCALE_TABLE_ENTRY_PAD = SCALE_TABLE_ENTRY_BYTES - 4 - 1  # int32 multiplier + uint8 shift


def packed_scale_table_bytes(out_channels: int, pe_rows: int) -> int:
    return _ceil_div(out_channels, pe_rows) * pe_rows * SCALE_TABLE_ENTRY_BYTES


def pack_scale_table(multipliers: tuple[int, ...], shifts: tuple[int, ...], pe_rows: int) -> bytes:
    """Per-output-channel `(multiplier, shift)` -> the `[OT][pe_rows]` scale
    table image of `cnn_accel_model.pack_scale_table_for_hw` (see module
    doc). `shifts` are descriptor shifts (already minus `implicit_shift`),
    each in `0..255`; multipliers are signed int32."""
    if len(multipliers) != len(shifts):
        raise ValueError(f"scale table multiplier count {len(multipliers)} != shift count {len(shifts)}")
    oc = len(multipliers)
    n_padded = _ceil_div(oc, pe_rows) * pe_rows
    out = bytearray()
    for i in range(n_padded):
        m, s = (int(multipliers[i]), int(shifts[i])) if i < oc else (0, 0)
        if not -(2**31) <= m <= 2**31 - 1:
            raise ValueError(f"scale table entry {i} multiplier {m} outside int32")
        if not 0 <= s <= 0xFF:
            raise ValueError(f"scale table entry {i} shift {s} outside uint8")
        out += struct.pack("<iB", m, s)
        out += bytes(_SCALE_TABLE_ENTRY_PAD)
    return bytes(out)
