"""ISA v2.0 command/program format.

Authoritative source: `modules/cnn_accel/doc/cnn_accel_top_v2_arch.md`
section 5 (word layout, opcodes), section 9 (error codes). The v1.2
opcodes and the byte offsets of every field that already existed in ISA
v1.2 are shared with `cnn_accel_constants.ISA_LAYOUT` /
`cnn_accel_model.encode_instruction` by construction -- a v1.2-equivalent
`DescV2` (all spaces `SPACE_DDR`, `xfer_bytes = 0`) must `encode_desc` to
exactly the same 64 bytes `cnn_accel_model.encode_instruction` produces
for the equivalent `LayerDesc` (see `accel_v2/tests/test_isa.py`).

Only the standard library is used, matching the rest of the golden model
(plain `dataclass` + `struct`, no numpy), so this stays bit-comparable
with `cnn_accel_model.py`.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import cnn_accel_constants as _const

# ---------------------------------------------------------------------------
# Version / word size.
# ---------------------------------------------------------------------------

ISA_VERSION = _const.ISA_VERSION
INSTR_WORD_BYTES = _const.INSTR_WORD_BYTES

#: Bytes an `ACT` descriptor's standalone activation LUT occupies, and
#: therefore the DDR read one `ACT` costs beyond its operands: the table
#: is one int8 per raw input byte, fetched in full from `weight_addr`
#: every time the opcode runs (`cnn_accel_elementwise.vhd`'s
#: `c_lut_entries`, whose fill is armed per command -- there is no
#: caching across descriptors). Stated once here because `program.py`
#: allocates and writes exactly this many bytes, `planner.py` predicts
#: the read and `reference.py` charges it independently.
ACT_LUT_BYTES = 256

# ---------------------------------------------------------------------------
# Space tags (section 3): 2-bit tag per operand, packed into W0 bits
# [23:16] (see `_SPACE_SHIFT` below).
# ---------------------------------------------------------------------------

SPACE_DDR = _const.SPACES["DDR"]
SPACE_LOCAL_TENSOR = _const.SPACES["LOCAL_TENSOR"]
SPACE_LOCAL_WEIGHT = _const.SPACES["LOCAL_WEIGHT"]
SPACE_RESERVED = _const.SPACES["RESERVED"]

# ---------------------------------------------------------------------------
# Opcodes (section 5.2). The first six reuse the v1.2 values verbatim
# (`cnn_accel_constants.OPCODES`); v2.0 adds the DMA/elementwise family.
# ---------------------------------------------------------------------------

OPCODES: dict[str, int] = dict(_const.OPCODES)

for _name, _value in OPCODES.items():
    globals()[f"OPCODE_{_name}"] = _value
del _name, _value

# ---------------------------------------------------------------------------
# Flags (section 5.1, W0 bits [15:8]).
# ---------------------------------------------------------------------------

FLAGS: dict[str, int] = dict(_const.FLAGS)

for _name, _bit in FLAGS.items():
    globals()[f"FLAG_{_name}"] = _bit
del _name, _bit

# ---------------------------------------------------------------------------
# Error codes (section 9).
# ---------------------------------------------------------------------------

ERR_NONE = 0x0
ERR_UNSUPPORTED_OP = 0x1
ERR_BAD_SPACE = 0x2
ERR_MISALIGNED = 0x3
ERR_LOCAL_RANGE = 0x4
ERR_DDR_RANGE = 0x5
ERR_BAD_RESERVED = 0x6
ERR_BAD_GEOMETRY = 0x7
ERR_AXI = 0x8
ERR_TIMEOUT = 0x9
# ISA v2.3: CTRL.START written while BUSY=1 and a job is already queued
# (section 8's one-deep INPUT_ADDR/OUTPUT_ADDR queue). Raised by
# cnn_accel_csr, not cnn_accel_cmd_proc -- every other ERR_CODE above
# names a bad *program*; this one names a bad *host write*, and the
# job already running is completely unaffected by it.
ERR_QUEUE_FULL = 0xA

# ---------------------------------------------------------------------------
# Byte offsets for the 64-byte word (section 5.1), all derived from
# `cnn_accel_constants.ISA_LAYOUT`. The `spaces` byte at offset 2 packs
# [1:0]=space_src0 [3:2]=space_src1 [5:4]=space_dst [7:6]=space_wgt.
# ---------------------------------------------------------------------------

# The two `reserved, must be 0` gaps. Neither has an entry in
# `isa_field_offsets()` -- that function returns only named fields -- so
# these are derived from `isa_reserved_ranges()` instead of being stated,
# which keeps them correct if the layout table changes.
_RESERVED_RANGES = _const.isa_reserved_ranges()
_OFF_RESERVED_W0 = _RESERVED_RANGES[0][0]
_OFF_RESERVED_W10 = _RESERVED_RANGES[1][0]
_LEN_RESERVED_W10 = _RESERVED_RANGES[1][1] - _RESERVED_RANGES[1][0] + 1

# Derived, never hand-typed: walking `cnn_accel_constants.ISA_LAYOUT` is
# what guarantees this encoder and the RTL's generated `cnn_accel_isa_pkg`
# cannot drift apart (AGENTS.md: "edit the Python, never the generated
# .vhd"). `_OFF_<FIELD>` is injected for every named field in the table,
# so `spaces` and `xfer_bytes` appear here automatically now that they are
# real fields rather than reserved gaps.
_FIELD_OFFSETS: dict[str, int] = {
    f.name: f.offset_bytes for f in _const.isa_field_offsets()
}
_FIELD_WIDTHS: dict[str, int] = {
    f.name: f.width_bytes for f in _const.isa_field_offsets()
}

for _name, _offset in _FIELD_OFFSETS.items():
    globals()[f"_OFF_{_name.upper()}"] = _offset
del _name, _offset

_SPACE_SRC0_SHIFT = _const.SPACE_FIELDS["SRC0"]
_SPACE_SRC1_SHIFT = _const.SPACE_FIELDS["SRC1"]
_SPACE_DST_SHIFT = _const.SPACE_FIELDS["DST"]
_SPACE_WGT_SHIFT = _const.SPACE_FIELDS["WGT"]
_SPACE_MASK = (1 << _const.SPACE_TAG_BITS) - 1


@dataclass
class DescV2:
    """One decoded 64-byte ISA v2.0 descriptor.

    Field names mirror the section 5.1 word layout; every field defaults
    to 0 (a default-constructed `DescV2` is the all-zero `HALT`-with-DDR-
    spaces descriptor). Unsigned fields are plain non-negative `int`s;
    `requant_scale`, `output_offset`, `clamp_min`, `clamp_max` are signed.
    """

    opcode: int = 0
    flags: int = 0
    space_src0: int = 0
    space_src1: int = 0
    space_dst: int = 0
    space_wgt: int = 0
    in_addr: int = 0
    out_addr: int = 0
    weight_addr: int = 0
    bias_addr: int = 0
    in_width: int = 0
    in_height: int = 0
    in_channels: int = 0
    out_channels: int = 0
    kernel_h: int = 0
    kernel_w: int = 0
    stride_h: int = 0
    stride_w: int = 0
    pad_top: int = 0
    pad_bottom: int = 0
    pad_left: int = 0
    pad_right: int = 0
    requant_scale: int = 0
    requant_shift: int = 0
    pool_kernel_h: int = 0
    pool_kernel_w: int = 0
    pool_stride_h: int = 0
    pool_stride_w: int = 0
    next_instr_addr: int = 0
    output_offset: int = 0
    clamp_min: int = 0
    clamp_max: int = 0
    scale_addr: int = 0
    xfer_bytes: int = 0
    # ISA v2.1, W10 byte 41: signed int8 value a padded tap takes (the
    # input tensor's quantization zero-point). 0 -- the value every v2.0
    # program left in this then-reserved byte -- means "pad with zero",
    # i.e. exactly v2.0 behaviour. Consumed by every opcode that pads a
    # window: POOL_MAX/POOL_AVG and CONV2D/DWCONV2D/FC alike.
    pad_value: int = 0
    # ISA v2.2, W10 byte 42: DEPTH_TO_SPACE's upscale factor r. 0 -- the
    # value every v2.1 program left in this then-reserved byte -- is not a
    # legal factor, so it can never be misread as a real one; only
    # DEPTH_TO_SPACE reads this field. v1 hardware validates `r == 2`
    # only, per `cnn_accel_cmd_proc`.
    dts_factor: int = 0
    # ISA v2.3, W0 byte 3 bits [1:0]: relocate this descriptor's
    # `in_addr`/`out_addr` by the CSR's latched INPUT_ADDR/OUTPUT_ADDR
    # (section 6a) when its space is `SPACE_DDR` -- the mechanism behind
    # the streaming-inference interface (compile the network once, drive
    # per-frame buffer addresses through two registers instead of
    # recompiling the descriptor chain). `False` on every field of every
    # program before v2.3 (the bits were part of `reserved_w0`, always
    # zero), so relocating nothing reproduces every existing program's
    # addresses exactly.
    reloc_input: bool = False
    reloc_output: bool = False
    # The `reserved, must be 0` gap remaining after `reloc_input`/
    # `reloc_output` claimed W0 byte 3 bits [1:0] (W10 byte 43 is
    # untouched). This is the value of bits [7:2] of that byte, NOT the
    # raw byte -- `encode_desc`/`decode_desc` shift it so the two concepts
    # never share a bit position. A well-formed program always leaves it
    # zero, and it is exposed here for exactly one purpose: letting the
    # error-case tests emit a deliberately malformed program to prove the
    # hardware raises ERR_BAD_RESERVED instead of executing it. Do not use
    # it to smuggle data -- `cnn_accel_cmd_proc` rejects any descriptor
    # with either reserved gap set.
    reserved_w0: int = 0
    reserved_w10: int = 0

    @property
    def relu_en(self) -> bool:
        return bool((self.flags >> FLAG_RELU_EN) & 1)

    @property
    def bias_en(self) -> bool:
        return bool((self.flags >> FLAG_BIAS_EN) & 1)

    @property
    def requant_en(self) -> bool:
        return bool((self.flags >> FLAG_REQUANT_EN) & 1)

    @property
    def pad_en(self) -> bool:
        return bool((self.flags >> FLAG_PAD_EN) & 1)

    @property
    def clamp_en(self) -> bool:
        return bool((self.flags >> FLAG_CLAMP_EN) & 1)

    @property
    def per_channel_en(self) -> bool:
        return bool((self.flags >> FLAG_PER_CHANNEL_EN) & 1)

    @property
    def act_lut_en(self) -> bool:
        return bool((self.flags >> FLAG_ACT_LUT_EN) & 1)

    @property
    def weight_reuse(self) -> bool:
        return bool((self.flags >> FLAG_WEIGHT_REUSE) & 1)


def encode_desc(d: DescV2) -> bytes:
    """Encode one `DescV2` as the 64-byte little-endian ISA v2.0 word
    (section 5.1). Byte-identical to `cnn_accel_model.encode_instruction`
    for a v1.2-equivalent descriptor (all spaces `SPACE_DDR`, `xfer_bytes
    == 0`) -- see `accel_v2/tests/test_isa.py`."""
    buf = bytearray(INSTR_WORD_BYTES)
    buf[_OFF_OPCODE] = d.opcode & 0xFF
    buf[_OFF_FLAGS] = d.flags & 0xFF
    buf[_OFF_SPACES] = (
        ((d.space_src0 & _SPACE_MASK) << _SPACE_SRC0_SHIFT)
        | ((d.space_src1 & _SPACE_MASK) << _SPACE_SRC1_SHIFT)
        | ((d.space_dst & _SPACE_MASK) << _SPACE_DST_SHIFT)
        | ((d.space_wgt & _SPACE_MASK) << _SPACE_WGT_SHIFT)
    )
    buf[_OFF_RESERVED_W0] = (
        ((d.reserved_w0 & 0x3F) << 2) | (int(d.reloc_input) << 0) | (int(d.reloc_output) << 1)
    )
    struct.pack_into("<I", buf, _OFF_IN_ADDR, d.in_addr & 0xFFFFFFFF)
    struct.pack_into("<I", buf, _OFF_OUT_ADDR, d.out_addr & 0xFFFFFFFF)
    struct.pack_into("<I", buf, _OFF_WEIGHT_ADDR, d.weight_addr & 0xFFFFFFFF)
    struct.pack_into("<I", buf, _OFF_BIAS_ADDR, d.bias_addr & 0xFFFFFFFF)
    struct.pack_into("<H", buf, _OFF_IN_WIDTH, d.in_width & 0xFFFF)
    struct.pack_into("<H", buf, _OFF_IN_HEIGHT, d.in_height & 0xFFFF)
    struct.pack_into("<H", buf, _OFF_IN_CHANNELS, d.in_channels & 0xFFFF)
    struct.pack_into("<H", buf, _OFF_OUT_CHANNELS, d.out_channels & 0xFFFF)
    buf[_OFF_KERNEL_H] = d.kernel_h & 0xFF
    buf[_OFF_KERNEL_W] = d.kernel_w & 0xFF
    buf[_OFF_STRIDE_H] = d.stride_h & 0xFF
    buf[_OFF_STRIDE_W] = d.stride_w & 0xFF
    buf[_OFF_PAD_TOP] = d.pad_top & 0xFF
    buf[_OFF_PAD_BOTTOM] = d.pad_bottom & 0xFF
    buf[_OFF_PAD_LEFT] = d.pad_left & 0xFF
    buf[_OFF_PAD_RIGHT] = d.pad_right & 0xFF
    struct.pack_into("<i", buf, _OFF_REQUANT_SCALE, d.requant_scale)
    buf[_OFF_REQUANT_SHIFT] = d.requant_shift & 0xFF
    buf[_OFF_RESERVED_W10 : _OFF_RESERVED_W10 + _LEN_RESERVED_W10] = (
        d.reserved_w10 & ((1 << (8 * _LEN_RESERVED_W10)) - 1)
    ).to_bytes(_LEN_RESERVED_W10, "little")
    buf[_OFF_POOL_KERNEL_H] = d.pool_kernel_h & 0xFF
    buf[_OFF_POOL_KERNEL_W] = d.pool_kernel_w & 0xFF
    buf[_OFF_POOL_STRIDE_H] = d.pool_stride_h & 0xFF
    buf[_OFF_POOL_STRIDE_W] = d.pool_stride_w & 0xFF
    struct.pack_into("<I", buf, _OFF_NEXT_INSTR_ADDR, d.next_instr_addr & 0xFFFFFFFF)
    struct.pack_into("<h", buf, _OFF_OUTPUT_OFFSET, d.output_offset)
    struct.pack_into("<b", buf, _OFF_CLAMP_MIN, d.clamp_min)
    struct.pack_into("<b", buf, _OFF_CLAMP_MAX, d.clamp_max)
    struct.pack_into("<I", buf, _OFF_SCALE_ADDR, d.scale_addr & 0xFFFFFFFF)
    struct.pack_into("<I", buf, _OFF_XFER_BYTES, d.xfer_bytes & 0xFFFFFFFF)
    struct.pack_into("<b", buf, _OFF_PAD_VALUE, d.pad_value)
    buf[_OFF_DTS_FACTOR] = d.dts_factor & 0xFF
    return bytes(buf)


def decode_desc(data: bytes) -> DescV2:
    """Inverse of `encode_desc`."""
    if len(data) != INSTR_WORD_BYTES:
        raise ValueError(f"expected {INSTR_WORD_BYTES} bytes, got {len(data)}")

    spaces = data[_OFF_SPACES]
    return DescV2(
        opcode=data[_OFF_OPCODE],
        flags=data[_OFF_FLAGS],
        space_src0=(spaces >> _SPACE_SRC0_SHIFT) & _SPACE_MASK,
        space_src1=(spaces >> _SPACE_SRC1_SHIFT) & _SPACE_MASK,
        space_dst=(spaces >> _SPACE_DST_SHIFT) & _SPACE_MASK,
        space_wgt=(spaces >> _SPACE_WGT_SHIFT) & _SPACE_MASK,
        in_addr=struct.unpack_from("<I", data, _OFF_IN_ADDR)[0],
        out_addr=struct.unpack_from("<I", data, _OFF_OUT_ADDR)[0],
        weight_addr=struct.unpack_from("<I", data, _OFF_WEIGHT_ADDR)[0],
        bias_addr=struct.unpack_from("<I", data, _OFF_BIAS_ADDR)[0],
        in_width=struct.unpack_from("<H", data, _OFF_IN_WIDTH)[0],
        in_height=struct.unpack_from("<H", data, _OFF_IN_HEIGHT)[0],
        in_channels=struct.unpack_from("<H", data, _OFF_IN_CHANNELS)[0],
        out_channels=struct.unpack_from("<H", data, _OFF_OUT_CHANNELS)[0],
        kernel_h=data[_OFF_KERNEL_H],
        kernel_w=data[_OFF_KERNEL_W],
        stride_h=data[_OFF_STRIDE_H],
        stride_w=data[_OFF_STRIDE_W],
        pad_top=data[_OFF_PAD_TOP],
        pad_bottom=data[_OFF_PAD_BOTTOM],
        pad_left=data[_OFF_PAD_LEFT],
        pad_right=data[_OFF_PAD_RIGHT],
        requant_scale=struct.unpack_from("<i", data, _OFF_REQUANT_SCALE)[0],
        requant_shift=data[_OFF_REQUANT_SHIFT],
        reloc_input=bool(data[_OFF_RESERVED_W0] & 0x1),
        reloc_output=bool(data[_OFF_RESERVED_W0] & 0x2),
        reserved_w0=(data[_OFF_RESERVED_W0] >> 2) & 0x3F,
        reserved_w10=int.from_bytes(
            data[_OFF_RESERVED_W10 : _OFF_RESERVED_W10 + _LEN_RESERVED_W10],
            "little",
        ),
        pool_kernel_h=data[_OFF_POOL_KERNEL_H],
        pool_kernel_w=data[_OFF_POOL_KERNEL_W],
        pool_stride_h=data[_OFF_POOL_STRIDE_H],
        pool_stride_w=data[_OFF_POOL_STRIDE_W],
        next_instr_addr=struct.unpack_from("<I", data, _OFF_NEXT_INSTR_ADDR)[0],
        output_offset=struct.unpack_from("<h", data, _OFF_OUTPUT_OFFSET)[0],
        clamp_min=struct.unpack_from("<b", data, _OFF_CLAMP_MIN)[0],
        clamp_max=struct.unpack_from("<b", data, _OFF_CLAMP_MAX)[0],
        scale_addr=struct.unpack_from("<I", data, _OFF_SCALE_ADDR)[0],
        xfer_bytes=struct.unpack_from("<I", data, _OFF_XFER_BYTES)[0],
        pad_value=struct.unpack_from("<b", data, _OFF_PAD_VALUE)[0],
        dts_factor=data[_OFF_DTS_FACTOR],
    )


__all__ = [
    "ISA_VERSION",
    "ACT_LUT_BYTES",
    "INSTR_WORD_BYTES",
    "SPACE_DDR",
    "SPACE_LOCAL_TENSOR",
    "SPACE_LOCAL_WEIGHT",
    "SPACE_RESERVED",
    "OPCODES",
    "FLAGS",
    "ERR_NONE",
    "ERR_UNSUPPORTED_OP",
    "ERR_BAD_SPACE",
    "ERR_MISALIGNED",
    "ERR_LOCAL_RANGE",
    "ERR_DDR_RANGE",
    "ERR_BAD_RESERVED",
    "ERR_BAD_GEOMETRY",
    "ERR_AXI",
    "ERR_TIMEOUT",
    "ERR_QUEUE_FULL",
    "DescV2",
    "encode_desc",
    "decode_desc",
]
