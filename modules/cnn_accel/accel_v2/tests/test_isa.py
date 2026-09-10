"""Tests for `accel_v2.isa`: dataclass/codec round-trip and the critical
v1.2 byte-compatibility guarantee (section 5.1: `W15` was `reserved, must
be 0` in v1.2, and the new `space_*` tags live entirely in bits that were
`reserved, must be 0` too, so a v1.2-equivalent `DescV2` must encode
identically to `cnn_accel_model.encode_instruction`)."""

from __future__ import annotations

import random

import pytest

import cnn_accel_model as model

from accel_v2 import isa


def test_isa_version_and_word_size() -> None:
    # v2.2 adds the DEPTH_TO_SPACE opcode and its W10 'dts_factor' byte;
    # both were unassigned/reserved-must-be-0 in v2.1, so the word size is
    # unchanged and every v2.1 program is still valid.
    assert isa.ISA_VERSION == 0x0202
    assert isa.INSTR_WORD_BYTES == 64


def test_dts_factor_round_trips() -> None:
    for value in (0, 2, 3, 4, 255):
        d = isa.DescV2(opcode=isa.OPCODE_DEPTH_TO_SPACE, dts_factor=value)
        assert isa.decode_desc(isa.encode_desc(d)).dts_factor == value
    # Default-zero: a descriptor that never mentions dts_factor encodes the
    # byte as 0, i.e. exactly the v2.1 reserved-zero word -- and 0 is not a
    # legal factor, so an old program can never be misread as requesting one.
    assert isa.encode_desc(isa.DescV2())[42] == 0


def test_pad_value_round_trips_signed() -> None:
    for value in (-128, -1, 0, 1, 127):
        d = isa.DescV2(opcode=isa.OPCODE_POOL_MAX, pad_value=value)
        assert isa.decode_desc(isa.encode_desc(d)).pad_value == value
    # Default-zero: a descriptor that never mentions pad_value encodes the
    # byte as 0, i.e. exactly the v2.0 reserved-zero word.
    assert isa.encode_desc(isa.DescV2())[41] == 0


def test_space_tags() -> None:
    assert isa.SPACE_DDR == 0
    assert isa.SPACE_LOCAL_TENSOR == 1
    assert isa.SPACE_LOCAL_WEIGHT == 2
    assert isa.SPACE_RESERVED == 3


def test_opcodes_reuse_v12_values_and_add_v20_family() -> None:
    assert isa.OPCODES["HALT"] == 0x00
    assert isa.OPCODES["CONV2D"] == 0x01
    assert isa.OPCODES["DWCONV2D"] == 0x02
    assert isa.OPCODES["POOL_MAX"] == 0x03
    assert isa.OPCODES["POOL_AVG"] == 0x04
    assert isa.OPCODES["FC"] == 0x05
    assert isa.OPCODES["LOAD"] == 0x10
    assert isa.OPCODES["STORE"] == 0x11
    assert isa.OPCODES["LOADW"] == 0x12
    assert isa.OPCODES["ADD"] == 0x13
    assert isa.OPCODES["UPSAMPLE"] == 0x14
    assert isa.OPCODES["COPY"] == 0x15
    assert isa.OPCODES["ACT"] == 0x16
    assert isa.OPCODES["DEPTH_TO_SPACE"] == 0x17
    # Every v1.2 opcode value/name matches cnn_accel_constants exactly.
    import cnn_accel_constants as const

    for name, value in const.OPCODES.items():
        assert isa.OPCODES[name] == value


def test_flag_bit_positions() -> None:
    assert isa.FLAGS == {
        "RELU_EN": 0,
        "BIAS_EN": 1,
        "REQUANT_EN": 2,
        "PAD_EN": 3,
        "CLAMP_EN": 4,
        "PER_CHANNEL_EN": 5,
        "ACT_LUT_EN": 6,
        "WEIGHT_REUSE": 7,
    }


def test_error_codes() -> None:
    assert isa.ERR_NONE == 0
    assert isa.ERR_UNSUPPORTED_OP == 1
    assert isa.ERR_BAD_SPACE == 2
    assert isa.ERR_MISALIGNED == 3
    assert isa.ERR_LOCAL_RANGE == 4
    assert isa.ERR_DDR_RANGE == 5
    assert isa.ERR_BAD_RESERVED == 6
    assert isa.ERR_BAD_GEOMETRY == 7
    assert isa.ERR_AXI == 8
    assert isa.ERR_TIMEOUT == 9


def test_flag_convenience_properties() -> None:
    all_flags = sum(1 << bit for bit in isa.FLAGS.values())
    d = isa.DescV2(flags=all_flags)
    assert d.relu_en
    assert d.bias_en
    assert d.requant_en
    assert d.pad_en
    assert d.clamp_en
    assert d.per_channel_en
    assert d.act_lut_en
    assert d.weight_reuse

    d0 = isa.DescV2(flags=0)
    assert not any(
        [
            d0.relu_en,
            d0.bias_en,
            d0.requant_en,
            d0.pad_en,
            d0.clamp_en,
            d0.per_channel_en,
            d0.act_lut_en,
            d0.weight_reuse,
        ]
    )


def _full_desc() -> isa.DescV2:
    """A `DescV2` exercising every field with a distinct, non-degenerate
    value (so a field-transposition bug in the codec would be caught)."""
    return isa.DescV2(
        opcode=isa.OPCODES["CONV2D"],
        flags=(1 << isa.FLAG_BIAS_EN) | (1 << isa.FLAG_REQUANT_EN),
        space_src0=isa.SPACE_LOCAL_TENSOR,
        space_src1=isa.SPACE_DDR,
        space_dst=isa.SPACE_LOCAL_TENSOR,
        space_wgt=isa.SPACE_LOCAL_WEIGHT,
        in_addr=0x1000,
        out_addr=0x2000,
        weight_addr=0x3000,
        bias_addr=0x4000,
        in_width=32,
        in_height=16,
        in_channels=8,
        out_channels=24,
        kernel_h=3,
        kernel_w=3,
        stride_h=2,
        stride_w=2,
        pad_top=1,
        pad_bottom=1,
        pad_left=1,
        pad_right=1,
        requant_scale=-12345,
        requant_shift=13,
        pool_kernel_h=2,
        pool_kernel_w=2,
        pool_stride_h=2,
        pool_stride_w=2,
        next_instr_addr=0x1040,
        output_offset=-42,
        clamp_min=-128,
        clamp_max=127,
        scale_addr=0x5000,
        xfer_bytes=0x800,
    )


def test_encode_decode_round_trip() -> None:
    d = _full_desc()
    data = isa.encode_desc(d)
    assert len(data) == isa.INSTR_WORD_BYTES
    assert isa.decode_desc(data) == d


def test_encode_decode_round_trip_random(seed: int = 12345) -> None:
    rng = random.Random(seed)
    for _ in range(200):
        d = isa.DescV2(
            opcode=rng.randrange(0, 256),
            flags=rng.randrange(0, 256),
            space_src0=rng.randrange(0, 4),
            space_src1=rng.randrange(0, 4),
            space_dst=rng.randrange(0, 4),
            space_wgt=rng.randrange(0, 4),
            in_addr=rng.randrange(0, 2**32),
            out_addr=rng.randrange(0, 2**32),
            weight_addr=rng.randrange(0, 2**32),
            bias_addr=rng.randrange(0, 2**32),
            in_width=rng.randrange(0, 2**16),
            in_height=rng.randrange(0, 2**16),
            in_channels=rng.randrange(0, 2**16),
            out_channels=rng.randrange(0, 2**16),
            kernel_h=rng.randrange(0, 256),
            kernel_w=rng.randrange(0, 256),
            stride_h=rng.randrange(0, 256),
            stride_w=rng.randrange(0, 256),
            pad_top=rng.randrange(0, 256),
            pad_bottom=rng.randrange(0, 256),
            pad_left=rng.randrange(0, 256),
            pad_right=rng.randrange(0, 256),
            requant_scale=rng.randrange(-(2**31), 2**31),
            requant_shift=rng.randrange(0, 256),
            pool_kernel_h=rng.randrange(0, 256),
            pool_kernel_w=rng.randrange(0, 256),
            pool_stride_h=rng.randrange(0, 256),
            pool_stride_w=rng.randrange(0, 256),
            next_instr_addr=rng.randrange(0, 2**32),
            output_offset=rng.randrange(-(2**15), 2**15),
            clamp_min=rng.randrange(-128, 128),
            clamp_max=rng.randrange(-128, 128),
            scale_addr=rng.randrange(0, 2**32),
            xfer_bytes=rng.randrange(0, 2**32),
        )
        data = isa.encode_desc(d)
        assert len(data) == isa.INSTR_WORD_BYTES
        assert isa.decode_desc(data) == d


def _layer_desc_equivalent_of(d: isa.DescV2) -> model.LayerDesc:
    """Build the `LayerDesc` with identical field values to `d` for
    every field that exists in both ISA v1.2 and v2.0."""
    return model.LayerDesc(
        opcode=d.opcode,
        flags=d.flags,
        in_addr=d.in_addr,
        out_addr=d.out_addr,
        weight_addr=d.weight_addr,
        bias_addr=d.bias_addr,
        in_width=d.in_width,
        in_height=d.in_height,
        in_channels=d.in_channels,
        out_channels=d.out_channels,
        kernel_h=d.kernel_h,
        kernel_w=d.kernel_w,
        stride_h=d.stride_h,
        stride_w=d.stride_w,
        pad_top=d.pad_top,
        pad_bottom=d.pad_bottom,
        pad_left=d.pad_left,
        pad_right=d.pad_right,
        requant_scale=d.requant_scale,
        requant_shift=d.requant_shift,
        pool_kernel_h=d.pool_kernel_h,
        pool_kernel_w=d.pool_kernel_w,
        pool_stride_h=d.pool_stride_h,
        pool_stride_w=d.pool_stride_w,
        next_instr_addr=d.next_instr_addr,
        output_offset=d.output_offset,
        clamp_min=d.clamp_min,
        clamp_max=d.clamp_max,
        scale_addr=d.scale_addr,
    )


def test_v12_compatible_descriptor_encodes_byte_identical_to_golden_model() -> None:
    """CRITICAL: for any v1.2 descriptor (all spaces DDR, xfer_bytes 0),
    `encode_desc` must produce byte-identical output to
    `cnn_accel_model.encode_instruction` on the equivalent `LayerDesc`."""
    d = _full_desc()
    d.space_src0 = isa.SPACE_DDR
    d.space_src1 = isa.SPACE_DDR
    d.space_dst = isa.SPACE_DDR
    d.space_wgt = isa.SPACE_DDR
    d.xfer_bytes = 0

    layer = _layer_desc_equivalent_of(d)
    assert isa.encode_desc(d) == model.encode_instruction(layer)


def test_v12_compatible_descriptor_encodes_byte_identical_random(seed: int = 999) -> None:
    rng = random.Random(seed)
    for _ in range(200):
        clamp_min = rng.randrange(-128, 128)
        clamp_max = rng.randrange(clamp_min, 128)
        d = isa.DescV2(
            opcode=rng.choice(list(isa.OPCODES.values())),
            flags=rng.randrange(0, 64),  # only v1.2 flag bits (0-5)
            space_src0=isa.SPACE_DDR,
            space_src1=isa.SPACE_DDR,
            space_dst=isa.SPACE_DDR,
            space_wgt=isa.SPACE_DDR,
            in_addr=rng.randrange(0, 2**32),
            out_addr=rng.randrange(0, 2**32),
            weight_addr=rng.randrange(0, 2**32),
            bias_addr=rng.randrange(0, 2**32),
            in_width=rng.randrange(0, 2**16),
            in_height=rng.randrange(0, 2**16),
            in_channels=rng.randrange(0, 2**16),
            out_channels=rng.randrange(0, 2**16),
            kernel_h=rng.randrange(0, 256),
            kernel_w=rng.randrange(0, 256),
            stride_h=rng.randrange(0, 256),
            stride_w=rng.randrange(0, 256),
            pad_top=rng.randrange(0, 256),
            pad_bottom=rng.randrange(0, 256),
            pad_left=rng.randrange(0, 256),
            pad_right=rng.randrange(0, 256),
            requant_scale=rng.randrange(-(2**31), 2**31),
            requant_shift=rng.randrange(0, 256),
            pool_kernel_h=rng.randrange(0, 256),
            pool_kernel_w=rng.randrange(0, 256),
            pool_stride_h=rng.randrange(0, 256),
            pool_stride_w=rng.randrange(0, 256),
            next_instr_addr=rng.randrange(0, 2**32),
            output_offset=rng.randrange(-(2**15), 2**15),
            clamp_min=clamp_min,
            clamp_max=clamp_max,
            scale_addr=rng.randrange(0, 2**32),
            xfer_bytes=0,
        )
        layer = _layer_desc_equivalent_of(d)
        assert isa.encode_desc(d) == model.encode_instruction(layer)


def test_decode_rejects_wrong_length() -> None:
    with pytest.raises(ValueError):
        isa.decode_desc(b"\x00" * 63)
