"""Self-check of `cnn_accel_model.py` -- the golden model must be
validated on its own before it is trusted as the expected-value source for
the VUnit testbenches (per `shared/Vunit.md`: a golden model is "the
single source of expected-behavior truth", so a wrong model would make
every downstream RTL test wrong in the same way and pass regardless of a
real DUT bug).

Layout of this file mirrors the task list: fixed-point helpers, ISA
encode/decode (against hard-coded VHDL offsets, NOT re-derived from the
model), per-opcode reference math, the five required-fix regression
tests, and finally a from-scratch naive convolution cross-check.
"""

from __future__ import annotations

import random
import struct

import pytest

import itertools

import cnn_accel_constants
from cnn_accel_model import (
    AccumulatorOverflow,
    INSTR_WORD_BYTES,
    LayerDesc,
    OPCODE_CONV2D,
    OPCODE_DWCONV2D,
    OPCODE_FC,
    OPCODE_HALT,
    OPCODE_POOL_AVG,
    OPCODE_POOL_MAX,
    OFF_BIAS_ADDR,
    OFF_CLAMP_MAX,
    OFF_CLAMP_MIN,
    OFF_FLAGS,
    OFF_IN_ADDR,
    OFF_IN_CHANNELS,
    OFF_IN_HEIGHT,
    OFF_IN_WIDTH,
    OFF_KERNEL_H,
    OFF_KERNEL_W,
    OFF_NEXT_INSTR_ADDR,
    OFF_OPCODE,
    OFF_OUT_ADDR,
    OFF_OUT_CHANNELS,
    OFF_OUTPUT_OFFSET,
    OFF_PAD_BOTTOM,
    OFF_PAD_LEFT,
    OFF_PAD_RIGHT,
    OFF_PAD_TOP,
    OFF_POOL_KERNEL_H,
    OFF_POOL_KERNEL_W,
    OFF_POOL_STRIDE_H,
    OFF_POOL_STRIDE_W,
    OFF_REQUANT_SCALE,
    OFF_REQUANT_SHIFT,
    OFF_SCALE_ADDR,
    OFF_STRIDE_H,
    OFF_STRIDE_W,
    OFF_WEIGHT_ADDR,
    FLAG_BIAS_EN,
    FLAG_CLAMP_EN,
    FLAG_PAD_EN,
    FLAG_PER_CHANNEL_EN,
    FLAG_RELU_EN,
    FLAG_REQUANT_EN,
    activation_bytes,
    activation_plane_count,
    bias_requantize_relu,
    build_memory_image,
    conv2d,
    decode_instruction,
    dwconv2d,
    encode_instruction,
    encode_program,
    fc,
    pack_activation_planes,
    lane_requant_params,
    pack_bias_for_hw,
    pack_scale_table_for_hw,
    pack_weights_for_hw,
    packed_bias_count,
    packed_scale_table_bytes,
    packed_weight_count,
    pool_avg,
    pool_max,
    round_shift_right_signed,
    run_layer,
    run_program,
    saturate_signed,
    unpack_activation_planes,
    unpack_scale_table_from_hw,
    unpack_weights_from_hw,
)


# ---------------------------------------------------------------------------
# round_shift_right_signed: round-half-up table (H0: ties towards +inf,
# i.e. floor((value + 2**(shift-1)) / 2**shift), TOSA apply_scale_32).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, shift, expected",
    [
        # shift=1, divisor=2: exact ties round towards +infinity.
        (0, 1, 0),  # 0/2 = 0.0 -> 0
        (1, 1, 1),  # 1/2 = 0.5 tie -> 1
        (3, 1, 2),  # 3/2 = 1.5 tie -> 2
        (-1, 1, 0),  # -1/2 = -0.5 tie -> 0
        (-3, 1, -1),  # -3/2 = -1.5 tie -> -1
        # shift=2, divisor=4: exact ties at remainder==2.
        (2, 2, 1),  # 2/4 = 0.5 tie -> 1
        (6, 2, 2),  # 6/4 = 1.5 tie -> 2
        (-2, 2, 0),  # -2/4 = -0.5 tie -> 0
        (-6, 2, -1),  # -6/4 = -1.5 tie -> -1
        # shift=3, divisor=8: exact ties at remainder==4.
        (4, 3, 1),  # 0.5 -> 1
        (12, 3, 2),  # 1.5 -> 2
        (20, 3, 3),  # 2.5 -> 3
        (-4, 3, 0),  # -0.5 -> 0
        (-12, 3, -1),  # -1.5 -> -1
        # Non-tie cases: round to nearest.
        (5, 2, 1),  # 5/4 = 1.25 -> 1
        (7, 2, 2),  # 7/4 = 1.75 -> 2
        (-5, 2, -1),  # -5/4 = -1.25 -> -1
        (-7, 2, -2),  # -7/4 = -1.75 -> -2
        # shift <= 0: plain left shift, no rounding.
        (5, 0, 5),
        (5, -1, 10),
        (-3, -2, -12),
    ],
)
def test_round_shift_right_signed_table(value: int, shift: int, expected: int) -> None:
    assert round_shift_right_signed(value, shift) == expected


def test_round_shift_right_signed_convergent_true_ties_to_even() -> None:
    # convergent=True: the pre-H0 round-to-even rule, kept for reference.
    assert round_shift_right_signed(1, 1, convergent=True) == 0
    assert round_shift_right_signed(3, 1, convergent=True) == 2
    assert round_shift_right_signed(-1, 1, convergent=True) == 0
    assert round_shift_right_signed(-3, 1, convergent=True) == -2


@pytest.mark.parametrize("shift", [1, 5, 15, 23, 40])
@pytest.mark.parametrize("value", [v * 977 - 60000 for v in range(0, 128, 7)] + [-1, 0, 1, 2**31 - 1, -(2**31)])
def test_round_shift_right_signed_matches_tosa_apply_scale_formula(value: int, shift: int) -> None:
    # TOSA apply_scale_32 (SINGLE_ROUND) = (value + (1 << (shift - 1))) >> shift with floor shift.
    assert round_shift_right_signed(value, shift) == (value + (1 << (shift - 1))) >> shift


# ---------------------------------------------------------------------------
# saturate_signed: boundary values.
# ---------------------------------------------------------------------------


def test_saturate_signed_int8_boundaries() -> None:
    assert saturate_signed(-128, 8) == -128
    assert saturate_signed(127, 8) == 127
    assert saturate_signed(-129, 8) == -128
    assert saturate_signed(128, 8) == 127
    assert saturate_signed(-1000, 8) == -128
    assert saturate_signed(1000, 8) == 127
    assert saturate_signed(0, 8) == 0


def test_saturate_signed_other_widths() -> None:
    assert saturate_signed(-32768, 16) == -32768
    assert saturate_signed(32767, 16) == 32767
    assert saturate_signed(-32769, 16) == -32768
    assert saturate_signed(32768, 16) == 32767


# ---------------------------------------------------------------------------
# bias_requantize_relu: all 8 (bias_en, requant_en, relu_en) combinations.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bias_en", [False, True])
@pytest.mark.parametrize("requant_en", [False, True])
@pytest.mark.parametrize("relu_en", [False, True])
def test_bias_requantize_relu_all_combinations(bias_en: bool, requant_en: bool, relu_en: bool) -> None:
    acc = -10
    bias = 5
    requant_scale = 1 << 15  # scale factor 1.0 in Q15
    requant_shift = 0
    result = bias_requantize_relu(
        acc,
        bias,
        bias_en=bias_en,
        requant_en=requant_en,
        relu_en=relu_en,
        requant_scale=requant_scale,
        requant_shift=requant_shift,
    )
    total = acc + (bias if bias_en else 0)
    if requant_en:
        expected = round_shift_right_signed(total * requant_scale, 15 + requant_shift)
    else:
        expected = total
    if relu_en:
        expected = max(expected, 0)
    expected = saturate_signed(expected, 8)
    assert result == expected
    assert -128 <= result <= 127


def test_bias_requantize_relu_int8_extremes_both_paths() -> None:
    # requant_en=True path: large positive accumulator saturates at +127.
    r = bias_requantize_relu(
        1_000_000, 0, bias_en=False, requant_en=True, relu_en=False,
        requant_scale=1 << 15, requant_shift=0,
    )
    assert r == 127
    # requant_en=True path: large negative saturates at -128.
    r = bias_requantize_relu(
        -1_000_000, 0, bias_en=False, requant_en=True, relu_en=False,
        requant_scale=1 << 15, requant_shift=0,
    )
    assert r == -128
    # requant_en=False (bypass) path must ALSO saturate, not wrap (fix #1).
    r = bias_requantize_relu(
        300, 0, bias_en=False, requant_en=False, relu_en=False,
        requant_scale=0, requant_shift=0,
    )
    assert r == 127
    r = bias_requantize_relu(
        -300, 0, bias_en=False, requant_en=False, relu_en=False,
        requant_scale=0, requant_shift=0,
    )
    assert r == -128


def test_bias_requantize_relu_relu_applied_before_saturate() -> None:
    # A large negative accumulator with ReLU enabled must clamp to 0, not
    # wrap/saturate to some negative int8 first. And a huge positive value
    # with ReLU enabled must still saturate at +127 (ReLU's upper bound is
    # unbounded -- the int8 saturate is what actually caps it).
    r = bias_requantize_relu(
        -50, 0, bias_en=False, requant_en=False, relu_en=True,
        requant_scale=0, requant_shift=0,
    )
    assert r == 0
    r = bias_requantize_relu(
        10_000_000, 0, bias_en=False, requant_en=True, relu_en=True,
        requant_scale=1 << 15, requant_shift=0,
    )
    assert r == 127
    # Ordering proof: bias added first, THEN relu, THEN saturate. A
    # negative acc with a bias that pushes it positive must survive ReLU.
    r = bias_requantize_relu(
        -5, 10, bias_en=True, requant_en=False, relu_en=True,
        requant_scale=0, requant_shift=0,
    )
    assert r == 5


def test_bias_requantize_relu_33bit_sum_boundary_positive() -> None:
    # D9: acc=INT32_MAX, bias=INT32_MAX. total = 2**32-2, which fits
    # exactly (and only just) inside the RTL's 33-bit c_sum_width -- a
    # 32-bit-wrapped add would instead see this as (2**32-2) - 2**32 = -2
    # (signed 32-bit two's complement), saturating to a *negative* int8.
    # The correct 33-bit-wide sum is unambiguously positive and saturates
    # at +127; asserting both the correct value and that it differs from
    # the wrong-width value pins the contract.
    acc = 2**31 - 1
    bias = 2**31 - 1
    wrong_32bit_wrapped_total = -2  # (acc + bias) reinterpreted as signed int32
    assert saturate_signed(wrong_32bit_wrapped_total, 8) == -2

    r_bypass = bias_requantize_relu(
        acc, bias, bias_en=True, requant_en=False, relu_en=False,
        requant_scale=0, requant_shift=0,
    )
    assert r_bypass == 127
    assert r_bypass != -2

    # requant_en=True with scale=1.0 (Q15 32768), shift=0: combined shift
    # is exactly 15, an exact division of total*32768 by 32768, so no
    # rounding is introduced and the result must match the bypass path.
    r_requant = bias_requantize_relu(
        acc, bias, bias_en=True, requant_en=True, relu_en=False,
        requant_scale=1 << 15, requant_shift=0,
    )
    assert r_requant == 127
    assert r_requant != -2


def test_bias_requantize_relu_33bit_sum_boundary_negative() -> None:
    # D9 mirror case: acc=INT32_MIN, bias=INT32_MIN. total = -2**32,
    # which a 32-bit-wrapped add would see as -2**32 mod 2**32 = 0
    # (signed int32), saturating to 0 -- a completely different (and
    # wrong) result from the correct 33-bit-wide sum, which is
    # unambiguously very negative and saturates at -128.
    acc = -(2**31)
    bias = -(2**31)
    wrong_32bit_wrapped_total = 0
    assert saturate_signed(wrong_32bit_wrapped_total, 8) == 0

    r_bypass = bias_requantize_relu(
        acc, bias, bias_en=True, requant_en=False, relu_en=False,
        requant_scale=0, requant_shift=0,
    )
    assert r_bypass == -128
    assert r_bypass != 0

    r_requant = bias_requantize_relu(
        acc, bias, bias_en=True, requant_en=True, relu_en=False,
        requant_scale=1 << 15, requant_shift=0,
    )
    assert r_requant == -128
    assert r_requant != 0


# ---------------------------------------------------------------------------
# encode_instruction/decode_instruction round-trip + hard-coded VHDL
# offsets (from modules/cnn_accel/src/cnn_accel_pkg.vhd lines 37-66,
# NOT re-derived from the model -- this is meant to catch drift).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# ISA v1.1 (H1): output_offset + general clamp, doc/tosa_compiler_plan.md
# section 5 extension 1. `s = round_shift(total*scale); s += output_offset;
# y = clamp(s, lo, hi)`.
# ---------------------------------------------------------------------------


def _legacy_bias_requantize_relu(acc, bias, *, bias_en, requant_en, relu_en, requant_scale, requant_shift):
    """Verbatim v1.0 epilogue (pre-H1 `bias_requantize_relu` body), kept
    here as the independent oracle for the `clamp_en=0` bit-identity
    contract."""
    total = acc + (bias if bias_en else 0)
    scaled = round_shift_right_signed(total * requant_scale, 15 + requant_shift) if requant_en else total
    if relu_en:
        scaled = max(scaled, 0)
    return saturate_signed(scaled, 8)


def test_bias_requantize_relu_clamp_en_zero_is_legacy() -> None:
    """With `output_offset=0, clamp_en=False` the H1 function must be
    bit-identical to the v1.0 epilogue for every flag combination,
    whatever `clamp_min`/`clamp_max` hold (they are ignored)."""
    rng = random.Random(0x481)
    for _ in range(2000):
        kwargs = dict(
            bias_en=rng.random() < 0.5,
            requant_en=rng.random() < 0.5,
            relu_en=rng.random() < 0.5,
            requant_scale=rng.randint(-(2**31), 2**31 - 1),
            requant_shift=rng.randint(0, 31),
        )
        acc = rng.choice([rng.randint(-(2**31), 2**31 - 1), rng.randint(-300, 300)])
        bias = rng.choice([rng.randint(-(2**31), 2**31 - 1), rng.randint(-300, 300)])
        expected = _legacy_bias_requantize_relu(acc, bias, **kwargs)
        assert bias_requantize_relu(acc, bias, **kwargs) == expected
        assert (
            bias_requantize_relu(
                acc, bias, **kwargs, output_offset=0, clamp_en=False,
                clamp_min=rng.randint(-128, 127), clamp_max=rng.randint(-128, 127),
            )
            == expected
        )


_IDENTITY = dict(bias_en=False, requant_en=True, relu_en=False, requant_scale=1 << 15, requant_shift=0)


def test_bias_requantize_relu_output_offset_added_after_shift() -> None:
    # Offset is exact (not scaled): scale 0.5, acc=10 -> s=5, +3 -> 8.
    r = bias_requantize_relu(10, 0, **{**_IDENTITY, "requant_scale": 1 << 14}, output_offset=3)
    assert r == 8
    # Offset is applied after the rounding: acc=1 at scale 0.5 is a tie
    # rounding to 1; the offset must not turn it into 1.5-ish territory.
    r = bias_requantize_relu(1, 0, **{**_IDENTITY, "requant_scale": 1 << 14}, output_offset=-1)
    assert r == 0
    # Offset also applies on the requant_en=0 bypass path.
    r = bias_requantize_relu(5, 0, **{**_IDENTITY, "requant_en": False}, output_offset=-7)
    assert r == -2
    # Offset before the legacy ReLU/saturate: -5 + 10 survives ReLU.
    r = bias_requantize_relu(-5, 0, **{**_IDENTITY, "relu_en": True}, output_offset=10)
    assert r == 5
    # ... and a positive s dragged negative by the offset is ReLU'd to 0.
    r = bias_requantize_relu(5, 0, **{**_IDENTITY, "relu_en": True}, output_offset=-10)
    assert r == 0


@pytest.mark.parametrize(
    ("acc", "offset", "expected"),
    [
        (100, 100, 127),        # s+offset above int8 -> hi
        (127, 1, 127),
        (0, 32767, 127),        # max offset
        (-100, -100, -128),     # s+offset below int8 -> lo
        (-128, -1, -128),
        (0, -32768, -128),      # min offset
        (200, -100, 100),       # s alone beyond int8, offset brings it back: NOT pre-saturated
        (-200, 100, -100),
        (1000, -1000, 0),
    ],
)
def test_bias_requantize_relu_offset_beyond_int8_both_sides(acc: int, offset: int, expected: int) -> None:
    """The offset add happens on the unbounded rounded value, before any
    clamp: `s` beyond int8 is not saturated first (200-100 = 100, not
    127-100 = 27)."""
    assert bias_requantize_relu(acc, 0, **_IDENTITY, output_offset=offset) == expected


def test_bias_requantize_relu_general_clamp() -> None:
    clamp = dict(clamp_en=True, clamp_min=5, clamp_max=100)
    assert bias_requantize_relu(50, 0, **_IDENTITY, **clamp) == 50
    assert bias_requantize_relu(3, 0, **_IDENTITY, **clamp) == 5
    assert bias_requantize_relu(-1000, 0, **_IDENTITY, **clamp) == 5
    assert bias_requantize_relu(101, 0, **_IDENTITY, **clamp) == 100
    assert bias_requantize_relu(10_000_000, 0, **_IDENTITY, **clamp) == 100
    # Negative bounds, both sides.
    clamp = dict(clamp_en=True, clamp_min=-100, clamp_max=-10)
    assert bias_requantize_relu(0, 0, **_IDENTITY, **clamp) == -10
    assert bias_requantize_relu(-128, 0, **_IDENTITY, **clamp) == -100
    # CLAMP_EN overrides RELU_EN: relu would give 0, clamp gives lo=-20.
    assert bias_requantize_relu(-50, 0, **{**_IDENTITY, "relu_en": True}, clamp_en=True, clamp_min=-20, clamp_max=20) == -20
    # Offset then clamp: 90 + 20 = 110 -> hi=100.
    assert bias_requantize_relu(90, 0, **_IDENTITY, output_offset=20, clamp_en=True, clamp_min=5, clamp_max=100) == 100
    # Bypass path honours the clamp too.
    assert bias_requantize_relu(3, 0, **{**_IDENTITY, "requant_en": False}, clamp_en=True, clamp_min=5, clamp_max=100) == 5


def test_bias_requantize_relu_clamp_lo_equals_hi() -> None:
    for value in (-1000, -1, 0, 1, 42, 1000):
        assert bias_requantize_relu(value, 0, **_IDENTITY, clamp_en=True, clamp_min=42, clamp_max=42) == 42
        assert bias_requantize_relu(value, 0, **_IDENTITY, clamp_en=True, clamp_min=-128, clamp_max=-128) == -128


def test_encode_rejects_clamp_min_gt_max_and_out_of_range_w13() -> None:
    base = dict(opcode=OPCODE_CONV2D, flags=1 << FLAG_CLAMP_EN)
    with pytest.raises(ValueError, match="clamp_min"):
        encode_instruction(LayerDesc(**base, clamp_min=10, clamp_max=9))
    # lo > hi is rejected even when CLAMP_EN is clear (the bytes still land in W13).
    with pytest.raises(ValueError, match="clamp_min"):
        encode_instruction(LayerDesc(opcode=OPCODE_CONV2D, clamp_min=1, clamp_max=0))
    # lo == hi is legal.
    encode_instruction(LayerDesc(**base, clamp_min=9, clamp_max=9))
    for bad in (dict(output_offset=32768), dict(output_offset=-32769), dict(clamp_max=128), dict(clamp_min=-129, clamp_max=0)):
        with pytest.raises(ValueError, match="outside signed range"):
            encode_instruction(LayerDesc(**base, **bad))


def test_layer_desc_clamp_en_property_and_conv2d_uses_w13() -> None:
    """`conv2d` (and hence `run_layer`) must actually read the new fields:
    a 1x1 conv with weight 1, scale 1.0, offset -128, clamp_en and
    clamp [-128, 0] maps x -> clamp(x - 128, -128, 0)."""
    desc = LayerDesc(
        opcode=OPCODE_CONV2D,
        flags=(1 << FLAG_REQUANT_EN) | (1 << FLAG_CLAMP_EN) | (1 << FLAG_RELU_EN),
        in_width=4, in_height=1, in_channels=1, out_channels=1,
        requant_scale=1 << 15, requant_shift=0,
        output_offset=-128, clamp_min=-128, clamp_max=0,
    )
    assert desc.clamp_en
    assert conv2d([-128, -1, 0, 127], [1], [0], desc) == [-128, -128, -128, -1]
    # Same descriptor with CLAMP_EN clear: RELU_EN then rules, offset still applies.
    desc.flags &= ~(1 << FLAG_CLAMP_EN)
    assert not desc.clamp_en
    assert conv2d([-128, -1, 0, 127], [1], [0], desc) == [0, 0, 0, 0]


# ---------------------------------------------------------------------------
# ISA v1.2 (H2): per-channel requantization.
# ---------------------------------------------------------------------------


def test_lane_requant_params_per_channel_en_zero_is_broadcast() -> None:
    """With PER_CHANNEL_EN clear the lane vector is the descriptor's
    (requant_scale, requant_shift) broadcast to every channel and any
    table is ignored -- the RTL's legacy `cfg_requant_*` path."""
    desc = LayerDesc(opcode=OPCODE_CONV2D, flags=1 << FLAG_REQUANT_EN, out_channels=3,
                     requant_scale=1234, requant_shift=5)
    assert not desc.per_channel_en
    assert lane_requant_params(desc, None) == [(1234, 5)] * 3
    assert lane_requant_params(desc, [(1, 1)] * 3) == [(1234, 5)] * 3


def test_lane_requant_params_per_channel_en_uses_table_and_requires_it() -> None:
    desc = LayerDesc(opcode=OPCODE_CONV2D, flags=(1 << FLAG_REQUANT_EN) | (1 << FLAG_PER_CHANNEL_EN),
                     out_channels=3, requant_scale=1234, requant_shift=5)
    assert desc.per_channel_en
    table = [(10, 1), (20, 2), (30, 3), (40, 4)]  # padded entries beyond out_c are dropped
    assert lane_requant_params(desc, table) == table[:3]
    with pytest.raises(ValueError):
        lane_requant_params(desc, None)
    with pytest.raises(ValueError):
        lane_requant_params(desc, table[:2])


def test_conv2d_per_channel_lanes_apply_distinct_scale_and_shift() -> None:
    """A 1x1 conv, 4 output channels each with weight 1 on a single input
    channel: lane `oc` must be requantised with ITS OWN (multiplier,
    shift), i.e. `bias_requantize_relu(x, ..., scale[oc], shift[oc])`."""
    desc = LayerDesc(
        opcode=OPCODE_CONV2D,
        flags=(1 << FLAG_REQUANT_EN) | (1 << FLAG_PER_CHANNEL_EN),
        in_width=3, in_height=1, in_channels=1, out_channels=4,
        requant_scale=1 << 15, requant_shift=0,  # must be ignored
    )
    table = [(1 << 15, 0), (1 << 14, 0), (1 << 15, 1), (-(1 << 15), 0)]
    xs = [100, -50, 7]
    out = conv2d(xs, [1, 1, 1, 1], [0, 0, 0, 0], desc, table)
    expected = []
    for x in xs:
        for mult, shift in table:
            expected.append(
                bias_requantize_relu(x, 0, bias_en=False, requant_en=True, relu_en=False,
                                     requant_scale=mult, requant_shift=shift)
            )
    assert out == expected
    # Sanity on the actual numbers: x, x/2, x/2, -x.
    assert out[:4] == [100, 50, 50, -100]


def test_conv2d_per_channel_en_zero_is_legacy_bit_identical() -> None:
    """Same random layer, once with the flag clear (table present but
    ignored) and once via the old call signature: byte-identical."""
    rng = random.Random(0x4A2)
    in_w, in_h, in_c, out_c, k = 4, 3, 3, 5, 3
    desc = LayerDesc(
        opcode=OPCODE_CONV2D,
        flags=(1 << FLAG_REQUANT_EN) | (1 << FLAG_BIAS_EN) | (1 << FLAG_RELU_EN) | (1 << FLAG_PAD_EN),
        in_width=in_w, in_height=in_h, in_channels=in_c, out_channels=out_c,
        kernel_h=k, kernel_w=k, pad_top=1, pad_bottom=1, pad_left=1, pad_right=1,
        requant_scale=rng.randint(1, 2**31 - 1), requant_shift=rng.randint(0, 12),
    )
    xs = [rng.randint(-128, 127) for _ in range(in_w * in_h * in_c)]
    ws = [rng.randint(-128, 127) for _ in range(out_c * k * k * in_c)]
    bs = [rng.randint(-5000, 5000) for _ in range(out_c)]
    stale_table = [(rng.randint(-(2**31), 2**31 - 1), rng.randint(0, 255)) for _ in range(out_c)]
    assert conv2d(xs, ws, bs, desc, stale_table) == conv2d(xs, ws, bs, desc)


def test_instruction_offsets_match_vhdl_pkg_literals() -> None:
    expected = {
        "OFF_OPCODE": 0,
        "OFF_FLAGS": 1,
        "OFF_IN_ADDR": 4,
        "OFF_OUT_ADDR": 8,
        "OFF_WEIGHT_ADDR": 12,
        "OFF_BIAS_ADDR": 16,
        "OFF_IN_WIDTH": 20,
        "OFF_IN_HEIGHT": 22,
        "OFF_IN_CHANNELS": 24,
        "OFF_OUT_CHANNELS": 26,
        "OFF_KERNEL_H": 28,
        "OFF_KERNEL_W": 29,
        "OFF_STRIDE_H": 30,
        "OFF_STRIDE_W": 31,
        "OFF_PAD_TOP": 32,
        "OFF_PAD_BOTTOM": 33,
        "OFF_PAD_LEFT": 34,
        "OFF_PAD_RIGHT": 35,
        "OFF_REQUANT_SCALE": 36,
        "OFF_REQUANT_SHIFT": 40,
        "OFF_POOL_KERNEL_H": 44,
        "OFF_POOL_KERNEL_W": 45,
        "OFF_POOL_STRIDE_H": 46,
        "OFF_POOL_STRIDE_W": 47,
        "OFF_NEXT_INSTR_ADDR": 48,
        "OFF_OUTPUT_OFFSET": 52,
        "OFF_CLAMP_MIN": 54,
        "OFF_CLAMP_MAX": 55,
        "OFF_SCALE_ADDR": 56,
    }
    actual = {
        "OFF_OPCODE": OFF_OPCODE,
        "OFF_FLAGS": OFF_FLAGS,
        "OFF_IN_ADDR": OFF_IN_ADDR,
        "OFF_OUT_ADDR": OFF_OUT_ADDR,
        "OFF_WEIGHT_ADDR": OFF_WEIGHT_ADDR,
        "OFF_BIAS_ADDR": OFF_BIAS_ADDR,
        "OFF_IN_WIDTH": OFF_IN_WIDTH,
        "OFF_IN_HEIGHT": OFF_IN_HEIGHT,
        "OFF_IN_CHANNELS": OFF_IN_CHANNELS,
        "OFF_OUT_CHANNELS": OFF_OUT_CHANNELS,
        "OFF_KERNEL_H": OFF_KERNEL_H,
        "OFF_KERNEL_W": OFF_KERNEL_W,
        "OFF_STRIDE_H": OFF_STRIDE_H,
        "OFF_STRIDE_W": OFF_STRIDE_W,
        "OFF_PAD_TOP": OFF_PAD_TOP,
        "OFF_PAD_BOTTOM": OFF_PAD_BOTTOM,
        "OFF_PAD_LEFT": OFF_PAD_LEFT,
        "OFF_PAD_RIGHT": OFF_PAD_RIGHT,
        "OFF_REQUANT_SCALE": OFF_REQUANT_SCALE,
        "OFF_REQUANT_SHIFT": OFF_REQUANT_SHIFT,
        "OFF_POOL_KERNEL_H": OFF_POOL_KERNEL_H,
        "OFF_POOL_KERNEL_W": OFF_POOL_KERNEL_W,
        "OFF_POOL_STRIDE_H": OFF_POOL_STRIDE_H,
        "OFF_POOL_STRIDE_W": OFF_POOL_STRIDE_W,
        "OFF_NEXT_INSTR_ADDR": OFF_NEXT_INSTR_ADDR,
        "OFF_OUTPUT_OFFSET": OFF_OUTPUT_OFFSET,
        "OFF_CLAMP_MIN": OFF_CLAMP_MIN,
        "OFF_CLAMP_MAX": OFF_CLAMP_MAX,
        "OFF_SCALE_ADDR": OFF_SCALE_ADDR,
    }
    assert actual == expected
    assert INSTR_WORD_BYTES == 64
    assert FLAG_CLAMP_EN == 4
    assert FLAG_PER_CHANNEL_EN == 5
    assert cnn_accel_constants.SCALE_TABLE_ENTRY_BYTES == 8
    assert cnn_accel_constants.SCALE_BUFFER_ENTRY_BITS == 40


def test_isa_layout_self_consistent() -> None:
    """Structural self-check of `cnn_accel_constants.ISA_LAYOUT` itself
    (as opposed to `test_instruction_offsets_match_vhdl_pkg_literals`
    above, which cross-checks the model's *exported* `OFF_*` names against
    independently hand-typed literals): no two fields overlap, no field
    (named or reserved) crosses the 64-byte word boundary, and the
    documented reserved gaps (W0 byte 3 and W10 bytes 41-43 --
    doc/cnn_accel_arch.md's ISA table; W13 became the ISA v1.1
    output_offset/clamp_min/clamp_max fields in H1, W14 the ISA v1.2
    scale_addr in H2, and in ISA v2.0 W0 byte 2 became the `spaces`
    tag byte and W15 bytes 60-63 became `xfer_bytes` --
    doc/cnn_accel_top_v2_arch.md section 5) land exactly where specified. This is the guarantee that replaces the old hand-maintained
    "cnn_accel_model.py's encoder must agree with these byte-for-byte"
    comment with something a test actually enforces."""
    occupied = bytearray(cnn_accel_constants.INSTR_WORD_BYTES)

    offset = 0
    for field in cnn_accel_constants.ISA_LAYOUT:
        start, end = offset, offset + field.width_bytes  # end exclusive
        assert end <= cnn_accel_constants.INSTR_WORD_BYTES, (
            f"field '{field.name}' at bytes [{start}, {end}) crosses the "
            f"{cnn_accel_constants.INSTR_WORD_BYTES}-byte word boundary"
        )
        for byte in range(start, end):
            assert occupied[byte] == 0, (
                f"field '{field.name}' overlaps another field at byte {byte}"
            )
            occupied[byte] = 1
        offset = end

    assert offset == cnn_accel_constants.INSTR_WORD_BYTES
    assert all(occupied), "instruction word has unaccounted-for byte(s)"

    assert cnn_accel_constants.isa_reserved_ranges() == [(3, 3), (42, 43)]

    # Every non-reserved field name is unique and does not collide with the
    # `RESERVED` sentinel.
    names = [f.name for f in cnn_accel_constants.ISA_LAYOUT if f.name != cnn_accel_constants.RESERVED]
    assert len(names) == len(set(names))


def test_encode_decode_round_trip() -> None:
    desc = LayerDesc(
        opcode=OPCODE_CONV2D,
        flags=(1 << FLAG_RELU_EN) | (1 << FLAG_BIAS_EN) | (1 << FLAG_REQUANT_EN) | (1 << FLAG_PAD_EN),
        in_addr=0x1000,
        out_addr=0x2000,
        weight_addr=0x3000,
        bias_addr=0x4000,
        in_width=8,
        in_height=8,
        in_channels=3,
        out_channels=4,
        kernel_h=3,
        kernel_w=3,
        stride_h=1,
        stride_w=1,
        pad_top=1,
        pad_bottom=1,
        pad_left=1,
        pad_right=1,
        requant_scale=-12345,
        requant_shift=7,
        pool_kernel_h=2,
        pool_kernel_w=2,
        pool_stride_h=2,
        pool_stride_w=2,
        next_instr_addr=0x100,
        output_offset=-129,
        clamp_min=-100,
        clamp_max=100,
        scale_addr=0xDEADBEE8,
    )
    encoded = encode_instruction(desc)
    assert len(encoded) == INSTR_WORD_BYTES
    assert decode_instruction(encoded) == desc
    # W13 byte placement (little-endian, signed).
    assert encoded[52:54] == (-129).to_bytes(2, "little", signed=True)
    assert encoded[54] == (-100) & 0xFF
    assert encoded[55] == 100
    # W14 scale_addr (ISA v1.2), little-endian unsigned.
    assert encoded[56:60] == (0xDEADBEE8).to_bytes(4, "little")
    # Reserved W15 untouched.
    assert encoded[60:64] == bytes(4)


def test_encode_v11_style_desc_leaves_w14_zero() -> None:
    """A descriptor that never mentions `scale_addr` must encode W14 as 0
    -- the reserved-must-be-0 guarantee that makes every pre-H2 program
    byte-identical under the v1.2 encoder."""
    encoded = encode_instruction(LayerDesc(opcode=OPCODE_CONV2D, flags=1 << FLAG_REQUANT_EN))
    assert encoded[56:64] == bytes(8)
    assert not decode_instruction(encoded).per_channel_en


def test_encode_decode_round_trip_randomized() -> None:
    rng = random.Random(12345)
    for _ in range(50):
        desc = LayerDesc(
            opcode=rng.choice([OPCODE_HALT, OPCODE_CONV2D, OPCODE_DWCONV2D, OPCODE_POOL_MAX, OPCODE_POOL_AVG, OPCODE_FC]),
            flags=rng.randint(0, 0xFF),
            in_addr=rng.randint(0, 0xFFFFFFFF),
            out_addr=rng.randint(0, 0xFFFFFFFF),
            weight_addr=rng.randint(0, 0xFFFFFFFF),
            bias_addr=rng.randint(0, 0xFFFFFFFF),
            in_width=rng.randint(0, 0xFFFF),
            in_height=rng.randint(0, 0xFFFF),
            in_channels=rng.randint(0, 0xFFFF),
            out_channels=rng.randint(0, 0xFFFF),
            kernel_h=rng.randint(0, 0xFF),
            kernel_w=rng.randint(0, 0xFF),
            stride_h=rng.randint(0, 0xFF),
            stride_w=rng.randint(0, 0xFF),
            pad_top=rng.randint(0, 0xFF),
            pad_bottom=rng.randint(0, 0xFF),
            pad_left=rng.randint(0, 0xFF),
            pad_right=rng.randint(0, 0xFF),
            requant_scale=rng.randint(-(2**31), 2**31 - 1),
            requant_shift=rng.randint(0, 0xFF),
            pool_kernel_h=rng.randint(0, 0xFF),
            pool_kernel_w=rng.randint(0, 0xFF),
            pool_stride_h=rng.randint(0, 0xFF),
            pool_stride_w=rng.randint(0, 0xFF),
            next_instr_addr=rng.randint(0, 0xFFFFFFFF),
            output_offset=rng.randint(-(2**15), 2**15 - 1),
            clamp_min=(clamp_pair := sorted((rng.randint(-128, 127), rng.randint(-128, 127))))[0],
            clamp_max=clamp_pair[1],
            scale_addr=rng.randint(0, 0xFFFFFFFF),
        )
        assert decode_instruction(encode_instruction(desc)) == desc


# ---------------------------------------------------------------------------
# Helpers shared by conv2d/dwconv2d tests.
# ---------------------------------------------------------------------------


def _desc(**kwargs) -> LayerDesc:
    base = dict(
        opcode=OPCODE_CONV2D,
        flags=0,
        in_width=1,
        in_height=1,
        in_channels=1,
        out_channels=1,
        kernel_h=1,
        kernel_w=1,
        stride_h=1,
        stride_w=1,
    )
    base.update(kwargs)
    return LayerDesc(**base)


def _hwc(values, w, h, c):
    assert len(values) == w * h * c
    return list(values)


# ---------------------------------------------------------------------------
# conv2d
# ---------------------------------------------------------------------------


def test_conv2d_1x1_identity() -> None:
    desc = _desc(in_width=2, in_height=2, in_channels=1, out_channels=1, kernel_h=1, kernel_w=1)
    inp = [1, -2, 3, -4]
    weights = [2]
    bias = [0]
    out = conv2d(inp, weights, bias, desc)
    assert out == [2, -4, 6, -8]


def test_conv2d_3x3_stride1_no_pad() -> None:
    desc = _desc(in_width=3, in_height=3, in_channels=1, out_channels=1, kernel_h=3, kernel_w=3)
    inp = list(range(1, 10))  # 1..9
    weights = [1] * 9  # sum filter
    bias = [0]
    out = conv2d(inp, weights, bias, desc)
    assert out == [sum(range(1, 10))]  # single output position, sum of all 9


def test_conv2d_3x3_stride2_with_pad() -> None:
    desc = _desc(
        in_width=5, in_height=5, in_channels=1, out_channels=1, kernel_h=3, kernel_w=3,
        stride_h=2, stride_w=2, flags=(1 << FLAG_PAD_EN),
        pad_top=1, pad_bottom=1, pad_left=1, pad_right=1,
    )
    inp = [1] * 25
    weights = [1] * 9
    bias = [0]
    out = conv2d(inp, weights, bias, desc)
    # padded 7x7, kernel3 stride2 -> out_h=out_w=3
    assert len(out) == 9
    # Corner window has only a 2x2 in-bounds overlap (4 taps of value 1).
    assert out[0] == 4


def test_conv2d_padding_on_vs_off_changes_output() -> None:
    desc_no_pad = _desc(in_width=4, in_height=4, in_channels=1, out_channels=1, kernel_h=3, kernel_w=3)
    desc_pad = _desc(
        in_width=4, in_height=4, in_channels=1, out_channels=1, kernel_h=3, kernel_w=3,
        flags=(1 << FLAG_PAD_EN), pad_top=1, pad_bottom=1, pad_left=1, pad_right=1,
    )
    inp = list(range(16))
    weights = [1] * 9
    bias = [0]
    out_no_pad = conv2d(inp, weights, bias, desc_no_pad)
    out_pad = conv2d(inp, weights, bias, desc_pad)
    assert len(out_no_pad) == 4  # out 2x2
    assert len(out_pad) == 16  # out 4x4
    assert out_no_pad != out_pad[: len(out_no_pad)]


def test_conv2d_odd_input_dims() -> None:
    desc = _desc(in_width=5, in_height=3, in_channels=1, out_channels=1, kernel_h=3, kernel_w=3)
    inp = list(range(15))
    weights = [1] * 9
    bias = [0]
    out = conv2d(inp, weights, bias, desc)
    assert len(out) == 3  # out_h=1, out_w=3


def test_conv2d_multi_in_channel() -> None:
    desc = _desc(in_width=1, in_height=1, in_channels=3, out_channels=1, kernel_h=1, kernel_w=1)
    inp = [1, 2, 3]
    weights = [10, 20, 30]  # OHWI: oc=0
    bias = [0]
    out = conv2d(inp, weights, bias, desc)
    assert out == [saturate_signed(1 * 10 + 2 * 20 + 3 * 30, 8)]


def test_conv2d_multi_out_channel() -> None:
    desc = _desc(in_width=1, in_height=1, in_channels=1, out_channels=2, kernel_h=1, kernel_w=1)
    inp = [3]
    weights = [2, -2]  # oc0, oc1
    bias = [0, 0]
    out = conv2d(inp, weights, bias, desc)
    assert out == [6, -6]


def test_conv2d_int8_extremes_all_min() -> None:
    desc = _desc(in_width=1, in_height=1, in_channels=4, out_channels=1, kernel_h=1, kernel_w=1)
    inp = [-128] * 4
    weights = [-128] * 4
    bias = [0]
    out = conv2d(inp, weights, bias, desc)
    # acc = 4 * (-128*-128) = 65536, saturates to +127
    assert out == [127]


def test_conv2d_int8_extremes_all_max() -> None:
    desc = _desc(in_width=1, in_height=1, in_channels=4, out_channels=1, kernel_h=1, kernel_w=1)
    inp = [127] * 4
    weights = [127] * 4
    bias = [0]
    out = conv2d(inp, weights, bias, desc)
    assert out == [127]


def test_conv2d_negative_weights_and_bias() -> None:
    desc = _desc(
        in_width=1, in_height=1, in_channels=1, out_channels=1, kernel_h=1, kernel_w=1,
        flags=(1 << FLAG_BIAS_EN),
    )
    inp = [10]
    weights = [-3]
    bias = [-5]
    out = conv2d(inp, weights, bias, desc)
    assert out == [10 * -3 + -5]  # -35, within int8 range


# ---------------------------------------------------------------------------
# dwconv2d: channel independence.
# ---------------------------------------------------------------------------


def test_dwconv2d_channel_independence() -> None:
    channels = 4
    desc = LayerDesc(
        opcode=OPCODE_DWCONV2D, in_width=3, in_height=3, in_channels=channels,
        out_channels=channels, kernel_h=3, kernel_w=3,
    )
    rng = random.Random(7)
    inp = [rng.randint(-128, 127) for _ in range(3 * 3 * channels)]
    weights = [rng.randint(-128, 127) for _ in range(channels * 3 * 3)]
    bias = [0] * channels
    baseline = dwconv2d(inp, weights, bias, desc)

    for k in range(channels):
        perturbed = list(inp)
        # perturb every occurrence of channel k
        for idx in range(k, len(perturbed), channels):
            perturbed[idx] = saturate_signed(perturbed[idx] + 1, 8) if perturbed[idx] < 127 else perturbed[idx] - 1
        out = dwconv2d(perturbed, weights, bias, desc)
        for ch in range(channels):
            if ch == k:
                continue
            assert out[ch] == baseline[ch], f"perturbing channel {k} changed output channel {ch}"


def test_dwconv2d_requires_matching_channels() -> None:
    desc = LayerDesc(opcode=OPCODE_DWCONV2D, in_width=1, in_height=1, in_channels=2, out_channels=3, kernel_h=1, kernel_w=1)
    with pytest.raises(ValueError):
        dwconv2d([1, 2], [1, 1], [0, 0, 0], desc)


# ---------------------------------------------------------------------------
# fc: degenerate case works, non-degenerate raises via run_layer.
# ---------------------------------------------------------------------------


def test_fc_degenerate_case_works() -> None:
    desc = LayerDesc(
        opcode=OPCODE_FC, in_width=1, in_height=1, in_channels=3, out_channels=2, kernel_h=1, kernel_w=1,
    )
    inp = [1, 2, 3]
    weights = [1, 1, 1, 2, 2, 2]  # OHWI, oc=0 then oc=1
    bias = [0, 0]
    out = fc(inp, weights, bias, desc)
    assert out == [6, 12]


def test_fc_non_degenerate_raises_via_run_layer() -> None:
    desc = LayerDesc(
        opcode=OPCODE_FC, in_addr=0, out_addr=100, weight_addr=200,
        in_width=2, in_height=1, in_channels=1, out_channels=1, kernel_h=1, kernel_w=1,
    )
    mem = bytearray(1000)
    with pytest.raises(ValueError):
        run_layer(mem, desc)


# ---------------------------------------------------------------------------
# pool_max / pool_avg
# ---------------------------------------------------------------------------


def test_pool_max_basic() -> None:
    desc = LayerDesc(
        opcode=OPCODE_POOL_MAX, in_width=4, in_height=4, in_channels=1,
        pool_kernel_h=2, pool_kernel_w=2, pool_stride_h=2, pool_stride_w=2,
    )
    inp = [
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16,
    ]
    out = pool_max(inp, desc)
    assert out == [6, 8, 14, 16]


def test_pool_avg_rounding_behaviour() -> None:
    # 2x2 average pool, sum=10 over 4 taps -> true average 2.5.
    # requant_scale chosen as Q15 scale factor 1/4 = 8192; requant_shift=0
    # so combined shift is 15; round-half-up applies to sum*scale.
    desc = LayerDesc(
        opcode=OPCODE_POOL_AVG, flags=(1 << FLAG_REQUANT_EN),
        in_width=2, in_height=2, in_channels=1,
        pool_kernel_h=2, pool_kernel_w=2, pool_stride_h=2, pool_stride_w=2,
        requant_scale=8192, requant_shift=0,
    )
    inp = [1, 2, 3, 4]  # sum = 10, true avg = 2.5
    out = pool_avg(inp, desc)
    scaled = round_shift_right_signed(10 * 8192, 15)
    assert out == [scaled]
    # 2.5 * 8192 = 20480, /32768 = 0.625 -> rounds to 1 (not a tie at this
    # scale); this test only proves the sum->scale->round chain, exact
    # value pinned via the same round_shift_right_signed call above.


def test_pool_avg_no_bias_applied() -> None:
    desc = LayerDesc(
        opcode=OPCODE_POOL_AVG, flags=0,
        in_width=2, in_height=2, in_channels=1,
        pool_kernel_h=2, pool_kernel_w=2, pool_stride_h=2, pool_stride_w=2,
        requant_scale=0, requant_shift=0,
    )
    inp = [1, 2, 3, 4]
    out = pool_avg(inp, desc)
    assert out == [10]  # requant_en=False bypass: sum passes through (saturated)


# ---------------------------------------------------------------------------
# Fix-specific regression tests -- each would FAIL against the
# pre-fix behaviour described in the task.
# ---------------------------------------------------------------------------


def test_fix1_bypass_path_saturates_not_wraps() -> None:
    # Pre-fix: requant_en=0 branch wrapped two's-complement to 8 bits, so
    # acc=300 -> (300 & 0xFF) = 44. Post-fix: saturates to +127.
    result = bias_requantize_relu(
        300, 0, bias_en=False, requant_en=False, relu_en=False,
        requant_scale=0, requant_shift=0,
    )
    assert result == 127
    assert result != 44  # the old (wrong) wrapped value


def test_fix2_accumulator_overflow_raises_with_context() -> None:
    # 1x1 conv, in_channels large enough that -128*-128 summed over all
    # channels exceeds the int32 accumulator range.
    in_c = 140_000
    desc = _desc(in_width=1, in_height=1, in_channels=in_c, out_channels=1, kernel_h=1, kernel_w=1)
    inp = [-128] * in_c
    weights = [-128] * in_c
    bias = [0]
    with pytest.raises(AccumulatorOverflow) as excinfo:
        conv2d(inp, weights, bias, desc)
    msg = str(excinfo.value)
    assert "out_channel=0" in msg
    assert "x=0" in msg and "y=0" in msg
    assert "0x01" in msg  # OPCODE_CONV2D


def test_fix2_accumulator_within_range_does_not_raise() -> None:
    desc = _desc(in_width=1, in_height=1, in_channels=4, out_channels=1, kernel_h=1, kernel_w=1)
    conv2d([-128] * 4, [-128] * 4, [0], desc)  # must not raise


def test_fix3_fc_run_layer_rejects_non_degenerate_original_desc() -> None:
    desc = LayerDesc(
        opcode=OPCODE_FC, in_addr=0, out_addr=64, weight_addr=128,
        in_width=3, in_height=2, in_channels=1, out_channels=1, kernel_h=2, kernel_w=3,
    )
    mem = bytearray(1000)
    with pytest.raises(ValueError):
        run_layer(mem, desc)


def test_fix3_fc_run_layer_accepts_degenerate_desc() -> None:
    desc = LayerDesc(
        opcode=OPCODE_FC, in_addr=0, out_addr=10, weight_addr=20, bias_addr=0,
        in_width=1, in_height=1, in_channels=2, out_channels=1, kernel_h=1, kernel_w=1,
    )
    mem = bytearray(100)
    mem[0:2] = bytes([1, 2])
    mem[20:22] = bytes([3, 4])
    run_layer(mem, desc)  # must not raise
    assert mem[10] == (1 * 3 + 2 * 4) & 0xFF


@pytest.mark.parametrize("stride_field", ["stride_h", "stride_w"])
def test_fix4_zero_conv_stride_raises(stride_field: str) -> None:
    desc = _desc(in_width=3, in_height=3, in_channels=1, out_channels=1, kernel_h=1, kernel_w=1)
    setattr(desc, stride_field, 0)
    with pytest.raises(ValueError):
        conv2d([1] * 9, [1], [0], desc)


@pytest.mark.parametrize("stride_field", ["pool_stride_h", "pool_stride_w"])
def test_fix4_zero_pool_stride_raises(stride_field: str) -> None:
    desc = LayerDesc(
        opcode=OPCODE_POOL_MAX, in_width=3, in_height=3, in_channels=1,
        pool_kernel_h=1, pool_kernel_w=1, pool_stride_h=1, pool_stride_w=1,
    )
    setattr(desc, stride_field, 0)
    with pytest.raises(ValueError):
        pool_max([1] * 9, desc)


def test_fix4_nonpositive_output_dims_raises() -> None:
    # kernel bigger than the (unpadded) input -> negative computed out dims.
    desc = _desc(in_width=2, in_height=2, in_channels=1, out_channels=1, kernel_h=3, kernel_w=3)
    with pytest.raises(ValueError):
        conv2d([1] * 4, [1] * 9, [0], desc)


def _naive_pool_max(
    values: list[int],
    in_h: int,
    in_w: int,
    channels: int,
    kernel: tuple[int, int],
    stride: tuple[int, int],
    pad: tuple[int, int, int, int],
    pad_value: int,
) -> list[int]:
    """From-scratch max pooling, written without touching any
    `cnn_accel_model` internals, so it is an independent check of
    `pool_max` rather than a restatement of it."""
    k_h, k_w = kernel
    s_h, s_w = stride
    pad_t, pad_b, pad_l, pad_r = pad
    out_h = (in_h + pad_t + pad_b - k_h) // s_h + 1
    out_w = (in_w + pad_l + pad_r - k_w) // s_w + 1
    out: list[int] = []
    for oy in range(out_h):
        for ox in range(out_w):
            for ch in range(channels):
                best = -129
                for ky in range(k_h):
                    for kx in range(k_w):
                        y = oy * s_h - pad_t + ky
                        x = ox * s_w - pad_l + kx
                        if 0 <= y < in_h and 0 <= x < in_w:
                            tap = values[(y * in_w + x) * channels + ch]
                        else:
                            tap = pad_value
                        best = max(best, tap)
                out.append(best)
    return out


def test_pool_padding_uses_pad_value() -> None:
    """ISA v2.1: pooling honours PAD_EN, and a padded tap takes
    `pad_value` (the tensor's zero-point), not 0. The exact YOLOv8n SPPF
    shape: 5x5, stride 1, padding 2."""
    random.seed(1234)
    in_h = in_w = 6
    channels = 2
    values = [random.randint(-128, 127) for _ in range(in_h * in_w * channels)]

    def desc_for(pad_value: int) -> LayerDesc:
        return LayerDesc(
            opcode=OPCODE_POOL_MAX,
            flags=(1 << FLAG_PAD_EN),
            in_width=in_w, in_height=in_h, in_channels=channels,
            pool_kernel_h=5, pool_kernel_w=5, pool_stride_h=1, pool_stride_w=1,
            pad_top=2, pad_bottom=2, pad_left=2, pad_right=2,
            pad_value=pad_value,
        )

    for pad_value in (-128, 0, 40):
        got = pool_max(values, desc_for(pad_value))
        expected = _naive_pool_max(
            values, in_h, in_w, channels, (5, 5), (1, 1), (2, 2, 2, 2), pad_value
        )
        assert got == expected, f"pad_value={pad_value}"
        # stride 1 with 'same' padding is shape-preserving
        assert len(got) == in_h * in_w * channels

    # The whole reason the field exists. With zero_point = -128 -- the
    # YOLOv8n activation case -- every real activation of a ReLU-like
    # tensor is a *negative* int8, so a zero-filled pad tap (0) is larger
    # than all of them and wins every border max, while the correct
    # zero-point pad (-128) never does.
    relu_like = [random.randint(-128, -40) for _ in range(in_h * in_w * channels)]
    with_zp = pool_max(relu_like, desc_for(-128))
    with_zero = pool_max(relu_like, desc_for(0))
    assert with_zp != with_zero
    # Every output position whose window touches the pad reads back a
    # spurious 0 in the zero-filled version -- pure corruption -- while
    # the zero-point version stays in the tensor's real value range.
    assert 0 in with_zero
    assert max(with_zp) < 0
    assert with_zp == _naive_pool_max(
        relu_like, in_h, in_w, channels, (5, 5), (1, 1), (2, 2, 2, 2), -128
    )


def test_pool_padding_defaults_to_zero_fill() -> None:
    """`pad_value` defaults to 0, so a v2.0-style pooling descriptor keeps
    its exact previous behaviour (and an unpadded pool ignores the field
    entirely)."""
    random.seed(99)
    values = [random.randint(-128, 127) for _ in range(4 * 4)]
    unpadded = LayerDesc(
        opcode=OPCODE_POOL_MAX,
        in_width=4, in_height=4, in_channels=1,
        pool_kernel_h=2, pool_kernel_w=2, pool_stride_h=2, pool_stride_w=2,
        pad_top=1, pad_bottom=1, pad_left=1, pad_right=1,  # ignored: no PAD_EN
        pad_value=-128,
    )
    assert pool_max(values, unpadded) == _naive_pool_max(
        values, 4, 4, 1, (2, 2), (2, 2), (0, 0, 0, 0), 0
    )

    padded_default = LayerDesc(
        opcode=OPCODE_POOL_MAX,
        flags=(1 << FLAG_PAD_EN),
        in_width=4, in_height=4, in_channels=1,
        pool_kernel_h=3, pool_kernel_w=3, pool_stride_h=1, pool_stride_w=1,
        pad_top=1, pad_bottom=1, pad_left=1, pad_right=1,
    )
    assert pool_max(values, padded_default) == _naive_pool_max(
        values, 4, 4, 1, (3, 3), (1, 1), (1, 1, 1, 1), 0
    )


def test_pool_avg_padding_counts_padded_taps() -> None:
    """`POOL_AVG` sums the same padded window `POOL_MAX` maxes over: the
    padded taps are `pad_value`, and they are part of the sum (the divide
    is a fixed `requant_scale`, so the model cannot and does not do
    count-exclude averaging)."""
    values = [10] * 16
    desc = LayerDesc(
        opcode=OPCODE_POOL_AVG,
        flags=(1 << FLAG_PAD_EN),
        in_width=4, in_height=4, in_channels=1,
        pool_kernel_h=3, pool_kernel_w=3, pool_stride_h=3, pool_stride_w=3,
        pad_top=1, pad_bottom=1, pad_left=1, pad_right=1,
        pad_value=-2,
        requant_scale=0, requant_shift=0,
    )
    # out dims: (4 + 2 - 3)//3 + 1 = 2 in each axis.
    # Top-left window covers 1 pad row + 1 pad column: 5 padded taps of -2
    # and 4 real taps of 10 -> 30. REQUANT_EN is clear, so the sum is
    # emitted (clamped to int8) unchanged.
    assert pool_avg(values, desc)[0] == 30


# ---------------------------------------------------------------------------
# Naive, from-scratch reference convolution -- deliberately NOT reusing
# any cnn_accel_model internals/helpers, to cross-check conv2d/dwconv2d
# over randomized seeded cases.
# ---------------------------------------------------------------------------


def _naive_round_half_up(value: int, shift: int) -> int:
    # Independent formulation of the H0 rule (TOSA apply_scale_32):
    # add half, then floor-shift. Ties go towards +infinity.
    if shift <= 0:
        return value << (-shift)
    return (value + (1 << (shift - 1))) >> shift


def _naive_saturate_int8(value: int) -> int:
    if value < -128:
        return -128
    if value > 127:
        return 127
    return value


def _naive_conv_generic(
    input_values, weights, bias, *,
    in_w, in_h, in_c, out_c, k_h, k_w, s_h, s_w,
    pad_top, pad_bottom, pad_left, pad_right,
    bias_en, requant_en, relu_en, requant_scale, requant_shift,
    depthwise: bool,
) -> list[int]:
    padded_h = in_h + pad_top + pad_bottom
    padded_w = in_w + pad_left + pad_right
    padded = [[[0] * in_c for _ in range(padded_w)] for _ in range(padded_h)]
    for r in range(in_h):
        for c in range(in_w):
            for ch in range(in_c):
                padded[r + pad_top][c + pad_left][ch] = input_values[(r * in_w + c) * in_c + ch]

    out_h = (padded_h - k_h) // s_h + 1
    out_w = (padded_w - k_w) // s_w + 1

    result = []
    for orow in range(out_h):
        for ocol in range(out_w):
            for oc in range(out_c):
                acc = 0
                if depthwise:
                    for kr in range(k_h):
                        for kc in range(k_w):
                            prow = orow * s_h + kr
                            pcol = ocol * s_w + kc
                            tap = padded[prow][pcol][oc]
                            w = weights[(oc * k_h + kr) * k_w + kc]
                            acc += tap * w
                else:
                    for kr in range(k_h):
                        for kc in range(k_w):
                            prow = orow * s_h + kr
                            pcol = ocol * s_w + kc
                            for ic in range(in_c):
                                tap = padded[prow][pcol][ic]
                                widx = ((oc * k_h + kr) * k_w + kc) * in_c + ic
                                acc += tap * weights[widx]
                total = acc + (bias[oc] if bias_en else 0)
                if requant_en:
                    scaled = _naive_round_half_up(total * requant_scale, 15 + requant_shift)
                else:
                    scaled = total
                if relu_en:
                    scaled = max(scaled, 0)
                result.append(_naive_saturate_int8(scaled))
    return result


def _random_layer_case(rng: random.Random, *, depthwise: bool, asymmetric_pad: bool = False):
    kernel = rng.choice([1, 3])
    stride = rng.choice([1, 2])
    if asymmetric_pad and kernel > 1:
        # 4 independent pad fields, deliberately allowed to differ on
        # every side (including a zero on some sides) -- the ISA has no
        # constraint tying pad_top/pad_bottom/pad_left/pad_right together.
        pad_top = rng.randint(0, 2)
        pad_bottom = rng.randint(0, 2)
        pad_left = rng.randint(0, 2)
        pad_right = rng.randint(0, 2)
    else:
        pad = rng.choice([0, 1]) if kernel > 1 else 0
        pad_top = pad_bottom = pad_left = pad_right = pad
    in_w = rng.randint(max(kernel, 3), 8)
    in_h = rng.randint(max(kernel, 3), 8)
    in_c = rng.randint(1, 8)
    out_c = in_c if depthwise else rng.randint(1, 8)

    padded_h = in_h + pad_top + pad_bottom
    padded_w = in_w + pad_left + pad_right
    out_h = (padded_h - kernel) // stride + 1
    out_w = (padded_w - kernel) // stride + 1
    if out_h <= 0 or out_w <= 0:
        return None  # skip an invalid combo (caller retries with a new seed draw)

    inp = [rng.randint(-128, 127) for _ in range(in_w * in_h * in_c)]
    if depthwise:
        weights = [rng.randint(-128, 127) for _ in range(out_c * kernel * kernel)]
    else:
        weights = [rng.randint(-128, 127) for _ in range(out_c * kernel * kernel * in_c)]
    bias_en = rng.choice([True, False])
    bias = [rng.randint(-1000, 1000) for _ in range(out_c)] if bias_en else [0] * out_c
    relu_en = rng.choice([True, False])
    requant_en = rng.choice([True, False])
    requant_scale = rng.randint(-(1 << 16), (1 << 16) - 1) if requant_en else 0
    # Full uint8 ISA field range (0..255), not just the small 0..8 window
    # that leaves most of the field unexercised.
    requant_shift = rng.randint(0, 255) if requant_en else 0
    pad_en = bool(pad_top or pad_bottom or pad_left or pad_right)

    desc = LayerDesc(
        opcode=OPCODE_DWCONV2D if depthwise else OPCODE_CONV2D,
        flags=(
            (int(relu_en) << FLAG_RELU_EN)
            | (int(bias_en) << FLAG_BIAS_EN)
            | (int(requant_en) << FLAG_REQUANT_EN)
            | (int(pad_en) << FLAG_PAD_EN)
        ),
        in_width=in_w, in_height=in_h, in_channels=in_c, out_channels=out_c,
        kernel_h=kernel, kernel_w=kernel, stride_h=stride, stride_w=stride,
        pad_top=pad_top, pad_bottom=pad_bottom, pad_left=pad_left, pad_right=pad_right,
        requant_scale=requant_scale, requant_shift=requant_shift,
    )
    naive_kwargs = dict(
        in_w=in_w, in_h=in_h, in_c=in_c, out_c=out_c, k_h=kernel, k_w=kernel,
        s_h=stride, s_w=stride,
        pad_top=pad_top, pad_bottom=pad_bottom, pad_left=pad_left, pad_right=pad_right,
        bias_en=bias_en, requant_en=requant_en, relu_en=relu_en,
        requant_scale=requant_scale, requant_shift=requant_shift,
    )
    return desc, inp, weights, bias, naive_kwargs


def test_conv2d_matches_naive_reference_randomized() -> None:
    rng = random.Random(20240601)
    checked = 0
    attempts = 0
    while checked < 200 and attempts < 5000:
        attempts += 1
        case = _random_layer_case(rng, depthwise=False)
        if case is None:
            continue
        desc, inp, weights, bias, naive_kwargs = case
        model_out = conv2d(inp, weights, bias, desc)
        naive_out = _naive_conv_generic(inp, weights, bias, depthwise=False, **naive_kwargs)
        assert model_out == naive_out, (desc, naive_kwargs)
        checked += 1
    assert checked >= 200


def test_dwconv2d_matches_naive_reference_randomized() -> None:
    rng = random.Random(20240602)
    checked = 0
    attempts = 0
    while checked < 200 and attempts < 5000:
        attempts += 1
        case = _random_layer_case(rng, depthwise=True)
        if case is None:
            continue
        desc, inp, weights, bias, naive_kwargs = case
        model_out = dwconv2d(inp, weights, bias, desc)
        naive_out = _naive_conv_generic(inp, weights, bias, depthwise=True, **naive_kwargs)
        assert model_out == naive_out, (desc, naive_kwargs)
        checked += 1
    assert checked >= 200


# ---------------------------------------------------------------------------
# Asymmetric padding: pad_top/pad_bottom/pad_left/pad_right independently
# different. The ISA encodes 4 separate byte fields (OFF_PAD_TOP/BOTTOM/
# LEFT/RIGHT) with no constraint that they match, but nothing above
# exercised them differing until now.
# ---------------------------------------------------------------------------


def test_conv2d_asymmetric_padding_top_and_right_only() -> None:
    # kernel=1x1 keeps the math trivial to hand-verify: every padded
    # position not covered by a real input pixel must be exactly 0
    # (zero-pad tap through an identity weight), and real positions must
    # land at (row+pad_top, col+pad_left) in the padded/output grid.
    desc = _desc(
        in_width=2, in_height=2, in_channels=1, out_channels=1, kernel_h=1, kernel_w=1,
        flags=(1 << FLAG_PAD_EN), pad_top=1, pad_bottom=0, pad_left=0, pad_right=1,
    )
    inp = [1, 2, 3, 4]  # row0=[1,2], row1=[3,4]
    weights = [1]
    bias = [0]
    out = conv2d(inp, weights, bias, desc)
    # padded 3x3 grid: row0 all-zero (pad_top), row1=[1,2,0] (pad_right),
    # row2=[3,4,0] (pad_right); kernel=1 stride=1 -> output == padded grid.
    assert out == [0, 0, 0, 1, 2, 0, 3, 4, 0]


def test_conv2d_asymmetric_padding_all_four_sides_different() -> None:
    # pad_top != pad_bottom AND pad_left != pad_right simultaneously,
    # cross-checked against the from-scratch naive reference (already
    # validated independently by the randomized fuzz tests below).
    desc = _desc(
        in_width=4, in_height=4, in_channels=1, out_channels=1, kernel_h=3, kernel_w=3,
        flags=(1 << FLAG_PAD_EN), pad_top=1, pad_bottom=2, pad_left=0, pad_right=3,
    )
    rng = random.Random(999)
    inp = [rng.randint(-128, 127) for _ in range(16)]
    weights = [rng.randint(-128, 127) for _ in range(9)]
    bias = [0]
    out = conv2d(inp, weights, bias, desc)
    naive_out = _naive_conv_generic(
        inp, weights, bias,
        in_w=4, in_h=4, in_c=1, out_c=1, k_h=3, k_w=3, s_h=1, s_w=1,
        pad_top=1, pad_bottom=2, pad_left=0, pad_right=3,
        bias_en=False, requant_en=False, relu_en=False,
        requant_scale=0, requant_shift=0, depthwise=False,
    )
    # padded 7x7 (4+1+2 rows, 4+0+3 cols), kernel3 stride1 -> out 5x5.
    assert len(out) == 25
    assert out == naive_out


def test_conv2d_matches_naive_reference_randomized_asymmetric_padding() -> None:
    rng = random.Random(20240701)
    checked = 0
    attempts = 0
    while checked < 150 and attempts < 5000:
        attempts += 1
        case = _random_layer_case(rng, depthwise=False, asymmetric_pad=True)
        if case is None:
            continue
        desc, inp, weights, bias, naive_kwargs = case
        model_out = conv2d(inp, weights, bias, desc)
        naive_out = _naive_conv_generic(inp, weights, bias, depthwise=False, **naive_kwargs)
        assert model_out == naive_out, (desc, naive_kwargs)
        checked += 1
    assert checked >= 150
    # Sanity: the sweep actually produced at least one genuinely
    # asymmetric case (not all degenerate to symmetric/no padding).
    rng = random.Random(20240701)
    asymmetric_seen = False
    attempts = 0
    while attempts < 5000 and not asymmetric_seen:
        attempts += 1
        case = _random_layer_case(rng, depthwise=False, asymmetric_pad=True)
        if case is None:
            continue
        desc, *_ = case
        if desc.pad_top != desc.pad_bottom or desc.pad_left != desc.pad_right:
            asymmetric_seen = True
    assert asymmetric_seen


def test_dwconv2d_matches_naive_reference_randomized_asymmetric_padding() -> None:
    rng = random.Random(20240702)
    checked = 0
    attempts = 0
    while checked < 150 and attempts < 5000:
        attempts += 1
        case = _random_layer_case(rng, depthwise=True, asymmetric_pad=True)
        if case is None:
            continue
        desc, inp, weights, bias, naive_kwargs = case
        model_out = dwconv2d(inp, weights, bias, desc)
        naive_out = _naive_conv_generic(inp, weights, bias, depthwise=True, **naive_kwargs)
        assert model_out == naive_out, (desc, naive_kwargs)
        checked += 1
    assert checked >= 150


# ---------------------------------------------------------------------------
# requant_shift: full uint8 ISA range (not just 0..8). A shift large
# enough to push the scaled value's magnitude far below one part in the
# divisor must round to exactly 0 (not raise, not saturate to a boundary
# by accident, not return garbage).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("requant_shift", [60, 100, 200, 255])
@pytest.mark.parametrize("relu_en", [False, True])
def test_requant_shift_extreme_values_shift_entirely_away_to_zero(
    requant_shift: int, relu_en: bool
) -> None:
    rng = random.Random(555 + requant_shift)
    for _ in range(20):
        acc = rng.randint(-(2**31), 2**31 - 1)
        bias = rng.randint(-(2**31), 2**31 - 1)
        scale = rng.randint(-(2**31), 2**31 - 1)
        result = bias_requantize_relu(
            acc, bias, bias_en=True, requant_en=True, relu_en=relu_en,
            requant_scale=scale, requant_shift=requant_shift,
        )
        # combined_shift = 15 + requant_shift is large enough that
        # |total * scale| (bounded by ~2**63 given int32 total/scale)
        # is always far below divisor/2 (>= 2**54), so the rounded
        # shift is exactly 0 regardless of sign -- and relu_en/saturate
        # leave 0 unchanged either way.
        assert result == 0


def test_requant_shift_full_range_conv_sweep_included() -> None:
    # The shared randomized-fuzz helper now draws requant_shift from the
    # full uint8 range; assert that range is actually exercised (not
    # just the old 0..8 window) so this doesn't silently regress back.
    rng = random.Random(778899)
    seen_large_shift = False
    attempts = 0
    while attempts < 2000 and not seen_large_shift:
        attempts += 1
        case = _random_layer_case(rng, depthwise=False)
        if case is None:
            continue
        desc, *_ = case
        if desc.requant_en and desc.requant_shift > 8:
            seen_large_shift = True
    assert seen_large_shift


# ---------------------------------------------------------------------------
# pool_max/pool_avg: from-scratch naive reference (independent of
# cnn_accel_model's _pool_windows/_at helpers) + randomized fuzz sweep.
# ---------------------------------------------------------------------------


def _naive_pool(
    input_values: list[int],
    *,
    in_w: int,
    in_h: int,
    channels: int,
    k_h: int,
    k_w: int,
    s_h: int,
    s_w: int,
    mode: str,
    relu_en: bool,
    requant_en: bool,
    requant_scale: int,
    requant_shift: int,
) -> list[int]:
    out_h = (in_h - k_h) // s_h + 1
    out_w = (in_w - k_w) // s_w + 1
    result: list[int] = []
    for orow in range(out_h):
        for ocol in range(out_w):
            for ch in range(channels):
                taps = [
                    input_values[((orow * s_h + kr) * in_w + (ocol * s_w + kc)) * channels + ch]
                    for kr in range(k_h)
                    for kc in range(k_w)
                ]
                if mode == "max":
                    result.append(max(taps))
                else:
                    total = sum(taps)
                    if requant_en:
                        scaled = _naive_round_half_up(total * requant_scale, 15 + requant_shift)
                    else:
                        scaled = total
                    if relu_en:
                        scaled = max(scaled, 0)
                    result.append(_naive_saturate_int8(scaled))
    return result


def _random_pool_case(rng: random.Random, *, mode: str):
    k_h = rng.randint(1, 4)
    k_w = rng.randint(1, 4)
    s_h = rng.randint(1, 4)
    s_w = rng.randint(1, 4)
    in_w = rng.randint(k_w, k_w + 4)
    in_h = rng.randint(k_h, k_h + 4)
    channels = rng.randint(1, 3)

    values = [rng.randint(-128, 127) for _ in range(in_w * in_h * channels)]
    # Force INT8 extremes to be present in every case.
    values[0] = -128
    if len(values) > 1:
        values[1] = 127

    relu_en = rng.choice([True, False]) if mode == "avg" else False
    requant_en = rng.choice([True, False]) if mode == "avg" else False
    requant_scale = rng.randint(-(1 << 16), (1 << 16) - 1) if requant_en else 0
    requant_shift = rng.randint(0, 12) if requant_en else 0

    flags = (
        (int(relu_en) << FLAG_RELU_EN)
        | (int(requant_en) << FLAG_REQUANT_EN)
    )
    desc = LayerDesc(
        opcode=OPCODE_POOL_MAX if mode == "max" else OPCODE_POOL_AVG,
        flags=flags,
        in_width=in_w, in_height=in_h, in_channels=channels,
        pool_kernel_h=k_h, pool_kernel_w=k_w, pool_stride_h=s_h, pool_stride_w=s_w,
        requant_scale=requant_scale, requant_shift=requant_shift,
    )
    naive_kwargs = dict(
        in_w=in_w, in_h=in_h, channels=channels, k_h=k_h, k_w=k_w, s_h=s_h, s_w=s_w,
        mode=mode, relu_en=relu_en, requant_en=requant_en,
        requant_scale=requant_scale, requant_shift=requant_shift,
    )
    return desc, values, naive_kwargs


def test_pool_max_matches_naive_reference_randomized() -> None:
    rng = random.Random(313131)
    for _ in range(300):
        desc, values, naive_kwargs = _random_pool_case(rng, mode="max")
        model_out = pool_max(values, desc)
        naive_out = _naive_pool(values, **naive_kwargs)
        assert model_out == naive_out, (desc, naive_kwargs)


def test_pool_avg_matches_naive_reference_randomized() -> None:
    rng = random.Random(414141)
    for _ in range(300):
        desc, values, naive_kwargs = _random_pool_case(rng, mode="avg")
        model_out = pool_avg(values, desc)
        naive_out = _naive_pool(values, **naive_kwargs)
        assert model_out == naive_out, (desc, naive_kwargs)


def test_pool_max_int8_extremes() -> None:
    desc = LayerDesc(
        opcode=OPCODE_POOL_MAX, in_width=2, in_height=1, in_channels=1,
        pool_kernel_h=1, pool_kernel_w=2, pool_stride_h=1, pool_stride_w=2,
    )
    assert pool_max([-128, 127], desc) == [127]
    assert pool_max([127, -128], desc) == [127]
    assert pool_max([-128, -128], desc) == [-128]


def test_pool_avg_rounding_half_up_explicit() -> None:
    # Two windows whose sums are both odd (so total*scale lands exactly
    # on a rounding tie at combined_shift=16) with quotients of opposite
    # parity -- demonstrates round-HALF-UP (H0: every tie goes towards
    # +infinity regardless of parity) for pool_avg specifically, not just
    # round_shift_right_signed in isolation. Negative tie too.
    desc = LayerDesc(
        opcode=OPCODE_POOL_AVG, flags=(1 << FLAG_REQUANT_EN),
        in_width=2, in_height=1, in_channels=1,
        pool_kernel_h=1, pool_kernel_w=2, pool_stride_h=1, pool_stride_w=2,
        requant_scale=1 << 15, requant_shift=1,  # combined shift = 16
    )
    # window sum = 5: 5*32768=163840, /65536 -> q=2 r=32768 (tie) -> 3.
    assert pool_avg([2, 3], desc) == [3]
    # window sum = 3: 3*32768=98304, /65536 -> q=1 r=32768 (tie) -> 2.
    assert pool_avg([1, 2], desc) == [2]
    # window sum = -3: -1.5 -> floor q=-2, tie -> -1.
    assert pool_avg([-1, -2], desc) == [-1]


# ---------------------------------------------------------------------------
# End-to-end run_layer/run_program: execute a layer through the actual
# byte-level memory image (build_memory_image + encode_instruction),
# rather than only calling conv2d/pool_*/fc directly on Python lists.
# ---------------------------------------------------------------------------


def _to_signed_bytes(values: list[int]) -> bytes:
    return bytes(v & 0xFF for v in values)


def _from_signed_bytes(data: bytes) -> list[int]:
    return [b - 256 if b >= 128 else b for b in data]


# `run_layer`/`run_program` speak the ratified DDR layout (decision S6 for
# activations, D10 for weights/bias), NOT the LOGICAL flat-list layout
# `conv2d`/`pool_*`/`fc` take/return -- these three helpers build/read the
# DDR-side `memory[]` chunks a real `memory_image.csv` would contain, using
# the module's own default TILE_CHANNELS/PE_ROWS (what `run_layer` itself
# assumes).
def _activation_chunk(values_hwc: list[int], width: int, height: int, channels: int) -> bytes:
    return _to_signed_bytes(pack_activation_planes(values_hwc, width, height, channels))


def _read_activation(mem: bytearray, addr: int, width: int, height: int, channels: int) -> list[int]:
    n = activation_bytes(width, height, channels)
    return unpack_activation_planes(_from_signed_bytes(bytes(mem[addr : addr + n])), width, height, channels)


def _weight_chunk(weights: list[int], desc: LayerDesc) -> bytes:
    return _to_signed_bytes(
        pack_weights_for_hw(weights, desc, cnn_accel_constants.TILE_CHANNELS, cnn_accel_constants.PE_ROWS)
    )


def _bias_chunk(bias: list[int], desc: LayerDesc) -> bytes:
    packed = pack_bias_for_hw(bias, desc, cnn_accel_constants.PE_ROWS)
    return struct.pack(f"<{len(packed)}i", *packed)


def test_run_layer_per_channel_reads_scale_table_from_ddr() -> None:
    """ISA v1.2 end to end: `run_layer` with PER_CHANNEL_EN must read
    `pack_scale_table_for_hw`'s image at `scale_addr` and match `conv2d`
    given the logical table; with the flag clear the same memory image
    (table still present) must match the legacy `conv2d` exactly."""
    rng = random.Random(0xC5A1)
    in_w, in_h, in_c, out_c, k = 4, 4, 3, 12, 3  # 12 -> 2 output tiles at PE_ROWS=8, padded
    xs = [rng.randint(-128, 127) for _ in range(in_w * in_h * in_c)]
    ws = [rng.randint(-128, 127) for _ in range(out_c * k * k * in_c)]
    bs = [rng.randint(-1000, 1000) for _ in range(out_c)]
    table = [(rng.randint(1 << 10, 1 << 20), rng.randint(0, 6)) for _ in range(out_c)]

    in_addr, weight_addr, bias_addr, scale_addr, out_addr = 0x1000, 0x2000, 0x3000, 0x3800, 0x4000
    desc = LayerDesc(
        opcode=OPCODE_CONV2D,
        flags=(1 << FLAG_REQUANT_EN) | (1 << FLAG_BIAS_EN) | (1 << FLAG_PAD_EN) | (1 << FLAG_PER_CHANNEL_EN),
        in_addr=in_addr, out_addr=out_addr, weight_addr=weight_addr, bias_addr=bias_addr,
        scale_addr=scale_addr,
        in_width=in_w, in_height=in_h, in_channels=in_c, out_channels=out_c,
        kernel_h=k, kernel_w=k, pad_top=1, pad_bottom=1, pad_left=1, pad_right=1,
        requant_scale=1 << 15, requant_shift=3,
    )
    table_image = pack_scale_table_for_hw(table, desc, cnn_accel_constants.PE_ROWS)
    assert len(table_image) == packed_scale_table_bytes(desc, cnn_accel_constants.PE_ROWS) == 16 * 8
    chunks = {
        in_addr: _activation_chunk(xs, in_w, in_h, in_c),
        weight_addr: _weight_chunk(ws, desc),
        bias_addr: _bias_chunk(bs, desc),
        scale_addr: table_image,
    }

    mem = build_memory_image(chunks, size=0x5000)
    run_layer(mem, desc)
    assert _read_activation(mem, out_addr, in_w, in_h, out_c) == conv2d(xs, ws, bs, desc, table)

    # Flag clear, same image: legacy result, scale_addr never dereferenced.
    legacy = LayerDesc(**{**desc.__dict__, "flags": desc.flags & ~(1 << FLAG_PER_CHANNEL_EN)})
    mem = build_memory_image(chunks, size=0x5000)
    run_layer(mem, legacy)
    assert _read_activation(mem, out_addr, in_w, in_h, out_c) == conv2d(xs, ws, bs, legacy)
    assert conv2d(xs, ws, bs, legacy) != conv2d(xs, ws, bs, desc, table)  # the test has teeth


def test_run_layer_conv2d_end_to_end_via_memory_image() -> None:
    rng = random.Random(42001)
    in_w, in_h, in_c, out_c, k = 4, 4, 2, 3, 3
    input_values = [rng.randint(-128, 127) for _ in range(in_w * in_h * in_c)]
    weights = [rng.randint(-128, 127) for _ in range(out_c * k * k * in_c)]
    bias = [rng.randint(-1000, 1000) for _ in range(out_c)]

    in_addr, weight_addr, bias_addr, out_addr = 0x1000, 0x2000, 0x3000, 0x4000
    desc = LayerDesc(
        opcode=OPCODE_CONV2D,
        flags=(1 << FLAG_RELU_EN) | (1 << FLAG_BIAS_EN) | (1 << FLAG_REQUANT_EN) | (1 << FLAG_PAD_EN),
        in_addr=in_addr, out_addr=out_addr, weight_addr=weight_addr, bias_addr=bias_addr,
        in_width=in_w, in_height=in_h, in_channels=in_c, out_channels=out_c,
        kernel_h=k, kernel_w=k, stride_h=1, stride_w=1,
        pad_top=1, pad_bottom=1, pad_left=1, pad_right=1,
        requant_scale=1 << 13, requant_shift=1,
    )
    chunks = {
        in_addr: _activation_chunk(input_values, in_w, in_h, in_c),
        weight_addr: _weight_chunk(weights, desc),
        bias_addr: _bias_chunk(bias, desc),
    }
    mem = build_memory_image(chunks, size=0x5000)
    run_layer(mem, desc)

    expected = conv2d(input_values, weights, bias, desc)
    out_w, out_h = in_w, in_h  # same-padding, stride 1 -> unchanged spatial size
    actual = _read_activation(mem, out_addr, out_w, out_h, out_c)
    assert actual == expected


def test_run_layer_pool_end_to_end_via_memory_image() -> None:
    rng = random.Random(42002)
    in_w, in_h, in_c = 4, 4, 2
    input_values = [rng.randint(-128, 127) for _ in range(in_w * in_h * in_c)]

    in_addr, out_addr = 0x1000, 0x2000
    desc = LayerDesc(
        opcode=OPCODE_POOL_MAX,
        in_addr=in_addr, out_addr=out_addr,
        in_width=in_w, in_height=in_h, in_channels=in_c,
        pool_kernel_h=2, pool_kernel_w=2, pool_stride_h=2, pool_stride_w=2,
    )
    mem = build_memory_image({in_addr: _activation_chunk(input_values, in_w, in_h, in_c)}, size=0x3000)
    run_layer(mem, desc)

    expected = pool_max(input_values, desc)
    out_w, out_h = in_w // 2, in_h // 2
    actual = _read_activation(mem, out_addr, out_w, out_h, in_c)
    assert actual == expected


def test_run_layer_fc_end_to_end_via_memory_image() -> None:
    rng = random.Random(42003)
    in_c, out_c = 5, 3
    input_values = [rng.randint(-128, 127) for _ in range(in_c)]
    weights = [rng.randint(-128, 127) for _ in range(out_c * in_c)]
    bias = [rng.randint(-1000, 1000) for _ in range(out_c)]

    in_addr, weight_addr, bias_addr, out_addr = 0x1000, 0x2000, 0x3000, 0x4000
    desc = LayerDesc(
        opcode=OPCODE_FC,
        flags=(1 << FLAG_BIAS_EN),
        in_addr=in_addr, out_addr=out_addr, weight_addr=weight_addr, bias_addr=bias_addr,
        in_width=1, in_height=1, in_channels=in_c, out_channels=out_c, kernel_h=1, kernel_w=1,
    )
    chunks = {
        in_addr: _activation_chunk(input_values, 1, 1, in_c),
        weight_addr: _weight_chunk(weights, desc),
        bias_addr: _bias_chunk(bias, desc),
    }
    mem = build_memory_image(chunks, size=0x5000)
    run_layer(mem, desc)

    expected = fc(input_values, weights, bias, desc)
    actual = _read_activation(mem, out_addr, 1, 1, out_c)
    assert actual == expected


def test_run_program_chains_conv_then_pool_and_halts() -> None:
    rng = random.Random(42004)
    in_w, in_h, in_c, out_c, k = 4, 4, 2, 2, 3
    input_values = [rng.randint(-128, 127) for _ in range(in_w * in_h * in_c)]
    weights = [rng.randint(-128, 127) for _ in range(out_c * k * k * in_c)]
    bias = [rng.randint(-1000, 1000) for _ in range(out_c)]

    program_addr = 0x0000
    in_addr, weight_addr, bias_addr = 0x1000, 0x2000, 0x3000
    mid_addr, final_addr = 0x4000, 0x5000

    conv_desc = LayerDesc(
        opcode=OPCODE_CONV2D,
        flags=(1 << FLAG_BIAS_EN) | (1 << FLAG_PAD_EN),
        in_addr=in_addr, out_addr=mid_addr, weight_addr=weight_addr, bias_addr=bias_addr,
        in_width=in_w, in_height=in_h, in_channels=in_c, out_channels=out_c,
        kernel_h=k, kernel_w=k, stride_h=1, stride_w=1,
        pad_top=1, pad_bottom=1, pad_left=1, pad_right=1,
    )
    # conv output is same spatial size as input (same-padding, stride 1)
    # with out_c channels -> pool consumes an (in_w, in_h, out_c) tensor.
    pool_desc = LayerDesc(
        opcode=OPCODE_POOL_MAX,
        in_addr=mid_addr, out_addr=final_addr,
        in_width=in_w, in_height=in_h, in_channels=out_c,
        pool_kernel_h=2, pool_kernel_w=2, pool_stride_h=2, pool_stride_w=2,
    )
    halt_desc = LayerDesc(opcode=OPCODE_HALT)

    program = encode_program([conv_desc, pool_desc, halt_desc], program_addr=program_addr)
    chunks = {
        program_addr: program,
        in_addr: _activation_chunk(input_values, in_w, in_h, in_c),
        weight_addr: _weight_chunk(weights, conv_desc),
        bias_addr: _bias_chunk(bias, conv_desc),
    }
    mem = build_memory_image(chunks, size=0x6000)

    executed = run_program(mem, program_addr)
    assert executed == 2  # conv + pool, HALT not counted

    conv_expected = conv2d(input_values, weights, bias, conv_desc)
    mid_actual = _read_activation(mem, mid_addr, in_w, in_h, out_c)
    assert mid_actual == conv_expected  # layer 2 really consumed layer 1's output

    final_w, final_h = in_w // 2, in_h // 2
    pool_expected = pool_max(conv_expected, pool_desc)
    final_actual = _read_activation(mem, final_addr, final_w, final_h, out_c)
    assert final_actual == pool_expected

    # (e) D1's alignment argument: every activation buffer's DDR byte size
    # AND its own address are multiples of ACTIVATION_PLANE_CHANNELS,
    # across this whole two-layer chain -- because every request is a
    # whole plane, an AXI bus of at most ACTIVATION_PLANE_CHANNELS bytes
    # is aligned unconditionally (doc/cnn_accel_arch.md "Bus-width bound
    # (decision D1)").
    t = cnn_accel_constants.ACTIVATION_PLANE_CHANNELS
    for addr, w, h, c in (
        (in_addr, in_w, in_h, in_c),
        (mid_addr, in_w, in_h, out_c),
        (final_addr, final_w, final_h, out_c),
    ):
        assert addr % t == 0
        assert activation_bytes(w, h, c) % t == 0


def test_run_program_conv2d_multi_tile_matches_direct_conv2d_in_channels_partial() -> None:
    """Multi-input-tile case (`in_channels=3` not a multiple of
    `TILE_CHANNELS=8`, `out_channels=8` exactly one output tile): a full
    `run_program` round trip through the ratified S6/D10 DDR layout must
    match `conv2d()` called directly on the LOGICAL lists."""
    rng = random.Random(0x5106)
    in_w, in_h, in_c, out_c, k = 5, 5, 3, 8, 3
    input_values = [rng.randint(-128, 127) for _ in range(in_w * in_h * in_c)]
    weights = [rng.randint(-128, 127) for _ in range(out_c * k * k * in_c)]
    bias = [rng.randint(-1000, 1000) for _ in range(out_c)]

    program_addr = 0x0000
    in_addr, weight_addr, bias_addr, out_addr = 0x1000, 0x2000, 0x8000, 0x9000
    conv_desc = LayerDesc(
        opcode=OPCODE_CONV2D,
        flags=(1 << FLAG_BIAS_EN) | (1 << FLAG_PAD_EN),
        in_addr=in_addr, out_addr=out_addr, weight_addr=weight_addr, bias_addr=bias_addr,
        in_width=in_w, in_height=in_h, in_channels=in_c, out_channels=out_c,
        kernel_h=k, kernel_w=k, stride_h=1, stride_w=1,
        pad_top=1, pad_bottom=1, pad_left=1, pad_right=1,
    )
    halt_desc = LayerDesc(opcode=OPCODE_HALT)
    program = encode_program([conv_desc, halt_desc], program_addr=program_addr)
    chunks = {
        program_addr: program,
        in_addr: _activation_chunk(input_values, in_w, in_h, in_c),
        weight_addr: _weight_chunk(weights, conv_desc),
        bias_addr: _bias_chunk(bias, conv_desc),
    }
    mem = build_memory_image(chunks, size=0xA000)

    executed = run_program(mem, program_addr)
    assert executed == 1

    expected = conv2d(input_values, weights, bias, conv_desc)
    actual = _read_activation(mem, out_addr, in_w, in_h, out_c)
    assert actual == expected


def test_run_program_conv2d_multi_tile_matches_direct_conv2d_both_dims() -> None:
    """Multi-tile on BOTH sides (`in_channels=16` = 2 input tiles,
    `out_channels=24` = 3 output tiles at `PE_ROWS=8`): same acceptance
    check as above, exercising the full `OT x T` sweep of
    `pack_weights_for_hw`/`unpack_weights_from_hw`."""
    rng = random.Random(0x1624)
    in_w, in_h, in_c, out_c, k = 4, 4, 16, 24, 3
    input_values = [rng.randint(-128, 127) for _ in range(in_w * in_h * in_c)]
    weights = [rng.randint(-128, 127) for _ in range(out_c * k * k * in_c)]
    bias = [rng.randint(-1000, 1000) for _ in range(out_c)]

    program_addr = 0x0000
    in_addr, weight_addr, bias_addr, out_addr = 0x1000, 0x2000, 0x8000, 0x9000
    conv_desc = LayerDesc(
        opcode=OPCODE_CONV2D,
        flags=(1 << FLAG_BIAS_EN) | (1 << FLAG_PAD_EN),
        in_addr=in_addr, out_addr=out_addr, weight_addr=weight_addr, bias_addr=bias_addr,
        in_width=in_w, in_height=in_h, in_channels=in_c, out_channels=out_c,
        kernel_h=k, kernel_w=k, stride_h=1, stride_w=1,
        pad_top=1, pad_bottom=1, pad_left=1, pad_right=1,
    )
    halt_desc = LayerDesc(opcode=OPCODE_HALT)
    program = encode_program([conv_desc, halt_desc], program_addr=program_addr)
    chunks = {
        program_addr: program,
        in_addr: _activation_chunk(input_values, in_w, in_h, in_c),
        weight_addr: _weight_chunk(weights, conv_desc),
        bias_addr: _bias_chunk(bias, conv_desc),
    }
    mem = build_memory_image(chunks, size=0xA000)

    executed = run_program(mem, program_addr)
    assert executed == 1

    expected = conv2d(input_values, weights, bias, conv_desc)
    actual = _read_activation(mem, out_addr, in_w, in_h, out_c)
    assert actual == expected


# ---------------------------------------------------------------------------
# Activation channel-tiled planes (decision S6): pack/unpack round trip,
# zero-padding, and the byte-offset formula itself.
# ---------------------------------------------------------------------------


def test_pack_activation_planes_round_trips_and_zero_pads() -> None:
    """`unpack_activation_planes(pack_activation_planes(x, w, h, c), w, h,
    c) == x` for every x, and every padded lane (`c_tile*T + t >=
    channels`, `T = ACTIVATION_PLANE_CHANNELS`) in the packed image is
    exactly 0 -- swept over channel counts spanning partial and exact
    tiles of `T = 8`."""
    rng = random.Random(0x5678)
    t = cnn_accel_constants.ACTIVATION_PLANE_CHANNELS
    width, height = 3, 2
    for channels in (3, 5, 8, 9, 16):
        values = [rng.randint(-128, 127) for _ in range(width * height * channels)]
        packed = pack_activation_planes(values, width, height, channels)
        n_tiles = activation_plane_count(channels)
        assert len(packed) == n_tiles * t * width * height
        assert len(packed) == activation_bytes(width, height, channels)

        for c_tile in range(n_tiles):
            for y in range(height):
                for x in range(width):
                    base = ((c_tile * height + y) * width + x) * t
                    for lane in range(t):
                        c = c_tile * t + lane
                        if c >= channels:
                            assert packed[base + lane] == 0

        assert unpack_activation_planes(packed, width, height, channels) == values


def test_pack_activation_planes_byte_formula_spot_check() -> None:
    """Direct spot-check of decision S6's own byte formula:
    `byte_offset = ((c_tile*H + y)*W + x)*T + t`."""
    t = cnn_accel_constants.ACTIVATION_PLANE_CHANNELS
    width, height, channels = 5, 4, 20  # ceil(20/8) = 3 planes
    rng = random.Random(0xF0F0)
    values = [rng.randint(-128, 127) for _ in range(width * height * channels)]
    packed = pack_activation_planes(values, width, height, channels)

    for _ in range(200):
        c_tile = rng.randrange(activation_plane_count(channels))
        y = rng.randrange(height)
        x = rng.randrange(width)
        lane = rng.randrange(t)
        offset = ((c_tile * height + y) * width + x) * t + lane
        c = c_tile * t + lane
        expected = values[(y * width + x) * channels + c] if c < channels else 0
        assert packed[offset] == expected


# ---------------------------------------------------------------------------
# unpack_weights_from_hw (D10): exact inverse of pack_weights_for_hw on the
# valid region.
# ---------------------------------------------------------------------------


def test_unpack_weights_from_hw_inverts_pack_weights_for_hw() -> None:
    """`unpack_weights_from_hw(pack_weights_for_hw(w, desc, T, R), desc,
    T, R) == w` (the LOGICAL OHWI array) for a shape sweep including
    `out_channels` not a multiple of `pe_rows` and `in_channels` not a
    multiple of `tile_channels`."""
    rng = random.Random(0xABCD)
    for in_c, out_c, kernel, tile_channels, pe_rows in itertools.product(
        [3, 8, 11, 16], [5, 8, 10, 16], [1, 3], [4, 8], [4, 8]
    ):
        desc = _desc(in_channels=in_c, out_channels=out_c, kernel_h=kernel, kernel_w=kernel)
        weights = [rng.randint(-128, 127) for _ in range(out_c * kernel * kernel * in_c)]
        packed = pack_weights_for_hw(weights, desc, tile_channels, pe_rows)
        assert len(packed) == packed_weight_count(desc, tile_channels, pe_rows)
        assert unpack_weights_from_hw(packed, desc, tile_channels, pe_rows) == weights


def test_unpack_weights_from_hw_rejects_dwconv2d() -> None:
    desc = _desc(opcode=OPCODE_DWCONV2D, in_channels=4, out_channels=4, kernel_h=3, kernel_w=3)
    packed = [0] * (4 * 3 * 3 * 8 * 8)
    with pytest.raises(ValueError):
        unpack_weights_from_hw(packed, desc, tile_channels=8, pe_rows=8)


# ---------------------------------------------------------------------------
# pack_weights_for_hw / pack_bias_for_hw (D10/D11): compile-time weight
# repack for the tiled PE array, ratified in
# doc/cnn_accel_tiled_dataflow_proposal.md section 4. Two kinds of tests:
# (1) a static equivalence sweep against the logical OHWI array (this
# section), and (2) a full hardware-order consumption simulation that
# closes the loop against conv2d() (next section, the acceptance gate).
# ---------------------------------------------------------------------------

# in_channels/out_channels sweep values: 1 (degenerate), 3 (the target
# network's real layer-1 partial-tile case, D11), a small prime (7/9,
# non-multiples of both 4 and 8), the tile/lane widths themselves (8),
# a multiple-of-both (16), and 20 (a large partial tile for every swept
# tile_channels/pe_rows). out_channels additionally includes 12 (a
# partial tile only at pe_rows=8, exact at pe_rows=4).
_PACK_IN_CHANNELS_SWEEP = [1, 3, 7, 8, 9, 16, 20]
_PACK_OUT_CHANNELS_SWEEP = [1, 3, 8, 9, 12, 16, 20]
_PACK_KERNELS_SWEEP = [1, 3]
_PACK_TILE_CHANNELS_SWEEP = [4, 8]
_PACK_PE_ROWS_SWEEP = [4, 8]


def test_pack_weights_for_hw_matches_logical_ohwi_and_pads_zero() -> None:
    """D10/D11 equivalence property: for every (oc, kr, kc, ic) with
    oc < out_channels and ic < in_channels, `pack_weights_for_hw`'s
    packed image holds the same weight value as the logical OHWI array
    at the offset its own docstring computes; every padded lane
    (oc >= out_channels or ic >= in_channels) is exactly 0; and the
    total packed length matches the documented
    `OT*T*kernel_h*kernel_w*tile_channels*pe_rows` formula. Swept over
    kernel 1 and 3, in_channels/out_channels spanning 1..20 (including 3
    and non-multiples of 8), and tile_channels/pe_rows in {4, 8}."""
    rng = random.Random(0xD10)
    for in_c, out_c, kernel, tile_channels, pe_rows in itertools.product(
        _PACK_IN_CHANNELS_SWEEP,
        _PACK_OUT_CHANNELS_SWEEP,
        _PACK_KERNELS_SWEEP,
        _PACK_TILE_CHANNELS_SWEEP,
        _PACK_PE_ROWS_SWEEP,
    ):
        desc = _desc(in_channels=in_c, out_channels=out_c, kernel_h=kernel, kernel_w=kernel)
        weights = [rng.randint(-128, 127) for _ in range(out_c * kernel * kernel * in_c)]
        packed = pack_weights_for_hw(weights, desc, tile_channels, pe_rows)

        n_in_tiles = -(-in_c // tile_channels)  # ceil
        n_out_tiles = -(-out_c // pe_rows)  # ceil
        expected_len = n_out_tiles * n_in_tiles * kernel * kernel * tile_channels * pe_rows
        assert len(packed) == expected_len

        idx = 0
        for ot in range(n_out_tiles):
            for t in range(n_in_tiles):
                for kr in range(kernel):
                    for kc in range(kernel):
                        for r in range(pe_rows):
                            oc = ot * pe_rows + r
                            for c in range(tile_channels):
                                ic = t * tile_channels + c
                                if ic < in_c and oc < out_c:
                                    expected = weights[
                                        ((oc * kernel + kr) * kernel + kc) * in_c + ic
                                    ]
                                else:
                                    expected = 0  # D11: padded lane must be zero
                                assert packed[idx] == expected, (
                                    f"in_c={in_c} out_c={out_c} kernel={kernel} "
                                    f"tile_channels={tile_channels} pe_rows={pe_rows} "
                                    f"ot={ot} t={t} kr={kr} kc={kc} c={c} r={r} "
                                    f"ic={ic} oc={oc}"
                                )
                                idx += 1


def test_pack_weights_for_hw_rejects_dwconv2d() -> None:
    """DWCONV2D's `(channels, kernel_h, kernel_w)` layout has no
    cross-channel reduction to gather -- out of D10's ratified scope."""
    desc = _desc(opcode=OPCODE_DWCONV2D, in_channels=4, out_channels=4, kernel_h=3, kernel_w=3)
    weights = [0] * (4 * 3 * 3)
    with pytest.raises(ValueError):
        pack_weights_for_hw(weights, desc, tile_channels=8, pe_rows=8)


def test_pack_weights_for_hw_lane_matches_pe_array_rtl_indexing() -> None:
    """Guard against the D10 lane-order defect regressing silently: pins
    `pack_weights_for_hw`'s within-row lane order to
    `cnn_accel_pe_array.vhd`'s own `compute_partial_sums()` indexing
    (`weight_lane := r * g_pe_cols + c`, `cnn_accel_pe_array.vhd:258`),
    via an independent re-derivation rather than a copy of either
    function's own loop nesting: for every lane in a packed row, DECODE
    `(r, c)` by the RTL's formula (`r = lane // tile_channels`,
    `c = lane % tile_channels` -- the inverse of `r*tile_channels + c`)
    and check the packed value at that lane against the logical OHWI
    weight at the `(oc, ic)` that `(r, c)` implies. Kernel is fixed at
    1x1 here (`kr=kc=0`) -- the row-offset arithmetic for kernel > 1 is
    already exhaustively covered by
    `test_pack_weights_for_hw_matches_logical_ohwi_and_pads_zero` above;
    this test's only job is the lane formula itself, swept over both a
    partial input tile and a partial+multi output tile so an off-by-one
    in `oc`/`ic` padding can't hide the lane formula being wrong.

    If `pack_weights_for_hw` (or `cnn_accel_pe_array.vhd`) ever
    reintroduces the transposed `c*pe_rows + r` convention, this test
    fails without needing a simulator; `tb_cnn_accel_pe_array_from_
    vectors.vhd` additionally closes the loop through the real RTL."""
    rng = random.Random(0xBEEF)
    for tile_channels, pe_rows in itertools.product([4, 8], [4, 8]):
        in_c, out_c = 11, 10  # partial input tile and partial+multi output tile
        desc = _desc(in_channels=in_c, out_channels=out_c, kernel_h=1, kernel_w=1)
        weights = [rng.randint(-128, 127) for _ in range(out_c * in_c)]
        packed = pack_weights_for_hw(weights, desc, tile_channels, pe_rows)

        n_in_tiles = -(-in_c // tile_channels)
        n_out_tiles = -(-out_c // pe_rows)
        row_len = tile_channels * pe_rows

        for ot in range(n_out_tiles):
            for t in range(n_in_tiles):
                row_idx = ot * n_in_tiles + t
                row = packed[row_idx * row_len : (row_idx + 1) * row_len]
                for lane in range(row_len):
                    # Independent re-derivation of cnn_accel_pe_array.vhd's own
                    # 'weight_lane := r * g_pe_cols + c' -- NOT pack_weights_for_hw's
                    # own (r, c) loop variables.
                    r = lane // tile_channels
                    c = lane % tile_channels
                    ic = t * tile_channels + c
                    oc = ot * pe_rows + r
                    expected = weights[oc * in_c + ic] if (ic < in_c and oc < out_c) else 0
                    assert row[lane] == expected, (
                        f"tile_channels={tile_channels} pe_rows={pe_rows} "
                        f"ot={ot} t={t} lane={lane} r={r} c={c} ic={ic} oc={oc}"
                    )


def test_pack_bias_for_hw_matches_and_pads_zero() -> None:
    """Bias padding must line up with `pack_weights_for_hw`'s `ot -> r`
    output-channel tiling exactly (`oc = ot*pe_rows + r`); total length
    is always `ceil(out_channels/pe_rows) * pe_rows`."""
    rng = random.Random(0xB1A5)
    for out_c, pe_rows in itertools.product(_PACK_OUT_CHANNELS_SWEEP, _PACK_PE_ROWS_SWEEP):
        desc = _desc(out_channels=out_c)
        bias = [rng.randint(-100_000, 100_000) for _ in range(out_c)]
        packed = pack_bias_for_hw(bias, desc, pe_rows)

        n_out_tiles = -(-out_c // pe_rows)
        assert len(packed) == n_out_tiles * pe_rows

        idx = 0
        for ot in range(n_out_tiles):
            for r in range(pe_rows):
                oc = ot * pe_rows + r
                expected = bias[oc] if oc < out_c else 0
                assert packed[idx] == expected
                idx += 1


def test_pack_scale_table_for_hw_format_padding_and_round_trip() -> None:
    """ISA v1.2 table image: 8 bytes per entry (`<i` multiplier, `B`
    shift, 3 zero bytes), padded to whole `pe_rows` tiles with the same
    `oc = ot*pe_rows + r` tiling as the bias image, and
    `unpack_scale_table_from_hw` inverts it."""
    rng = random.Random(0x5CA1)
    for out_c, pe_rows in itertools.product(_PACK_OUT_CHANNELS_SWEEP, _PACK_PE_ROWS_SWEEP):
        desc = _desc(out_channels=out_c)
        table = [(rng.randint(-(2**31), 2**31 - 1), rng.randint(0, 255)) for _ in range(out_c)]
        packed = pack_scale_table_for_hw(table, desc, pe_rows)

        n_out_tiles = -(-out_c // pe_rows)
        assert len(packed) == n_out_tiles * pe_rows * 8 == packed_scale_table_bytes(desc, pe_rows)
        assert len(packed) == packed_bias_count(desc, pe_rows) * cnn_accel_constants.SCALE_TABLE_ENTRY_BYTES

        for i in range(n_out_tiles * pe_rows):
            entry = packed[8 * i : 8 * i + 8]
            mult, shift = table[i] if i < out_c else (0, 0)
            assert entry[0:4] == struct.pack("<i", mult)
            assert entry[4] == shift
            assert entry[5:8] == bytes(3)

        assert unpack_scale_table_from_hw(packed, out_c) == table
        assert unpack_scale_table_from_hw(packed, n_out_tiles * pe_rows) == table + [(0, 0)] * (
            n_out_tiles * pe_rows - out_c
        )


def test_pack_scale_table_for_hw_rejects_bad_input() -> None:
    desc = _desc(out_channels=2)
    with pytest.raises(ValueError):
        pack_scale_table_for_hw([(1, 0)], desc, 8)  # too short
    with pytest.raises(ValueError):
        pack_scale_table_for_hw([(2**31, 0), (1, 0)], desc, 8)  # multiplier not int32
    with pytest.raises(ValueError):
        pack_scale_table_for_hw([(1, 256), (1, 0)], desc, 8)  # shift not uint8
    with pytest.raises(ValueError):
        pack_scale_table_for_hw([(1, 0), (1, 0)], desc, 0)
    good = pack_scale_table_for_hw([(1, 0), (1, 0)], desc, 8)
    with pytest.raises(ValueError):
        unpack_scale_table_from_hw(good[:8], 2)  # too short
    bad = bytearray(good)
    bad[5] = 1  # reserved byte must be zero
    with pytest.raises(ValueError):
        unpack_scale_table_from_hw(bytes(bad), 2)


# ---------------------------------------------------------------------------
# Hardware-order PE array simulation (the acceptance gate): proves the
# packed weight order (D10/D11) and the tiled dataflow (proposal §2/§3/
# §4) are mutually consistent BEFORE any RTL is written, by walking the
# packed image strictly sequentially -- exactly the order
# `weight_rd_addr` increments in `cnn_accel_weight_buffer.vhd` -- against
# an independently-simulated tiled window generator, and checking the
# int32-accumulated, bias/requant/relu'd result against conv2d() itself.
# ---------------------------------------------------------------------------


def _tap(
    input_values: list[int], row: int, col: int, channel: int, width: int, height: int, channels: int
) -> int:
    """Same zero-padding convention as `cnn_accel_model._at`, reimplemented
    independently here (this file avoids reaching into the model's
    private helpers, matching its own naive-reference-implementation
    convention elsewhere in this file)."""
    if row < 0 or row >= height or col < 0 or col >= width:
        return 0
    return input_values[(row * width + col) * channels + channel]


def _simulate_pe_array_tiled_dataflow(
    input_values: list[int],
    weights: list[int],
    bias: list[int],
    desc: LayerDesc,
    *,
    tile_channels: int,
    pe_rows: int,
) -> list[int]:
    """Simulates the RTL's own weight/window consumption order for
    `OPCODE_CONV2D`, ratified by D10:

    - Packs `weights` with `pack_weights_for_hw` (compile-time gather,
      never in hardware) and reads it back strictly sequentially, one
      `tile_channels * pe_rows`-lane row at a time -- mirroring
      `weight_rd_addr`, reset to 0 only when a new output-channel tile's
      sweep starts over a pixel, then incrementing once per `(t, kr,
      kc)` group (`cnn_accel_weight_buffer.vhd` addressing, proposal §4).
    - Independently generates the tiled window taps per `(t, kr, kc)`
      (mirrors `cnn_accel_window_gen`'s new tile loop, proposal §2),
      using the same zero-padding convention as the golden model
      (`_tap` above) and treating channels `>= in_channels` in a
      partial input tile as the zero activation D11 relies on.
    - Accumulates int32 per output lane across all `T` tiles of a pixel
      before ever calling `bias_requantize_relu` (mirrors the
      `first_tile`-clears/`last_tile`-commits accumulator carry, §3).

    Returns output in the same `(out_row, out_col, out_channel)` raster
    order as `conv2d()`, so a direct list equality checks the whole
    pipeline (packing order + tiled accumulation + quantization) end to
    end for every in-range output pixel/channel.
    """
    in_w, in_h, in_c = desc.in_width, desc.in_height, desc.in_channels
    out_c = desc.out_channels
    k_h, k_w = desc.kernel_h, desc.kernel_w
    s_h, s_w = desc.stride_h, desc.stride_w
    pad_top = desc.pad_top if desc.pad_en else 0
    pad_left = desc.pad_left if desc.pad_en else 0
    pad_bottom = desc.pad_bottom if desc.pad_en else 0
    pad_right = desc.pad_right if desc.pad_en else 0
    padded_h = in_h + pad_top + pad_bottom
    padded_w = in_w + pad_left + pad_right
    out_h = (padded_h - k_h) // s_h + 1
    out_w = (padded_w - k_w) // s_w + 1

    n_in_tiles = -(-in_c // tile_channels)  # ceil: T
    n_out_tiles = -(-out_c // pe_rows)  # ceil: OT
    groups_per_out_tile = n_in_tiles * k_h * k_w
    row_len = tile_channels * pe_rows

    packed = pack_weights_for_hw(weights, desc, tile_channels, pe_rows)
    assert len(packed) == n_out_tiles * groups_per_out_tile * row_len

    output = [0] * (out_h * out_w * out_c)
    for out_row in range(out_h):
        for out_col in range(out_w):
            base_row = out_row * s_h - pad_top
            base_col = out_col * s_w - pad_left
            for ot in range(n_out_tiles):
                acc = [0] * pe_rows  # first_tile: clear all pe_rows accumulators
                weight_rd_addr = ot * groups_per_out_tile
                for t in range(n_in_tiles):
                    for kr in range(k_h):
                        for kc in range(k_w):
                            row = packed[
                                weight_rd_addr * row_len : (weight_rd_addr + 1) * row_len
                            ]
                            weight_rd_addr += 1
                            for c in range(tile_channels):
                                ic = t * tile_channels + c
                                tap = (
                                    _tap(
                                        input_values,
                                        base_row + kr,
                                        base_col + kc,
                                        ic,
                                        in_w,
                                        in_h,
                                        in_c,
                                    )
                                    if ic < in_c
                                    else 0  # D11: partial-tile channel, zero activation
                                )
                                if tap == 0:
                                    continue  # zero tap contributes zero to every lane
                                for r in range(pe_rows):
                                    # lane = r*tile_channels + c (row-major, pinned to
                                    # cnn_accel_pe_array.vhd's own weight_lane indexing --
                                    # see pack_weights_for_hw's docstring).
                                    acc[r] += tap * row[r * tile_channels + c]
                # last_tile: commit accumulators to the quantized output.
                for r in range(pe_rows):
                    oc = ot * pe_rows + r
                    if oc >= out_c:
                        continue
                    bias_value = bias[oc] if desc.bias_en else 0
                    output[(out_row * out_w + out_col) * out_c + oc] = bias_requantize_relu(
                        acc[r],
                        bias_value,
                        bias_en=desc.bias_en,
                        requant_en=desc.requant_en,
                        relu_en=desc.relu_en,
                        requant_scale=desc.requant_scale,
                        requant_shift=desc.requant_shift,
                    )
    return output


# in_channels: 3 (target network's real layer-1 partial tile, D11), 8
# (exact fit, no padding at all), 20 (partial at every swept
# tile_channels). out_channels: 8 (exact at pe_rows=8), 12 (partial at
# pe_rows=8, exact at pe_rows=4). kernel 1x1 and 3x3. stride 1 and 2.
# padding on/off. tile_channels/pe_rows both the recommended (8, 8)
# default and a second (4, 4) pair to prove the simulation/packing is
# not accidentally hard-coded to one width.
_PE_SIM_IN_W = _PE_SIM_IN_H = 5  # smallest size giving a positive out_h/out_w for every combo below
_PE_SIM_SHAPES = list(
    itertools.product(
        [3, 8, 20],  # in_channels
        [8, 12],  # out_channels
        [1, 3],  # kernel
        [1, 2],  # stride
        [False, True],  # padding on/off
        [(8, 8), (4, 4)],  # (tile_channels, pe_rows)
    )
)


@pytest.mark.parametrize(
    "in_c, out_c, kernel, stride, pad_on, tile_pe",
    _PE_SIM_SHAPES,
    ids=[
        f"ic{ic}_oc{oc}_k{k}_s{s}_pad{int(p)}_t{tc}pe{pr}"
        for ic, oc, k, s, p, (tc, pr) in _PE_SIM_SHAPES
    ],
)
def test_pe_array_tiled_dataflow_matches_conv2d(in_c, out_c, kernel, stride, pad_on, tile_pe) -> None:
    """THE acceptance gate (per task): the hardware-order simulation
    must reproduce conv2d() exactly for every swept shape, BEFORE any
    RTL is written."""
    tile_channels, pe_rows = tile_pe
    seed = f"pe_sim-{in_c}-{out_c}-{kernel}-{stride}-{pad_on}-{tile_channels}-{pe_rows}"
    rng = random.Random(seed)

    in_w, in_h = _PE_SIM_IN_W, _PE_SIM_IN_H
    pad = 1 if pad_on else 0
    desc = LayerDesc(
        opcode=OPCODE_CONV2D,
        flags=(
            (1 << FLAG_BIAS_EN)
            | (1 << FLAG_RELU_EN)
            | (1 << FLAG_REQUANT_EN)
            | ((1 << FLAG_PAD_EN) if pad_on else 0)
        ),
        in_width=in_w, in_height=in_h, in_channels=in_c, out_channels=out_c,
        kernel_h=kernel, kernel_w=kernel, stride_h=stride, stride_w=stride,
        pad_top=pad, pad_bottom=pad, pad_left=pad, pad_right=pad,
        requant_scale=1 << 13, requant_shift=1,
    )
    input_values = [rng.randint(-128, 127) for _ in range(in_w * in_h * in_c)]
    weights = [rng.randint(-128, 127) for _ in range(out_c * kernel * kernel * in_c)]
    bias = [rng.randint(-1000, 1000) for _ in range(out_c)]

    golden = conv2d(input_values, weights, bias, desc)
    simulated = _simulate_pe_array_tiled_dataflow(
        input_values, weights, bias, desc, tile_channels=tile_channels, pe_rows=pe_rows
    )
    assert simulated == golden


# ---------------------------------------------------------------------------
# Frame budget (flow_status.md S5): the section-6 cycle model from
# doc/cnn_accel_sizing_proposal.md, as a real, currently-shortfall-aware
# test instead of a static markdown table -- `cnn_accel_constants.py` is
# the single source of truth for `CLOCK_HZ`/`TARGET_FPS`/`BACKBONE_TARGET`
# and the `cycles_per_frame()`/`frame_budget_cycles()` model.
# ---------------------------------------------------------------------------


def test_frame_budget_cycle_model_matches_sizing_proposal() -> None:
    """Pins `cycles_per_frame()` to the exact per-config totals in
    doc/cnn_accel_sizing_proposal.md section 3/4 at the 320x240 target, so
    a silent change to `BACKBONE_TARGET` or the cycle formula is caught."""
    assert cnn_accel_constants.cycles_per_frame(8) == 4_300_800
    assert cnn_accel_constants.cycles_per_frame(16) == 2_150_400
    assert cnn_accel_constants.frame_budget_cycles() == 2_500_000
    assert cnn_accel_constants.frame_budget_cycles(fps=30) == 5_000_000


def test_frame_budget_scaled_pe_rows_meets_60fps() -> None:
    """`PE_ROWS_SCALED` (16) is the decision S1-S7 ratified as the 60 fps,
    150 MHz point on the 320x240 target: this must have headroom, not
    just clear the bar."""
    cycles = cnn_accel_constants.cycles_per_frame(cnn_accel_constants.PE_ROWS_SCALED)
    budget = cnn_accel_constants.frame_budget_cycles()
    assert cycles <= budget, (
        f"pe_rows={cnn_accel_constants.PE_ROWS_SCALED}: {cycles} cycles/frame "
        f"exceeds the {budget}-cycle budget for {cnn_accel_constants.TARGET_FPS} fps "
        f"at {cnn_accel_constants.CLOCK_HZ} Hz by {cycles / budget:.3f}x"
    )
    headroom = 1 - cycles / budget
    assert headroom == pytest.approx(0.14, abs=0.01)


def test_frame_budget_default_pe_rows_misses_60fps_by_known_shortfall() -> None:
    """Documents, as a running assertion rather than prose, that the
    shipped default (`PE_ROWS`, 8 rows, ~30 fps) does NOT meet the 60 fps
    target -- doc/cnn_accel_sizing_proposal.md section 4's "1.72x over"
    verdict. If this ever starts passing, `PE_ROWS`'s default changed and
    flow_status.md's S1-S7 block needs updating, not this test."""
    cycles = cnn_accel_constants.cycles_per_frame(cnn_accel_constants.PE_ROWS)
    budget = cnn_accel_constants.frame_budget_cycles()
    shortfall = cycles / budget
    assert shortfall == pytest.approx(1.72, abs=0.01), (
        f"pe_rows={cnn_accel_constants.PE_ROWS}: {cycles} cycles/frame is "
        f"{shortfall:.3f}x the {budget}-cycle, "
        f"{cnn_accel_constants.TARGET_FPS} fps @ {cnn_accel_constants.CLOCK_HZ} Hz budget"
    )


def test_frame_budget_default_pe_rows_meets_30fps() -> None:
    """The shipped default does meet 30 fps at 150 MHz, per
    doc/cnn_accel_sizing_proposal.md section 4: "8x8 does meet 30 FPS"."""
    cycles = cnn_accel_constants.cycles_per_frame(cnn_accel_constants.PE_ROWS)
    budget = cnn_accel_constants.frame_budget_cycles(fps=30)
    assert cycles <= budget, (
        f"pe_rows={cnn_accel_constants.PE_ROWS}: {cycles} cycles/frame "
        f"exceeds the {budget}-cycle, 30 fps budget by {cycles / budget:.3f}x"
    )
