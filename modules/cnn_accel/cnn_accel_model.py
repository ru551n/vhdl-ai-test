"""Python golden/reference model for the `cnn_accel` CNN inference IP.

Single source of "expected behavior" truth for every `modules/cnn_accel/`
VUnit testbench, per `shared/Vunit.md` ("Python reference models (golden
models)") — same role `canny_model.py` plays for the Canny IP. Covers:

- The ISA byte encoding/decoding (`encode_instruction`/`decode_instruction`),
  which must agree byte-for-byte with `modules/cnn_accel/src/cnn_accel_pkg.vhd`
  and `doc/cnn_accel_arch.md`'s "Instruction Set (v1)" table.
- Per-opcode reference math (`conv2d`, `dwconv2d`, `pool_max`, `pool_avg`,
  `fc`) and the shared output-quantization step (`bias_requantize_relu`),
  matching `cnn_accel_bias_requant`'s documented
  `int32 -> (+bias) -> (x requant_scale) -> (>>requant_shift) -> saturate
  -> (optional ReLU)` pipeline bit-exactly (round-half-up shift, ties
  towards +infinity, matching TOSA `apply_scale_32`; HW milestone H0).
- An ISA interpreter (`run_program`) executing a full instruction stream
  against a flat `bytearray` "DDR image", and a memory-image builder
  (`build_memory_image`) used by every testbench's VUnit `pre_config` to
  produce the `memory_image.csv`/`expected_output.csv` pair (byte-address,
  byte-value rows).

Tensor layout conventions -- two layers, LOGICAL vs DDR, now that the DDR
byte image is ratified (activations: decision S6; weights/bias: decision
D10). `conv2d`/`dwconv2d`/`pool_max`/`pool_avg`/`fc` operate on the
LOGICAL layout only -- that is also what `generate_vectors.py` and the
VHDL testbenches feed them directly, unaffected by anything below.
`run_layer`/`run_program` are the only functions that speak the DDR
layout; they convert to/from LOGICAL at the `memory[]` boundary via
`pack_activation_planes`/`unpack_activation_planes` and
`pack_weights_for_hw`/`unpack_weights_from_hw`/`pack_bias_for_hw`.

- Activations, LOGICAL layout (what `conv2d`/`dwconv2d`/`pool_*`/`fc`
  take and return): row-major `(height, width, channels)` (HWC), one
  signed int8 byte per element, no padding between elements.
- Activations, DDR layout (decision S6 -- what `run_layer`/`run_program`
  actually read/write, every opcode's ifmap AND ofmap, FC included with
  its degenerate `1x1xC`): channel-tiled planes of `T =
  ACTIVATION_PLANE_CHANNELS` (8) channels each,
  `byte_offset = ((c_tile*height + y)*width + x)*T + t`,
  `c_tile = 0 .. ceil(channels/T)-1`, channels `>= channels` inside the
  last tile zero-padded. Total length always
  `ceil(channels/T)*T*width*height` bytes (`activation_bytes()`). See
  `pack_activation_planes`/`unpack_activation_planes` and
  `doc/cnn_accel_arch.md`'s "Off-chip activation layout (decision S6)".
- Weights, LOGICAL layout: row-major
  `(out_channels, kernel_h, kernel_w, in_channels)` (OHWI) for
  `CONV2D`/`FC`; `(channels, kernel_h, kernel_w)` for `DWCONV2D` (one
  filter per channel, no cross-channel dimension).
- Weights, DDR layout for `CONV2D`/`FC` (decision D10 -- what
  `run_layer` actually reads): exactly `pack_weights_for_hw`'s tile-major
  byte image (see that function's docstring), built with this module's
  own `TILE_CHANNELS`/`PE_ROWS` constants because that is what the
  currently loaded bitstream expects (the host reads `HW_INFO` to learn
  the real `PE_ROWS` before packing on a real system; this golden model
  has one fixed pair). `DWCONV2D`'s weight layout is UNRATIFIED (decision
  D2 -- the compiler rejects the opcode); `run_layer` reads it as the
  raw, un-tiled LOGICAL layout, unchanged -- only its activations moved
  to S6 planes.
- Bias, LOGICAL layout: one little-endian signed int32 per output
  channel, contiguous. Bias, DDR layout for `CONV2D`/`FC`: exactly
  `pack_bias_for_hw`'s `OT*PE_ROWS`-int32 image (`OT =
  ceil(out_channels/PE_ROWS)`, padded lanes zero); `run_layer` uses the
  first `out_channels` of it.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

from cnn_accel_constants import (
    ACTIVATION_PLANE_CHANNELS,
    FLAGS,
    INSTR_WORD_BYTES,
    OPCODES,
    PE_ROWS,
    TILE_CHANNELS,
    isa_field_offsets,
)

# ---------------------------------------------------------------------------
# ISA constants, derived from cnn_accel_constants.py -- the single Python
# source of truth also consumed by cnn_accel_isa_generator.py to generate
# modules/cnn_accel/src's regs_src/cnn_accel_isa_pkg.vhd (`use`d from
# cnn_accel_pkg.vhd). This encoder and that generated VHDL package agree
# byte-for-byte by construction (same source table), not by hand-matched
# comment, per doc/cnn_accel_arch.md "Instruction Set (v1)".
#
# OPCODE_*/FLAG_*/OFF_* names below are generated at import time from
# OPCODES/FLAGS/isa_field_offsets() so every existing usage in this file
# (and in test_cnn_accel_model.py) keeps working unchanged.
# ---------------------------------------------------------------------------

for _name, _value in OPCODES.items():
    globals()[f"OPCODE_{_name}"] = _value

for _name, _bit in FLAGS.items():
    globals()[f"FLAG_{_name}"] = _bit

for _field in isa_field_offsets():
    globals()[f"OFF_{_field.name.upper()}"] = _field.offset_bytes

del _name, _value, _bit, _field
OFF_NEXT_INSTR_ADDR = 48


@dataclass
class LayerDesc:
    """One decoded 64-byte instruction. Field names/types mirror
    `cnn_accel_pkg.layer_desc_t` exactly (unsigned fields as plain
    non-negative `int`, `requant_scale` as a signed `int`)."""

    opcode: int
    flags: int = 0
    in_addr: int = 0
    out_addr: int = 0
    weight_addr: int = 0
    bias_addr: int = 0
    in_width: int = 0
    in_height: int = 0
    in_channels: int = 0
    out_channels: int = 0
    kernel_h: int = 1
    kernel_w: int = 1
    stride_h: int = 1
    stride_w: int = 1
    pad_top: int = 0
    pad_bottom: int = 0
    pad_left: int = 0
    pad_right: int = 0
    requant_scale: int = 0
    requant_shift: int = 0
    pool_kernel_h: int = 1
    pool_kernel_w: int = 1
    pool_stride_h: int = 1
    pool_stride_w: int = 1
    next_instr_addr: int = 0
    # ISA v1.1 (H1) epilogue fields, W13. Defaults of 0 make a v1.0-style
    # `LayerDesc(...)` bit-identical to a v1.0 program (the bytes were
    # reserved-must-be-0 before).
    output_offset: int = 0
    clamp_min: int = 0
    clamp_max: int = 0

    @property
    def relu_en(self) -> bool:
        return bool((self.flags >> FLAG_RELU_EN) & 1)

    @property
    def clamp_en(self) -> bool:
        return bool((self.flags >> FLAG_CLAMP_EN) & 1)

    @property
    def bias_en(self) -> bool:
        return bool((self.flags >> FLAG_BIAS_EN) & 1)

    @property
    def requant_en(self) -> bool:
        return bool((self.flags >> FLAG_REQUANT_EN) & 1)

    @property
    def pad_en(self) -> bool:
        return bool((self.flags >> FLAG_PAD_EN) & 1)


def encode_instruction(desc: LayerDesc) -> bytes:
    """Encode one `LayerDesc` as the 64-byte ISA word.

    Rejects `clamp_min > clamp_max` (an empty clamp range has no defined
    HW result -- see `bias_requantize_relu`) and any W13 field outside
    its signed width, rather than silently wrapping: these are the only
    descriptor fields whose misuse cannot be caught by the RTL."""
    if desc.clamp_min > desc.clamp_max:
        raise ValueError(
            f"clamp_min ({desc.clamp_min}) > clamp_max ({desc.clamp_max}): empty clamp range"
        )
    for name, value, lo, hi in (
        ("output_offset", desc.output_offset, -(2**15), 2**15 - 1),
        ("clamp_min", desc.clamp_min, -128, 127),
        ("clamp_max", desc.clamp_max, -128, 127),
    ):
        if not lo <= value <= hi:
            raise ValueError(f"{name}={value} outside signed range [{lo}, {hi}]")

    buf = bytearray(INSTR_WORD_BYTES)
    buf[OFF_OPCODE] = desc.opcode & 0xFF
    buf[OFF_FLAGS] = desc.flags & 0xFF
    struct.pack_into("<I", buf, OFF_IN_ADDR, desc.in_addr & 0xFFFFFFFF)
    struct.pack_into("<I", buf, OFF_OUT_ADDR, desc.out_addr & 0xFFFFFFFF)
    struct.pack_into("<I", buf, OFF_WEIGHT_ADDR, desc.weight_addr & 0xFFFFFFFF)
    struct.pack_into("<I", buf, OFF_BIAS_ADDR, desc.bias_addr & 0xFFFFFFFF)
    struct.pack_into("<H", buf, OFF_IN_WIDTH, desc.in_width & 0xFFFF)
    struct.pack_into("<H", buf, OFF_IN_HEIGHT, desc.in_height & 0xFFFF)
    struct.pack_into("<H", buf, OFF_IN_CHANNELS, desc.in_channels & 0xFFFF)
    struct.pack_into("<H", buf, OFF_OUT_CHANNELS, desc.out_channels & 0xFFFF)
    buf[OFF_KERNEL_H] = desc.kernel_h & 0xFF
    buf[OFF_KERNEL_W] = desc.kernel_w & 0xFF
    buf[OFF_STRIDE_H] = desc.stride_h & 0xFF
    buf[OFF_STRIDE_W] = desc.stride_w & 0xFF
    buf[OFF_PAD_TOP] = desc.pad_top & 0xFF
    buf[OFF_PAD_BOTTOM] = desc.pad_bottom & 0xFF
    buf[OFF_PAD_LEFT] = desc.pad_left & 0xFF
    buf[OFF_PAD_RIGHT] = desc.pad_right & 0xFF
    struct.pack_into("<i", buf, OFF_REQUANT_SCALE, desc.requant_scale)
    buf[OFF_REQUANT_SHIFT] = desc.requant_shift & 0xFF
    buf[OFF_POOL_KERNEL_H] = desc.pool_kernel_h & 0xFF
    buf[OFF_POOL_KERNEL_W] = desc.pool_kernel_w & 0xFF
    buf[OFF_POOL_STRIDE_H] = desc.pool_stride_h & 0xFF
    buf[OFF_POOL_STRIDE_W] = desc.pool_stride_w & 0xFF
    struct.pack_into("<I", buf, OFF_NEXT_INSTR_ADDR, desc.next_instr_addr & 0xFFFFFFFF)
    struct.pack_into("<h", buf, OFF_OUTPUT_OFFSET, desc.output_offset)
    struct.pack_into("<b", buf, OFF_CLAMP_MIN, desc.clamp_min)
    struct.pack_into("<b", buf, OFF_CLAMP_MAX, desc.clamp_max)
    return bytes(buf)


def decode_instruction(data: bytes) -> LayerDesc:
    """Inverse of `encode_instruction`, for the encode/decode cross-check
    test and for `run_program`'s instruction fetch."""
    if len(data) != INSTR_WORD_BYTES:
        raise ValueError(f"expected {INSTR_WORD_BYTES} bytes, got {len(data)}")

    return LayerDesc(
        opcode=data[OFF_OPCODE],
        flags=data[OFF_FLAGS],
        in_addr=struct.unpack_from("<I", data, OFF_IN_ADDR)[0],
        out_addr=struct.unpack_from("<I", data, OFF_OUT_ADDR)[0],
        weight_addr=struct.unpack_from("<I", data, OFF_WEIGHT_ADDR)[0],
        bias_addr=struct.unpack_from("<I", data, OFF_BIAS_ADDR)[0],
        in_width=struct.unpack_from("<H", data, OFF_IN_WIDTH)[0],
        in_height=struct.unpack_from("<H", data, OFF_IN_HEIGHT)[0],
        in_channels=struct.unpack_from("<H", data, OFF_IN_CHANNELS)[0],
        out_channels=struct.unpack_from("<H", data, OFF_OUT_CHANNELS)[0],
        kernel_h=data[OFF_KERNEL_H],
        kernel_w=data[OFF_KERNEL_W],
        stride_h=data[OFF_STRIDE_H],
        stride_w=data[OFF_STRIDE_W],
        pad_top=data[OFF_PAD_TOP],
        pad_bottom=data[OFF_PAD_BOTTOM],
        pad_left=data[OFF_PAD_LEFT],
        pad_right=data[OFF_PAD_RIGHT],
        requant_scale=struct.unpack_from("<i", data, OFF_REQUANT_SCALE)[0],
        requant_shift=data[OFF_REQUANT_SHIFT],
        pool_kernel_h=data[OFF_POOL_KERNEL_H],
        pool_kernel_w=data[OFF_POOL_KERNEL_W],
        pool_stride_h=data[OFF_POOL_STRIDE_H],
        pool_stride_w=data[OFF_POOL_STRIDE_W],
        next_instr_addr=struct.unpack_from("<I", data, OFF_NEXT_INSTR_ADDR)[0],
        output_offset=struct.unpack_from("<h", data, OFF_OUTPUT_OFFSET)[0],
        clamp_min=struct.unpack_from("<b", data, OFF_CLAMP_MIN)[0],
        clamp_max=struct.unpack_from("<b", data, OFF_CLAMP_MAX)[0],
    )


# ---------------------------------------------------------------------------
# Fixed-point helpers, matching hdl-modules math.truncate_round_signed
# / math.saturate_signed bit-exactly (the rounding rule is round-half-up
# since H0, see round_shift_right_signed; math.truncate_round_signed's
# round-to-even is no longer what the RTL implements).
# ---------------------------------------------------------------------------


def round_shift_right_signed(value: int, shift: int, convergent: bool = False) -> int:
    """Round `value / 2**shift` to the nearest integer. Ties round towards
    +infinity (`convergent=False`, the default since HW milestone H0:
    `floor((value + 2**(shift-1)) / 2**shift)`, identical to TOSA's
    `apply_scale_32` SINGLE_ROUND so the TOSA compiler can emit bit-exact
    programs) or to even (`convergent=True`, the pre-H0 hdl-modules
    `truncate_round_signed` convention, kept for reference only). `shift <=
    0` is a plain left-shift (no rounding needed)."""
    if shift <= 0:
        return value << (-shift)

    divisor = 1 << shift
    quotient, remainder = divmod(value, divisor)  # floor division; 0 <= remainder < divisor
    twice_remainder = 2 * remainder

    if twice_remainder < divisor:
        return quotient
    if twice_remainder > divisor:
        return quotient + 1
    # Exact tie.
    if convergent:
        return quotient if quotient % 2 == 0 else quotient + 1
    return quotient + 1


def saturate_signed(value: int, result_width: int) -> int:
    """Clamp `value` to the signed range representable in `result_width`
    bits, matching `math.saturate_signed`."""
    min_value = -(2 ** (result_width - 1))
    max_value = 2 ** (result_width - 1) - 1
    return max(min_value, min(max_value, value))


INT32_MIN = -(2**31)
INT32_MAX = 2**31 - 1


class AccumulatorOverflow(ValueError):
    """Raised when a partial-sum value would leave the signed 32-bit
    range the RTL accumulator actually implements. The golden model uses
    unbounded Python ints for accumulation, so without this guard it
    would silently diverge from any real (wrapping) 32-bit RTL
    accumulator for programs whose true mathematical accumulation
    exceeds int32 -- this makes that divergence a loud failure instead
    of a silent bit-exactness mismatch."""


def _check_accumulator(
    value: int, *, opcode: int, out_channel: int, out_row: int, out_col: int
) -> int:
    """Enforce the int32 accumulator contract at an accumulation
    boundary; returns `value` unchanged so call sites can wrap an
    expression in-place."""
    if value < INT32_MIN or value > INT32_MAX:
        raise AccumulatorOverflow(
            f"accumulator overflow: opcode=0x{opcode:02x} out_channel={out_channel} "
            f"x={out_col} y={out_row} value={value} outside int32 range "
            f"[{INT32_MIN}, {INT32_MAX}]"
        )
    return value


def bias_requantize_relu(
    acc: int,
    bias: int,
    *,
    bias_en: bool,
    requant_en: bool,
    relu_en: bool,
    requant_scale: int,
    requant_shift: int,
    output_offset: int = 0,
    clamp_en: bool = False,
    clamp_min: int = -128,
    clamp_max: int = 127,
) -> int:
    """`cnn_accel_bias_requant`'s per-lane reference function:
    `int32 accumulator -> (+ bias) -> (x requant_scale, Q15) ->
    (>> requant_shift, arithmetic, rounded) -> (+ output_offset) ->
    clamp(lo, hi)`, where `(lo, hi) = (clamp_min, clamp_max)` when
    `clamp_en` and otherwise the legacy `(0 if relu_en else -128, 127)`
    -- i.e. the v1.0 `ReLU -> saturate to int8` (ISA v1.1, HW milestone
    H1, doc/tosa_compiler_plan.md section 5 extension 1).

    `output_offset` is added AFTER the rounded shift (so it is exact, not
    scaled), on an unbounded value, before any clamping: `s + offset`
    beyond int8 in either direction clamps to `hi`/`lo`. With
    `output_offset=0, clamp_en=False` the function is bit-identical to
    the v1.0 epilogue for every input (`relu_en` then selects `lo`;
    `clamp_min`/`clamp_max` are ignored). `clamp_min > clamp_max` is
    rejected by `encode_instruction`; here it is not checked, and the
    result follows the RTL's `min(max(s, lo), hi)` order (= `hi`).

    `requant_scale` is a signed Q15 fixed-point multiplier (i.e. the
    "mathematical" scale factor is `requant_scale / 2**15`); the combined
    shift removing both the Q15 fraction and `requant_shift` is applied in
    one rounding step, matching the RTL's documented option to "fold into
    one combined shift amount at vhdesign time".

    Ratified 33-bit bias-sum contract (D9), ratified against the RTL, not
    a model choice: `cnn_accel_bias_requant.vhd` declares
    `c_sum_width = g_accum_width + 1` (line 81, one guard bit on top of
    the 32-bit accumulator) specifically "so 'accum + bias' cannot
    overflow", and `total_l <= resize(accum_l, c_sum_width) +
    resize(bias_l, c_sum_width) ...` (line ~211) does the add at that
    width. Both operands are full int32 (`acc` is the MAC accumulator,
    range-checked to int32 by `_check_accumulator`/`AccumulatorOverflow`
    at every call site before it ever reaches this function; `bias` is
    the int32 value decoded from the bias buffer). int32 + int32 always
    fits exactly in 33 bits -- there is no width in this pipeline at
    which the bias add can wrap or need to saturate; the only saturating,
    lossy step anywhere in `bias_requantize_relu` is the final int8
    `saturate_signed` call below. Consequently this function's `total =
    acc + (bias if bias_en else 0)` uses plain unbounded Python-int
    addition deliberately (not a 33-bit or 32-bit masked/wrapped add) and
    that is already bit-exact with the RTL: any width narrower than 33
    bits (e.g. accidentally wrapping to int32 here) would silently
    diverge from the RTL for exactly the acc=INT32_MAX/bias=INT32_MAX
    (or INT32_MIN/INT32_MIN) corner, which is why that corner has a
    dedicated regression test below.

    When `requant_en=False`, per `cnn_accel_bias_requant_req.md`: the
    pipeline still applies bias/ReLU but passes the value through
    unscaled -- SATURATED to int8, identical semantics (and ordering:
    bias -> ReLU -> saturate) to the `requant_en=True` path, just without
    the scale/shift step. (Ratified fix: an earlier revision of this
    model wrapped/truncated to the low 8 bits here instead of saturating,
    which is a bit-exactness trap -- two different overflow semantics
    for what is otherwise the same pipeline.)
    """
    total = acc + (bias if bias_en else 0)

    if requant_en:
        scaled = round_shift_right_signed(total * requant_scale, 15 + requant_shift)
    else:
        scaled = total

    scaled += output_offset

    if clamp_en:
        lo, hi = clamp_min, clamp_max
    else:
        lo, hi = (0 if relu_en else -128), 127
    return min(max(scaled, lo), hi)


# ---------------------------------------------------------------------------
# Tensor helpers: flat, row-major int8 activation buffers as list[int],
# each element already in the signed -128..127 range.
# ---------------------------------------------------------------------------


def _at(
    values: list[int],
    row: int,
    col: int,
    channel: int,
    width: int,
    height: int,
    channels: int,
) -> int:
    """HWC-indexed read from a flat activation buffer, 0 outside
    `[0, width) x [0, height)` (the zero-padding convention -- callers pass
    already-shifted `row`/`col` so this only needs to bounds-check)."""
    if row < 0 or row >= height or col < 0 or col >= width:
        return 0
    return values[(row * width + col) * channels + channel]


def _conv2d_generic(
    input_values: list[int],
    weights: list[int],
    bias: list[int],
    desc: LayerDesc,
    *,
    depthwise: bool,
) -> list[int]:
    """Shared implementation for `conv2d`/`dwconv2d`/`fc`: int8 x int8 MAC
    over the padded, strided window, then `bias_requantize_relu` per
    output element. `weights` layout: OHWI for `depthwise=False`,
    `(channels, kernel_h, kernel_w)` for `depthwise=True`."""
    in_w, in_h, in_c = desc.in_width, desc.in_height, desc.in_channels
    out_c = desc.out_channels
    k_h, k_w = desc.kernel_h, desc.kernel_w
    s_h, s_w = desc.stride_h, desc.stride_w
    pad_top = desc.pad_top if desc.pad_en else 0
    pad_left = desc.pad_left if desc.pad_en else 0
    pad_bottom = desc.pad_bottom if desc.pad_en else 0
    pad_right = desc.pad_right if desc.pad_en else 0

    if s_h == 0 or s_w == 0:
        raise ValueError(f"conv: stride_h/stride_w must be nonzero, got ({s_h}, {s_w})")

    padded_h = in_h + pad_top + pad_bottom
    padded_w = in_w + pad_left + pad_right
    out_h = (padded_h - k_h) // s_h + 1
    out_w = (padded_w - k_w) // s_w + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError(
            f"conv: computed output dims must be positive, got out_h={out_h} out_w={out_w} "
            f"(padded {padded_h}x{padded_w}, kernel {k_h}x{k_w}, stride {s_h}x{s_w})"
        )

    output: list[int] = []
    for out_row in range(out_h):
        for out_col in range(out_w):
            base_row = out_row * s_h - pad_top
            base_col = out_col * s_w - pad_left
            for oc in range(out_c):
                acc = 0
                if depthwise:
                    for kr in range(k_h):
                        for kc in range(k_w):
                            tap = _at(
                                input_values, base_row + kr, base_col + kc, oc, in_w, in_h, in_c
                            )
                            w = weights[(oc * k_h + kr) * k_w + kc]
                            acc += tap * w
                else:
                    for kr in range(k_h):
                        for kc in range(k_w):
                            for ic in range(in_c):
                                tap = _at(
                                    input_values,
                                    base_row + kr,
                                    base_col + kc,
                                    ic,
                                    in_w,
                                    in_h,
                                    in_c,
                                )
                                w = weights[((oc * k_h + kr) * k_w + kc) * in_c + ic]
                                acc += tap * w

                _check_accumulator(
                    acc, opcode=desc.opcode, out_channel=oc, out_row=out_row, out_col=out_col
                )
                bias_value = bias[oc] if desc.bias_en else 0
                output.append(
                    bias_requantize_relu(
                        acc,
                        bias_value,
                        bias_en=desc.bias_en,
                        requant_en=desc.requant_en,
                        relu_en=desc.relu_en,
                        requant_scale=desc.requant_scale,
                        requant_shift=desc.requant_shift,
                        output_offset=desc.output_offset,
                        clamp_en=desc.clamp_en,
                        clamp_min=desc.clamp_min,
                        clamp_max=desc.clamp_max,
                    )
                )
    return output


def conv2d(input_values: list[int], weights: list[int], bias: list[int], desc: LayerDesc) -> list[int]:
    """`OPCODE_CONV2D` reference: full cross-channel convolution.
    Weights: OHWI, `out_channels * kernel_h * kernel_w * in_channels` int8s.
    """
    return _conv2d_generic(input_values, weights, bias, desc, depthwise=False)


def dwconv2d(input_values: list[int], weights: list[int], bias: list[int], desc: LayerDesc) -> list[int]:
    """`OPCODE_DWCONV2D` reference: one filter per channel, no
    cross-channel accumulation (`out_channels` must equal `in_channels`).
    Weights: `channels * kernel_h * kernel_w` int8s."""
    if desc.out_channels != desc.in_channels:
        raise ValueError("dwconv2d requires out_channels == in_channels")
    return _conv2d_generic(input_values, weights, bias, desc, depthwise=True)


def fc(input_values: list[int], weights: list[int], bias: list[int], desc: LayerDesc) -> list[int]:
    """`OPCODE_FC` reference: degenerate 1x1-spatial `CONV2D`
    (`in_width=in_height=kernel_h=kernel_w=1`), per
    `doc/cnn_accel_arch.md`'s ISA section."""
    fc_desc = LayerDesc(**{**desc.__dict__, "in_width": 1, "in_height": 1, "kernel_h": 1, "kernel_w": 1})
    return conv2d(input_values, weights, bias, fc_desc)


def activation_plane_count(channels: int) -> int:
    """Number of `T = ACTIVATION_PLANE_CHANNELS`-channel planes (decision
    S6) a `channels`-channel activation occupies in DDR:
    `ceil(channels / T)`."""
    if channels <= 0:
        raise ValueError(f"channels must be positive, got channels={channels}")
    return -(-channels // ACTIVATION_PLANE_CHANNELS)  # ceil


def activation_bytes(width: int, height: int, channels: int) -> int:
    """Total DDR byte length of a `width x height x channels` activation
    stored as decision-S6 channel-tiled planes:
    `ceil(channels/T) * T * width * height`, `T =
    ACTIVATION_PLANE_CHANNELS` -- the zero-padded channels in the last
    plane (`t >= channels - c_tile*T`) still occupy bytes, so this is
    always a multiple of `T * width * height` (decision D1's alignment
    argument)."""
    return activation_plane_count(channels) * ACTIVATION_PLANE_CHANNELS * width * height


def pack_activation_planes(values_hwc: list[int], width: int, height: int, channels: int) -> list[int]:
    """Repack a LOGICAL HWC activation (module docstring) into the
    decision-S6 DDR byte image: channel-tiled planes of `T =
    ACTIVATION_PLANE_CHANNELS` channels each,

        byte_offset = ((c_tile * height + y) * width + x) * T + t
        c_tile = 0 .. ceil(channels/T) - 1,  t = 0 .. T-1

    with `t >= channels - c_tile*T` (i.e. `c_tile*T + t >= channels`)
    zero-padded. Output length is always
    `activation_bytes(width, height, channels)`."""
    n_tiles = activation_plane_count(channels)
    packed = [0] * (n_tiles * ACTIVATION_PLANE_CHANNELS * width * height)
    for c_tile in range(n_tiles):
        for y in range(height):
            for x in range(width):
                base = ((c_tile * height + y) * width + x) * ACTIVATION_PLANE_CHANNELS
                for t in range(ACTIVATION_PLANE_CHANNELS):
                    c = c_tile * ACTIVATION_PLANE_CHANNELS + t
                    if c < channels:
                        packed[base + t] = values_hwc[(y * width + x) * channels + c]
    return packed


def unpack_activation_planes(values_planes: list[int], width: int, height: int, channels: int) -> list[int]:
    """Exact inverse of `pack_activation_planes`: reads a decision-S6
    channel-tiled-plane DDR image (`activation_bytes(width, height,
    channels)` elements) back into a LOGICAL HWC list
    (`width*height*channels` elements), dropping the zero-padding lanes
    (`c_tile*T + t >= channels`)."""
    n_tiles = activation_plane_count(channels)
    hwc = [0] * (width * height * channels)
    for c_tile in range(n_tiles):
        for y in range(height):
            for x in range(width):
                base = ((c_tile * height + y) * width + x) * ACTIVATION_PLANE_CHANNELS
                for t in range(ACTIVATION_PLANE_CHANNELS):
                    c = c_tile * ACTIVATION_PLANE_CHANNELS + t
                    if c < channels:
                        hwc[(y * width + x) * channels + c] = values_planes[base + t]
    return hwc


def pack_weights_for_hw(
    weights: list[int], desc: LayerDesc, tile_channels: int, pe_rows: int
) -> list[int]:
    """Compile-time repack of `CONV2D`/`FC` weights (logical OHWI, see the
    module docstring) into the accelerator-NATIVE byte image the RTL
    weight buffer/`pe_array` actually stream sequentially -- ratified
    D10 (`doc/cnn_accel_tiled_dataflow_proposal.md` §4): the PE array
    sweeps input channels TILE-MAJOR (`tile_idx` hoisted above
    `kr,kc`, §4/§2), which disagrees with OHWI's `ic` fastest-varying
    *inside* `kr,kc` -- so the gather happens once, here, in Python,
    never in hardware.

    LAYOUT (outermost to innermost; every leaf is one `int8` weight):

        for ot in 0 .. OT-1:                      # output-channel tile
          for t in 0 .. T-1:                       # input-channel tile
            for kr in 0 .. kernel_h-1:
              for kc in 0 .. kernel_w-1:
                for r in 0 .. pe_rows-1:          # output lane within output tile
                  for c in 0 .. tile_channels-1:      # channel within input tile
                    ic = t*tile_channels + c
                    oc = ot*pe_rows + r
                    yield weights[((oc*kernel_h+kr)*kernel_w+kc)*in_channels+ic]
                          if ic < in_channels and oc < out_channels else 0   # D11

    where `OT = ceil(out_channels/pe_rows)`, `T =
    ceil(in_channels/tile_channels)`. This is exactly the order
    `weight_rd_addr` walks in `cnn_accel_weight_buffer.vhd`: one flat
    row address, reset to 0 only on `first_tile` of a new pixel (§4),
    incrementing once per `(t, kr, kc)` group thereafter, wrapping to
    the next output-channel tile only when the whole ifmap is
    re-streamed (D6, §5) -- so within one `pe_array` pass over one
    output-channel tile, the row sweep is `t -> kr -> kc`, and each row
    holds `tile_channels * pe_rows` int8 lanes. Within a row, lane `r`
    outer / `c` inner (`lane = r*tile_channels + c`) is NOT a free
    convention picked here -- it is pinned bit-for-bit to
    `cnn_accel_pe_array.vhd`'s own `compute_partial_sums()`, which reads
    weight lane `weight_lane := r * g_pe_cols + c` (`cnn_accel_pe_array.
    vhd:258`; `g_pe_cols` is this function's `tile_channels`). The two
    sides used to disagree -- this function previously used
    `c*pe_rows + r` (`c` outer, `r` inner), the transpose of the RTL's
    own indexing, so every `r != c` weight landed on the wrong PE lane.
    The RTL is the side verified by a passing 68/68 VUnit regression and
    is the expensive side to change; the row layout itself is an
    arbitrary convention with no other consumer, so this function is the
    one that moved to match `cnn_accel_pe_array.vhd`, not the reverse.
    Pinned against silent re-drift by
    `test_pack_weights_for_hw_lane_matches_pe_array_rtl_indexing`
    (`test_cnn_accel_model.py`, decodes `(r, c)` from a lane index by the
    RTL's own formula and checks the packed value independently of this
    function's loop nesting) and, end to end through the real RTL, by
    `tb_cnn_accel_pe_array_from_vectors.vhd`.

    D11 (zero padding for partial tiles): whenever `in_channels` is not
    a multiple of `tile_channels` or `out_channels` is not a multiple
    of `pe_rows`, every lane whose `ic >= in_channels` or `oc >=
    out_channels` is written as weight `0` -- never omitted, never
    garbage. This is what lets the hardware skip masking logic entirely
    (garbage activation x zero weight = 0 accumulates as a no-op).
    Layer 1 of the target network (`in_channels=3`, recommended
    `tile_channels=8`) is exactly this case.

    Total length is always exactly
    `OT * T * kernel_h * kernel_w * tile_channels * pe_rows`
    int8 values, regardless of padding.

    Only `OPCODE_CONV2D`/`OPCODE_FC`'s OHWI weight layout is in scope
    (per the ratifying proposal's own scoping, §4/§8 risk 1);
    `OPCODE_DWCONV2D`'s `(channels, kernel_h, kernel_w)` layout has no
    cross-channel reduction to gather -- each output channel already
    depends on exactly one input channel -- so it needs no repacking
    and this function rejects it rather than silently producing a
    layout no `vhdesign` has ratified.
    """
    if desc.opcode not in (OPCODE_CONV2D, OPCODE_FC):
        raise ValueError(
            "pack_weights_for_hw only supports OPCODE_CONV2D/OPCODE_FC's OHWI "
            f"weight layout (ratified D10 scope), got opcode=0x{desc.opcode:02x}"
        )
    if tile_channels <= 0 or pe_rows <= 0:
        raise ValueError(
            f"tile_channels and pe_rows must be positive, got tile_channels={tile_channels} "
            f"pe_rows={pe_rows}"
        )
    in_c, out_c = desc.in_channels, desc.out_channels
    k_h, k_w = desc.kernel_h, desc.kernel_w
    n_in_tiles = -(-in_c // tile_channels)  # ceil(in_c / tile_channels)
    n_out_tiles = -(-out_c // pe_rows)  # ceil(out_c / pe_rows)

    packed: list[int] = []
    for ot in range(n_out_tiles):
        for t in range(n_in_tiles):
            for kr in range(k_h):
                for kc in range(k_w):
                    for r in range(pe_rows):
                        oc = ot * pe_rows + r
                        for c in range(tile_channels):
                            ic = t * tile_channels + c
                            if ic < in_c and oc < out_c:
                                packed.append(
                                    weights[((oc * k_h + kr) * k_w + kc) * in_c + ic]
                                )
                            else:
                                packed.append(0)
    return packed


def packed_weight_count(desc: LayerDesc, tile_channels: int, pe_rows: int) -> int:
    """Element count of `pack_weights_for_hw`'s output, computable before
    packing (so `run_layer` can slice `memory[]` without materializing
    the image): `OT * T * kernel_h * kernel_w * tile_channels * pe_rows`,
    `OT = ceil(out_channels/pe_rows)`, `T = ceil(in_channels/
    tile_channels)` -- see that function's docstring for the layout."""
    if tile_channels <= 0 or pe_rows <= 0:
        raise ValueError(
            f"tile_channels and pe_rows must be positive, got tile_channels={tile_channels} "
            f"pe_rows={pe_rows}"
        )
    n_in_tiles = -(-desc.in_channels // tile_channels)  # ceil
    n_out_tiles = -(-desc.out_channels // pe_rows)  # ceil
    return n_out_tiles * n_in_tiles * desc.kernel_h * desc.kernel_w * tile_channels * pe_rows


def unpack_weights_from_hw(
    packed: list[int], desc: LayerDesc, tile_channels: int, pe_rows: int
) -> list[int]:
    """Exact inverse of `pack_weights_for_hw` on the valid (non-padded)
    region: reconstructs the LOGICAL OHWI weight array
    (`out_channels*kernel_h*kernel_w*in_channels` elements,
    decision D10) that `conv2d`/`fc` need from the tile-major DDR byte
    image `run_layer` actually reads. Padded lanes in `packed`
    (`ic >= in_channels` or `oc >= out_channels`, decision D11) are
    dropped, not asserted zero -- `pack_weights_for_hw`'s own tests cover
    that invariant on the packing side.

    `len(packed)` must be exactly `packed_weight_count(desc,
    tile_channels, pe_rows)`; only `OPCODE_CONV2D`/`OPCODE_FC` are in
    scope (same D10 scoping as `pack_weights_for_hw`)."""
    if desc.opcode not in (OPCODE_CONV2D, OPCODE_FC):
        raise ValueError(
            "unpack_weights_from_hw only supports OPCODE_CONV2D/OPCODE_FC's OHWI "
            f"weight layout (ratified D10 scope), got opcode=0x{desc.opcode:02x}"
        )
    if tile_channels <= 0 or pe_rows <= 0:
        raise ValueError(
            f"tile_channels and pe_rows must be positive, got tile_channels={tile_channels} "
            f"pe_rows={pe_rows}"
        )
    in_c, out_c = desc.in_channels, desc.out_channels
    k_h, k_w = desc.kernel_h, desc.kernel_w
    n_in_tiles = -(-in_c // tile_channels)  # ceil
    n_out_tiles = -(-out_c // pe_rows)  # ceil

    ohwi = [0] * (out_c * k_h * k_w * in_c)
    idx = 0
    for ot in range(n_out_tiles):
        for t in range(n_in_tiles):
            for kr in range(k_h):
                for kc in range(k_w):
                    for r in range(pe_rows):
                        oc = ot * pe_rows + r
                        for c in range(tile_channels):
                            ic = t * tile_channels + c
                            if ic < in_c and oc < out_c:
                                ohwi[((oc * k_h + kr) * k_w + kc) * in_c + ic] = packed[idx]
                            idx += 1
    return ohwi


def pack_bias_for_hw(bias: list[int], desc: LayerDesc, pe_rows: int) -> list[int]:
    """Zero-pad `bias` (one int32 per output channel, see the module
    docstring) to a whole number of `pe_rows`-wide output-channel tiles,
    matching `pack_weights_for_hw`'s `ot -> r` (`oc = ot*pe_rows + r`)
    tiling exactly -- `cnn_accel_bias_requant`'s lane `l` reads bias row
    `ot` lane `l`, so the padded lanes (`oc >= out_channels`) must be
    present (as `0`) for the addressing to line up, even though those
    lanes' MAC results are already `0` from D11's zero weight padding.

    Total length is always exactly `OT * pe_rows` int32 values, where
    `OT = ceil(out_channels / pe_rows)`.
    """
    if pe_rows <= 0:
        raise ValueError(f"pe_rows must be positive, got pe_rows={pe_rows}")
    out_c = desc.out_channels
    n_out_tiles = -(-out_c // pe_rows)  # ceil(out_c / pe_rows)
    packed: list[int] = []
    for ot in range(n_out_tiles):
        for r in range(pe_rows):
            oc = ot * pe_rows + r
            packed.append(bias[oc] if oc < out_c else 0)
    return packed


def packed_bias_count(desc: LayerDesc, pe_rows: int) -> int:
    """Element count of `pack_bias_for_hw`'s output, computable before
    packing: `OT * pe_rows` int32s, `OT = ceil(out_channels / pe_rows)`."""
    if pe_rows <= 0:
        raise ValueError(f"pe_rows must be positive, got pe_rows={pe_rows}")
    n_out_tiles = -(-desc.out_channels // pe_rows)  # ceil
    return n_out_tiles * pe_rows


def _pool_windows(input_values: list[int], desc: LayerDesc) -> list[list[int]]:
    """Per-channel pooling windows in output-raster order, one list of
    `pool_kernel_h * pool_kernel_w` int8 taps per (output position,
    channel).

    LIMITATION (v1): zero-padding is NOT modeled for pooling -- the v1
    ISA has no pooling-specific padding fields, and `pad_en` is only
    meaningful for `CONV2D`/`DWCONV2D`/`FC`. If `desc.pad_en` is set for
    a pooling instruction this is a program-authoring error (the bit
    would be silently ignored otherwise) and raises `ValueError`."""
    if desc.pad_en:
        raise ValueError(
            "pooling has no padding support in v1 (pad_en must be 0 for "
            "POOL_MAX/POOL_AVG)"
        )
    in_w, in_h, channels = desc.in_width, desc.in_height, desc.in_channels
    k_h, k_w = desc.pool_kernel_h, desc.pool_kernel_w
    s_h, s_w = desc.pool_stride_h, desc.pool_stride_w
    if s_h == 0 or s_w == 0:
        raise ValueError(f"pool: pool_stride_h/pool_stride_w must be nonzero, got ({s_h}, {s_w})")
    out_h = (in_h - k_h) // s_h + 1
    out_w = (in_w - k_w) // s_w + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError(
            f"pool: computed output dims must be positive, got out_h={out_h} out_w={out_w} "
            f"(input {in_h}x{in_w}, kernel {k_h}x{k_w}, stride {s_h}x{s_w})"
        )

    windows: list[list[int]] = []
    for out_row in range(out_h):
        for out_col in range(out_w):
            base_row = out_row * s_h
            base_col = out_col * s_w
            for ch in range(channels):
                taps = [
                    _at(input_values, base_row + kr, base_col + kc, ch, in_w, in_h, channels)
                    for kr in range(k_h)
                    for kc in range(k_w)
                ]
                windows.append(taps)
    return windows


def pool_max(input_values: list[int], desc: LayerDesc) -> list[int]:
    """`OPCODE_POOL_MAX` reference: int8 max over each window, already a
    valid int8 result (no `bias_requantize_relu` step -- bypasses
    `cnn_accel_bias_requant` in the RTL too)."""
    return [max(taps) for taps in _pool_windows(input_values, desc)]


def pool_avg(input_values: list[int], desc: LayerDesc) -> list[int]:
    """`OPCODE_POOL_AVG` reference: int32 sum over each window, then
    `bias_requantize_relu` with `bias_en=False` (division by the pool area
    is `requant_scale`/`requant_shift`, per
    `cnn_accel_pool_req.md`/`doc/cnn_accel_arch.md`)."""
    windows = _pool_windows(input_values, desc)
    channels = desc.in_channels
    # windows is in (out_row, out_col, channel) raster order, see _pool_windows.
    out_h = (desc.in_height - desc.pool_kernel_h) // desc.pool_stride_h + 1
    out_w = (desc.in_width - desc.pool_kernel_w) // desc.pool_stride_w + 1
    output: list[int] = []
    idx = 0
    for out_row in range(out_h):
        for out_col in range(out_w):
            for _ch in range(channels):
                acc_sum = sum(windows[idx])
                _check_accumulator(
                    acc_sum,
                    opcode=desc.opcode,
                    out_channel=_ch,
                    out_row=out_row,
                    out_col=out_col,
                )
                output.append(
                    bias_requantize_relu(
                        acc_sum,
                        0,
                        bias_en=False,
                        requant_en=desc.requant_en,
                        relu_en=desc.relu_en,
                        requant_scale=desc.requant_scale,
                        requant_shift=desc.requant_shift,
                        output_offset=desc.output_offset,
                        clamp_en=desc.clamp_en,
                        clamp_min=desc.clamp_min,
                        clamp_max=desc.clamp_max,
                    )
                )
                idx += 1
    return output


def _conv_output_dims(desc: LayerDesc) -> tuple[int, int]:
    """`(out_width, out_height)` for `CONV2D`/`DWCONV2D`/`FC`, duplicating
    `_conv2d_generic`'s own formula (kept separate rather than factored
    out so `conv2d`/`dwconv2d`/`fc` stay untouched by the S6 DDR-layout
    plumbing) -- needed by `run_layer` to know the output activation's
    spatial shape before it can call `pack_activation_planes`."""
    pad_top = desc.pad_top if desc.pad_en else 0
    pad_left = desc.pad_left if desc.pad_en else 0
    pad_bottom = desc.pad_bottom if desc.pad_en else 0
    pad_right = desc.pad_right if desc.pad_en else 0
    padded_h = desc.in_height + pad_top + pad_bottom
    padded_w = desc.in_width + pad_left + pad_right
    out_h = (padded_h - desc.kernel_h) // desc.stride_h + 1
    out_w = (padded_w - desc.kernel_w) // desc.stride_w + 1
    return out_w, out_h


def _pool_output_dims(desc: LayerDesc) -> tuple[int, int]:
    """`(out_width, out_height)` for `POOL_MAX`/`POOL_AVG`, duplicating
    `_pool_windows`'s own formula for the same reason as
    `_conv_output_dims`."""
    out_h = (desc.in_height - desc.pool_kernel_h) // desc.pool_stride_h + 1
    out_w = (desc.in_width - desc.pool_kernel_w) // desc.pool_stride_w + 1
    return out_w, out_h


def run_layer(memory: bytearray, desc: LayerDesc) -> None:
    """Execute one decoded instruction against `memory` (read input/
    weights/bias, write output), in place, through the ratified DDR byte
    layout (module docstring):

    - ifmap/ofmap (every opcode, including FC's degenerate `1x1xC`):
      decision-S6 channel-tiled planes via `activation_bytes`/
      `pack_activation_planes`/`unpack_activation_planes`.
    - `CONV2D`/`FC` weights: exactly `pack_weights_for_hw`'s byte image
      (`packed_weight_count`/`unpack_weights_from_hw`), using this
      module's `TILE_CHANNELS`/`PE_ROWS` constants -- the layout the
      currently loaded bitstream expects (a real host learns these from
      `HW_INFO` before packing; this golden model has one fixed pair).
    - `DWCONV2D` weights: UNRATIFIED (decision D2 -- the compiler rejects
      the opcode) -- read as the raw, un-tiled `(channels, kernel_h,
      kernel_w)` layout unchanged; only its activations moved to S6
      planes.
    - `CONV2D`/`FC` bias: exactly `pack_bias_for_hw`'s byte image
      (`packed_bias_count`), first `out_channels` int32s used.
    """
    in_w, in_h, in_c = desc.in_width, desc.in_height, desc.in_channels

    if desc.opcode == OPCODE_FC and (
        desc.in_width != 1 or desc.in_height != 1 or desc.kernel_h != 1 or desc.kernel_w != 1
    ):
        # fc() silently rebuilds its own degenerate 1x1-spatial LayerDesc
        # internally; if the ORIGINAL desc (used here to slice memory) has
        # non-degenerate in_width/in_height/kernel_h/kernel_w, the input
        # byte count computed below (activation_bytes(in_w, in_h, in_c))
        # would not match what fc()/conv2d() actually consume (in_c only),
        # silently dropping input bytes. Callers must emit the degenerate
        # form explicitly.
        raise ValueError(
            "OPCODE_FC requires in_width=in_height=kernel_h=kernel_w=1 in the "
            f"encoded instruction itself, got in_width={desc.in_width} "
            f"in_height={desc.in_height} kernel_h={desc.kernel_h} kernel_w={desc.kernel_w}"
        )

    if desc.opcode in (OPCODE_CONV2D, OPCODE_FC, OPCODE_DWCONV2D):
        in_size = activation_bytes(in_w, in_h, in_c)
        input_bytes = memory[desc.in_addr : desc.in_addr + in_size]
        input_planes = [b - 256 if b >= 128 else b for b in input_bytes]
        input_values = unpack_activation_planes(input_planes, in_w, in_h, in_c)

        out_c = desc.out_channels
        if desc.opcode == OPCODE_DWCONV2D:
            # D2: DWCONV2D has no ratified weight layout -- raw, un-tiled
            # (channels, kernel_h, kernel_w), unchanged from before S6.
            weight_count = out_c * desc.kernel_h * desc.kernel_w
            weight_bytes = memory[desc.weight_addr : desc.weight_addr + weight_count]
            weights = [b - 256 if b >= 128 else b for b in weight_bytes]
        else:
            weight_count = packed_weight_count(desc, TILE_CHANNELS, PE_ROWS)
            weight_bytes = memory[desc.weight_addr : desc.weight_addr + weight_count]
            packed_weights = [b - 256 if b >= 128 else b for b in weight_bytes]
            weights = unpack_weights_from_hw(packed_weights, desc, TILE_CHANNELS, PE_ROWS)

        bias = [0] * out_c
        if desc.bias_en:
            bias_count = packed_bias_count(desc, PE_ROWS)
            bias_bytes = memory[desc.bias_addr : desc.bias_addr + 4 * bias_count]
            packed_bias_values = struct.unpack(f"<{bias_count}i", bytes(bias_bytes))
            bias = list(packed_bias_values[:out_c])

        if desc.opcode == OPCODE_CONV2D:
            output = conv2d(input_values, weights, bias, desc)
        elif desc.opcode == OPCODE_DWCONV2D:
            output = dwconv2d(input_values, weights, bias, desc)
        else:
            output = fc(input_values, weights, bias, desc)
        out_w, out_h = _conv_output_dims(desc)

    elif desc.opcode in (OPCODE_POOL_MAX, OPCODE_POOL_AVG):
        in_size = activation_bytes(in_w, in_h, in_c)
        input_bytes = memory[desc.in_addr : desc.in_addr + in_size]
        input_planes = [b - 256 if b >= 128 else b for b in input_bytes]
        input_values = unpack_activation_planes(input_planes, in_w, in_h, in_c)
        output = pool_max(input_values, desc) if desc.opcode == OPCODE_POOL_MAX else pool_avg(
            input_values, desc
        )
        out_c = in_c  # pooling is channel-preserving
        out_w, out_h = _pool_output_dims(desc)

    else:
        raise ValueError(f"run_layer: unsupported opcode 0x{desc.opcode:02x}")

    output_planes = pack_activation_planes(output, out_w, out_h, out_c)
    out_bytes = bytes((v & 0xFF) for v in output_planes)
    memory[desc.out_addr : desc.out_addr + len(out_bytes)] = out_bytes


def run_program(memory: bytearray, program_addr: int, *, max_instructions: int = 1000) -> int:
    """ISA interpreter: fetch-decode-execute from `program_addr` until
    `OPCODE_HALT`, following each instruction's own `next_instr_addr`.
    Returns the number of non-HALT instructions executed. Raises if
    `max_instructions` is exceeded (guards against a malformed
    `next_instr_addr` loop in test authoring, not part of the RTL's own
    semantics -- the RTL has no such cap)."""
    pc = program_addr
    executed = 0
    for _ in range(max_instructions + 1):
        instr_bytes = bytes(memory[pc : pc + INSTR_WORD_BYTES])
        desc = decode_instruction(instr_bytes)
        if desc.opcode == OPCODE_HALT:
            return executed
        run_layer(memory, desc)
        executed += 1
        pc = desc.next_instr_addr
    raise RuntimeError(
        f"run_program: exceeded max_instructions={max_instructions} without HALT "
        "(check next_instr_addr chain)"
    )


def encode_program(descs: list[LayerDesc], program_addr: int = 0) -> bytes:
    """Encode a straight-line instruction list, auto-filling
    `next_instr_addr = program_addr + (i+1)*64` for any non-`HALT`
    instruction that left it at the dataclass default (0); explicit
    `next_instr_addr` values (e.g. for branching test cases) are left
    untouched."""
    out = bytearray()
    for i, desc in enumerate(descs):
        if desc.opcode != OPCODE_HALT and desc.next_instr_addr == 0:
            desc.next_instr_addr = program_addr + (i + 1) * INSTR_WORD_BYTES
        out += encode_instruction(desc)
    return bytes(out)


def build_memory_image(chunks: dict[int, bytes], size: int | None = None) -> bytearray:
    """Lay out `{byte_address: byte_data}` chunks (instructions, weights,
    bias, input activations, pre-zeroed output region, ...) into one flat
    `bytearray`, zero-filled elsewhere. `size` defaults to the highest
    touched address + 1."""
    if size is None:
        size = max((addr + len(data) for addr, data in chunks.items()), default=0)
    image = bytearray(size)
    for addr, data in chunks.items():
        image[addr : addr + len(data)] = data
    return image


if __name__ == "__main__":
    # Encode/decode cross-check: every field round-trips exactly, for a
    # descriptor that exercises every field (including a negative
    # requant_scale, since it is the one signed field).
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
    )
    round_trip = decode_instruction(encode_instruction(desc))
    assert round_trip == desc, f"encode/decode mismatch:\n{desc}\nvs\n{round_trip}"
    print("PASS: encode_instruction/decode_instruction round-trip")
