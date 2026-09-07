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
  -> (optional ReLU)` pipeline bit-exactly (convergent/round-to-even
  shift, per `hdl-modules` `math.truncate_round_signed`'s default).
- An ISA interpreter (`run_program`) executing a full instruction stream
  against a flat `bytearray` "DDR image", and a memory-image builder
  (`build_memory_image`) used by every testbench's VUnit `pre_config` to
  produce the `memory_image.csv`/`expected_output.csv` pair (byte-address,
  byte-value rows).

Tensor layout conventions (golden-model choice, not yet constrained by any
committed RTL microarchitecture -- `cnn_accel_pe_array`/
`cnn_accel_weight_buffer`'s exact internal addressing is deferred to their
own `vhdesign`; this is the DDR-resident byte layout the compiler/DMA
engines must agree on regardless of internal PE sequencing):

- Activations (input and output): row-major `(height, width, channels)`
  (HWC), one signed int8 byte per element, no padding between elements.
- Weights: row-major `(out_channels, kernel_h, kernel_w, in_channels)`
  (OHWI) for `CONV2D`/`FC`; `(channels, kernel_h, kernel_w)` for
  `DWCONV2D` (one filter per channel, no cross-channel dimension).
- Bias: one little-endian signed int32 per output channel, contiguous.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

from cnn_accel_constants import FLAGS, INSTR_WORD_BYTES, OPCODES, isa_field_offsets

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


def encode_instruction(desc: LayerDesc) -> bytes:
    """Encode one `LayerDesc` as the 64-byte ISA word."""
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
    )


# ---------------------------------------------------------------------------
# Fixed-point helpers, matching hdl-modules math.truncate_round_signed
# (default convergent_rounding=true) / math.saturate_signed bit-exactly.
# ---------------------------------------------------------------------------


def round_shift_right_signed(value: int, shift: int, convergent: bool = True) -> int:
    """Round `value / 2**shift` to the nearest integer. Ties round to even
    (`convergent=True`, the hdl-modules default) or towards +infinity
    (`convergent=False`). `shift <= 0` is a plain left-shift (no rounding
    needed)."""
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
) -> int:
    """`cnn_accel_bias_requant`'s per-lane reference function:
    `int32 accumulator -> (+ bias) -> (x requant_scale, Q15) ->
    (>> requant_shift, arithmetic, rounded) -> saturate to int8 ->
    (optional ReLU clamp at 0, applied before the int8 saturate)`.

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

    if relu_en:
        scaled = max(scaled, 0)
    return saturate_signed(scaled, 8)


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
                    )
                )
                idx += 1
    return output


def run_layer(memory: bytearray, desc: LayerDesc) -> None:
    """Execute one decoded instruction against `memory` (read input/
    weights/bias, write output), in place."""
    in_w, in_h, in_c = desc.in_width, desc.in_height, desc.in_channels

    if desc.opcode == OPCODE_FC and (
        desc.in_width != 1 or desc.in_height != 1 or desc.kernel_h != 1 or desc.kernel_w != 1
    ):
        # fc() silently rebuilds its own degenerate 1x1-spatial LayerDesc
        # internally; if the ORIGINAL desc (used here to slice memory) has
        # non-degenerate in_width/in_height/kernel_h/kernel_w, the input
        # byte count computed below (in_w*in_h*in_c) would not match what
        # fc()/conv2d() actually consume (in_c only), silently dropping
        # input bytes. Callers must emit the degenerate form explicitly.
        raise ValueError(
            "OPCODE_FC requires in_width=in_height=kernel_h=kernel_w=1 in the "
            f"encoded instruction itself, got in_width={desc.in_width} "
            f"in_height={desc.in_height} kernel_h={desc.kernel_h} kernel_w={desc.kernel_w}"
        )

    if desc.opcode in (OPCODE_CONV2D, OPCODE_FC, OPCODE_DWCONV2D):
        in_size = in_w * in_h * in_c
        input_bytes = memory[desc.in_addr : desc.in_addr + in_size]
        input_values = [b - 256 if b >= 128 else b for b in input_bytes]

        out_c = desc.out_channels
        if desc.opcode == OPCODE_DWCONV2D:
            weight_count = out_c * desc.kernel_h * desc.kernel_w
        else:
            weight_count = out_c * desc.kernel_h * desc.kernel_w * in_c
        weight_bytes = memory[desc.weight_addr : desc.weight_addr + weight_count]
        weights = [b - 256 if b >= 128 else b for b in weight_bytes]

        bias = [0] * out_c
        if desc.bias_en:
            bias_bytes = memory[desc.bias_addr : desc.bias_addr + 4 * out_c]
            bias = list(struct.unpack(f"<{out_c}i", bytes(bias_bytes)))

        if desc.opcode == OPCODE_CONV2D:
            output = conv2d(input_values, weights, bias, desc)
        elif desc.opcode == OPCODE_DWCONV2D:
            output = dwconv2d(input_values, weights, bias, desc)
        else:
            output = fc(input_values, weights, bias, desc)

    elif desc.opcode in (OPCODE_POOL_MAX, OPCODE_POOL_AVG):
        in_size = in_w * in_h * in_c
        input_bytes = memory[desc.in_addr : desc.in_addr + in_size]
        input_values = [b - 256 if b >= 128 else b for b in input_bytes]
        output = pool_max(input_values, desc) if desc.opcode == OPCODE_POOL_MAX else pool_avg(
            input_values, desc
        )

    else:
        raise ValueError(f"run_layer: unsupported opcode 0x{desc.opcode:02x}")

    out_bytes = bytes((v & 0xFF) for v in output)
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
