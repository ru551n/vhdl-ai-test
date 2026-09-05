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
  byte-value rows, mirroring `canny_model.py`'s `_write_csv` idiom).

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

# ---------------------------------------------------------------------------
# ISA constants -- must match modules/cnn_accel/src/cnn_accel_pkg.vhd and
# doc/cnn_accel_arch.md "Instruction Set (v1)" byte-for-byte.
# ---------------------------------------------------------------------------

OPCODE_HALT = 0x00
OPCODE_CONV2D = 0x01
OPCODE_DWCONV2D = 0x02
OPCODE_POOL_MAX = 0x03
OPCODE_POOL_AVG = 0x04
OPCODE_FC = 0x05

FLAG_RELU_EN = 0
FLAG_BIAS_EN = 1
FLAG_REQUANT_EN = 2
FLAG_PAD_EN = 3

INSTR_WORD_BYTES = 64

OFF_OPCODE = 0
OFF_FLAGS = 1
OFF_IN_ADDR = 4
OFF_OUT_ADDR = 8
OFF_WEIGHT_ADDR = 12
OFF_BIAS_ADDR = 16
OFF_IN_WIDTH = 20
OFF_IN_HEIGHT = 22
OFF_IN_CHANNELS = 24
OFF_OUT_CHANNELS = 26
OFF_KERNEL_H = 28
OFF_KERNEL_W = 29
OFF_STRIDE_H = 30
OFF_STRIDE_W = 31
OFF_PAD_TOP = 32
OFF_PAD_BOTTOM = 33
OFF_PAD_LEFT = 34
OFF_PAD_RIGHT = 35
OFF_REQUANT_SCALE = 36
OFF_REQUANT_SHIFT = 40
OFF_POOL_KERNEL_H = 44
OFF_POOL_KERNEL_W = 45
OFF_POOL_STRIDE_H = 46
OFF_POOL_STRIDE_W = 47
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

    When `requant_en=False`, per `cnn_accel_bias_requant_req.md`: "the
    pipeline still applies bias/ReLU but passes the low 8 bits through
    unscaled (debug/bypass path, not expected in normal compiled
    programs)" -- i.e. two's-complement truncation to 8 bits, not
    saturation.
    """
    total = acc + (bias if bias_en else 0)

    if requant_en:
        scaled = round_shift_right_signed(total * requant_scale, 15 + requant_shift)
        if relu_en:
            scaled = max(scaled, 0)
        return saturate_signed(scaled, 8)

    # Bypass path: bias/ReLU still applied, no scaling; low 8 bits pass through.
    if relu_en:
        total = max(total, 0)
    value = total & 0xFF
    return value - 256 if value >= 128 else value


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

    padded_h = in_h + pad_top + pad_bottom
    padded_w = in_w + pad_left + pad_right
    out_h = (padded_h - k_h) // s_h + 1
    out_w = (padded_w - k_w) // s_w + 1

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


def _pool_windows(input_values: list[int], desc: LayerDesc) -> list[list[int]]:
    """Per-channel pooling windows in output-raster order, one list of
    `pool_kernel_h * pool_kernel_w` int8 taps per (output position,
    channel). Zero-padding is not modeled for pooling (v1 ISA has no
    pooling-specific padding fields; `pad_en` only applies to
    `CONV2D`/`DWCONV2D`/`FC`)."""
    in_w, in_h, channels = desc.in_width, desc.in_height, desc.in_channels
    k_h, k_w = desc.pool_kernel_h, desc.pool_kernel_w
    s_h, s_w = desc.pool_stride_h, desc.pool_stride_w
    out_h = (in_h - k_h) // s_h + 1
    out_w = (in_w - k_w) // s_w + 1

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
    return [
        bias_requantize_relu(
            sum(taps),
            0,
            bias_en=False,
            requant_en=desc.requant_en,
            relu_en=desc.relu_en,
            requant_scale=desc.requant_scale,
            requant_shift=desc.requant_shift,
        )
        for taps in _pool_windows(input_values, desc)
    ]


def run_layer(memory: bytearray, desc: LayerDesc) -> None:
    """Execute one decoded instruction against `memory` (read input/
    weights/bias, write output), in place."""
    in_w, in_h, in_c = desc.in_width, desc.in_height, desc.in_channels

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
