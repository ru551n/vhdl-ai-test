"""Deterministic test-vector export for `cnn_accel_model.py`.

Writes `<out_dir>/<case>/{desc,input,weights,bias,expected}.txt` -- plain
ASCII, one signed decimal integer per line (no headers), directly readable
by VHDL `std.textio` integer reads; file formats in
`doc/cnn_accel_test_vectors.md`. Intended as a second, RTL-testbench-facing
consumer of the same golden model already covered by
`test_cnn_accel_model.py`; every value here comes straight from
`cnn_accel_model.py`'s public `conv2d`/`dwconv2d`, so it is bit-exact with
what the VUnit testbenches must reproduce.

Nothing written here is ever checked in: module_cnn_accel.py's VUnit
`pre_config` hooks call `generate_conv_core_cases` /
`generate_pe_array_xlang_case` straight into each test config's own
`output_path` (at that config's `pe_rows`) right before simulation. The
`__main__` entry (`python generate_vectors.py <out_dir> [pe_rows]`) is for
manual inspection only.
"""

from __future__ import annotations

import dataclasses
import random
from pathlib import Path
from typing import NamedTuple

import cnn_accel_constants
from cnn_accel_model import (
    FLAG_BIAS_EN,
    FLAG_CLAMP_EN,
    FLAG_PAD_EN,
    FLAG_RELU_EN,
    FLAG_REQUANT_EN,
    OPCODE_CONV2D,
    OPCODE_DWCONV2D,
    OPCODE_FC,
    OPCODE_POOL_AVG,
    OPCODE_POOL_MAX,
    LayerDesc,
    bias_requantize_relu,
    conv2d,
    dwconv2d,
    fc,
    pack_weights_for_hw,
    pool_avg,
    pool_max,
)

class HwPacking(NamedTuple):
    """One vector root's accelerator-native packing point (D10,
    doc/cnn_accel_tiled_dataflow_proposal.md section 4): the tile/lane
    widths every `weights_packed.txt` under `vectors_dir` is built with.
    `tile_channels` is fixed forever (S-decisions in flow_status.md);
    `pe_rows` is THE single scaling knob, so there is one root per legal
    value of it."""

    vectors_dir: Path
    tile_channels: int
    pe_rows: int


def hw_packing(vectors_dir: Path, *, pe_rows: int = cnn_accel_constants.PE_ROWS) -> HwPacking:
    """Packing point for one vector root. `pe_rows` must be one of
    `cnn_accel_constants.PE_ROWS_LEGAL`; `tile_channels` is always the
    fixed `TILE_CHANNELS` -- both from the single Python source of truth,
    never a literal here."""
    assert pe_rows in cnn_accel_constants.PE_ROWS_LEGAL, (
        f"pe_rows={pe_rows} not in {cnn_accel_constants.PE_ROWS_LEGAL}"
    )
    return HwPacking(
        vectors_dir=Path(vectors_dir),
        tile_channels=cnn_accel_constants.TILE_CHANNELS,
        pe_rows=pe_rows,
    )


def _flags(
    *, relu_en: bool, bias_en: bool, requant_en: bool, pad_en: bool, clamp_en: bool = False
) -> int:
    return (
        (int(relu_en) << FLAG_RELU_EN)
        | (int(bias_en) << FLAG_BIAS_EN)
        | (int(requant_en) << FLAG_REQUANT_EN)
        | (int(pad_en) << FLAG_PAD_EN)
        | (int(clamp_en) << FLAG_CLAMP_EN)
    )


def _write_int_lines(path: Path, values: list[int]) -> None:
    path.write_text("".join(f"{v}\n" for v in values))


def _write_desc(path: Path, desc: LayerDesc, *, extra: dict[str, int] | None = None) -> None:
    lines = []
    for field in dataclasses.fields(desc):
        lines.append(f"{field.name} {getattr(desc, field.name)}\n")
    # `extra` (D10): tile_channels/pe_rows are not LayerDesc/ISA fields --
    # they are the host-compiler-time packing parameters used to produce
    # this case's weights_packed.txt, appended after the standard
    # `LayerDesc` fields so existing readers that stop after the known
    # field count are unaffected.
    for key, value in (extra or {}).items():
        lines.append(f"{key} {value}\n")
    path.write_text("".join(lines))


def _build_case(
    name: str,
    *,
    hw: HwPacking,
    seed: int,
    depthwise: bool,
    in_w: int,
    in_h: int,
    in_c: int,
    out_c: int,
    kernel: int,
    stride: int,
    pad: int = 0,
    pad_top: int | None = None,
    pad_bottom: int | None = None,
    pad_left: int | None = None,
    pad_right: int | None = None,
    bias_en: bool,
    relu_en: bool,
    requant_en: bool,
    requant_scale: int,
    requant_shift: int,
    input_override: list[int] | None = None,
    weight_override: list[int] | None = None,
    # ISA v1.1 (H1) epilogue fields; the defaults are the v1.0 encoding, so
    # every pre-H1 case's desc.txt gains only three `... 0` records.
    output_offset: int = 0,
    clamp_en: bool = False,
    clamp_min: int = 0,
    clamp_max: int = 0,
) -> None:
    # `pad` is the symmetric (all 4 sides equal) shorthand used by the
    # original 5 cases; pad_top/bottom/left/right let a case override
    # individual sides independently (asymmetric padding) without
    # changing anything for callers that only pass `pad`.
    pad_top = pad if pad_top is None else pad_top
    pad_bottom = pad if pad_bottom is None else pad_bottom
    pad_left = pad if pad_left is None else pad_left
    pad_right = pad if pad_right is None else pad_right
    pad_en = bool(pad_top or pad_bottom or pad_left or pad_right)

    rng = random.Random(seed)

    if input_override is not None:
        input_values = list(input_override)
        assert len(input_values) == in_w * in_h * in_c
    else:
        input_values = [rng.randint(-128, 127) for _ in range(in_w * in_h * in_c)]

    if weight_override is not None:
        weights = list(weight_override)
    else:
        weight_count = (out_c * kernel * kernel) if depthwise else (out_c * kernel * kernel * in_c)
        weights = [rng.randint(-128, 127) for _ in range(weight_count)]

    bias = [rng.randint(-1000, 1000) for _ in range(out_c)] if bias_en else [0] * out_c

    desc = LayerDesc(
        opcode=OPCODE_DWCONV2D if depthwise else OPCODE_CONV2D,
        flags=_flags(
            relu_en=relu_en, bias_en=bias_en, requant_en=requant_en, pad_en=pad_en, clamp_en=clamp_en
        ),
        in_width=in_w,
        in_height=in_h,
        in_channels=in_c,
        out_channels=out_c,
        kernel_h=kernel,
        kernel_w=kernel,
        stride_h=stride,
        stride_w=stride,
        pad_top=pad_top,
        pad_bottom=pad_bottom,
        pad_left=pad_left,
        pad_right=pad_right,
        requant_scale=requant_scale,
        requant_shift=requant_shift,
        output_offset=output_offset,
        clamp_min=clamp_min,
        clamp_max=clamp_max,
    )

    expected = (
        dwconv2d(input_values, weights, bias, desc)
        if depthwise
        else conv2d(input_values, weights, bias, desc)
    )

    case_dir = hw.vectors_dir / name
    case_dir.mkdir(parents=True, exist_ok=True)
    # D10: DWCONV2D's (channels, kernel_h, kernel_w) weight layout has no
    # cross-channel reduction to gather -- it is out of the tiled-conv
    # weight_buffer addressing this repack ratifies (see
    # cnn_accel_model.pack_weights_for_hw's docstring) and uses a
    # different pe_cols-grouped row layout per
    # doc/cnn_accel_pe_array_proposal.md section 3.5, not implemented
    # here. Only CONV2D/FC (OHWI) cases get a desc.txt tile_channels/
    # pe_rows record and a weights_packed.txt export.
    if depthwise:
        _write_desc(case_dir / "desc.txt", desc)
    else:
        packed_weights = pack_weights_for_hw(weights, desc, hw.tile_channels, hw.pe_rows)
        _write_desc(
            case_dir / "desc.txt",
            desc,
            extra={"tile_channels": hw.tile_channels, "pe_rows": hw.pe_rows},
        )
        _write_int_lines(case_dir / "weights_packed.txt", packed_weights)
    _write_int_lines(case_dir / "input.txt", input_values)
    _write_int_lines(case_dir / "weights.txt", weights)
    _write_int_lines(case_dir / "bias.txt", bias)
    _write_int_lines(case_dir / "expected.txt", expected)
    print(f"wrote {case_dir} ({len(expected)} output values)")


def _build_pool_case(
    name: str,
    *,
    hw: HwPacking,
    seed: int,
    mode: str,  # "max" or "avg"
    in_w: int,
    in_h: int,
    in_c: int,
    pool_kernel_h: int,
    pool_kernel_w: int,
    pool_stride_h: int,
    pool_stride_w: int,
    relu_en: bool = False,
    requant_en: bool = False,
    requant_scale: int = 0,
    requant_shift: int = 0,
) -> None:
    """POOL_MAX/POOL_AVG have no weights/bias tensors in the ISA; weights.txt
    is written empty and bias.txt all-zero (one line per input channel,
    since pooling output channels == input channels) purely so every
    case directory has the same 5-file shape for a uniform VHDL reader."""
    rng = random.Random(seed)
    input_values = [rng.randint(-128, 127) for _ in range(in_w * in_h * in_c)]

    desc = LayerDesc(
        opcode=OPCODE_POOL_MAX if mode == "max" else OPCODE_POOL_AVG,
        flags=_flags(relu_en=relu_en, bias_en=False, requant_en=requant_en, pad_en=False),
        in_width=in_w,
        in_height=in_h,
        in_channels=in_c,
        out_channels=in_c,
        pool_kernel_h=pool_kernel_h,
        pool_kernel_w=pool_kernel_w,
        pool_stride_h=pool_stride_h,
        pool_stride_w=pool_stride_w,
        requant_scale=requant_scale,
        requant_shift=requant_shift,
    )

    expected = pool_max(input_values, desc) if mode == "max" else pool_avg(input_values, desc)

    case_dir = hw.vectors_dir / name
    case_dir.mkdir(parents=True, exist_ok=True)
    _write_desc(case_dir / "desc.txt", desc)
    _write_int_lines(case_dir / "input.txt", input_values)
    _write_int_lines(case_dir / "weights.txt", [])
    _write_int_lines(case_dir / "bias.txt", [0] * in_c)
    _write_int_lines(case_dir / "expected.txt", expected)
    print(f"wrote {case_dir} ({len(expected)} output values)")


def _build_fc_case(
    name: str,
    *,
    hw: HwPacking,
    seed: int,
    in_c: int,
    out_c: int,
    bias_en: bool,
    relu_en: bool,
    requant_en: bool,
    requant_scale: int,
    requant_shift: int,
) -> None:
    """FC is emitted in its required degenerate spatial form:
    in_width=in_height=kernel_h=kernel_w=1 (see `run_layer`'s OPCODE_FC
    check in cnn_accel_model.py -- a non-degenerate original desc is
    rejected there)."""
    rng = random.Random(seed)
    input_values = [rng.randint(-128, 127) for _ in range(in_c)]
    weights = [rng.randint(-128, 127) for _ in range(out_c * in_c)]  # OHWI, k=1x1
    bias = [rng.randint(-1000, 1000) for _ in range(out_c)] if bias_en else [0] * out_c

    desc = LayerDesc(
        opcode=OPCODE_FC,
        flags=_flags(relu_en=relu_en, bias_en=bias_en, requant_en=requant_en, pad_en=False),
        in_width=1,
        in_height=1,
        in_channels=in_c,
        out_channels=out_c,
        kernel_h=1,
        kernel_w=1,
        stride_h=1,
        stride_w=1,
        requant_scale=requant_scale,
        requant_shift=requant_shift,
    )

    expected = fc(input_values, weights, bias, desc)

    case_dir = hw.vectors_dir / name
    case_dir.mkdir(parents=True, exist_ok=True)
    # FC's weight layout is OHWI with a degenerate 1x1 kernel -- same D10
    # scope as CONV2D.
    packed_weights = pack_weights_for_hw(weights, desc, hw.tile_channels, hw.pe_rows)
    _write_desc(
        case_dir / "desc.txt",
        desc,
        extra={"tile_channels": hw.tile_channels, "pe_rows": hw.pe_rows},
    )
    _write_int_lines(case_dir / "weights_packed.txt", packed_weights)
    _write_int_lines(case_dir / "input.txt", input_values)
    _write_int_lines(case_dir / "weights.txt", weights)
    _write_int_lines(case_dir / "bias.txt", bias)
    _write_int_lines(case_dir / "expected.txt", expected)
    print(f"wrote {case_dir} ({len(expected)} output values)")


def _raw_pe_accum_pointwise(
    input_values: list[int],
    packed_weights: list[int],
    desc: LayerDesc,
    *,
    tile_channels: int,
    pe_rows: int,
) -> list[int]:
    """Pre-quantization per-pixel/per-lane accumulator for a 1x1
    ("pointwise") `CONV2D` case, single output-channel tile only
    (`out_channels <= pe_rows` -- the only shape `cnn_accel_pe_array` can
    be exercised at standalone today; multiple output-channel tiles need
    the whole ifmap re-streamed by the not-yet-built `layer_ctrl`/DMA,
    D6 \u00a75). Mirrors `cnn_accel_pe_array.vhd`'s own first_tile-clears /
    accumulate / last_tile-commits contract and `test_cnn_accel_model.
    py`'s `_simulate_pe_array_tiled_dataflow` structurally, but returns
    the raw int32 sums BEFORE `bias_requantize_relu` -- exactly what
    `m_accum_m2s.data` carries -- instead of the final quantized int8
    layer output `conv2d()` returns. This is a separate, minimal
    reimplementation (not a call into that test's private helper) kept
    deliberately small since kernel is fixed at 1x1 here (no `kr, kc`
    loop, no spatial padding); `_build_pe_array_xlang_case` below
    self-checks its output against `conv2d()` before ever writing a
    vector file, so a bug here cannot silently ship a wrong regression
    vector.
    """
    assert desc.kernel_h == 1 and desc.kernel_w == 1
    assert desc.out_channels <= pe_rows
    in_w, in_h, in_c = desc.in_width, desc.in_height, desc.in_channels
    n_in_tiles = -(-in_c // tile_channels)  # ceil
    row_len = tile_channels * pe_rows
    assert len(packed_weights) == n_in_tiles * row_len

    output: list[int] = []
    for pixel in range(in_w * in_h):
        acc = [0] * pe_rows
        for t in range(n_in_tiles):
            row = packed_weights[t * row_len : (t + 1) * row_len]
            for c in range(tile_channels):
                ic = t * tile_channels + c
                tap = input_values[pixel * in_c + ic] if ic < in_c else 0
                if tap == 0:
                    continue  # zero tap contributes zero to every lane
                for r in range(pe_rows):
                    # lane = r*tile_channels + c -- pinned to
                    # cnn_accel_pe_array.vhd's own weight_lane indexing,
                    # see pack_weights_for_hw's docstring.
                    acc[r] += tap * row[r * tile_channels + c]
        output.extend(acc)
    return output


def _build_pe_array_xlang_case(
    name: str,
    *,
    hw: HwPacking,
    seed: int,
    in_w: int,
    in_h: int,
    in_c: int,
    out_c: int,
    bias_en: bool,
    relu_en: bool,
    requant_en: bool,
    requant_scale: int,
    requant_shift: int,
) -> None:
    """The cross-language guard vector case for `tb_cnn_accel_pe_array_
    from_vectors.vhd` (the D10 weight-lane-order regression -- see
    `pack_weights_for_hw`'s docstring and `cnn_accel_pe_array.vhd`'s
    `compute_partial_sums` comment). A directed pointwise (1x1) `CONV2D`
    shape chosen so `cnn_accel_pe_array` can be exercised standalone
    (single output-channel tile, `out_c <= hw.pe_rows`) while still
    covering both D11 zero-padding cases: a partial input-channel tile
    (`in_c` not a multiple of `hw.tile_channels`) and a partial
    output-channel tile (`out_c < hw.pe_rows`).

    Exports the standard case-directory shape (for parity/tooling with
    every other generated case) PLUS `pe_array_raw_accum.txt`: the
    pre-quantization per-pixel/per-lane accumulator
    `cnn_accel_pe_array.m_accum_m2s` must reproduce bit-exactly when fed
    `weights_packed.txt` sequentially. That cross-check -- driving this
    function's own packed output through the real RTL and comparing
    against this function's own accumulator, rather than a second
    Python-side (or VHDL-side) golden model re-deriving the same
    arithmetic -- is what actually exercises the Python-packer/RTL-reader
    contract end to end.
    """
    rng = random.Random(seed)
    input_values = [rng.randint(-128, 127) for _ in range(in_w * in_h * in_c)]
    weights = [rng.randint(-128, 127) for _ in range(out_c * in_c)]  # 1x1 kernel, OHWI
    bias = [rng.randint(-1000, 1000) for _ in range(out_c)] if bias_en else [0] * out_c

    desc = LayerDesc(
        opcode=OPCODE_CONV2D,
        flags=_flags(relu_en=relu_en, bias_en=bias_en, requant_en=requant_en, pad_en=False),
        in_width=in_w,
        in_height=in_h,
        in_channels=in_c,
        out_channels=out_c,
        kernel_h=1,
        kernel_w=1,
        stride_h=1,
        stride_w=1,
        requant_scale=requant_scale,
        requant_shift=requant_shift,
    )

    packed_weights = pack_weights_for_hw(weights, desc, hw.tile_channels, hw.pe_rows)
    raw_accum = _raw_pe_accum_pointwise(
        input_values, packed_weights, desc, tile_channels=hw.tile_channels, pe_rows=hw.pe_rows
    )

    # Internal consistency check (not a file export): bias/requantize/relu'ing
    # the raw accumulator, per real output channel, must reproduce conv2d()'s
    # own bit-exact output -- proves _raw_pe_accum_pointwise (independent of
    # test_cnn_accel_model.py's _simulate_pe_array_tiled_dataflow) is not
    # itself wrong before its output gets checked into a vector file a VHDL
    # testbench will trust.
    expected = conv2d(input_values, weights, bias, desc)
    for pixel in range(in_w * in_h):
        for oc in range(out_c):
            bias_value = bias[oc] if bias_en else 0
            got = bias_requantize_relu(
                raw_accum[pixel * hw.pe_rows + oc],
                bias_value,
                bias_en=bias_en,
                requant_en=requant_en,
                relu_en=relu_en,
                requant_scale=requant_scale,
                requant_shift=requant_shift,
            )
            assert got == expected[pixel * out_c + oc], (
                f"{name}: _raw_pe_accum_pointwise disagrees with conv2d at pixel {pixel}, "
                f"oc {oc}: requantized raw accum {got} != expected {expected[pixel * out_c + oc]}"
            )

    case_dir = hw.vectors_dir / name
    case_dir.mkdir(parents=True, exist_ok=True)
    _write_desc(
        case_dir / "desc.txt",
        desc,
        extra={"tile_channels": hw.tile_channels, "pe_rows": hw.pe_rows},
    )
    _write_int_lines(case_dir / "weights_packed.txt", packed_weights)
    _write_int_lines(case_dir / "input.txt", input_values)
    _write_int_lines(case_dir / "weights.txt", weights)
    _write_int_lines(case_dir / "bias.txt", bias)
    _write_int_lines(case_dir / "expected.txt", expected)
    _write_int_lines(case_dir / "pe_array_raw_accum.txt", raw_accum)
    print(
        f"wrote {case_dir} ({len(raw_accum)} raw-accum values, {len(expected)} output values)"
    )


# Case names tb_cnn_accel_conv_core.vhd runs, in order. Kept here (not in
# the testbench) so the Python side that writes them and the VHDL side
# that reads them cannot silently disagree on the set: the testbench
# reads `cases.txt` (one name per line) from its vector root instead of
# hardcoding names.
CONV_CORE_CASES_FILE = "cases.txt"


def generate_conv_core_cases(hw: HwPacking) -> list[str]:
    """Every CONV2D/FC case tb_cnn_accel_conv_core.vhd runs at packing point
    `hw`, written under `hw.vectors_dir`, plus `cases.txt` listing them in
    run order. Called from module_cnn_accel.py's `pre_config` for each
    conv_core test config -- once per legal `pe_rows` -- into that test's
    own VUnit `output_path`; nothing here is ever checked in. Returns the
    case names written."""
    hw.vectors_dir.mkdir(parents=True, exist_ok=True)

    # conv1x1_c4_o4: pointwise conv, no spatial reduction.
    _build_case(
        "conv1x1_c4_o4",
        hw=hw,
        seed=1001,
        depthwise=False,
        in_w=4, in_h=4, in_c=4, out_c=4,
        kernel=1, stride=1, pad=0,
        bias_en=True, relu_en=True, requant_en=True,
        requant_scale=1 << 14, requant_shift=0,
    )

    # conv3x3_s1_c3_o8_pad1: typical first-layer-shaped conv (RGB-like
    # in_channels=3), stride 1, same-padding.
    _build_case(
        "conv3x3_s1_c3_o8_pad1",
        hw=hw,
        seed=2002,
        depthwise=False,
        in_w=6, in_h=6, in_c=3, out_c=8,
        kernel=3, stride=1, pad=1,
        bias_en=True, relu_en=True, requant_en=True,
        requant_scale=1 << 12, requant_shift=2,
    )

    # conv3x3_s2_c8_o8_pad1: downsampling conv, odd spatial dims, wider
    # channel counts.
    _build_case(
        "conv3x3_s2_c8_o8_pad1",
        hw=hw,
        seed=3003,
        depthwise=False,
        in_w=7, in_h=7, in_c=8, out_c=8,
        kernel=3, stride=2, pad=1,
        bias_en=True, relu_en=False, requant_en=True,
        requant_scale=1 << 13, requant_shift=1,
    )

    # conv3x3_s1_extremes: int8-extreme inputs/weights (all -128 and all
    # +127 taps present) to exercise accumulator range and int8 saturate,
    # with a negative bias and requant_en=0 (bypass-saturate) path.
    in_w = in_h = 4
    in_c = out_c = 2
    kernel = 3
    extreme_input = ([-128] * (in_w * in_h) + [127] * (in_w * in_h))[: in_w * in_h * in_c]
    extreme_weights = ([-128] * (kernel * kernel) + [127] * (kernel * kernel) * (out_c * in_c - 1))[
        : out_c * kernel * kernel * in_c
    ]
    _build_case(
        "conv3x3_s1_extremes",
        hw=hw,
        seed=4004,
        depthwise=False,
        in_w=in_w, in_h=in_h, in_c=in_c, out_c=out_c,
        kernel=kernel, stride=1, pad=1,
        bias_en=True, relu_en=True, requant_en=False,
        requant_scale=0, requant_shift=0,
        input_override=extreme_input,
        weight_override=extreme_weights,
    )

    # conv3x3_asymmetric_pad: pad_top != pad_bottom AND pad_left !=
    # pad_right in the same case (the 4 ISA pad fields are independent).
    _build_case(
        "conv3x3_asymmetric_pad",
        hw=hw,
        seed=6006,
        depthwise=False,
        in_w=6, in_h=6, in_c=3, out_c=4,
        kernel=3, stride=1,
        pad_top=1, pad_bottom=0, pad_left=0, pad_right=1,
        bias_en=True, relu_en=True, requant_en=True,
        requant_scale=1 << 13, requant_shift=1,
    )

    # conv3x3_negative_requant_scale: same shape family as
    # conv3x3_s1_c3_o8_pad1 but with a negative Q15 requant_scale
    # (requant_scale is the one signed ISA field).
    _build_case(
        "conv3x3_negative_requant_scale",
        hw=hw,
        seed=7007,
        depthwise=False,
        in_w=5, in_h=5, in_c=2, out_c=3,
        kernel=3, stride=1, pad=1,
        bias_en=True, relu_en=False, requant_en=True,
        requant_scale=-(1 << 13), requant_shift=1,
    )

    # conv3x3_offset_clamp (ISA v1.1, H1): the one conv_core case that
    # exercises the W13 epilogue fields end to end -- a non-zero
    # output_offset added after the rounded shift, and FLAG_CLAMP_EN with
    # a general [clamp_min, clamp_max] range replacing the legacy
    # ReLU/int8 saturate (relu_en stays '0' so the clamp alone sets the
    # lower bound). Shape family of conv3x3_negative_requant_scale.
    # The offset/clamp values are chosen so that both bounds are actually
    # hit by this seed's random data (bias range +-1000 at scale 2**-2
    # spreads the pre-clamp values well past both bounds).
    _build_case(
        "conv3x3_offset_clamp",
        hw=hw,
        seed=14014,
        depthwise=False,
        in_w=5, in_h=5, in_c=4, out_c=5,
        kernel=3, stride=1, pad=1,
        bias_en=True, relu_en=False, requant_en=True,
        requant_scale=1 << 13, requant_shift=6,
        output_offset=-7, clamp_en=True, clamp_min=-100, clamp_max=90,
    )

    # fc_in6_out4: FC layer in its required degenerate spatial form
    # (in_width=in_height=kernel_h=kernel_w=1).
    _build_fc_case(
        "fc_in6_out4",
        hw=hw,
        seed=10010,
        in_c=6, out_c=4,
        bias_en=True, relu_en=True, requant_en=True,
        requant_scale=1 << 14, requant_shift=0,
    )

    # conv3x3_c20_o6_multitile: in_channels=20 > tile_channels=8, so
    # T=ceil(20/8)=3 input-channel tiles/pixel -- exercises the
    # window_gen->pe_array first_tile/last_tile partial-sum-carry path
    # (doc/cnn_accel_tiled_dataflow_proposal.md section 2/3) end to end,
    # which none of the CONV2D cases above do (they all have
    # in_channels<=8, T=1). Added for tb_cnn_accel_conv_core.vhd (M6b) --
    # see that testbench's header comment. out_c=6 stays <=
    # pe_rows=8 (single output-channel tile, the only shape this
    # composition supports standalone, D6 section 5).
    _build_case(
        "conv3x3_c20_o6_multitile",
        hw=hw,
        seed=12012,
        depthwise=False,
        in_w=5, in_h=5, in_c=20, out_c=6,
        kernel=3, stride=1, pad=1,
        bias_en=True, relu_en=True, requant_en=True,
        requant_scale=1 << 13, requant_shift=1,
    )

    # conv3x3_c8_o16: the one case that exercises EVERY lane of a
    # 16-row PE array, and therefore the one case that can only exist in
    # the scaled root -- at pe_rows=8 it is two output-channel tiles,
    # which cnn_accel_conv_core cannot be driven with standalone (D6
    # section 5, run_case's own out_channels <= g_pe_rows assert). Shaped
    # like the target backbone's layer 1 (out_channels=16 is what fixes
    # the legal pe_rows set to {8, 16} -- flow_status.md S-decisions).
    # Skipped for any root whose pe_rows would need more than one output
    # tile for it.
    if hw.pe_rows >= 16:
        _build_case(
            "conv3x3_c8_o16",
            hw=hw,
            seed=13013,
            depthwise=False,
            in_w=5, in_h=5, in_c=8, out_c=16,
            kernel=3, stride=1, pad=1,
            bias_en=True, relu_en=True, requant_en=True,
            requant_scale=1 << 13, requant_shift=1,
        )

    names = sorted(
        d.name for d in hw.vectors_dir.iterdir() if (d / "weights_packed.txt").is_file()
    )
    (hw.vectors_dir / CONV_CORE_CASES_FILE).write_text("".join(f"{n}\n" for n in names))
    return names


def generate_pe_array_xlang_case(hw: HwPacking) -> None:
    """The one case tb_cnn_accel_pe_array_from_vectors.vhd reads, written
    under `hw.vectors_dir` from that test's `pre_config`."""
    # pe_array_xlang_check: cross-language (Python packer -> real RTL)
    # bit-exactness guard for cnn_accel_pe_array, see
    # tb_cnn_accel_pe_array_from_vectors.vhd and
    # _build_pe_array_xlang_case's own docstring. in_c=10 is a partial
    # input-channel tile at tile_channels=8 (D11); out_c=5 is a
    # partial output-channel tile at pe_rows=8 (D11), and the only
    # kind of output-channel-tile shape this DUT can be driven with
    # standalone (out_c <= pe_rows, no ifmap re-streaming).
    _build_pe_array_xlang_case(
        "pe_array_xlang_check",
        hw=hw,
        seed=11011,
        in_w=3, in_h=3, in_c=10, out_c=5,
        bias_en=True, relu_en=True, requant_en=True,
        requant_scale=1 << 13, requant_shift=1,
    )


def generate_pool_dwconv_cases(hw: HwPacking) -> None:
    """DWCONV2D/POOL cases -- pe_rows-independent (no weights_packed.txt),
    not read by any VHDL testbench today; emitted by `generate_all` for
    inspection/tooling parity only."""
    # dwconv3x3_c8: depthwise, out_channels == in_channels, largest
    # spatial dims allowed (8x8) to keep simulation fast.
    _build_case(
        "dwconv3x3_c8",
        hw=hw,
        seed=5005,
        depthwise=True,
        in_w=8, in_h=8, in_c=8, out_c=8,
        kernel=3, stride=1, pad=1,
        bias_en=True, relu_en=True, requant_en=True,
        requant_scale=1 << 13, requant_shift=1,
    )

    # pool_max_4x4: 2x2 max pool, stride 2, over an 8x8x2 input.
    _build_pool_case(
        "pool_max_4x4",
        hw=hw,
        seed=8008,
        mode="max",
        in_w=8, in_h=8, in_c=2,
        pool_kernel_h=2, pool_kernel_w=2, pool_stride_h=2, pool_stride_w=2,
    )

    # pool_avg_4x4: 2x2 average pool, stride 2, requant_en=1 (the
    # division-by-pool-area path), over an 8x8x2 input.
    _build_pool_case(
        "pool_avg_4x4",
        hw=hw,
        seed=9009,
        mode="avg",
        in_w=8, in_h=8, in_c=2,
        pool_kernel_h=2, pool_kernel_w=2, pool_stride_h=2, pool_stride_w=2,
        relu_en=True, requant_en=True,
        requant_scale=1 << 13, requant_shift=0,  # scale factor = 1/4 (pool area)
    )


def generate_all(hw: HwPacking) -> None:
    generate_conv_core_cases(hw)
    generate_pe_array_xlang_case(hw)
    generate_pool_dwconv_cases(hw)


if __name__ == "__main__":
    # Manual inspection only: `python generate_vectors.py <out_dir> [pe_rows]`.
    # The VUnit flow never calls this -- module_cnn_accel.py's pre_config
    # hooks call generate_conv_core_cases/generate_pe_array_xlang_case
    # straight into each test's output_path -- and nothing written here
    # belongs in the repository.
    import sys

    if len(sys.argv) not in (2, 3):
        sys.exit("usage: generate_vectors.py <out_dir> [pe_rows]")
    _pe_rows = int(sys.argv[2]) if len(sys.argv) == 3 else cnn_accel_constants.PE_ROWS
    generate_all(hw_packing(Path(sys.argv[1]), pe_rows=_pe_rows))
