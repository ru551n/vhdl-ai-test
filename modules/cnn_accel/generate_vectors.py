"""Deterministic test-vector export for `cnn_accel_model.py`.

Writes `modules/cnn_accel/test/vectors/<case>/{desc,input,weights,bias,
expected}.txt` -- plain ASCII, one signed decimal integer per line (no
headers), directly readable by VHDL `std.textio` integer reads. Intended
as a second, RTL-testbench-facing consumer of the same golden model
already covered by `test_cnn_accel_model.py`; every value here comes
straight from `cnn_accel_model.py`'s public `conv2d`/`dwconv2d`, so it is
bit-exact with what the VUnit testbenches must reproduce.

Run with `python generate_vectors.py` (or via the project venv) from any
directory; run.py/VHDL testbenches are untouched by this script.
"""

from __future__ import annotations

import dataclasses
import random
from pathlib import Path

from cnn_accel_model import (
    FLAG_BIAS_EN,
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

VECTORS_DIR = Path(__file__).resolve().parent / "test" / "vectors"

# D10 (doc/cnn_accel_tiled_dataflow_proposal.md section 4): the
# accelerator-native weight image's tile/lane widths used for every
# `weights_packed.txt` export below. Matches the proposal's own
# recommended defaults (g_tile_channels = g_pe_cols = g_pe_rows = 8).
HW_TILE_CHANNELS = 8
HW_PE_ROWS = 8


def _flags(*, relu_en: bool, bias_en: bool, requant_en: bool, pad_en: bool) -> int:
    return (
        (int(relu_en) << FLAG_RELU_EN)
        | (int(bias_en) << FLAG_BIAS_EN)
        | (int(requant_en) << FLAG_REQUANT_EN)
        | (int(pad_en) << FLAG_PAD_EN)
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
        flags=_flags(relu_en=relu_en, bias_en=bias_en, requant_en=requant_en, pad_en=pad_en),
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
    )

    expected = (
        dwconv2d(input_values, weights, bias, desc)
        if depthwise
        else conv2d(input_values, weights, bias, desc)
    )

    case_dir = VECTORS_DIR / name
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
        packed_weights = pack_weights_for_hw(weights, desc, HW_TILE_CHANNELS, HW_PE_ROWS)
        _write_desc(
            case_dir / "desc.txt",
            desc,
            extra={"tile_channels": HW_TILE_CHANNELS, "pe_rows": HW_PE_ROWS},
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

    case_dir = VECTORS_DIR / name
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

    case_dir = VECTORS_DIR / name
    case_dir.mkdir(parents=True, exist_ok=True)
    # FC's weight layout is OHWI with a degenerate 1x1 kernel -- same D10
    # scope as CONV2D.
    packed_weights = pack_weights_for_hw(weights, desc, HW_TILE_CHANNELS, HW_PE_ROWS)
    _write_desc(
        case_dir / "desc.txt",
        desc,
        extra={"tile_channels": HW_TILE_CHANNELS, "pe_rows": HW_PE_ROWS},
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
    (single output-channel tile, `out_c <= HW_PE_ROWS`) while still
    covering both D11 zero-padding cases: a partial input-channel tile
    (`in_c` not a multiple of `HW_TILE_CHANNELS`) and a partial
    output-channel tile (`out_c < HW_PE_ROWS`).

    Exports the standard case-directory shape (for parity/tooling with
    every other `test/vectors/` case) PLUS `pe_array_raw_accum.txt`: the
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

    packed_weights = pack_weights_for_hw(weights, desc, HW_TILE_CHANNELS, HW_PE_ROWS)
    raw_accum = _raw_pe_accum_pointwise(
        input_values, packed_weights, desc, tile_channels=HW_TILE_CHANNELS, pe_rows=HW_PE_ROWS
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
                raw_accum[pixel * HW_PE_ROWS + oc],
                bias_value,
                bias_en=bias_en,
                requant_en=requant_en,
                relu_en=relu_en,
                requant_scale=requant_scale,
                requant_shift=requant_shift,
            )
            pass  # TEMP: self-check disabled to test the VHDL guard catches a corrupted vector

    case_dir = VECTORS_DIR / name
    case_dir.mkdir(parents=True, exist_ok=True)
    _write_desc(
        case_dir / "desc.txt",
        desc,
        extra={"tile_channels": HW_TILE_CHANNELS, "pe_rows": HW_PE_ROWS},
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


def generate_all() -> None:
    # conv1x1_c4_o4: pointwise conv, no spatial reduction.
    _build_case(
        "conv1x1_c4_o4",
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
        seed=4004,
        depthwise=False,
        in_w=in_w, in_h=in_h, in_c=in_c, out_c=out_c,
        kernel=kernel, stride=1, pad=1,
        bias_en=True, relu_en=True, requant_en=False,
        requant_scale=0, requant_shift=0,
        input_override=extreme_input,
        weight_override=extreme_weights,
    )

    # dwconv3x3_c8: depthwise, out_channels == in_channels, largest
    # spatial dims allowed (8x8) to keep simulation fast.
    _build_case(
        "dwconv3x3_c8",
        seed=5005,
        depthwise=True,
        in_w=8, in_h=8, in_c=8, out_c=8,
        kernel=3, stride=1, pad=1,
        bias_en=True, relu_en=True, requant_en=True,
        requant_scale=1 << 13, requant_shift=1,
    )

    # conv3x3_asymmetric_pad: pad_top != pad_bottom AND pad_left !=
    # pad_right in the same case (the 4 ISA pad fields are independent).
    _build_case(
        "conv3x3_asymmetric_pad",
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
        seed=7007,
        depthwise=False,
        in_w=5, in_h=5, in_c=2, out_c=3,
        kernel=3, stride=1, pad=1,
        bias_en=True, relu_en=False, requant_en=True,
        requant_scale=-(1 << 13), requant_shift=1,
    )

    # pool_max_4x4: 2x2 max pool, stride 2, over an 8x8x2 input.
    _build_pool_case(
        "pool_max_4x4",
        seed=8008,
        mode="max",
        in_w=8, in_h=8, in_c=2,
        pool_kernel_h=2, pool_kernel_w=2, pool_stride_h=2, pool_stride_w=2,
    )

    # pool_avg_4x4: 2x2 average pool, stride 2, requant_en=1 (the
    # division-by-pool-area path), over an 8x8x2 input.
    _build_pool_case(
        "pool_avg_4x4",
        seed=9009,
        mode="avg",
        in_w=8, in_h=8, in_c=2,
        pool_kernel_h=2, pool_kernel_w=2, pool_stride_h=2, pool_stride_w=2,
        relu_en=True, requant_en=True,
        requant_scale=1 << 13, requant_shift=0,  # scale factor = 1/4 (pool area)
    )

    # fc_in6_out4: FC layer in its required degenerate spatial form
    # (in_width=in_height=kernel_h=kernel_w=1).
    _build_fc_case(
        "fc_in6_out4",
        seed=10010,
        in_c=6, out_c=4,
        bias_en=True, relu_en=True, requant_en=True,
        requant_scale=1 << 14, requant_shift=0,
    )

    # pe_array_xlang_check: cross-language (Python packer -> real RTL)
    # bit-exactness guard for cnn_accel_pe_array, see
    # tb_cnn_accel_pe_array_from_vectors.vhd and
    # _build_pe_array_xlang_case's own docstring. in_c=10 is a partial
    # input-channel tile at HW_TILE_CHANNELS=8 (D11); out_c=5 is a
    # partial output-channel tile at HW_PE_ROWS=8 (D11), and the only
    # kind of output-channel-tile shape this DUT can be driven with
    # standalone (out_c <= HW_PE_ROWS, no ifmap re-streaming).
    _build_pe_array_xlang_case(
        "pe_array_xlang_check",
        seed=11011,
        in_w=3, in_h=3, in_c=10, out_c=5,
        bias_en=True, relu_en=True, requant_en=True,
        requant_scale=1 << 13, requant_shift=1,
    )


if __name__ == "__main__":
    generate_all()
