"""Reproducible generator for the M9 multi-layer fixtures
(doc/tosa_compiler_plan.md M9): `two_layer.mlir` (16->32->32 channels,
3x3 s1 then 3x3 s2) and `first_layer_cin3.mlir` (Cin=3, 3x3 s2), and the
M11 epilogue fixtures: `out_zp_relu.mlir` (`rescale.output_zp = -128`,
identity clamp `[-128,127]` -- the asymmetric-output ReLU idiom) and
`clamp_5_100.mlir` (general clamp `[5,100]`, `output_zp = 0`), and the M12
per-channel fixtures `per_channel.mlir` (16 channels, `per_channel = true`,
distinct multipliers, some `shift < 15` to exercise per-element
legalization) and `per_channel_oc8.mlir` (the same idea at 8 channels, the
most the RTL conv_core testbench streams -- `out_channels <= g_pe_rows`).

Run with the repo's compiler venv from anywhere:

    .venv-compiler/bin/python compiler/tests/fixtures/gen_fixtures.py

It is pure stdlib + numpy (only used for the deterministic PRNG and
flattening, not for any conv/rescale math) and re-writes the six
`.mlir` files byte-identically every time it's run -- weights/bias are
derived from fixed `numpy.random.default_rng` seeds so the fixtures are
reproducible without checking in a separate data file.

Every `tosa.const` weight/bias tensor uses varied, non-splat values
(int8 weights in [-4, 4], int32 bias in [-20, 20]) so the fixtures
exercise real datapath behavior instead of degenerate all-equal
constants. Quantization (multiplier/shift) per layer was chosen by a
one-off numpy search (see `doc/tosa_compiler_plan.md` M9 notes / the
task report) so that, for input seeds 0-2, each layer's clamped int8
output has >= 16 distinct values and is not dominated by saturation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

FIXTURES_DIR = Path(__file__).parent


def _nest_dense_values(values: list[int], shape: tuple[int, ...]) -> str:
    """Format a flat, row-major list of dense element values with the
    shape-nested `[...]` bracketing real MLIR (`iree-opt`/`iree-compile`)
    requires for rank > 1 tensors -- mirrors
    `cnnc.frontend.mlir_generic._nest_dense_values`, duplicated here (not
    imported) so this generator has no dependency on `cnnc` internals."""
    if len(shape) <= 1:
        return "[" + ", ".join(str(v) for v in values) + "]"
    stride = 1
    for d in shape[1:]:
        stride *= d
    parts = [_nest_dense_values(values[i * stride : (i + 1) * stride], shape[1:]) for i in range(shape[0])]
    return "[" + ", ".join(parts) + "]"


def _weights(shape: tuple[int, ...], seed: int) -> np.ndarray:
    """Deterministic pseudo-random int8 weights in [-4, 4]."""
    rng = np.random.default_rng(seed)
    return rng.integers(-4, 5, size=shape, dtype=np.int64)


def _bias(n: int, seed: int, lo: int = -20, hi: int = 20) -> np.ndarray:
    """Deterministic pseudo-random int32 bias in [lo, hi]."""
    rng = np.random.default_rng(seed)
    return rng.integers(lo, hi + 1, size=n, dtype=np.int64)


def _dense(values: np.ndarray | int, ty: str) -> str:
    if isinstance(values, np.ndarray):
        flat = values.reshape(-1).tolist()
        body = _nest_dense_values(flat, values.shape) if values.ndim > 1 else f"[{', '.join(str(v) for v in flat)}]"
        return f"dense<{body}> : tensor<{ty}>"
    return f"dense<{values}> : tensor<{ty}>"


class _ValueCounter:
    def __init__(self) -> None:
        self.n = 0

    def next(self) -> str:
        v = f"%{self.n}"
        self.n += 1
        return v


class LayerSpec:
    """One `conv2d -> rescale -> clamp` stage. `mult`/`shift` are either a
    scalar (per-tensor rescale) or one value per output channel
    (`per_channel = true`, M12)."""

    def __init__(
        self,
        *,
        in_shape: tuple[int, int, int, int],
        out_channels: int,
        kernel: int,
        stride: tuple[int, int],
        pad: tuple[int, int, int, int],
        weight_seed: int,
        bias_seed: int,
        mult: int | tuple[int, ...],
        shift: int | tuple[int, ...],
        clamp: tuple[int, int],
        out_zp: int = 0,
    ) -> None:
        self.in_shape = in_shape
        self.out_channels = out_channels
        self.kernel = kernel
        self.stride = stride
        self.pad = pad
        self.weight_seed = weight_seed
        self.bias_seed = bias_seed
        self.mult = mult
        self.shift = shift
        self.clamp = clamp
        self.out_zp = out_zp
        self.per_channel = isinstance(mult, tuple)
        if self.per_channel:
            assert isinstance(shift, tuple) and len(mult) == len(shift) == out_channels
        else:
            assert isinstance(shift, int)

    @property
    def in_channels(self) -> int:
        return self.in_shape[3]

    @property
    def out_shape(self) -> tuple[int, int, int, int]:
        _, h, w, _ = self.in_shape
        pt, pb, pl, pr = self.pad
        sh, sw = self.stride
        out_h = (h + pt + pb - self.kernel) // sh + 1
        out_w = (w + pl + pr - self.kernel) // sw + 1
        return (1, out_h, out_w, self.out_channels)


def _emit_layer(spec: LayerSpec, input_value: str, counter: _ValueCounter, lines: list[str]) -> tuple[str, str]:
    """Append one layer's ops to `lines`; returns (clamp_result, out_type)."""
    in_type = "x".join(str(d) for d in spec.in_shape) + "xi8"
    w_shape = (spec.out_channels, spec.kernel, spec.kernel, spec.in_channels)
    w_type = "x".join(str(d) for d in w_shape) + "xi8"
    b_type = f"{spec.out_channels}xi32"
    acc_type = "x".join(str(d) for d in spec.out_shape) + "xi32"
    q_type = "x".join(str(d) for d in spec.out_shape) + "xi8"

    weight = _weights(w_shape, spec.weight_seed)
    bias = _bias(spec.out_channels, spec.bias_seed)
    n_scale = spec.out_channels if spec.per_channel else 1
    mult_ty, shift_ty = f"{n_scale}xi32", f"{n_scale}xi8"
    mult_dense = _dense(np.asarray(spec.mult, dtype=np.int64), mult_ty) if spec.per_channel else _dense(spec.mult, mult_ty)
    shift_dense = _dense(np.asarray(spec.shift, dtype=np.int64), shift_ty) if spec.per_channel else _dense(spec.shift, shift_ty)

    v_weight = counter.next()
    v_bias = counter.next()
    v_conv_in_zp = counter.next()
    v_conv_w_zp = counter.next()
    v_conv = counter.next()
    v_mult = counter.next()
    v_shift = counter.next()
    v_rescale_in_zp = counter.next()
    v_rescale_out_zp = counter.next()
    v_rescale = counter.next()
    v_clamp = counter.next()

    lines += [
        f'    {v_weight} = "tosa.const"() <{{values = {_dense(weight, w_type)}}}> : () -> tensor<{w_type}>',
        f'    {v_bias} = "tosa.const"() <{{values = {_dense(bias, b_type)}}}> : () -> tensor<{b_type}>',
        f'    {v_conv_in_zp} = "tosa.const"() <{{values = dense<0> : tensor<1xi8>}}> : () -> tensor<1xi8>',
        f'    {v_conv_w_zp} = "tosa.const"() <{{values = dense<0> : tensor<1xi8>}}> : () -> tensor<1xi8>',
        f'    {v_conv} = "tosa.conv2d"({input_value}, {v_weight}, {v_bias}, {v_conv_in_zp}, {v_conv_w_zp}) '
        f'<{{acc_type = i32, dilation = array<i64: 1, 1>, pad = array<i64: {spec.pad[0]}, {spec.pad[1]}, '
        f'{spec.pad[2]}, {spec.pad[3]}>, stride = array<i64: {spec.stride[0]}, {spec.stride[1]}>}}> : '
        f"(tensor<{in_type}>, tensor<{w_type}>, tensor<{b_type}>, tensor<1xi8>, tensor<1xi8>) -> tensor<{acc_type}>",
        f'    {v_mult} = "tosa.const"() <{{values = {mult_dense}}}> : () -> tensor<{mult_ty}>',
        f'    {v_shift} = "tosa.const"() <{{values = {shift_dense}}}> : () -> tensor<{shift_ty}>',
        f'    {v_rescale_in_zp} = "tosa.const"() <{{values = dense<0> : tensor<1xi32>}}> : () -> tensor<1xi32>',
        f'    {v_rescale_out_zp} = "tosa.const"() <{{values = dense<{spec.out_zp}> : tensor<1xi8>}}> : () -> tensor<1xi8>',
        f'    {v_rescale} = "tosa.rescale"({v_conv}, {v_mult}, {v_shift}, {v_rescale_in_zp}, {v_rescale_out_zp}) '
        f'<{{input_unsigned = false, output_unsigned = false, per_channel = {"true" if spec.per_channel else "false"}, '
        f'rounding_mode = #tosa.rounding_mode<SINGLE_ROUND>, scale32 = true}}> : '
        f"(tensor<{acc_type}>, tensor<{mult_ty}>, tensor<{shift_ty}>, tensor<1xi32>, tensor<1xi8>) -> tensor<{q_type}>",
        f'    {v_clamp} = "tosa.clamp"({v_rescale}) <{{max_val = {spec.clamp[1]} : i8, min_val = {spec.clamp[0]} : i8, '
        f'nan_mode = #tosa.nan_mode<PROPAGATE>}}> : (tensor<{q_type}>) -> tensor<{q_type}>',
    ]
    return v_clamp, q_type


def build_module(layers: list[LayerSpec]) -> str:
    in_type = "x".join(str(d) for d in layers[0].in_shape) + "xi8"
    out_type = "x".join(str(d) for d in layers[-1].out_shape) + "xi8"

    counter = _ValueCounter()
    lines = [
        '"builtin.module"() ({',
        f'  "func.func"() <{{function_type = (tensor<{in_type}>) -> tensor<{out_type}>, sym_name = "main"}}> ({{',
        f"  ^bb0(%arg0: tensor<{in_type}>):",
    ]
    value = "%arg0"
    for spec in layers:
        value, _ = _emit_layer(spec, value, counter, lines)
    lines += [
        f'    "func.return"({value}) : (tensor<{out_type}>) -> ()',
        "  }) : () -> ()",
        "}) : () -> ()",
    ]
    return "\n".join(lines) + "\n"


def two_layer_text() -> str:
    layer1 = LayerSpec(
        in_shape=(1, 8, 8, 16),
        out_channels=32,
        kernel=3,
        stride=(1, 1),
        pad=(1, 1, 1, 1),
        weight_seed=101,
        bias_seed=102,
        mult=1073741824,
        shift=34,
        clamp=(0, 127),
    )
    layer2 = LayerSpec(
        in_shape=layer1.out_shape,
        out_channels=32,
        kernel=3,
        stride=(2, 2),
        # Asymmetric (1, 0, 1, 0), not (1, 1, 1, 1): real TOSA (and
        # `iree-compile`'s legalizer) requires
        # `in_dim - 1 + pad_lo + pad_hi - (k - 1) * dilation` to be exactly
        # divisible by stride, which (1, 1, 1, 1) violates for in=8, k=3,
        # stride=2 (`7 / 2`) but (1, 0, 1, 0) satisfies (`6 / 2`), giving
        # the same 4x4 output as the plan's symmetric-pad arithmetic.
        pad=(1, 0, 1, 0),
        weight_seed=201,
        bias_seed=202,
        mult=1073741824,
        shift=35,
        clamp=(-128, 127),
    )
    return build_module([layer1, layer2])


def first_layer_cin3_text() -> str:
    layer = LayerSpec(
        in_shape=(1, 8, 8, 3),
        out_channels=8,
        kernel=3,
        stride=(2, 2),
        pad=(1, 0, 1, 0),  # see two_layer_text()'s layer2 comment on TOSA's stride-divisibility rule
        weight_seed=301,
        bias_seed=302,
        mult=1073741824,
        shift=34,
        clamp=(0, 127),
    )
    return build_module([layer])


def out_zp_relu_text() -> str:
    """M11: `output_zp = -128` with the identity clamp `[-128,127]`. In TOSA
    `y = clamp_i8(s - 128)`: everything the requantizer rounds to `<= 0`
    lands on -128 and the positive range maps onto `(-128, 127]` -- the
    asymmetric (uint8-style) quantized-ReLU idiom, which the v1.0 RELU_EN
    flag cannot express. Lowered as `output_offset=-128, CLAMP_EN,
    [-128,127]`. `shift=34` was picked (same one-off search as M9) so seeds
    0-2 give >= 100 distinct int8 outputs with ~half at the -128 floor."""
    layer = LayerSpec(
        in_shape=(1, 8, 8, 8),
        out_channels=8,
        kernel=3,
        stride=(1, 1),
        pad=(1, 1, 1, 1),
        weight_seed=401,
        bias_seed=402,
        mult=1073741824,
        shift=34,
        clamp=(-128, 127),
        out_zp=-128,
    )
    return build_module([layer])


def clamp_5_100_text() -> str:
    """M11: general clamp `[5,100]` (`output_zp = 0`), lowered as
    `CLAMP_EN, [5,100]`. `shift=35` gives seeds 0-2 ~75 distinct values with
    both bounds hit (~55% of outputs at 5 or 100), so neither bound is
    dead in the RTL cross-check."""
    layer = LayerSpec(
        in_shape=(1, 8, 8, 8),
        out_channels=8,
        kernel=3,
        stride=(1, 1),
        pad=(1, 1, 1, 1),
        weight_seed=501,
        bias_seed=502,
        mult=1073741824,
        shift=35,
        clamp=(5, 100),
    )
    return build_module([layer])


# M12 per-channel quantization. Every channel's `(mult, shift)` encodes a
# scale factor `mult / 2**shift` near the per-tensor fixtures' `2**30 /
# 2**34 = 1/16` (which the M9 search found gives a well-spread int8 output
# for these weights/inputs), but with a per-channel factor in ~[0.6, 1.5]
# so no two channels share a multiplier and the requantized ranges differ
# visibly per lane. The shifts cycle through `_PER_CHANNEL_SHIFTS`: three
# of them (10, 12, 14) are below `cnn_accel`'s `shift_min = 15`
# (`implicit_shift`), so `legalize_rescale` must rewrite those *elements*
# to `(mult << k, 15)` while leaving the others alone -- the per-element
# legalization M12 requires. All legalized multipliers stay far below
# 2**31.
_PER_CHANNEL_SHIFTS = (10, 12, 14, 20, 26, 30, 32, 34)


def _per_channel_scales(out_channels: int, seed: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    # An evenly spaced ramp of factors, shuffled deterministically: two
    # channels sharing a shift are then at least `0.9/(C-1)` apart in
    # factor, which at the coarsest shift (10, `2**6 * f`) still separates
    # their rounded multipliers for C <= 16.
    rng = np.random.default_rng(seed)
    factors = rng.permutation(np.linspace(0.6, 1.5, out_channels))
    shifts = tuple(_PER_CHANNEL_SHIFTS[i % len(_PER_CHANNEL_SHIFTS)] for i in range(out_channels))
    mults = tuple(int(round(2 ** (s - 4) * f)) for s, f in zip(shifts, factors))
    assert len(set(mults)) == out_channels, "per-channel multipliers must be distinct"
    return mults, shifts


def per_channel_text() -> str:
    """M12: 16-channel `per_channel = true` rescale (two PE_ROWS tiles of
    scale-table entries), distinct multipliers, shifts 10/12/14 below
    `shift_min` on six of the channels."""
    mults, shifts = _per_channel_scales(16, 603)
    layer = LayerSpec(
        in_shape=(1, 8, 8, 8),
        out_channels=16,
        kernel=3,
        stride=(1, 1),
        pad=(1, 1, 1, 1),
        weight_seed=601,
        bias_seed=602,
        mult=mults,
        shift=shifts,
        clamp=(-128, 127),
    )
    return build_module([layer])


def per_channel_oc8_text() -> str:
    """M12: 8-channel variant of `per_channel` -- exactly one PE_ROWS tile,
    so `tb_cnn_accel_conv_core`'s compiler-vector cross-check (which needs
    `out_channels <= g_pe_rows`) can stream its scale table too. Shifts
    10/12/14 land on channels 0-2."""
    mults, shifts = _per_channel_scales(8, 703)
    layer = LayerSpec(
        in_shape=(1, 8, 8, 8),
        out_channels=8,
        kernel=3,
        stride=(1, 1),
        pad=(1, 1, 1, 1),
        weight_seed=701,
        bias_seed=702,
        mult=mults,
        shift=shifts,
        clamp=(-128, 127),
    )
    return build_module([layer])


def main() -> None:
    (FIXTURES_DIR / "two_layer.mlir").write_text(two_layer_text())
    (FIXTURES_DIR / "first_layer_cin3.mlir").write_text(first_layer_cin3_text())
    (FIXTURES_DIR / "out_zp_relu.mlir").write_text(out_zp_relu_text())
    (FIXTURES_DIR / "clamp_5_100.mlir").write_text(clamp_5_100_text())
    (FIXTURES_DIR / "per_channel.mlir").write_text(per_channel_text())
    (FIXTURES_DIR / "per_channel_oc8.mlir").write_text(per_channel_oc8_text())


if __name__ == "__main__":
    main()
