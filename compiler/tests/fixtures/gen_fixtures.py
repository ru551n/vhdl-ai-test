"""Reproducible generator for the M9 multi-layer fixtures
(doc/tosa_compiler_plan.md M9): `two_layer.mlir` (16->32->32 channels,
3x3 s1 then 3x3 s2) and `first_layer_cin3.mlir` (Cin=3, 3x3 s2).

Run with the repo's compiler venv from anywhere:

    .venv-compiler/bin/python compiler/tests/fixtures/gen_fixtures.py

It is pure stdlib + numpy (only used for the deterministic PRNG and
flattening, not for any conv/rescale math) and re-writes the two
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
    """One `conv2d -> rescale -> clamp` stage."""

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
        mult: int,
        shift: int,
        clamp: tuple[int, int],
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
        f'    {v_mult} = "tosa.const"() <{{values = dense<{spec.mult}> : tensor<1xi32>}}> : () -> tensor<1xi32>',
        f'    {v_shift} = "tosa.const"() <{{values = dense<{spec.shift}> : tensor<1xi8>}}> : () -> tensor<1xi8>',
        f'    {v_rescale_in_zp} = "tosa.const"() <{{values = dense<0> : tensor<1xi32>}}> : () -> tensor<1xi32>',
        f'    {v_rescale_out_zp} = "tosa.const"() <{{values = dense<0> : tensor<1xi8>}}> : () -> tensor<1xi8>',
        f'    {v_rescale} = "tosa.rescale"({v_conv}, {v_mult}, {v_shift}, {v_rescale_in_zp}, {v_rescale_out_zp}) '
        f'<{{input_unsigned = false, output_unsigned = false, per_channel = false, '
        f'rounding_mode = #tosa.rounding_mode<SINGLE_ROUND>, scale32 = true}}> : '
        f"(tensor<{acc_type}>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<{q_type}>",
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


def main() -> None:
    (FIXTURES_DIR / "two_layer.mlir").write_text(two_layer_text())
    (FIXTURES_DIR / "first_layer_cin3.mlir").write_text(first_layer_cin3_text())


if __name__ == "__main__":
    main()
