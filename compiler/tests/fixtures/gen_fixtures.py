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


# ---------------------------------------------------------------------------
# M14 `tosa.table` fixtures: the 256-entry int8 -> int8 LUT that carries a
# general activation (SiLU, in YOLOv8n) to the accelerator's OPCODE_ACT.
# ---------------------------------------------------------------------------

#: Quantization the SiLU LUT below is built at: an int8 code `q` means the
#: real value `q * _SILU_STEP`, for both input and output. 16/128 puts the
#: representable range at [-8, +8), which covers SiLU's interesting region
#: (it is essentially linear above +4 and essentially 0 below -4).
_SILU_STEP = 16.0 / 128.0


def silu_lut() -> np.ndarray:
    """A real SiLU (`x * sigmoid(x)`) sampled onto the 256 int8 codes, in
    TOSA `TABLE` order: entry `i` answers input `i - 128`.

    Not a synthetic ramp: the point of the ACT LUT is a function the
    hardware cannot compute any other way, and SiLU is the one YOLOv8n
    actually needs. Values are rounded half-away-from-zero and clamped to
    int8, so the table is exactly representable and the fixture is
    reproducible from this function alone."""
    codes = np.arange(-128, 128, dtype=np.float64)
    x = codes * _SILU_STEP
    y = x / (1.0 + np.exp(-x))
    q = np.trunc(y / _SILU_STEP + np.copysign(0.5, y))
    return np.clip(q, -128, 127).astype(np.int64)


def _table_module(shape: tuple[int, int, int, int], n_tables: int) -> str:
    """A module of `n_tables` chained `tosa.table` ops over one input, all
    sharing one `tosa.const` LUT (which also pins that a const read by two
    ops is placed once, not twice)."""
    dims = "x".join(str(d) for d in shape)
    ty = f"tensor<{dims}xi8>"
    lines = [
        '"builtin.module"() ({',
        f'  "func.func"() <{{function_type = ({ty}) -> {ty}, sym_name = "main"}}> ({{',
        f"  ^bb0(%arg0: {ty}):",
        f'    %lut = "tosa.const"() <{{values = {_dense(silu_lut(), "256xi8")}}}> : () -> tensor<256xi8>',
    ]
    value = "%arg0"
    for i in range(n_tables):
        result = f"%t{i}"
        lines.append(
            f'    {result} = "tosa.table"({value}, %lut) : ({ty}, tensor<256xi8>) -> {ty}'
        )
        value = result
    lines += [
        f'    "func.return"({value}) : ({ty}) -> ()',
        "  }) : () -> ()",
        "}) : () -> ()",
        "",
    ]
    return "\n".join(lines)


def table_silu_text() -> str:
    """One SiLU LUT over a `1x4x6x12` tensor. C=12 is deliberately not a
    multiple of the 8-channel DDR plane, so the padding lanes ride along."""
    return _table_module((1, 4, 6, 12), 1)


def table_twice_text() -> str:
    """The same LUT applied twice, so the program is a real two-instruction
    chain through an intermediate buffer -- and both ACTs read one shared
    LUT const."""
    return _table_module((1, 8, 8, 8), 2)


# ---------------------------------------------------------------------------
# M14 nearest-2x upsample fixtures. TOSA has no upsample op, so a producer
# spells it as `reshape -> tile -> reshape -> tile -> reshape` over a
# rank-5 intermediate; this generator writes that chain in NHWC (the
# layout GIR uses), which is the same chain YOLOv8n's NCHW artifact
# contains with the spatial axes in the other place.
# ---------------------------------------------------------------------------


def _shape_const(name: str, values: list[int]) -> str:
    dims = ", ".join(str(v) for v in values)
    return (
        f'    {name} = "tosa.const_shape"() <{{values = dense<[{dims}]> : '
        f"tensor<{len(values)}xindex>}}> : () -> !tosa.shape<{len(values)}>"
    )


def upsample_chain_lines(
    source: str, shape: tuple[int, int, int, int], prefix: str, out_name: str
) -> tuple[list[str], tuple[int, int, int, int]]:
    """The five ops that spell nearest-2x upsample of an NHWC `shape`.

    H is replicated first (a rank-5 `[1, H, 1, W, C]` tiled on axis 2,
    then folded back into `[1, 2H, W, 1, C]`), then W. Written out rather
    than emitted by a helper op so the fixture is exactly what a real
    exporter produces -- the whole point is that the compiler recognises
    the chain, not a shorthand for it."""
    n, h, w, c = shape
    ty = lambda dims: "tensor<" + "x".join(str(d) for d in dims) + "xi8>"  # noqa: E731
    lines = [
        _shape_const(f"%{prefix}s0", [n, h, 1, w, c]),
        _shape_const(f"%{prefix}s1", [1, 1, 2, 1, 1]),
        _shape_const(f"%{prefix}s2", [n, 2 * h, w, 1, c]),
        _shape_const(f"%{prefix}s3", [1, 1, 1, 2, 1]),
        _shape_const(f"%{prefix}s4", [n, 2 * h, 2 * w, c]),
        f'    %{prefix}0 = "tosa.reshape"({source}, %{prefix}s0) : '
        f"({ty((n, h, w, c))}, !tosa.shape<5>) -> {ty((n, h, 1, w, c))}",
        f'    %{prefix}1 = "tosa.tile"(%{prefix}0, %{prefix}s1) : '
        f"({ty((n, h, 1, w, c))}, !tosa.shape<5>) -> {ty((n, h, 2, w, c))}",
        f'    %{prefix}2 = "tosa.reshape"(%{prefix}1, %{prefix}s2) : '
        f"({ty((n, h, 2, w, c))}, !tosa.shape<5>) -> {ty((n, 2 * h, w, 1, c))}",
        f'    %{prefix}3 = "tosa.tile"(%{prefix}2, %{prefix}s3) : '
        f"({ty((n, 2 * h, w, 1, c))}, !tosa.shape<5>) -> {ty((n, 2 * h, w, 2, c))}",
        f'    {out_name} = "tosa.reshape"(%{prefix}3, %{prefix}s4) : '
        f"({ty((n, 2 * h, w, 2, c))}, !tosa.shape<4>) -> {ty((n, 2 * h, 2 * w, c))}",
    ]
    return lines, (n, 2 * h, 2 * w, c)


def _upsample_module(shape: tuple[int, int, int, int], times: int) -> str:
    ty = lambda dims: "tensor<" + "x".join(str(d) for d in dims) + "xi8>"  # noqa: E731
    body: list[str] = []
    value, current = "%arg0", shape
    for i in range(times):
        out_name = f"%u{i}"
        lines, current = upsample_chain_lines(value, current, f"c{i}_", out_name)
        body += lines
        value = out_name
    return "\n".join(
        [
            '"builtin.module"() ({',
            f'  "func.func"() <{{function_type = ({ty(shape)}) -> {ty(current)}, sym_name = "main"}}> ({{',
            f"  ^bb0(%arg0: {ty(shape)}):",
            *body,
            f'    "func.return"({value}) : ({ty(current)}) -> ()',
            "  }) : () -> ()",
            "}) : () -> ()",
            "",
        ]
    )


def upsample2x_text() -> str:
    """One nearest-2x upsample of `1x3x5x8`. Non-square and odd spatial
    dims so a transposed or swapped-axis match cannot pass by symmetry."""
    return _upsample_module((1, 3, 5, 8), 1)


def upsample4x_text() -> str:
    """Two chained 2x upsamples -- a two-instruction program, and the
    shape YOLOv8n's neck reaches by upsampling twice."""
    return _upsample_module((1, 2, 3, 12), 2)


# ---------------------------------------------------------------------------
# Depth-to-space (pixel-shuffle / sub-pixel convolution) fixtures.
#
# `reshape -> transpose -> reshape` over a rank-6 intermediate, which is
# what an exporter emits for `nn.PixelShuffle(r)`: split the channel axis
# into three, interleave the two new `r` axes with H and W, fold back
# down. Written out op by op, like the upsample chain above and for the
# same reason -- the point is that the compiler RECOGNISES the chain.
#
# Two channel groupings, and the difference between them is the whole
# reason both fixtures exist:
#
#   plane-major    [1, H, W, r, r, out_c], perms (0, 1, 3, 2, 4, 5)
#   channel-major  [1, H, W, out_c, r, r], perms (0, 1, 4, 2, 5, 3)
#
# Both land on `[1, H, r, W, r, out_c]` and fold to the same shape; they
# are different FUNCTIONS of the same input. Plane-major is what the
# hardware does; channel-major is what `nn.PixelShuffle` means, and needs
# the producing convolution's weight rows permuted first
# (`passes.depth_to_space_channels`).
# ---------------------------------------------------------------------------


def depth_to_space_chain_lines(
    source: str,
    shape: tuple[int, int, int, int],
    factor: int,
    prefix: str,
    *,
    channel_major: bool,
) -> tuple[list[str], tuple[int, int, int, int]]:
    """The three ops that spell a depth-to-space of an NHWC `shape`."""
    n, h, w, cin = shape
    assert cin % (factor * factor) == 0
    out_c = cin // (factor * factor)
    mid = (
        (n, h, w, out_c, factor, factor) if channel_major else (n, h, w, factor, factor, out_c)
    )
    perms = (0, 1, 4, 2, 5, 3) if channel_major else (0, 1, 3, 2, 4, 5)
    permuted = tuple(mid[p] for p in perms)
    out_shape = (n, h * factor, w * factor, out_c)
    ty = lambda dims: "tensor<" + "x".join(str(d) for d in dims) + "xi8>"  # noqa: E731
    lines = [
        _shape_const(f"%{prefix}s0", list(mid)),
        _shape_const(f"%{prefix}s1", list(out_shape)),
        f'    %{prefix}0 = "tosa.reshape"({source}, %{prefix}s0) : '
        f"({ty(shape)}, !tosa.shape<6>) -> {ty(mid)}",
        f'    %{prefix}1 = "tosa.transpose"(%{prefix}0) '
        f"<{{perms = array<i32: {', '.join(str(p) for p in perms)}>}}> : "
        f"({ty(mid)}) -> {ty(permuted)}",
        f'    %{prefix}out = "tosa.reshape"(%{prefix}1, %{prefix}s1) : '
        f"({ty(permuted)}, !tosa.shape<4>) -> {ty(out_shape)}",
    ]
    return lines, out_shape


def depth_to_space2x_text() -> str:
    """A bare plane-major 2x pixel shuffle of `1x3x5x32` straight off the
    graph input: no convolution, so no permutation is involved and the
    chain must lower on its own. Non-square, odd spatial dims so a
    transposed match cannot pass by symmetry."""
    in_shape = (1, 3, 5, 32)
    lines, out_shape = depth_to_space_chain_lines(
        "%arg0", in_shape, 2, "d", channel_major=False
    )
    return _module_text(in_shape, out_shape, lines, "%dout")


def espcn_tail_text() -> str:
    """ESPCN's tail, and the reason the permutation pass exists: a 3x3
    convolution producing `r**2 * out_c` channels, ReLU, then a
    CHANNEL-major (`nn.PixelShuffle`) 2x shuffle down to 8 channels.

    8 output channels because the hardware moves whole activation-plane
    channel tiles and refuses an `out_channels` that is not a multiple of
    T=8 -- a real luma-only ESPCN's `out_channels = 1` is exactly the case
    `lower.to_hir` rejects by name."""
    in_shape = (1, 4, 6, 4)
    counter = _ValueCounter()
    lines: list[str] = []
    conv, conv_shape = _conv_layer_lines(
        "%arg0", in_shape, 32, counter, lines, weight_seed=901, bias_seed=902,
        # shift 34 chosen the same way as the M9 fixtures' quantization
        # (see this module's docstring): for input seeds 1-4 the conv's
        # clamped int8 output has >100 distinct values with only ~1.5% of
        # taps saturating, so the shuffle is permuting real data rather
        # than a near-constant tensor.
        mult=1073741824, shift=34,
    )
    chain, out_shape = depth_to_space_chain_lines(conv, conv_shape, 2, "d", channel_major=True)
    lines += chain
    return _module_text(in_shape, out_shape, lines, "%dout")


def espcn_luma_text() -> str:
    """A REAL luma-only ESPCN tail: a 3x3 convolution producing
    `r**2 * 1 = 4` channels, ReLU, then a CHANNEL-major
    (`nn.PixelShuffle`) 2x shuffle down to ONE output channel.

    `espcn_tail` above uses 8 output channels, which is already a whole
    activation channel tile; this one uses the count an actual
    super-resolution model has, and which the hardware cannot address
    directly -- `OPCODE_DEPTH_TO_SPACE` moves whole T=8-byte tiles, so a
    1-channel output would be one BYTE LANE of a tile. It is
    `passes.depth_to_space_pad` that makes this compile, by giving the
    convolution `4 * (8 - 1) = 28` dummy zero-weight output channels so
    the shuffle produces a whole 8-lane tile of which the graph's real
    single channel is lane 0. The stored tensor is 8 channels wide; the
    tensor's VALUE is still 1 channel, and `Tensor.logical_shape` ->
    manifest `logical_shape` -> `unpack_activation_planes` is what keeps
    those two facts apart.

    Same conv quantization as `espcn_tail` (see there for how `shift` was
    chosen), so the shuffle is permuting real data."""
    in_shape = (1, 4, 6, 4)
    counter = _ValueCounter()
    lines: list[str] = []
    conv, conv_shape = _conv_layer_lines(
        "%arg0", in_shape, 4, counter, lines, weight_seed=903, bias_seed=904,
        mult=1073741824, shift=34,
    )
    chain, out_shape = depth_to_space_chain_lines(conv, conv_shape, 2, "d", channel_major=True)
    lines += chain
    return _module_text(in_shape, out_shape, lines, "%dout")


# ---------------------------------------------------------------------------
# M14 `tosa.concat` / `tosa.slice` fixtures: the buffer-view lowering.
#
# Both are shaped like YOLOv8n's C2f block, which is where they actually
# occur: a tensor is SLICED into two halves along channels, one half is
# convolved, and the pieces are CONCATENATED back together. Written in
# NHWC (axis 3), the layout GIR uses; the shipped NCHW artifact says
# `axis = 1` for the same thing.
# ---------------------------------------------------------------------------


def _conv_layer_lines(
    in_value: str,
    in_shape: tuple[int, int, int, int],
    out_channels: int,
    counter: "_ValueCounter",
    lines: list[str],
    *,
    weight_seed: int,
    bias_seed: int,
    mult: int,
    shift: int,
) -> tuple[str, tuple[int, int, int, int]]:
    """A 3x3 same-padding conv + rescale + [0,127] clamp, appended to
    `lines`. Returns `(result value, output shape)`."""
    spec = LayerSpec(
        in_shape=in_shape,
        out_channels=out_channels,
        kernel=3,
        stride=(1, 1),
        pad=(1, 1, 1, 1),
        weight_seed=weight_seed,
        bias_seed=bias_seed,
        mult=mult,
        shift=shift,
        clamp=(0, 127),
    )
    value, _ty = _emit_layer(spec, in_value, counter, lines)
    n, h, w, _c = in_shape
    return value, (n, h, w, out_channels)


def _ty(shape: tuple[int, ...]) -> str:
    return "tensor<" + "x".join(str(d) for d in shape) + "xi8>"


def _slice_lines(
    source: str,
    in_shape: tuple[int, int, int, int],
    start_c: int,
    size_c: int,
    counter: "_ValueCounter",
    lines: list[str],
) -> tuple[str, tuple[int, int, int, int]]:
    n, h, w, _c = in_shape
    out_shape = (n, h, w, size_c)
    start = counter.next()
    size = counter.next()
    result = counter.next()
    lines.append(
        f'    {start} = "tosa.const_shape"() <{{values = dense<[0, 0, 0, {start_c}]> : '
        "tensor<4xindex>}> : () -> !tosa.shape<4>"
    )
    lines.append(
        f'    {size} = "tosa.const_shape"() <{{values = dense<[{n}, {h}, {w}, {size_c}]> : '
        "tensor<4xindex>}> : () -> !tosa.shape<4>"
    )
    lines.append(
        f'    {result} = "tosa.slice"({source}, {start}, {size}) : '
        f"({_ty(in_shape)}, !tosa.shape<4>, !tosa.shape<4>) -> {_ty(out_shape)}"
    )
    return result, out_shape


def _concat_lines(
    parts: list[tuple[str, tuple[int, int, int, int]]],
    counter: "_ValueCounter",
    lines: list[str],
) -> tuple[str, tuple[int, int, int, int]]:
    n, h, w, _c = parts[0][1]
    total = sum(shape[3] for _v, shape in parts)
    out_shape = (n, h, w, total)
    result = counter.next()
    operands = ", ".join(v for v, _s in parts)
    in_types = ", ".join(_ty(s) for _v, s in parts)
    lines.append(
        f'    {result} = "tosa.concat"({operands}) <{{axis = 3 : i32}}> : '
        f"({in_types}) -> {_ty(out_shape)}"
    )
    return result, out_shape


def _module_text(in_shape: tuple[int, ...], out_shape: tuple[int, ...], body: list[str], out_value: str) -> str:
    return "\n".join(
        [
            '"builtin.module"() ({',
            f'  "func.func"() <{{function_type = ({_ty(in_shape)}) -> {_ty(out_shape)}, '
            'sym_name = "main"}> ({',
            f"  ^bb0(%arg0: {_ty(in_shape)}):",
            *body,
            f'    "func.return"({out_value}) : ({_ty(out_shape)}) -> ()',
            "  }) : () -> ()",
            "}) : () -> ()",
            "",
        ]
    )


def concat_two_convs_text() -> str:
    """Two convolutions over one input, concatenated on channels: the
    plainest producer-directed placement there is -- each conv is told to
    write its own plane range of the result."""
    counter = _ValueCounter()
    lines: list[str] = []
    in_shape = (1, 8, 8, 8)
    a, a_shape = _conv_layer_lines(
        "%arg0", in_shape, 8, counter, lines, weight_seed=801, bias_seed=802, mult=1073741824, shift=38
    )
    b, b_shape = _conv_layer_lines(
        "%arg0", in_shape, 16, counter, lines, weight_seed=803, bias_seed=804, mult=1073741824, shift=38
    )
    out, out_shape = _concat_lines([(a, a_shape), (b, b_shape)], counter, lines)
    return _module_text(in_shape, out_shape, lines, out)


def concat_equal_halves_text() -> str:
    """Two 8-channel convolutions concatenated: both operands are exactly
    one channel plane, so their plane offsets can be SWAPPED and every
    address stays legal (contained, disjoint, tiling the result). That is
    the one concat mis-placement no structural check can catch, which is
    what makes this fixture worth having -- only reading the result back
    finds it."""
    counter = _ValueCounter()
    lines: list[str] = []
    in_shape = (1, 8, 8, 8)
    a, a_shape = _conv_layer_lines(
        "%arg0", in_shape, 8, counter, lines, weight_seed=821, bias_seed=822, mult=1073741824, shift=38
    )
    b, b_shape = _conv_layer_lines(
        "%arg0", in_shape, 8, counter, lines, weight_seed=823, bias_seed=824, mult=1073741824, shift=38
    )
    out, out_shape = _concat_lines([(a, a_shape), (b, b_shape)], counter, lines)
    return _module_text(in_shape, out_shape, lines, out)


def slice_concat_c2f_text() -> str:
    """YOLOv8n's C2f shape: one convolution's output is split in half on
    channels, the second half is convolved again, and all three pieces are
    concatenated. Exercises a view and producer-directed placement in the
    same graph, with a slice whose parent is itself a conv output."""
    counter = _ValueCounter()
    lines: list[str] = []
    in_shape = (1, 8, 8, 8)
    stem, stem_shape = _conv_layer_lines(
        "%arg0", in_shape, 16, counter, lines, weight_seed=811, bias_seed=812, mult=1073741824, shift=38
    )
    first, first_shape = _slice_lines(stem, stem_shape, 0, 8, counter, lines)
    second, second_shape = _slice_lines(stem, stem_shape, 8, 8, counter, lines)
    branch, branch_shape = _conv_layer_lines(
        second, second_shape, 8, counter, lines, weight_seed=813, bias_seed=814, mult=1073741824, shift=38
    )
    out, out_shape = _concat_lines(
        [(first, first_shape), (second, second_shape), (branch, branch_shape)], counter, lines
    )
    return _module_text(in_shape, out_shape, lines, out)


def main() -> None:
    (FIXTURES_DIR / "two_layer.mlir").write_text(two_layer_text())
    (FIXTURES_DIR / "first_layer_cin3.mlir").write_text(first_layer_cin3_text())
    (FIXTURES_DIR / "out_zp_relu.mlir").write_text(out_zp_relu_text())
    (FIXTURES_DIR / "clamp_5_100.mlir").write_text(clamp_5_100_text())
    (FIXTURES_DIR / "per_channel.mlir").write_text(per_channel_text())
    (FIXTURES_DIR / "per_channel_oc8.mlir").write_text(per_channel_oc8_text())
    (FIXTURES_DIR / "table_silu.mlir").write_text(table_silu_text())
    (FIXTURES_DIR / "table_twice.mlir").write_text(table_twice_text())
    (FIXTURES_DIR / "upsample2x.mlir").write_text(upsample2x_text())
    (FIXTURES_DIR / "upsample4x.mlir").write_text(upsample4x_text())
    (FIXTURES_DIR / "concat_two_convs.mlir").write_text(concat_two_convs_text())
    (FIXTURES_DIR / "slice_concat_c2f.mlir").write_text(slice_concat_c2f_text())
    (FIXTURES_DIR / "concat_equal_halves.mlir").write_text(concat_equal_halves_text())
    (FIXTURES_DIR / "depth_to_space2x.mlir").write_text(depth_to_space2x_text())
    (FIXTURES_DIR / "espcn_tail.mlir").write_text(espcn_tail_text())
    (FIXTURES_DIR / "espcn_luma.mlir").write_text(espcn_luma_text())


if __name__ == "__main__":
    main()
