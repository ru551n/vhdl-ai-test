"""M6 tests: `cnnc.lower.to_hir` (GIR -> HIR lowering with capability
checks, doc/tosa_compiler_plan.md §2.2, §4, §6, §13 M6)."""

from __future__ import annotations

from pathlib import Path

import pytest

from cnnc.errors import CapabilityError
from cnnc.frontend.mlir_generic import parse_module
from cnnc.frontend.tosa_import import import_tosa
from cnnc.gir.ir import ClampAttrs, ConvAttrs, FusedConvAttrs, Graph, Op, RescaleParams, Tensor
from cnnc.gir.verify import verify as verify_gir
from cnnc.hir.printer import print_hir
from cnnc.lower.to_hir import layer_env, select_unit, to_hir
from cnnc.passes import PassContext, default_pipeline, run_pipeline
from cnnc.target.constraints import check
from cnnc.target.contract import Target

GOLDEN_PATH = Path(__file__).parent / "golden" / "conv_rescale_clamp.hir.txt"


def _conv_op_shape(in_h: int, in_w: int, pad: tuple[int, int, int, int], k: int, stride: tuple[int, int]) -> tuple[int, int]:
    pad_t, pad_b, pad_l, pad_r = pad
    out_h = (in_h + pad_t + pad_b - (k - 1) - 1) // stride[0] + 1
    out_w = (in_w + pad_l + pad_r - (k - 1) - 1) // stride[1] + 1
    return out_h, out_w


def _dense(values: tuple[int, ...], ty: str) -> str:
    if len(values) == 1:
        return f"dense<{values[0]}> : tensor<{ty}>"
    return f"dense<[{', '.join(str(v) for v in values)}]> : tensor<{ty}>"


def _build_mlir(
    *,
    in_h: int = 8,
    in_w: int = 8,
    in_c: int = 4,
    out_c: int = 8,
    k: int = 3,
    stride: tuple[int, int] = (1, 1),
    pad: tuple[int, int, int, int] = (1, 1, 1, 1),
    conv_in_zp: int = 0,
    conv_w_zp: int = 0,
    bias: tuple[int, ...] | None = None,
    mult: tuple[int, ...] = (1073741824,),
    shift: tuple[int, ...] = (38,),
    per_channel: bool = False,
    rescale_in_zp: int = 0,
    rescale_out_zp: int = 0,
    clamp: tuple[int, int] | None = (0, 127),
) -> str:
    """A parameterized `conv2d -> rescale [-> clamp]` TOSA generic-form
    module, mirroring `fixtures/conv_rescale_clamp.mlir`'s structure with
    every dimension/quantization knob free to vary (used to build the
    to_hir capability-rejection fixtures below)."""
    out_h, out_w = _conv_op_shape(in_h, in_w, pad, k, stride)
    bias = bias if bias is not None else tuple([0] * out_c)
    assert len(bias) == out_c
    n_scale = out_c if per_channel else 1
    assert len(mult) == n_scale and len(shift) == n_scale

    in_type = f"1x{in_h}x{in_w}x{in_c}xi8"
    w_type = f"{out_c}x{k}x{k}x{in_c}xi8"
    b_type = f"{out_c}xi32"
    acc_type = f"1x{out_h}x{out_w}x{out_c}xi32"
    q_type = f"1x{out_h}x{out_w}x{out_c}xi8"

    lines = [
        '"builtin.module"() ({',
        f'  "func.func"() <{{function_type = (tensor<{in_type}>) -> tensor<{q_type}>, sym_name = "main"}}> ({{',
        "  ^bb0(%arg0: tensor<{}>):".format(in_type),
        f'    %0 = "tosa.const"() <{{values = {_dense((1,) * (out_c * k * k * in_c) if False else (1,), w_type)}}}> : () -> tensor<{w_type}>',
        f'    %1 = "tosa.const"() <{{values = {_dense(bias, b_type)}}}> : () -> tensor<{b_type}>',
        f'    %2 = "tosa.const"() <{{values = dense<{conv_in_zp}> : tensor<1xi8>}}> : () -> tensor<1xi8>',
        f'    %3 = "tosa.const"() <{{values = dense<{conv_w_zp}> : tensor<1xi8>}}> : () -> tensor<1xi8>',
        f'    %4 = "tosa.conv2d"(%arg0, %0, %1, %2, %3) <{{acc_type = i32, dilation = array<i64: 1, 1>, '
        f'pad = array<i64: {pad[0]}, {pad[1]}, {pad[2]}, {pad[3]}>, stride = array<i64: {stride[0]}, {stride[1]}>}}> : '
        f"(tensor<{in_type}>, tensor<{w_type}>, tensor<{b_type}>, tensor<1xi8>, tensor<1xi8>) -> tensor<{acc_type}>",
        f'    %5 = "tosa.const"() <{{values = {_dense(mult, f"{n_scale}xi32")}}}> : () -> tensor<{n_scale}xi32>',
        f'    %6 = "tosa.const"() <{{values = {_dense(shift, f"{n_scale}xi8")}}}> : () -> tensor<{n_scale}xi8>',
        f'    %7 = "tosa.const"() <{{values = dense<{rescale_in_zp}> : tensor<1xi32>}}> : () -> tensor<1xi32>',
        f'    %8 = "tosa.const"() <{{values = dense<{rescale_out_zp}> : tensor<1xi8>}}> : () -> tensor<1xi8>',
        f'    %9 = "tosa.rescale"(%4, %5, %6, %7, %8) <{{input_unsigned = false, output_unsigned = false, '
        f'per_channel = {"true" if per_channel else "false"}, rounding_mode = #tosa.rounding_mode<SINGLE_ROUND>, scale32 = true}}> : '
        f"(tensor<{acc_type}>, tensor<{n_scale}xi32>, tensor<{n_scale}xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<{q_type}>",
    ]
    if clamp is not None:
        lines.append(
            f'    %10 = "tosa.clamp"(%9) <{{max_val = {clamp[1]} : i8, min_val = {clamp[0]} : i8, '
            f'nan_mode = #tosa.nan_mode<PROPAGATE>}}> : (tensor<{q_type}>) -> tensor<{q_type}>'
        )
        ret = "%10"
    else:
        ret = "%9"
    lines += [
        f'    "func.return"({ret}) : (tensor<{q_type}>) -> ()',
        "  }) : () -> ()",
        "}) : () -> ()",
    ]
    return "\n".join(lines)


def _graph(**kwargs) -> Graph:
    return import_tosa(parse_module(_build_mlir(**kwargs)))


def _fuse(graph: Graph, target) -> Graph:
    ctx = PassContext(target=target)
    return run_pipeline(graph, default_pipeline(target), ctx)


def _to_hir(target, **kwargs):
    return to_hir(_fuse(_graph(**kwargs), target), target)


def _fixture_graph() -> Graph:
    return import_tosa(parse_module(_build_mlir()))


# --------------------------------------------------------------------------
# Golden dump + params/buffers/entry io
# --------------------------------------------------------------------------


def test_fixture_matches_golden_hir(target):
    module = _to_hir(target)
    assert print_hir(module) == GOLDEN_PATH.read_text()


def test_fixture_params(target):
    module = _to_hir(target)
    params = module.ops[0].params
    assert params["requant_shift"] == 23  # 38 - 15
    assert params["requant_scale"] == 1073741824
    # ISA v1.1 (M11) epilogue encoding: the fixture's ReLU clamp [0,127] is
    # CLAMP_EN,[0,127] with relu_en=0 (behaviourally identical to the v1.0
    # RELU_EN encoding, see test_fixtures_m11.py); out_zp=0 -> output_offset=0.
    assert params["relu_en"] is False
    assert params["clamp_en"] is True
    assert (params["clamp_min"], params["clamp_max"]) == (0, 127)
    assert params["output_offset"] == 0
    assert params["pad_en"] is True
    assert params["bias_en"] is True
    assert params["requant_en"] is True
    assert params["in_width"] == 8 and params["in_height"] == 8 and params["in_channels"] == 4
    assert params["out_channels"] == 8 and params["kernel_h"] == 3 and params["kernel_w"] == 3
    assert params["stride_h"] == 1 and params["stride_w"] == 1
    assert (params["pad_top"], params["pad_bottom"], params["pad_left"], params["pad_right"]) == (1, 1, 1, 1)


def test_fixture_buffers(target):
    module = _to_hir(target)
    weight = module.buffer("%0")
    bias = module.buffer("%1")
    # 8x3x3x4 all-ones weights, tiled 8 (cin) x 8 (cout): one output tile, one
    # input tile whose lanes c=4..7 are D11 zero padding -> 3*3*8*8 = 576 B.
    assert len(weight.data) == 576
    rows = [weight.data[i : i + 8] for i in range(0, 576, 8)]
    assert all(row == bytes([1, 1, 1, 1, 0, 0, 0, 0]) for row in rows)
    assert bias.data == bytes(32)
    assert weight.layout == "TILED_OHWI" and bias.layout == "I32_TILED"
    assert module.buffer("%arg0").size_bytes == 512  # 4 channels padded to one 8-channel plane
    assert module.buffer("%10").size_bytes == 512
    assert module.buffer("%arg0").role == "input"
    assert module.buffer("%10").role == "output"


def test_fixture_deps_and_entry_io(target):
    module = _to_hir(target)
    assert module.ops[0].deps == ()
    assert module.entry_inputs == ("%arg0",)
    assert module.entry_outputs == ("%10",)
    assert module.stage == "mapped"


# --------------------------------------------------------------------------
# Rounding gate: real (half_up) target passes, synthetic half_even is rejected
# --------------------------------------------------------------------------


def test_real_target_passes_rounding_gate(target):
    graph = _fuse(_fixture_graph(), target)
    module = to_hir(graph, target)
    assert module.stage == "mapped"


def test_half_even_target_rejected(target):
    from conftest import half_even_target

    synthetic = half_even_target(target)
    graph = _fuse(_fixture_graph(), synthetic)
    with pytest.raises(CapabilityError) as exc_info:
        to_hir(graph, synthetic)
    message = str(exc_info.value)
    assert "half_up" in message
    assert "half_even" in message


# --------------------------------------------------------------------------
# Rejections: capability checks
# --------------------------------------------------------------------------


def test_cout_not_divisible_rejected(target):
    with pytest.raises(CapabilityError) as exc_info:
        _to_hir(target, out_c=12, bias=tuple([0] * 12))
    assert "%4" in str(exc_info.value) or "%10" in str(exc_info.value)
    assert exc_info.value.constraint == "out_channels"


def test_row_tile_words_exceeded_rejected(target):
    with pytest.raises(CapabilityError) as exc_info:
        _to_hir(target, in_w=600, in_c=8, out_c=8, bias=tuple([0] * 8))
    assert "in_width" in str(exc_info.value.constraint)


def test_5x5_kernel_rejected(target):
    with pytest.raises(CapabilityError) as exc_info:
        _to_hir(target, k=5, pad=(2, 2, 2, 2))
    assert exc_info.value.constraint == "kernels"


def test_stride_300_exceeds_field_width_rejected(target):
    # `verify()` (compiler/cnnc/gir/verify.py) requires the stride to
    # evenly divide the padded input extent minus the dilated kernel span;
    # with the default 8x8/k3/pad1111 shape, stride 300 leaves a remainder
    # and is rejected there before reaching to_hir. Use a shape where the
    # division is exact (in=303, k=3, pad=0: (303-1-2) % 300 == 0) so this
    # test still reaches the intended CapabilityError about field width.
    with pytest.raises(CapabilityError) as exc_info:
        _to_hir(target, stride=(300, 300), in_h=303, in_w=303, pad=(0, 0, 0, 0))
    assert "stride" in exc_info.value.constraint


def test_stride_3_passes_capability_checks(target):
    # `unit.strides` is discovered as "any" (no elaboration-time RTL
    # bound); stride 3 is well within the 1-byte ISA field, so it is
    # only bounded by the general `max stride_h/stride_w` constraint
    # (255), not rejected outright like the M6 plan's original static
    # JSON example assumed.
    # in=9, k=3, pad=0: (9-1-2) % 3 == 0, so `verify()`'s stride-divisibility
    # check (compiler/cnnc/gir/verify.py) is satisfied too.
    module = _to_hir(target, stride=(3, 3), pad=(0, 0, 0, 0), in_h=9, in_w=9)
    assert module.ops[0].params["stride_h"] == 3


def test_conv_in_zp_rejected(target):
    with pytest.raises(CapabilityError) as exc_info:
        _to_hir(target, conv_in_zp=1)
    # Nonzero zero points don't block fusion, so this reaches to_hir as a
    # fused_conv whose id is the fused chain's output id (clamp's %10).
    assert "%10" in str(exc_info.value)
    assert exc_info.value.constraint == "conv.zero_point"


def test_per_channel_rescale_stays_unfused_and_rejected(target):
    # MVP `epilogue.rescale.per_channel=False`: `FusePass` refuses to fuse
    # a per_channel rescale in the first place (doc/tosa_compiler_plan.md
    # §9), so `to_hir` sees a standalone `conv2d`, not a `fused_conv` with
    # `per_channel=True`.
    with pytest.raises(CapabilityError) as exc_info:
        _to_hir(target, per_channel=True, mult=(1073741824,) * 8, shift=(38,) * 8)
    assert "%4" in str(exc_info.value)


def test_out_zp_stays_unfused_and_rejected_on_isa_v10_target(target):
    # Pre-H1 target (`output_zp: false`): FusePass leaves the rescale
    # standalone, so `to_hir` rejects the bare conv2d `%4`.
    from conftest import v10_target

    with pytest.raises(CapabilityError) as exc_info:
        _to_hir(v10_target(target), rescale_out_zp=5)
    assert "%4" in str(exc_info.value)


def test_out_zp_lowers_to_output_offset_on_isa_v11_target(target):
    # Real ISA v1.1 target (H1/M11): the out_zp rescale is fused and lowered
    # onto `output_offset`; the ReLU clamp rides on CLAMP_EN,[0,127].
    module = _to_hir(target, rescale_out_zp=5)
    params = module.ops[0].params
    assert params["output_offset"] == 5
    assert params["clamp_en"] is True and params["relu_en"] is False
    assert (params["clamp_min"], params["clamp_max"]) == (0, 127)


def test_legacy_relu_encoding_on_isa_v10_target(target):
    # ISA v1.0 keeps the RELU_EN encoding and never emits W13 params.
    from conftest import v10_target

    module = _to_hir(v10_target(target))
    params = module.ops[0].params
    assert params["relu_en"] is True
    assert not any(key in params for key in ("clamp_en", "clamp_min", "clamp_max", "output_offset"))

    module = _to_hir(v10_target(target), clamp=(-128, 127))
    assert module.ops[0].params["relu_en"] is False


def test_no_clamp_lowers_to_int8_range_on_isa_v11_target(target):
    # Rule: clamp bounds = following clamp if present else [-128,127].
    module = _to_hir(target, clamp=None)
    params = module.ops[0].params
    assert params["clamp_en"] is True and params["relu_en"] is False
    assert (params["clamp_min"], params["clamp_max"]) == (-128, 127)


def test_unfused_clamp_5_100_rejected_on_isa_v10_target(target):
    from conftest import v10_target

    with pytest.raises(CapabilityError) as exc_info:
        _to_hir(v10_target(target), clamp=(5, 100))
    assert "%10" in str(exc_info.value)


def test_fused_clamp_5_100_lowers_to_clamp_fields_on_isa_v11_target(target):
    # Real ISA v1.1 target: the general clamp is fused (clamp_ranges "any")
    # and lowered onto CLAMP_EN/clamp_min/clamp_max (M11).
    module = _to_hir(target, clamp=(5, 100))
    params = module.ops[0].params
    assert params["clamp_en"] is True and params["relu_en"] is False
    assert (params["clamp_min"], params["clamp_max"]) == (5, 100)
    assert params["output_offset"] == 0


def test_direct_general_clamp_rejected_on_isa_v10_target(target):
    # Bypass FusePass: a hand-built fused_conv with a general clamp on a
    # v1.0 unit has no field to carry it -> CapabilityError at the fused op.
    from conftest import v10_target

    rescale = RescaleParams(
        multiplier=(1073741824,), shift=(38,), per_channel=False, in_zp=0, out_zp=0,
        rounding="SINGLE_ROUND", scale32=True, input_unsigned=False, output_unsigned=False,
    )
    graph = _direct_fused_graph(rescale, clamp=ClampAttrs(min=5, max=100))
    verify_gir(graph)
    with pytest.raises(CapabilityError) as exc_info:
        to_hir(graph, v10_target(target))
    assert exc_info.value.constraint == "clamp"
    assert exc_info.value.op_id == "%10"
    assert "clamp_ranges" in str(exc_info.value)

    # Inconsistent target (advertises "any" clamp but is still ISA v1.0):
    # the isa_version gate, not clamp_ranges, rejects it.
    data = v10_target(target).to_dict()
    for unit in data["units"]:
        unit["epilogue"]["clamp_ranges"] = "any"
    with pytest.raises(CapabilityError) as exc_info:
        to_hir(graph, Target.from_dict(data))
    assert exc_info.value.constraint == "clamp"
    assert "isa_version >= 1.1" in str(exc_info.value)


# --------------------------------------------------------------------------
# to_hir's own rescale capability checks (direct GIR construction,
# bypassing `FusePass`'s upstream admissibility filter -- see
# `test_per_channel_rescale_stays_unfused_and_rejected` /
# `test_out_zp_stays_unfused_and_rejected` above for why the normal
# pipeline never reaches these checks with an MVP target).
# --------------------------------------------------------------------------


def _direct_fused_graph(rescale: RescaleParams, clamp: ClampAttrs | None = ClampAttrs(min=0, max=127)) -> Graph:
    conv = ConvAttrs(pad=(1, 1, 1, 1), stride=(1, 1), dilation=(1, 1), in_zp=0, w_zp=0, acc_dtype="i32")
    attrs = FusedConvAttrs(conv=conv, rescale=rescale, clamp=clamp)
    tensors = {
        "arg0": Tensor(id="arg0", shape=(1, 8, 8, 4), dtype="i8"),
        "w": Tensor(id="w", shape=(8, 3, 3, 4), dtype="i8", values=tuple([1] * (8 * 3 * 3 * 4))),
        "b": Tensor(id="b", shape=(8,), dtype="i32", values=tuple([0] * 8)),
        "10": Tensor(id="10", shape=(1, 8, 8, 8), dtype="i8"),
    }
    ops = (
        Op(id="%w", kind="const", inputs=(), outputs=("w",), attrs=None),
        Op(id="%b", kind="const", inputs=(), outputs=("b",), attrs=None),
        Op(id="%10", kind="fused_conv", inputs=("arg0", "w", "b"), outputs=("10",), attrs=attrs),
    )
    return Graph(name="main", tensors=tensors, ops=ops, inputs=("arg0",), outputs=("10",))


def test_to_hir_rejects_per_channel_directly(target):
    rescale = RescaleParams(
        multiplier=(1073741824,) * 8, shift=(38,) * 8, per_channel=True, in_zp=0, out_zp=0,
        rounding="SINGLE_ROUND", scale32=True, input_unsigned=False, output_unsigned=False,
    )
    graph = _direct_fused_graph(rescale)
    verify_gir(graph)
    with pytest.raises(CapabilityError) as exc_info:
        to_hir(graph, target)
    assert exc_info.value.constraint == "rescale.per_channel"
    assert exc_info.value.op_id == "%10"


def test_to_hir_rejects_out_zp_directly_on_isa_v10_target(target):
    # M11 acceptance: `isa_version 1.0` target + `out_zp != 0` -> CapabilityError
    # (here at the fused op, bypassing FusePass's own `output_zp` filter).
    from conftest import v10_target

    rescale = RescaleParams(
        multiplier=(1073741824,), shift=(38,), per_channel=False, in_zp=0, out_zp=5,
        rounding="SINGLE_ROUND", scale32=True, input_unsigned=False, output_unsigned=False,
    )
    graph = _direct_fused_graph(rescale)
    verify_gir(graph)
    with pytest.raises(CapabilityError) as exc_info:
        to_hir(graph, v10_target(target))
    assert exc_info.value.constraint == "rescale.out_zp"
    assert exc_info.value.op_id == "%10"


def test_to_hir_rejects_out_zp_outside_int8_directly(target):
    rescale = RescaleParams(
        multiplier=(1073741824,), shift=(38,), per_channel=False, in_zp=0, out_zp=200,
        rounding="SINGLE_ROUND", scale32=True, input_unsigned=False, output_unsigned=False,
    )
    graph = _direct_fused_graph(rescale)
    with pytest.raises(CapabilityError) as exc_info:
        to_hir(graph, target)
    assert exc_info.value.constraint == "rescale.out_zp"


# --------------------------------------------------------------------------
# Accumulator overflow note (contract D3)
# --------------------------------------------------------------------------


def test_accumulator_note_for_worst_case_bias(target):
    bias = tuple([2**31 - 1] + [0] * 7)
    module = _to_hir(target, in_c=8, out_c=8, k=3, bias=bias)
    assert any("worst-case accumulation" in note and "int32" in note for note in module.notes)
    # The note references the fused_conv op's id, i.e. the fused chain's
    # output id (the clamp's %10), not the original conv2d's %4.
    assert any("%10" in note for note in module.notes)


def test_no_accumulator_note_for_fixture(target):
    module = _to_hir(target)
    assert module.notes == ()


# --------------------------------------------------------------------------
# select_unit / layer_env / constraint evaluation
# --------------------------------------------------------------------------


def test_select_unit_finds_conv_engine(target):
    unit = select_unit(target, "conv2d")
    assert unit.name == "conv_engine"


def test_select_unit_raises_for_unknown_kind(target):
    with pytest.raises(CapabilityError):
        select_unit(target, "pool2d")


def test_layer_env_extracts_shape_fields():
    params = {
        "in_width": 8, "in_height": 8, "in_channels": 4, "out_channels": 8,
        "kernel_h": 3, "kernel_w": 3, "stride_h": 1, "stride_w": 1,
        "pad_top": 1, "pad_bottom": 1, "pad_left": 1, "pad_right": 1,
        "bias_en": True, "relu_en": True, "requant_scale": 1073741824,
    }
    env = layer_env(params)
    assert env == {
        "in_width": 8, "in_height": 8, "in_channels": 4, "out_channels": 8,
        "kernel_h": 3, "kernel_w": 3, "stride_h": 1, "stride_w": 1,
        "pad_top": 1, "pad_bottom": 1, "pad_left": 1, "pad_right": 1,
    }


def test_constraint_check_flags_divisible_violation(target):
    unit = select_unit(target, "conv2d")
    divisible = next(c for c in unit.constraints if c.kind == "divisible")
    env = layer_env(
        {
            "in_width": 8, "in_height": 8, "in_channels": 4, "out_channels": 12,
            "kernel_h": 3, "kernel_w": 3, "stride_h": 1, "stride_w": 1,
            "pad_top": 1, "pad_bottom": 1, "pad_left": 1, "pad_right": 1,
        }
    )
    violation = check(divisible, env)
    assert violation is not None
    assert violation.actual == 12


def test_constraint_check_passes_for_fixture(target):
    unit = select_unit(target, "conv2d")
    env = layer_env(
        {
            "in_width": 8, "in_height": 8, "in_channels": 4, "out_channels": 8,
            "kernel_h": 3, "kernel_w": 3, "stride_h": 1, "stride_w": 1,
            "pad_top": 1, "pad_bottom": 1, "pad_left": 1, "pad_right": 1,
        }
    )
    for constraint in unit.constraints:
        assert check(constraint, env) is None
