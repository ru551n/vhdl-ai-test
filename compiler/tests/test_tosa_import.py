"""M2 tests: `cnnc.frontend.tosa_import` (TOSA -> GIR) and its verifier
integration (doc/tosa_compiler_plan.md §13, M2 acceptance criteria)."""

from __future__ import annotations

from pathlib import Path

import pytest

from cnnc.errors import TosaImportError, UnsupportedAttribute, UnsupportedOp, VerifyError
from cnnc.frontend.mlir_generic import parse_module
from cnnc.frontend.tosa_import import import_tosa, load_tosa_file
from cnnc.gir.ir import ClampAttrs, ConvAttrs, RescaleParams, conv2d_output_shape
from cnnc.gir.printer import print_gir

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "conv_rescale_clamp.mlir"
GOLDEN_PATH = Path(__file__).parent / "golden" / "conv_rescale_clamp.gir.txt"


def _fixture_text() -> str:
    return FIXTURE_PATH.read_text()


def _mutated(old: str, new: str, *, count: int = 1) -> str:
    """Return the fixture text with an exact, unique textual substitution
    applied (used to build negative TOSA IR without hand-writing whole
    fixture files, per the M2 task spec)."""
    text = _fixture_text()
    occurrences = text.count(old)
    assert occurrences == count, f"expected {count} occurrence(s) of {old!r}, found {occurrences}"
    return text.replace(old, new)


def _import(text: str):
    return import_tosa(parse_module(text))


# --------------------------------------------------------------------------
# Golden dump
# --------------------------------------------------------------------------


def test_fixture_import_matches_golden():
    graph = load_tosa_file(FIXTURE_PATH)
    assert print_gir(graph) == GOLDEN_PATH.read_text()


def test_fixture_import_is_verified_on_return():
    # import_tosa calls verify() internally; this just re-asserts the
    # graph it hands back is structurally sound (no exception raised).
    graph = load_tosa_file(FIXTURE_PATH)
    assert graph.inputs == ("arg0",)
    assert graph.outputs == ("10",)
    assert [op.kind for op in graph.ops] == ["const", "const", "conv2d", "rescale", "clamp"]


# --------------------------------------------------------------------------
# Attribute assertions (>= 6, per acceptance criteria)
# --------------------------------------------------------------------------


def test_conv2d_attrs():
    graph = load_tosa_file(FIXTURE_PATH)
    conv = next(op for op in graph.ops if op.kind == "conv2d")
    attrs: ConvAttrs = conv.attrs
    assert attrs.pad == (1, 1, 1, 1)
    assert attrs.stride == (1, 1)
    assert attrs.dilation == (1, 1)
    assert attrs.in_zp == 0
    assert attrs.w_zp == 0
    assert attrs.acc_dtype == "i32"
    out = graph.tensor(conv.outputs[0])
    assert out.dtype == "i32"
    assert out.shape == (1, 8, 8, 8)


def test_rescale_attrs():
    graph = load_tosa_file(FIXTURE_PATH)
    rescale = next(op for op in graph.ops if op.kind == "rescale")
    attrs: RescaleParams = rescale.attrs
    assert attrs.multiplier == (1073741824,)
    assert attrs.shift == (38,)
    assert attrs.per_channel is False
    assert attrs.in_zp == 0
    assert attrs.out_zp == 0
    assert attrs.rounding == "SINGLE_ROUND"
    assert attrs.scale32 is True
    assert attrs.input_unsigned is False
    assert attrs.output_unsigned is False
    out = graph.tensor(rescale.outputs[0])
    assert out.dtype == "i8"


def test_clamp_attrs():
    graph = load_tosa_file(FIXTURE_PATH)
    clamp = next(op for op in graph.ops if op.kind == "clamp")
    attrs: ClampAttrs = clamp.attrs
    assert attrs.min == 0
    assert attrs.max == 127
    out = graph.tensor(clamp.outputs[0])
    assert out.dtype == "i8"
    assert out.shape == (1, 8, 8, 8)


# --------------------------------------------------------------------------
# conv2d output-shape formula (stride 1/2 x pad {0,1} x kernel {1,3},
# including odd input sizes), verified against a brute-force reference.
# --------------------------------------------------------------------------


def _brute_force_out_dim(in_dim: int, pad_lo: int, pad_hi: int, kernel: int, stride: int, dilation: int) -> int:
    """Count valid window positions in the (conceptually zero-padded)
    input: independent reference implementation for `conv2d_output_shape`."""
    padded = in_dim + pad_lo + pad_hi
    count = 0
    o = 0
    while True:
        start = o * stride
        end = start + dilation * (kernel - 1)
        if end >= padded:
            break
        count += 1
        o += 1
    return count


@pytest.mark.parametrize("in_size", [5, 6, 7, 8, 9, 15])
@pytest.mark.parametrize("kernel", [1, 3])
@pytest.mark.parametrize("pad", [0, 1])
@pytest.mark.parametrize("stride", [1, 2])
def test_conv2d_output_shape_matches_brute_force(in_size, kernel, pad, stride):
    attrs = ConvAttrs(pad=(pad, pad, pad, pad), stride=(stride, stride), dilation=(1, 1), in_zp=0, w_zp=0, acc_dtype="i32")
    in_shape = (1, in_size, in_size, 4)
    w_shape = (8, kernel, kernel, 4)
    got = conv2d_output_shape(in_shape, w_shape, attrs)
    expected_h = _brute_force_out_dim(in_size, pad, pad, kernel, stride, 1)
    expected_w = _brute_force_out_dim(in_size, pad, pad, kernel, stride, 1)
    assert got == (1, expected_h, expected_w, 8)


def test_conv2d_output_shape_asymmetric_pad():
    attrs = ConvAttrs(pad=(0, 1, 1, 0), stride=(2, 2), dilation=(1, 1), in_zp=0, w_zp=0, acc_dtype="i32")
    got = conv2d_output_shape((1, 7, 7, 4), (8, 3, 3, 4), attrs)
    expected_h = _brute_force_out_dim(7, 0, 1, 3, 2, 1)
    expected_w = _brute_force_out_dim(7, 1, 0, 3, 2, 1)
    assert got == (1, expected_h, expected_w, 8)


# --------------------------------------------------------------------------
# Rejection tests: exception type + op id in the message.
# --------------------------------------------------------------------------


def test_reject_float_tensor():
    text = _mutated("%arg0: tensor<1x8x8x4xi8>", "%arg0: tensor<1x8x8x4xf32>")
    with pytest.raises(TosaImportError) as exc:
        _import(text)
    assert "%arg0" in str(exc.value)


def test_reject_batch_not_one():
    text = _mutated("%arg0: tensor<1x8x8x4xi8>", "%arg0: tensor<2x8x8x4xi8>")
    with pytest.raises(UnsupportedAttribute) as exc:
        _import(text)
    assert "%4" in str(exc.value)


def test_reject_dilation_not_one():
    text = _mutated("dilation = array<i64: 1, 1>", "dilation = array<i64: 2, 1>")
    with pytest.raises(UnsupportedAttribute) as exc:
        _import(text)
    assert "%4" in str(exc.value)


def test_reject_scale32_false():
    text = _mutated("scale32 = true", "scale32 = false")
    with pytest.raises(UnsupportedAttribute) as exc:
        _import(text)
    assert "%9" in str(exc.value)


def test_reject_input_unsigned_true():
    text = _mutated("input_unsigned = false", "input_unsigned = true")
    with pytest.raises(UnsupportedAttribute) as exc:
        _import(text)
    assert "%9" in str(exc.value)


def test_reject_unknown_op():
    # tosa.add is a real TOSA op with no importer support in M2 (it is
    # scheduled for M14).
    text = _mutated('"tosa.clamp"', '"tosa.add"')
    with pytest.raises(UnsupportedOp) as exc:
        _import(text)
    assert "%10" in str(exc.value)
    assert "tosa.add" in str(exc.value)


def test_reject_zero_point_non_const_operand():
    text = _mutated("(%arg0, %0, %1, %2, %3)", "(%arg0, %0, %1, %arg0, %3)")
    with pytest.raises(TosaImportError) as exc:
        _import(text)
    assert "%4" in str(exc.value)
    assert "must be a constant" in str(exc.value)


def test_reject_declared_result_shape_mismatch():
    text = _mutated(
        "(tensor<1x8x8x4xi8>, tensor<8x3x3x4xi8>, tensor<8xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x8x8x8xi32>",
        "(tensor<1x8x8x4xi8>, tensor<8x3x3x4xi8>, tensor<8xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x8x8x16xi32>",
    )
    with pytest.raises(TosaImportError) as exc:
        _import(text)
    assert "%4" in str(exc.value)
    assert "!=" in str(exc.value)


def test_reject_multiplier_too_large():
    text = _mutated("dense<1073741824> : tensor<1xi32>", "dense<2147483648> : tensor<1xi32>")
    with pytest.raises(VerifyError) as exc:
        _import(text)
    assert "%9" in str(exc.value)


def test_reject_shift_out_of_range():
    text = _mutated("dense<38> : tensor<1xi8>", "dense<70> : tensor<1xi8>")
    with pytest.raises(VerifyError) as exc:
        _import(text)
    assert "%9" in str(exc.value)


# --------------------------------------------------------------------------
# Const folding: multi-use, dead, shared, and dual data/param roles.
# --------------------------------------------------------------------------

_SHARED_BIAS_MLIR = """
"builtin.module"() ({
  "func.func"() <{function_type = (tensor<1x2x2x4xi8>) -> tensor<1x2x2x1xi8>, sym_name = "main"}> ({
  ^bb0(%arg0: tensor<1x2x2x4xi8>):
    %w = "tosa.const"() <{values = dense<1> : tensor<1x1x1x4xi8>}> : () -> tensor<1x1x1x4xi8>
    %b = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %izp = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %wzp = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %dead = "tosa.const"() <{values = dense<9> : tensor<1xi32>}> : () -> tensor<1xi32>
    %c1 = "tosa.conv2d"(%arg0, %w, %b, %izp, %wzp) <{acc_type = i32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>}> : (tensor<1x2x2x4xi8>, tensor<1x1x1x4xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x2x2x1xi32>
    %mult = "tosa.const"() <{values = dense<1073741824> : tensor<1xi32>}> : () -> tensor<1xi32>
    %shift = "tosa.const"() <{values = dense<38> : tensor<1xi8>}> : () -> tensor<1xi8>
    %rizp = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %rozp = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %r1 = "tosa.rescale"(%c1, %mult, %shift, %rizp, %rozp) <{input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = #tosa.rounding_mode<SINGLE_ROUND>, scale32 = true}> : (tensor<1x2x2x1xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x2x2x1xi8>
    %c2 = "tosa.conv2d"(%arg0, %w, %b, %izp, %wzp) <{acc_type = i32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>}> : (tensor<1x2x2x4xi8>, tensor<1x1x1x4xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x2x2x1xi32>
    %r2 = "tosa.rescale"(%c2, %mult, %shift, %rizp, %rozp) <{input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = #tosa.rounding_mode<SINGLE_ROUND>, scale32 = true}> : (tensor<1x2x2x1xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x2x2x1xi8>
    "func.return"(%r1, %r2) : (tensor<1x2x2x1xi8>, tensor<1x2x2x1xi8>) -> ()
  }) : () -> ()
}) : () -> ()
"""


def test_shared_weight_and_bias_const_emitted_once():
    graph = _import(_SHARED_BIAS_MLIR)
    const_ops = [op for op in graph.ops if op.kind == "const"]
    # %w and %b each feed two conv2d ops but must be materialised only once.
    const_outputs = [op.outputs[0] for op in const_ops]
    assert const_outputs.count("w") == 1
    assert const_outputs.count("b") == 1
    convs = [op for op in graph.ops if op.kind == "conv2d"]
    assert len(convs) == 2
    assert convs[0].inputs[1] == convs[1].inputs[1] == "w"
    assert convs[0].inputs[2] == convs[1].inputs[2] == "b"


def test_dead_const_is_dropped_silently():
    graph = _import(_SHARED_BIAS_MLIR)
    assert "dead" not in graph.tensors
    assert not any(op.outputs and op.outputs[0] == "dead" for op in graph.ops)


def test_op_order_preserved_with_lazy_const_emission():
    graph = _import(_SHARED_BIAS_MLIR)
    kinds = [op.kind for op in graph.ops]
    # w, b materialise before the first conv2d; the second conv2d reuses
    # them (no re-emission) and precedes its own rescale.
    assert kinds == ["const", "const", "conv2d", "rescale", "conv2d", "rescale"]


_PER_CHANNEL_MLIR = """
"builtin.module"() ({
  "func.func"() <{function_type = (tensor<1x2x2x2xi8>) -> tensor<1x2x2x2xi8>, sym_name = "main"}> ({
  ^bb0(%arg0: tensor<1x2x2x2xi8>):
    %w = "tosa.const"() <{values = dense<1> : tensor<2x1x1x2xi8>}> : () -> tensor<2x1x1x2xi8>
    %b = "tosa.const"() <{values = dense<0> : tensor<2xi32>}> : () -> tensor<2xi32>
    %izp = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %wzp = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %mult = "tosa.const"() <{values = dense<[1073741824, 536870912]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %shift = "tosa.const"() <{values = dense<[38, 40]> : tensor<2xi8>}> : () -> tensor<2xi8>
    %rizp = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %rozp = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %c1 = "tosa.conv2d"(%arg0, %w, %b, %izp, %wzp) <{acc_type = i32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>}> : (tensor<1x2x2x2xi8>, tensor<2x1x1x2xi8>, tensor<2xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x2x2x2xi32>
    %r = "tosa.rescale"(%c1, %mult, %shift, %rizp, %rozp) <{input_unsigned = false, output_unsigned = false, per_channel = true, rounding_mode = #tosa.rounding_mode<SINGLE_ROUND>, scale32 = true}> : (tensor<1x2x2x2xi32>, tensor<2xi32>, tensor<2xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x2x2x2xi8>
    "func.return"(%r) : (tensor<1x2x2x2xi8>) -> ()
  }) : () -> ()
}) : () -> ()
"""


def test_per_channel_rescale_multiplier_and_shift_are_read_verbatim():
    graph = _import(_PER_CHANNEL_MLIR)
    rescale = next(op for op in graph.ops if op.kind == "rescale")
    assert rescale.attrs.per_channel is True
    assert rescale.attrs.multiplier == (1073741824, 536870912)
    assert rescale.attrs.shift == (38, 40)  # read verbatim from the i8 dense, not wrapped


def test_per_channel_length_mismatch_is_rejected():
    text = _PER_CHANNEL_MLIR.replace(
        "dense<[38, 40]> : tensor<2xi8>", "dense<[38, 40, 41]> : tensor<3xi8>"
    ).replace(
        "tensor<2xi32>, tensor<2xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x2x2x2xi8>",
        "tensor<2xi32>, tensor<3xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x2x2x2xi8>",
    )
    with pytest.raises(VerifyError, match="length"):
        _import(text)


def test_shift_value_out_of_i8_range_is_rejected_not_wrapped():
    # 200 does not fit signed i8 and is also outside the legal shift range
    # [2, 62]; either way it must be rejected, never silently wrapped
    # (e.g. to -56 via two's-complement truncation).
    text = _mutated("dense<38> : tensor<1xi8>", "dense<200> : tensor<1xi8>")
    with pytest.raises(VerifyError, match="out of range") as exc:
        _import(text)
    assert "200" in str(exc.value)
