"""Tests for cnnc.frontend.mlir_generic: MLIR generic-form parser/printer."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from cnnc.frontend.mlir_generic import (
    BoolAttr,
    DenseArrayAttr,
    DenseElementsAttr,
    DenseResourceAttr,
    EnumAttr,
    FloatAttr,
    FunctionTypeAttr,
    GenericFormRequired,
    IntAttr,
    MlirParseError,
    ResourceDecodeError,
    ScalarType,
    ShapeType,
    StringAttr,
    TensorType,
    TypeAttr,
    UndefinedValue,
    UnsupportedConstruct,
    UnsupportedLiteral,
    parse_module,
    parse_resources,
    print_generic,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "conv_rescale_clamp.mlir"


def _fixture_text() -> str:
    return FIXTURE_PATH.read_text()


# --------------------------------------------------------------------------
# Round-trip
# --------------------------------------------------------------------------


def test_fixture_round_trip_byte_equal():
    text = _fixture_text()
    module = parse_module(text)
    printed = print_generic(module)
    assert printed.rstrip("\n") == text.rstrip("\n")


# --------------------------------------------------------------------------
# Op-level assertions on the parsed fixture
# --------------------------------------------------------------------------


def test_fixture_structure():
    module = parse_module(_fixture_text())
    assert len(module.funcs) == 1
    func = module.funcs[0]
    assert func.name == "main"
    assert func.arg_names == ("arg0",)
    assert func.arg_types == (TensorType((1, 8, 8, 4), ScalarType("i8")),)
    assert func.result_types == (TensorType((1, 8, 8, 8), ScalarType("i8")),)
    assert len(func.ops) == 12


def test_fixture_conv2d_operands():
    module = parse_module(_fixture_text())
    func = module.funcs[0]
    conv = func.ops[4]
    assert conv.name == "tosa.conv2d"
    assert conv.operands == ("arg0", "0", "1", "2", "3")
    assert conv.attrs["pad"] == DenseArrayAttr("i64", (1, 1, 1, 1))
    assert conv.attrs["dilation"] == DenseArrayAttr("i64", (1, 1))
    assert conv.attrs["stride"] == DenseArrayAttr("i64", (1, 1))
    assert conv.attrs["acc_type"] == TypeAttr(ScalarType("i32"))


def test_fixture_rescale_rounding_mode():
    module = parse_module(_fixture_text())
    func = module.funcs[0]
    rescale = func.ops[9]
    assert rescale.name == "tosa.rescale"
    assert rescale.attrs["rounding_mode"] == EnumAttr("tosa", "rounding_mode", "SINGLE_ROUND")
    assert rescale.attrs["scale32"] == BoolAttr(True)


def test_fixture_clamp_max_val():
    module = parse_module(_fixture_text())
    func = module.funcs[0]
    clamp = func.ops[10]
    assert clamp.name == "tosa.clamp"
    assert clamp.attrs["max_val"] == IntAttr(127, "i8")
    assert clamp.attrs["min_val"] == IntAttr(0, "i8")
    assert clamp.attrs["nan_mode"] == EnumAttr("tosa", "nan_mode", "PROPAGATE")


def test_fixture_block_args_and_func_return():
    module = parse_module(_fixture_text())
    func = module.funcs[0]
    assert func.arg_names[0] == "arg0"
    assert func.arg_types[0] == TensorType((1, 8, 8, 4), ScalarType("i8"))
    ret = func.ops[-1]
    assert ret.name == "func.return"
    assert ret.operands == ("10",)
    assert ret.results == ()


# --------------------------------------------------------------------------
# One unit test per Attr variant
# --------------------------------------------------------------------------


def _wrap(op_body: str) -> str:
    return (
        '"builtin.module"() ({\n'
        '  "func.func"() <{function_type = () -> (), sym_name = "main"}> ({\n'
        "  ^bb0():\n"
        f"    {op_body}\n"
        '    "func.return"() : () -> ()\n'
        "  }) : () -> ()\n"
        "}) : () -> ()\n"
    )


def _single_op(text: str):
    module = parse_module(_wrap(text))
    return module.funcs[0].ops[0]


def test_attr_int_attr():
    op = _single_op('%0 = "t.op"() <{v = 127 : i8}> : () -> i8')
    assert op.attrs["v"] == IntAttr(127, "i8")


def test_attr_int_attr_negative():
    op = _single_op('%0 = "t.op"() <{v = -3 : i32}> : () -> i32')
    assert op.attrs["v"] == IntAttr(-3, "i32")


def test_attr_bool_attr():
    op = _single_op('%0 = "t.op"() <{v = true}> : () -> i1')
    assert op.attrs["v"] == BoolAttr(True)
    op2 = _single_op('%0 = "t.op"() <{v = false}> : () -> i1')
    assert op2.attrs["v"] == BoolAttr(False)


def test_attr_type_attr():
    op = _single_op('%0 = "t.op"() <{v = i32}> : () -> i32')
    assert op.attrs["v"] == TypeAttr(ScalarType("i32"))


def test_attr_dense_array_attr():
    op = _single_op('%0 = "t.op"() <{v = array<i64: 1, 1>}> : () -> i32')
    assert op.attrs["v"] == DenseArrayAttr("i64", (1, 1))


def test_attr_dense_array_attr_empty():
    op = _single_op('%0 = "t.op"() <{v = array<i64>}> : () -> i32')
    assert op.attrs["v"] == DenseArrayAttr("i64", ())


def test_attr_dense_elements_splat():
    op = _single_op('%0 = "t.op"() <{v = dense<1> : tensor<8xi32>}> : () -> i32')
    assert op.attrs["v"] == DenseElementsAttr(TensorType((8,), ScalarType("i32")), None, 1)


def test_attr_dense_elements_list():
    op = _single_op('%0 = "t.op"() <{v = dense<[1, -2, 3]> : tensor<3xi8>}> : () -> i32')
    assert op.attrs["v"] == DenseElementsAttr(TensorType((3,), ScalarType("i8")), (1, -2, 3), None)


def test_attr_dense_elements_nested():
    op = _single_op('%0 = "t.op"() <{v = dense<[[1, 2], [3, 4]]> : tensor<2x2xi8>}> : () -> i32')
    assert op.attrs["v"] == DenseElementsAttr(TensorType((2, 2), ScalarType("i8")), (1, 2, 3, 4), None)


def test_attr_enum_attr():
    op = _single_op('%0 = "t.op"() <{v = #tosa.rounding_mode<SINGLE_ROUND>}> : () -> i32')
    assert op.attrs["v"] == EnumAttr("tosa", "rounding_mode", "SINGLE_ROUND")


def test_attr_string_attr():
    op = _single_op('%0 = "t.op"() <{v = "hello"}> : () -> i32')
    assert op.attrs["v"] == StringAttr("hello")


def test_attr_function_type_attr():
    op = _single_op('%0 = "t.op"() <{v = (i32, i32) -> i32}> : () -> i32')
    assert op.attrs["v"] == FunctionTypeAttr((ScalarType("i32"), ScalarType("i32")), (ScalarType("i32"),))


def test_attr_older_form_no_angle_brackets():
    op = _single_op('%0 = "t.op"() {v = 127 : i8} : () -> i8')
    assert op.attrs["v"] == IntAttr(127, "i8")


# --------------------------------------------------------------------------
# Negative tests
# --------------------------------------------------------------------------


def test_custom_form_op_raises_generic_form_required():
    text = _wrap("%0 = tosa.clamp %arg0")
    with pytest.raises(GenericFormRequired) as exc:
        parse_module(text)
    assert ":" in str(exc.value)
    assert exc.value.line == 4


def test_custom_form_top_level_module_raises():
    text = 'module {\n}\n'
    with pytest.raises(GenericFormRequired):
        parse_module(text)


def test_undefined_ssa_value_raises():
    text = _wrap('%0 = "t.op"(%missing) : (i32) -> i32')
    with pytest.raises(UndefinedValue):
        parse_module(text)


def test_float_dense_parses_as_float_values():
    """Float literals PARSE (they used to raise `UnsupportedLiteral` right
    here). Being integer-only is a property of the compiler, not of MLIR,
    so the refusal belongs to `tosa_import` -- which can name the op and
    say "floating point dtype 'f32' is not supported" -- and not to a
    lexer that dies on the first constant of a pre-quantization model."""
    module = parse_module(_wrap('%0 = "t.op"() <{v = dense<1.0> : tensor<1xf32>}> : () -> i32'))
    attr = module.funcs[0].ops[0].attrs["v"]
    assert attr.splat == 1.0
    assert attr.elements() == (1.0,)


def test_float_scalar_attr_parses_as_float_attr():
    module = parse_module(_wrap('%0 = "t.op"() <{v = 1.5 : f32}> : () -> i32'))
    attr = module.funcs[0].ops[0].attrs["v"]
    assert isinstance(attr, FloatAttr)
    assert (attr.value, attr.type) == (1.5, "f32")


def test_integer_literal_with_float_type_is_a_float_attr():
    """MLIR prints `0 : f32` without a decimal point; the *type* is what
    makes it a float, not the literal's spelling."""
    module = parse_module(_wrap('%0 = "t.op"() <{v = 0 : f32}> : () -> i32'))
    assert module.funcs[0].ops[0].attrs["v"] == FloatAttr(0.0, "f32")


def test_float_literal_in_tensor_shape_still_raises():
    with pytest.raises(UnsupportedLiteral):
        parse_module(_wrap('%0 = "t.op"() <{v = dense<1> : tensor<1.5xi8>}> : () -> i32'))


def test_multi_result_raises_unsupported_construct():
    text = _wrap('%0:2 = "t.op"() : () -> i32')
    with pytest.raises(UnsupportedConstruct):
        parse_module(text)


def test_unterminated_string_raises_parse_error():
    text = '"builtin.module"() ({\n  "func.func\n'
    with pytest.raises(MlirParseError):
        parse_module(text)


# --------------------------------------------------------------------------
# iree-opt cross-check (optional, requires iree-opt in the venv)
# --------------------------------------------------------------------------


def _find_iree_opt() -> str | None:
    venv_bin = Path(sys.executable).parent
    candidate = venv_bin / "iree-opt"
    if candidate.exists():
        return str(candidate)
    return shutil.which("iree-opt")


@pytest.mark.iree
def test_iree_opt_accepts_fixture_and_round_trip():
    iree_opt = _find_iree_opt()
    if iree_opt is None:
        pytest.skip("iree-opt not found")

    result = subprocess.run(
        [iree_opt, "--mlir-print-op-generic", str(FIXTURE_PATH)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.rstrip("\n") == _fixture_text().rstrip("\n")

    module = parse_module(_fixture_text())
    printed = print_generic(module)
    result2 = subprocess.run(
        [iree_opt, "--mlir-print-op-generic", "-"],
        input=printed,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result2.returncode == 0, result2.stderr


# --------------------------------------------------------------------------
# Constructs real (non-IREE-hand-written) MLIR producers emit: TOSA-1.0
# shape types, `dense_resource` + the trailing `{-# ... #-}` block,
# `dense<"0x...">` hex blobs, and `index`-typed shapes. Every one of these
# used to be a raw traceback or a hard parse error.
# --------------------------------------------------------------------------


def test_tosa_shape_type_parses():
    text = _wrap(
        '%0 = "tosa.const_shape"() <{values = dense<[1, 16, 160, 160]> : tensor<4xindex>}> '
        ": () -> !tosa.shape<4>"
    )
    op = parse_module(text).funcs[0].ops[0]
    assert op.results[0][1] == ShapeType(rank=4)
    assert op.attrs["values"].tensor_type == TensorType((4,), ScalarType("index"))


def test_tosa_shape_typed_operand_parses():
    text = _wrap(
        '%0 = "tosa.const_shape"() <{values = dense<0> : tensor<4xindex>}> : () -> !tosa.shape<4>\n'
        '    %1 = "tosa.slice"(%0, %0, %0) : '
        "(!tosa.shape<4>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<1x8x8x4xi8>"
    )
    ops = parse_module(text).funcs[0].ops
    assert ops[1].operands == ("0", "0", "0")


def test_index_element_type_is_not_split_on_its_own_x():
    """`tensor<4xindex>`: 'index' ends in an 'x', which the old
    split-on-every-'x' shape parser turned into ('inde', '')."""
    text = _wrap('%0 = "t.op"() <{v = dense<[1, 2]> : tensor<2xindex>}> : () -> i32')
    attr = parse_module(text).funcs[0].ops[0].attrs["v"]
    assert attr.tensor_type == TensorType((2,), ScalarType("index"))


def test_rank_zero_tensor_type_parses():
    text = _wrap('%0 = "t.op"() <{v = dense<3> : tensor<i32>}> : () -> i32')
    attr = parse_module(text).funcs[0].ops[0].attrs["v"]
    assert attr.tensor_type == TensorType((), ScalarType("i32"))
    assert attr.elements() == (3,)


_RESOURCE_MODULE = (
    _wrap('%0 = "tosa.const"() <{values = dense_resource<w0> : tensor<4xi8>}> : () -> tensor<4xi8>')
    + '\n{-#\n  dialect_resources: {\n    builtin: {\n      w0: "0x0400000001FF0207"\n    }\n  }\n#-}\n'
)


def test_dialect_resources_block_is_skipped_and_collected():
    module = parse_module(_RESOURCE_MODULE)
    assert set(module.resources) == {"w0"}
    assert module.resources["w0"] == bytes.fromhex("0400000001FF0207")


def test_dense_resource_decodes_to_signed_elements():
    module = parse_module(_RESOURCE_MODULE)
    attr = module.funcs[0].ops[0].attrs["values"]
    assert isinstance(attr, DenseResourceAttr)
    # The blob's leading 4 bytes are MLIR's alignment header, not data.
    assert attr.decode(module.resources) == (1, -1, 2, 7)


def test_dense_resource_missing_key_raises_resource_decode_error():
    module = parse_module(_RESOURCE_MODULE)
    attr = module.funcs[0].ops[0].attrs["values"]
    with pytest.raises(ResourceDecodeError) as exc:
        attr.decode({})
    assert "w0" in str(exc.value)


def test_dense_resource_wrong_blob_length_raises():
    module = parse_module(_RESOURCE_MODULE)
    attr = module.funcs[0].ops[0].attrs["values"]
    with pytest.raises(ResourceDecodeError) as exc:
        attr.decode({"w0": bytes.fromhex("0400000001FF")})
    assert "2 bytes" in str(exc.value)


def test_unterminated_metadata_block_raises_parse_error():
    with pytest.raises(MlirParseError):
        parse_module(_wrap('%0 = "t.op"() : () -> i32') + "\n{-#\n  dialect_resources: {\n")


def test_dense_hex_blob_decodes():
    text = _wrap('%0 = "t.op"() <{v = dense<"0x01FF0207"> : tensor<4xi8>}> : () -> i32')
    attr = parse_module(text).funcs[0].ops[0].attrs["v"]
    # An inline dense blob has NO alignment header, unlike a resource blob.
    assert attr.blob == bytes.fromhex("01FF0207")
    assert attr.elements() == (1, -1, 2, 7)


def test_dense_hex_blob_i32_is_little_endian():
    text = _wrap('%0 = "t.op"() <{v = dense<"0x01000000FFFFFFFF"> : tensor<2xi32>}> : () -> i32')
    assert parse_module(text).funcs[0].ops[0].attrs["v"].elements() == (1, -1)


def test_dense_hex_blob_odd_digit_count_raises():
    with pytest.raises(MlirParseError):
        parse_module(_wrap('%0 = "t.op"() <{v = dense<"0x01F"> : tensor<2xi8>}> : () -> i32'))


def test_parse_resources_reads_flat_entries():
    resources = parse_resources('dialect_resources: { builtin: { a.b_1: "0x00FF", c: "0x" } }')
    assert resources == {"a.b_1": b"\x00\xff", "c": b""}
