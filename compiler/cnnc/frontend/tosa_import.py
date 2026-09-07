"""TOSA (MLIR generic form) -> GIR importer (doc/tosa_compiler_plan.md §2.1,
§6, M2). Supports `tosa.const`, `tosa.conv2d`, `tosa.rescale`, `tosa.clamp`
and `func.return`; anything else raises `UnsupportedOp`.

Zero points, `rescale` multiplier/shift are TOSA *operands*, not
attributes; they must resolve to a `tosa.const` (folded into GIR attrs,
never materialised as a GIR tensor) or import fails with `TosaImportError`.
A `tosa.const` that *is* used as data (weight, bias, ...) becomes a GIR
`const` op, emitted lazily the first time it is referenced that way so
`print_gir` output only ever shows tensors the graph actually carries.
"""

from __future__ import annotations

from pathlib import Path

from cnnc.errors import TosaImportError, UnsupportedAttribute, UnsupportedOp
from cnnc.frontend.mlir_generic import (
    BoolAttr,
    DenseArrayAttr,
    DenseElementsAttr,
    EnumAttr,
    IntAttr,
    MlirFunc,
    MlirModule,
    MlirOp,
    TensorType,
    TypeAttr,
    parse_file,
)
from cnnc.gir.ir import DTYPES, ClampAttrs, ConvAttrs, Graph, Op, RescaleParams, Tensor, conv2d_output_shape
from cnnc.gir.verify import verify

_STAGE = "import"


def _check_dtype(op_id: str, dtype: str) -> None:
    if dtype in DTYPES:
        return
    if dtype.startswith("f") or dtype == "bf16":
        raise TosaImportError(f"floating point dtype {dtype!r} is not supported", op_id=op_id, stage=_STAGE)
    raise TosaImportError(f"unsupported dtype {dtype!r} (expected one of {DTYPES})", op_id=op_id, stage=_STAGE)


def _tensor_from_const_op(const_op: MlirOp) -> Tensor:
    name, ttype = const_op.results[0]
    op_id = f"%{name}"
    if not isinstance(ttype, TensorType):
        raise TosaImportError("tosa.const result must be a tensor", op_id=op_id, stage=_STAGE)
    _check_dtype(op_id, str(ttype.dtype))
    dea = const_op.attrs.get("values")
    if not isinstance(dea, DenseElementsAttr):
        raise TosaImportError("tosa.const missing 'values' dense attribute", op_id=op_id, stage=_STAGE)
    numel = ttype.numel
    if dea.splat is not None:
        values = tuple([dea.splat] * numel)
    else:
        values = tuple(dea.values) if dea.values else ()
    return Tensor(id=name, shape=ttype.shape, dtype=str(ttype.dtype), values=values)


def _resolve_data_operand(
    name: str,
    tensors: dict[str, Tensor],
    const_defs: dict[str, MlirOp],
    used_consts: list[str],
    op_id: str,
    what: str,
) -> Tensor:
    if name in tensors:
        return tensors[name]
    if name in const_defs:
        t = _tensor_from_const_op(const_defs[name])
        tensors[name] = t
        used_consts.append(name)
        return t
    raise TosaImportError(f"{what} operand %{name} is not defined", op_id=op_id, stage=_STAGE)


def _array_const_values(const_defs: dict[str, MlirOp], name: str, op_id: str, what: str) -> tuple[int, ...]:
    const_op = const_defs.get(name)
    if const_op is None:
        raise TosaImportError(f"{what} must be a constant", op_id=op_id, stage=_STAGE)
    _, ttype = const_op.results[0]
    if not isinstance(ttype, TensorType):
        raise TosaImportError(f"{what} constant must be a tensor", op_id=op_id, stage=_STAGE)
    dea = const_op.attrs.get("values")
    if not isinstance(dea, DenseElementsAttr):
        raise TosaImportError(f"{what} constant missing 'values' attribute", op_id=op_id, stage=_STAGE)
    if dea.splat is not None:
        return tuple([dea.splat] * ttype.numel)
    return tuple(dea.values) if dea.values else ()


def _scalar_const_value(const_defs: dict[str, MlirOp], name: str, op_id: str, what: str) -> int:
    const_op = const_defs.get(name)
    if const_op is None:
        raise TosaImportError(f"{what} must be a constant", op_id=op_id, stage=_STAGE)
    _, ttype = const_op.results[0]
    if not isinstance(ttype, TensorType) or ttype.numel != 1:
        raise TosaImportError(f"{what} must be a constant with exactly one element", op_id=op_id, stage=_STAGE)
    values = _array_const_values(const_defs, name, op_id, what)
    return values[0]


def _require_dense_array(attrs: dict, key: str, op_id: str, length: int) -> tuple[int, ...]:
    attr = attrs.get(key)
    if not isinstance(attr, DenseArrayAttr):
        raise TosaImportError(f"missing or invalid {key!r} attribute", op_id=op_id, stage=_STAGE)
    if len(attr.values) != length:
        raise TosaImportError(
            f"{key!r} must have {length} elements, got {len(attr.values)}", op_id=op_id, stage=_STAGE
        )
    return tuple(attr.values)


def _require_bool_attr(attrs: dict, key: str, op_id: str) -> bool:
    attr = attrs.get(key)
    if not isinstance(attr, BoolAttr):
        raise TosaImportError(f"missing or invalid {key!r} attribute", op_id=op_id, stage=_STAGE)
    return attr.value


def _import_conv2d(op: MlirOp, tensors: dict[str, Tensor], const_defs: dict[str, MlirOp], used_consts: list[str]) -> Op:
    result_name, result_type = op.results[0]
    op_id = f"%{result_name}"
    if len(op.operands) != 5:
        raise TosaImportError(f"tosa.conv2d expects 5 operands, got {len(op.operands)}", op_id=op_id, stage=_STAGE)
    x_name, w_name, b_name, izp_name, wzp_name = op.operands

    x = _resolve_data_operand(x_name, tensors, const_defs, used_consts, op_id, "input")
    _check_dtype(op_id, x.dtype)
    if len(x.shape) != 4:
        raise TosaImportError(f"conv2d input rank {len(x.shape)} != 4", op_id=op_id, stage=_STAGE)
    if x.shape[0] != 1:
        raise UnsupportedAttribute(f"batch size {x.shape[0]} != 1", op_id=op_id, stage=_STAGE)

    w = _resolve_data_operand(w_name, tensors, const_defs, used_consts, op_id, "weight")
    _check_dtype(op_id, w.dtype)
    b = _resolve_data_operand(b_name, tensors, const_defs, used_consts, op_id, "bias")
    _check_dtype(op_id, b.dtype)

    in_zp = _scalar_const_value(const_defs, izp_name, op_id, "input zero point")
    w_zp = _scalar_const_value(const_defs, wzp_name, op_id, "weight zero point")

    pad = _require_dense_array(op.attrs, "pad", op_id, 4)
    stride = _require_dense_array(op.attrs, "stride", op_id, 2)
    dilation = _require_dense_array(op.attrs, "dilation", op_id, 2)
    if dilation != (1, 1):
        raise UnsupportedAttribute(f"dilation {dilation} != (1, 1)", op_id=op_id, stage=_STAGE)

    acc_attr = op.attrs.get("acc_type")
    if not isinstance(acc_attr, TypeAttr) or str(acc_attr.type) != "i32":
        raise UnsupportedAttribute(f"acc_type must be i32, got {acc_attr}", op_id=op_id, stage=_STAGE)

    attrs = ConvAttrs(pad=pad, stride=stride, dilation=dilation, in_zp=in_zp, w_zp=w_zp, acc_dtype="i32")
    computed = conv2d_output_shape(x.shape, w.shape, attrs)

    if not isinstance(result_type, TensorType):
        raise TosaImportError("tosa.conv2d result must be a tensor", op_id=op_id, stage=_STAGE)
    if result_type.shape != computed:
        raise TosaImportError(
            f"declared result type {result_type.shape} != computed {computed}", op_id=op_id, stage=_STAGE
        )
    _check_dtype(op_id, str(result_type.dtype))

    out = Tensor(id=result_name, shape=computed, dtype=str(result_type.dtype))
    tensors[result_name] = out
    return Op(id=op_id, kind="conv2d", inputs=(x.id, w.id, b.id), outputs=(result_name,), attrs=attrs)


def _import_rescale(op: MlirOp, tensors: dict[str, Tensor], const_defs: dict[str, MlirOp], used_consts: list[str]) -> Op:
    result_name, result_type = op.results[0]
    op_id = f"%{result_name}"
    if len(op.operands) != 5:
        raise TosaImportError(f"tosa.rescale expects 5 operands, got {len(op.operands)}", op_id=op_id, stage=_STAGE)
    x_name, mult_name, shift_name, izp_name, ozp_name = op.operands

    x = _resolve_data_operand(x_name, tensors, const_defs, used_consts, op_id, "input")
    _check_dtype(op_id, x.dtype)

    scale32 = _require_bool_attr(op.attrs, "scale32", op_id)
    if not scale32:
        raise UnsupportedAttribute("scale32 must be true", op_id=op_id, stage=_STAGE)
    per_channel = _require_bool_attr(op.attrs, "per_channel", op_id)
    input_unsigned = _require_bool_attr(op.attrs, "input_unsigned", op_id)
    if input_unsigned:
        raise UnsupportedAttribute("input_unsigned must be false", op_id=op_id, stage=_STAGE)
    output_unsigned = _require_bool_attr(op.attrs, "output_unsigned", op_id)
    if output_unsigned:
        raise UnsupportedAttribute("output_unsigned must be false", op_id=op_id, stage=_STAGE)

    rounding_attr = op.attrs.get("rounding_mode")
    if not isinstance(rounding_attr, EnumAttr):
        raise TosaImportError("missing or invalid 'rounding_mode' attribute", op_id=op_id, stage=_STAGE)
    rounding = rounding_attr.value

    multiplier = _array_const_values(const_defs, mult_name, op_id, "multiplier")
    shift = _array_const_values(const_defs, shift_name, op_id, "shift")
    in_zp = _scalar_const_value(const_defs, izp_name, op_id, "input zero point")
    out_zp = _scalar_const_value(const_defs, ozp_name, op_id, "output zero point")

    if not isinstance(result_type, TensorType):
        raise TosaImportError("tosa.rescale result must be a tensor", op_id=op_id, stage=_STAGE)
    _check_dtype(op_id, str(result_type.dtype))

    attrs = RescaleParams(
        multiplier=multiplier,
        shift=shift,
        per_channel=per_channel,
        in_zp=in_zp,
        out_zp=out_zp,
        rounding=rounding,
        scale32=scale32,
        input_unsigned=input_unsigned,
        output_unsigned=output_unsigned,
    )
    out = Tensor(id=result_name, shape=result_type.shape, dtype=str(result_type.dtype))
    tensors[result_name] = out
    return Op(id=op_id, kind="rescale", inputs=(x.id,), outputs=(result_name,), attrs=attrs)


def _import_clamp(op: MlirOp, tensors: dict[str, Tensor], const_defs: dict[str, MlirOp], used_consts: list[str]) -> Op:
    result_name, result_type = op.results[0]
    op_id = f"%{result_name}"
    if len(op.operands) != 1:
        raise TosaImportError(f"tosa.clamp expects 1 operand, got {len(op.operands)}", op_id=op_id, stage=_STAGE)

    x = _resolve_data_operand(op.operands[0], tensors, const_defs, used_consts, op_id, "input")
    _check_dtype(op_id, x.dtype)

    min_attr = op.attrs.get("min_val")
    max_attr = op.attrs.get("max_val")
    if not isinstance(min_attr, IntAttr) or not isinstance(max_attr, IntAttr):
        raise UnsupportedAttribute("clamp min_val/max_val must be integer-typed", op_id=op_id, stage=_STAGE)
    min_val, max_val = min_attr.value, max_attr.value

    if not isinstance(result_type, TensorType):
        raise TosaImportError("tosa.clamp result must be a tensor", op_id=op_id, stage=_STAGE)
    _check_dtype(op_id, str(result_type.dtype))

    out = Tensor(id=result_name, shape=result_type.shape, dtype=str(result_type.dtype))
    tensors[result_name] = out
    attrs = ClampAttrs(min=min_val, max=max_val)
    return Op(id=op_id, kind="clamp", inputs=(x.id,), outputs=(result_name,), attrs=attrs)


def _select_func(module: MlirModule, func_name: str | None) -> MlirFunc:
    if func_name is not None:
        for f in module.funcs:
            if f.name == func_name:
                return f
        raise TosaImportError(f"function {func_name!r} not found in module", stage=_STAGE)
    if len(module.funcs) != 1:
        raise TosaImportError(
            f"expected exactly one function, found {len(module.funcs)}; pass func_name= to select one",
            stage=_STAGE,
        )
    return module.funcs[0]


def import_tosa(module: MlirModule, func_name: str | None = None) -> Graph:
    func = _select_func(module, func_name)

    tensors: dict[str, Tensor] = {}
    for name, t in zip(func.arg_names, func.arg_types):
        arg_id = f"%{name}"
        if not isinstance(t, TensorType):
            raise TosaImportError("function argument must be a tensor", op_id=arg_id, stage=_STAGE)
        _check_dtype(arg_id, str(t.dtype))
        tensors[name] = Tensor(id=name, shape=t.shape, dtype=str(t.dtype))
    inputs = tuple(func.arg_names)

    const_defs: dict[str, MlirOp] = {op.results[0][0]: op for op in func.ops if op.name == "tosa.const"}

    ops: list[Op] = []
    outputs: tuple[str, ...] | None = None
    for op in func.ops:
        if op.name == "tosa.const":
            continue  # materialised lazily, only if referenced as data
        if op.name == "func.return":
            outputs = tuple(op.operands)
            continue

        result_name = op.results[0][0] if op.results else None
        op_id = f"%{result_name}" if result_name is not None else f"<{op.name}>"
        used_consts: list[str] = []
        if op.name == "tosa.conv2d":
            gir_op = _import_conv2d(op, tensors, const_defs, used_consts)
        elif op.name == "tosa.rescale":
            gir_op = _import_rescale(op, tensors, const_defs, used_consts)
        elif op.name == "tosa.clamp":
            gir_op = _import_clamp(op, tensors, const_defs, used_consts)
        else:
            raise UnsupportedOp(f"unsupported op {op.name!r}", op_id=op_id, stage=_STAGE)

        for cname in used_consts:
            ops.append(Op(id=f"%{cname}", kind="const", inputs=(), outputs=(cname,), attrs=None))
        ops.append(gir_op)

    if outputs is None:
        raise TosaImportError("function has no 'func.return'", stage=_STAGE)

    graph = Graph(name=func.name, tensors=tensors, ops=tuple(ops), inputs=inputs, outputs=outputs)
    verify(graph)
    return graph


def load_tosa_file(path: str | Path, func_name: str | None = None) -> Graph:
    return import_tosa(parse_file(path), func_name=func_name)
