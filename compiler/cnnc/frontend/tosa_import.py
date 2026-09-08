"""TOSA (MLIR generic form) -> GIR importer (doc/tosa_compiler_plan.md §2.1,
§6, M2). `_IMPORTERS` (surfaced as `supported_ops()`) is the authoritative
list of what is accepted -- today `tosa.conv2d`, `tosa.rescale`,
`tosa.clamp`, `tosa.max_pool2d`, plus `tosa.const` and `func.return`, which
are structural rather than computational.

Everything else raises `UnsupportedOp`, and the message distinguishes three
genuinely different situations, because the reader's next action differs:

* `_HOST_SIDE_HEAD_OPS` -- YOLOv8n detection-head ops (softmax pieces,
  elementwise multiply, reshape/transpose). The accelerator is a
  backbone/neck engine by design; the fix is to split the graph, not to
  write more compiler.
* `_INEXACT_OPS` -- the accelerator HAS the opcode, but it does not compute
  what TOSA specifies (`tosa.avg_pool2d` today). The fix is a numeric
  equivalence argument, and emitting the opcode meanwhile would be
  silently wrong.
* everything else -- simply not implemented yet.

Zero points, `rescale` multiplier/shift are TOSA *operands*, not
attributes; they must resolve to a `tosa.const` (folded into GIR attrs,
never materialised as a GIR tensor) or import fails with `TosaImportError`.
A `tosa.const` that *is* used as data (weight, bias, ...) becomes a GIR
`const` op, emitted lazily the first time it is referenced that way so
`print_gir` output only ever shows tensors the graph actually carries.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from cnnc.errors import TosaImportError, UnsupportedAttribute, UnsupportedOp
from cnnc.frontend.mlir_generic import (
    BoolAttr,
    DenseArrayAttr,
    DenseElementsAttr,
    DenseResourceAttr,
    EnumAttr,
    IntAttr,
    MlirFunc,
    MlirModule,
    MlirOp,
    ResourceDecodeError,
    TensorType,
    TypeAttr,
    parse_file,
)
from cnnc.gir.ir import (
    DTYPES,
    ClampAttrs,
    ConvAttrs,
    Graph,
    Op,
    PoolAttrs,
    RescaleParams,
    Tensor,
    conv2d_output_shape,
    dtype_range,
    pool2d_output_shape,
)
from cnnc.gir.verify import verify

_STAGE = "import"

# TOSA ops that belong to YOLOv8n's *detection head*, which by design runs
# on the host, not on the accelerator (doc/tosa_compiler_plan.md: the
# accelerator is a backbone/neck engine -- it has no softmax, no
# elementwise multiply, and no data-layout instruction). Listed separately
# from "not implemented yet" so the diagnostic can say which it is: there
# is nothing to add here, the graph has to be split.
_HOST_SIDE_HEAD_OPS = {
    "tosa.mul": "elementwise multiply",
    "tosa.sigmoid": "sigmoid",
    "tosa.exp": "exponential",
    "tosa.reciprocal": "reciprocal",
    "tosa.reduce_sum": "sum reduction",
    "tosa.reduce_max": "max reduction",
    "tosa.reshape": "reshape",
    "tosa.transpose": "transpose",
}


def _op_id(op: MlirOp) -> str:
    """The diagnostic id for `op`: its SSA result name plus the op name and
    source location, so a message about a 477-op real-world module says
    exactly which line to look at (`%123 (tosa.conv2d at 208:11)`)."""
    line, col = op.loc
    name = f"%{op.results[0][0]}" if op.results else f"<{op.name}>"
    return f"{name} ({op.name} at {line}:{col})"


def _check_dtype(op_id: str, dtype: str) -> None:
    if dtype in DTYPES:
        return
    if dtype.startswith("f") or dtype == "bf16":
        raise TosaImportError(f"floating point dtype {dtype!r} is not supported", op_id=op_id, stage=_STAGE)
    raise TosaImportError(f"unsupported dtype {dtype!r} (expected one of {DTYPES})", op_id=op_id, stage=_STAGE)


def _const_elements(const_op: MlirOp, op_id: str, resources: dict[str, bytes], what: str) -> tuple[int, ...]:
    """The flat, row-major elements of a `tosa.const`'s `values`
    attribute, in whichever of the four printed forms it took: a splat, a
    value list, an inline `dense<"0x...">` hex blob, or a
    `dense_resource<key>` pointing into the file's trailing
    `dialect_resources` block. A blob that cannot be decoded is a
    `TosaImportError` naming the op, never a `struct.error` traceback."""
    attr = const_op.attrs.get("values")
    try:
        if isinstance(attr, DenseResourceAttr):
            return attr.decode(resources)
        if isinstance(attr, DenseElementsAttr):
            return attr.elements()
    except ResourceDecodeError as exc:
        raise TosaImportError(f"{what}: {exc}", op_id=op_id, stage=_STAGE) from exc
    raise TosaImportError(f"{what} missing a 'values' dense attribute", op_id=op_id, stage=_STAGE)


def _tensor_from_const_op(const_op: MlirOp, resources: dict[str, bytes]) -> Tensor:
    name, ttype = const_op.results[0]
    op_id = _op_id(const_op)
    if not isinstance(ttype, TensorType):
        raise TosaImportError("tosa.const result must be a tensor", op_id=op_id, stage=_STAGE)
    _check_dtype(op_id, str(ttype.dtype))
    values = _const_elements(const_op, op_id, resources, "tosa.const")
    return Tensor(id=name, shape=ttype.shape, dtype=str(ttype.dtype), values=values)


def _resolve_data_operand(
    name: str,
    tensors: dict[str, Tensor],
    const_defs: dict[str, MlirOp],
    used_consts: list[str],
    op_id: str,
    what: str,
    resources: dict[str, bytes],
) -> Tensor:
    if name in tensors:
        return tensors[name]
    if name in const_defs:
        t = _tensor_from_const_op(const_defs[name], resources)
        tensors[name] = t
        used_consts.append(name)
        return t
    raise TosaImportError(f"{what} operand %{name} is not defined", op_id=op_id, stage=_STAGE)


def _array_const_values(
    const_defs: dict[str, MlirOp], name: str, op_id: str, what: str, resources: dict[str, bytes]
) -> tuple[int, ...]:
    const_op = const_defs.get(name)
    if const_op is None:
        raise TosaImportError(f"{what} must be a constant", op_id=op_id, stage=_STAGE)
    _, ttype = const_op.results[0]
    if not isinstance(ttype, TensorType):
        raise TosaImportError(f"{what} constant must be a tensor", op_id=op_id, stage=_STAGE)
    return _const_elements(const_op, op_id, resources, f"{what} constant")


def _scalar_const_value(
    const_defs: dict[str, MlirOp], name: str, op_id: str, what: str, resources: dict[str, bytes]
) -> int:
    const_op = const_defs.get(name)
    if const_op is None:
        raise TosaImportError(f"{what} must be a constant", op_id=op_id, stage=_STAGE)
    _, ttype = const_op.results[0]
    if not isinstance(ttype, TensorType) or ttype.numel != 1:
        raise TosaImportError(f"{what} must be a constant with exactly one element", op_id=op_id, stage=_STAGE)
    values = _array_const_values(const_defs, name, op_id, what, resources)
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


@dataclasses.dataclass
class _Ctx:
    """Everything an `_import_*` helper needs beyond the op itself.

    `used_consts` is per-op scratch (the `tosa.const`s this op pulled in
    as data, so the caller can emit their GIR `const` ops just before it)
    and is cleared by `take_used_consts` after each op; the other three
    fields live for the whole function."""

    tensors: dict[str, Tensor]
    const_defs: dict[str, MlirOp]
    resources: dict[str, bytes]
    used_consts: list[str] = dataclasses.field(default_factory=list)

    def take_used_consts(self) -> list[str]:
        used, self.used_consts = self.used_consts, []
        return used

    def data_operand(self, name: str, op_id: str, what: str) -> Tensor:
        return _resolve_data_operand(
            name, self.tensors, self.const_defs, self.used_consts, op_id, what, self.resources
        )

    def array_const(self, name: str, op_id: str, what: str) -> tuple[int, ...]:
        return _array_const_values(self.const_defs, name, op_id, what, self.resources)

    def scalar_const(self, name: str, op_id: str, what: str) -> int:
        return _scalar_const_value(self.const_defs, name, op_id, what, self.resources)

    def define(self, tensor: Tensor) -> Tensor:
        self.tensors[tensor.id] = tensor
        return tensor


def _import_conv2d(op: MlirOp, ctx: _Ctx) -> Op:
    result_name, result_type = op.results[0]
    op_id = _op_id(op)
    if len(op.operands) != 5:
        raise TosaImportError(f"tosa.conv2d expects 5 operands, got {len(op.operands)}", op_id=op_id, stage=_STAGE)
    x_name, w_name, b_name, izp_name, wzp_name = op.operands

    x = ctx.data_operand(x_name, op_id, "input")
    _check_dtype(op_id, x.dtype)
    if len(x.shape) != 4:
        raise TosaImportError(f"conv2d input rank {len(x.shape)} != 4", op_id=op_id, stage=_STAGE)
    if x.shape[0] != 1:
        raise UnsupportedAttribute(f"batch size {x.shape[0]} != 1", op_id=op_id, stage=_STAGE)

    w = ctx.data_operand(w_name, op_id, "weight")
    _check_dtype(op_id, w.dtype)
    b = ctx.data_operand(b_name, op_id, "bias")
    _check_dtype(op_id, b.dtype)

    in_zp = ctx.scalar_const(izp_name, op_id, "input zero point")
    w_zp = ctx.scalar_const(wzp_name, op_id, "weight zero point")

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
    ctx.define(out)
    return Op(id=f"%{result_name}", kind="conv2d", inputs=(x.id, w.id, b.id), outputs=(result_name,), attrs=attrs)


def _import_rescale(op: MlirOp, ctx: _Ctx) -> Op:
    result_name, result_type = op.results[0]
    op_id = _op_id(op)
    if len(op.operands) != 5:
        raise TosaImportError(f"tosa.rescale expects 5 operands, got {len(op.operands)}", op_id=op_id, stage=_STAGE)
    x_name, mult_name, shift_name, izp_name, ozp_name = op.operands

    x = ctx.data_operand(x_name, op_id, "input")
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

    multiplier = ctx.array_const(mult_name, op_id, "multiplier")
    shift = ctx.array_const(shift_name, op_id, "shift")
    in_zp = ctx.scalar_const(izp_name, op_id, "input zero point")
    out_zp = ctx.scalar_const(ozp_name, op_id, "output zero point")

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
    ctx.define(out)
    return Op(id=f"%{result_name}", kind="rescale", inputs=(x.id,), outputs=(result_name,), attrs=attrs)


def _import_max_pool2d(op: MlirOp, ctx: _Ctx) -> Op:
    """`tosa.max_pool2d` -> GIR `pool` (mode `"max"`).

    TOSA pads MAX_POOL2D with the *minimum representable value* of the
    element type, not with zero and not with a zero-point, so `pad_value`
    is derived from the input dtype here and pinned by the GIR verifier.
    That is exactly the accelerator's ISA v2.1 `pad_value` semantics, so
    padded max pooling (YOLOv8n's SPPF is 5x5/stride 1/pad 2) lowers with
    no approximation."""
    result_name, result_type = op.results[0]
    op_id = _op_id(op)
    if len(op.operands) != 1:
        raise TosaImportError(
            f"tosa.max_pool2d expects 1 operand, got {len(op.operands)}", op_id=op_id, stage=_STAGE
        )

    x = ctx.data_operand(op.operands[0], op_id, "input")
    _check_dtype(op_id, x.dtype)
    if len(x.shape) != 4:
        raise TosaImportError(f"max_pool2d input rank {len(x.shape)} != 4", op_id=op_id, stage=_STAGE)
    if x.shape[0] != 1:
        raise UnsupportedAttribute(f"batch size {x.shape[0]} != 1", op_id=op_id, stage=_STAGE)

    kernel = _require_dense_array(op.attrs, "kernel", op_id, 2)
    stride = _require_dense_array(op.attrs, "stride", op_id, 2)
    pad = _require_dense_array(op.attrs, "pad", op_id, 4)

    # `nan_mode` only distinguishes NaN handling, which cannot arise for an
    # integer element type; accepted (and ignored) for any value.
    attrs = PoolAttrs(mode="max", kernel=kernel, stride=stride, pad=pad, pad_value=dtype_range(x.dtype)[0])
    computed = pool2d_output_shape(x.shape, attrs)

    if not isinstance(result_type, TensorType):
        raise TosaImportError("tosa.max_pool2d result must be a tensor", op_id=op_id, stage=_STAGE)
    _check_dtype(op_id, str(result_type.dtype))
    if result_type.shape != computed:
        raise TosaImportError(
            f"declared result type {result_type.shape} != computed {computed}", op_id=op_id, stage=_STAGE
        )

    ctx.define(Tensor(id=result_name, shape=computed, dtype=str(result_type.dtype)))
    return Op(id=f"%{result_name}", kind="pool", inputs=(x.id,), outputs=(result_name,), attrs=attrs)


def _import_clamp(op: MlirOp, ctx: _Ctx) -> Op:
    result_name, result_type = op.results[0]
    op_id = _op_id(op)
    if len(op.operands) != 1:
        raise TosaImportError(f"tosa.clamp expects 1 operand, got {len(op.operands)}", op_id=op_id, stage=_STAGE)

    x = ctx.data_operand(op.operands[0], op_id, "input")
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
    ctx.define(out)
    attrs = ClampAttrs(min=min_val, max=max_val)
    return Op(id=f"%{result_name}", kind="clamp", inputs=(x.id,), outputs=(result_name,), attrs=attrs)


#: TOSA op name -> importer. The single dispatch table `import_tosa` and
#: `supported_ops` both read, so "what does this frontend accept?" has one
#: answer rather than an `elif` chain and a docstring that can drift.
_IMPORTERS = {
    "tosa.conv2d": _import_conv2d,
    "tosa.rescale": _import_rescale,
    "tosa.clamp": _import_clamp,
    "tosa.max_pool2d": _import_max_pool2d,
}

#: TOSA ops the accelerator has an opcode for, but whose TOSA semantics
#: that opcode does not reproduce. Kept apart from `_HOST_SIDE_HEAD_OPS`
#: (which the accelerator is not meant to run at all) and from the
#: catch-all "not implemented" case, because the right response is
#: different again: these need a numeric equivalence argument, not more
#: code. Emitting the opcode anyway would be silently wrong.
_INEXACT_OPS = {
    "tosa.avg_pool2d": (
        "the target's POOL_AVG is count-include-pad and divides through the half_up epilogue "
        "requantizer, whereas TOSA excludes padded taps from the count and divides with "
        "apply_scale_32(reciprocal_scale(count)). Emitting POOL_AVG would be silently wrong on "
        "every padded window, and differ by a rounding step even without padding"
    ),
}


def supported_ops() -> tuple[str, ...]:
    """Every TOSA op name this frontend can import, sorted."""
    return tuple(sorted(_IMPORTERS))


def _reject_op(op: MlirOp) -> None:
    """Refuse `op` with a diagnostic that says *why* it is refused --
    host-side detection head vs simply not implemented -- and where it is."""
    op_id = _op_id(op)
    inexact = _INEXACT_OPS.get(op.name)
    if inexact is not None:
        raise UnsupportedOp(f"{op.name} is not lowered: {inexact}", op_id=op_id, stage=_STAGE)
    what = _HOST_SIDE_HEAD_OPS.get(op.name)
    if what is not None:
        raise UnsupportedOp(
            f"{op.name} ({what}) belongs to the host-side detection head, which by design does not "
            f"run on the accelerator: it has no instruction for it. Split the graph so the "
            f"backbone/neck ends before this op and run the head on the host.",
            op_id=op_id, stage=_STAGE,
        )
    raise UnsupportedOp(
        f"unsupported op {op.name!r}; this frontend imports {', '.join(supported_ops())}",
        op_id=op_id, stage=_STAGE,
    )


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
    ctx = _Ctx(tensors=tensors, const_defs=const_defs, resources=module.resources)

    ops: list[Op] = []
    outputs: tuple[str, ...] | None = None
    for op in func.ops:
        if op.name == "tosa.const":
            continue  # materialised lazily, only if referenced as data
        if op.name == "func.return":
            outputs = tuple(op.operands)
            continue

        importer = _IMPORTERS.get(op.name)
        if importer is None:
            _reject_op(op)
        gir_op = importer(op, ctx)

        for cname in ctx.take_used_consts():
            ops.append(Op(id=f"%{cname}", kind="const", inputs=(), outputs=(cname,), attrs=None))
        ops.append(gir_op)

    if outputs is None:
        raise TosaImportError("function has no 'func.return'", stage=_STAGE)

    graph = Graph(name=func.name, tensors=tensors, ops=tuple(ops), inputs=inputs, outputs=outputs)
    verify(graph)
    return graph


def load_tosa_file(path: str | Path, func_name: str | None = None) -> Graph:
    return import_tosa(parse_file(path), func_name=func_name)
