"""TOSA (MLIR generic form) -> GIR importer (doc/tosa_compiler_plan.md §2.1,
§6, M2). `_IMPORTERS` (surfaced as `supported_ops()`) is the authoritative
list of what is accepted -- today `tosa.conv2d`, `tosa.rescale`,
`tosa.clamp`, `tosa.max_pool2d`, `tosa.add`, `tosa.table`, `tosa.concat`
and `tosa.slice`, plus `tosa.const`/`tosa.const_shape` and `func.return`,
which are structural rather than computational. A nearest-2x upsample is
also imported, but as a `reshape`/`tile` CHAIN rather than as one op --
see `_match_upsample_chains`.

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
    CHANNEL_MAJOR,
    DTYPES,
    PLANE_MAJOR,
    AddAttrs,
    ClampAttrs,
    ConcatAttrs,
    ConvAttrs,
    DepthToSpaceAttrs,
    Graph,
    Op,
    PoolAttrs,
    RescaleParams,
    SliceAttrs,
    Tensor,
    UpsampleAttrs,
    conv2d_output_shape,
    depth_to_space_output_shape,
    dtype_range,
    pool2d_output_shape,
    upsample_output_shape,
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
    #: `tosa.const_shape` definitions. Separate from `const_defs` because
    #: a `!tosa.shape<N>` is never tensor data -- it is always folded into
    #: an attribute (a slice's start/size), never materialised as a GIR
    #: tensor.
    shape_defs: dict[str, MlirOp] = dataclasses.field(default_factory=dict)
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

    def shape_const(self, name: str, op_id: str, what: str) -> tuple[int, ...]:
        shape_op = self.shape_defs.get(name)
        if shape_op is None:
            raise TosaImportError(
                f"{what} operand %{name} must be a tosa.const_shape", op_id=op_id, stage=_STAGE
            )
        return _const_elements(shape_op, op_id, self.resources, what)

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


#: The identity rescale in TOSA `apply_scale_32` terms:
#: `(v * 2**15 + 2**14) >> 15 == v` for every `v`, exactly. A bare
#: `tosa.add` rescales neither operand, so this is what it imports as;
#: `passes.fuse` replaces it when it folds two real `tosa.rescale`s in.
_IDENTITY_RESCALE = (1 << 15, 15)

#: A TOSA int8 `TABLE` operand is exactly `2**8` entries; so is the
#: accelerator's ACT LUT (`lower.layout.ACT_LUT_ENTRIES`).
_ACT_LUT_ENTRIES = 256


def _import_add(op: MlirOp, ctx: _Ctx) -> Op:
    """`tosa.add` -> GIR `add` (the residual/skip-connection shortcut).

    Only the int8 form is imported, and only with both operands the same
    shape: the accelerator's `OPCODE_ADD` walks two ifmaps of one
    geometry lane by lane, so it has no broadcasting to offer, and TOSA's
    rank-1 broadcast would need a materialised copy the compiler does not
    emit.

    A bare add carries the identity rescale. A *quantized* residual add
    reaches here as `rescale(a) -> rescale(b) -> add`, and `passes.fuse`
    folds those two rescales into this op's `AddAttrs` -- which is the
    only shape the hardware can execute, since its descriptor holds one
    `(requant_scale, requant_shift)` pair for both operands.
    """
    result_name, result_type = op.results[0]
    op_id = _op_id(op)
    if len(op.operands) != 2:
        raise TosaImportError(f"tosa.add expects 2 operands, got {len(op.operands)}", op_id=op_id, stage=_STAGE)

    a = ctx.data_operand(op.operands[0], op_id, "lhs")
    b = ctx.data_operand(op.operands[1], op_id, "rhs")
    for what, t in (("lhs", a), ("rhs", b)):
        _check_dtype(op_id, t.dtype)
        if t.dtype != "i8":
            raise UnsupportedAttribute(
                f"tosa.add {what} dtype {t.dtype!r} != 'i8': the accelerator's ADD is an int8 "
                "elementwise opcode. An i32 add (the unfused quantized idiom "
                "'rescale i8->i32, add i32, rescale i32->i8') has no instruction; quantize the "
                "add itself to int8 first",
                op_id=op_id, stage=_STAGE,
            )
    if a.shape != b.shape:
        raise UnsupportedAttribute(
            f"tosa.add operand shapes {a.shape} != {b.shape}: the accelerator's ADD has no "
            "broadcasting (one geometry, two ifmaps)",
            op_id=op_id, stage=_STAGE,
        )
    if len(a.shape) != 4:
        raise TosaImportError(f"tosa.add operand rank {len(a.shape)} != 4 (NHWC)", op_id=op_id, stage=_STAGE)
    if a.shape[0] != 1:
        raise UnsupportedAttribute(f"batch size {a.shape[0]} != 1", op_id=op_id, stage=_STAGE)

    if not isinstance(result_type, TensorType):
        raise TosaImportError("tosa.add result must be a tensor", op_id=op_id, stage=_STAGE)
    _check_dtype(op_id, str(result_type.dtype))
    if result_type.shape != a.shape:
        raise TosaImportError(
            f"declared result type {result_type.shape} != operand shape {a.shape}", op_id=op_id, stage=_STAGE
        )

    multiplier, shift = _IDENTITY_RESCALE
    ctx.define(Tensor(id=result_name, shape=a.shape, dtype=str(result_type.dtype)))
    return Op(
        id=f"%{result_name}",
        kind="add",
        inputs=(a.id, b.id),
        outputs=(result_name,),
        attrs=AddAttrs(multiplier=multiplier, shift=shift),
    )


def _import_table(op: MlirOp, ctx: _Ctx) -> Op:
    """`tosa.table` -> GIR `table` (the accelerator's `OPCODE_ACT` LUT).

    This is how a general activation reaches the hardware: SiLU, sigmoid,
    tanh and friends have no closed-form instruction, but an int8 -> int8
    function is only 256 answers, and both TOSA and the accelerator say
    exactly that. Only the int8 form is imported (a 512-entry int16 TABLE
    would need interpolation the hardware does not do).

    The table stays a GIR *tensor* rather than becoming an attribute --
    like conv weights, it is 256 bytes of compile-time data that ends up
    as a const buffer in DDR, and keeping it a tensor is what lets the
    memory planner place it.
    """
    result_name, result_type = op.results[0]
    op_id = _op_id(op)
    if len(op.operands) != 2:
        raise TosaImportError(f"tosa.table expects 2 operands, got {len(op.operands)}", op_id=op_id, stage=_STAGE)

    x = ctx.data_operand(op.operands[0], op_id, "input")
    _check_dtype(op_id, x.dtype)
    if x.dtype != "i8":
        raise UnsupportedAttribute(
            f"tosa.table input dtype {x.dtype!r} != 'i8': the accelerator's ACT LUT is a "
            "256-entry int8 -> int8 table, with no int16 interpolation path",
            op_id=op_id, stage=_STAGE,
        )
    table = ctx.data_operand(op.operands[1], op_id, "table")
    _check_dtype(op_id, table.dtype)
    if table.dtype != "i8" or table.shape != (_ACT_LUT_ENTRIES,):
        raise UnsupportedAttribute(
            f"tosa.table table operand must be tensor<{_ACT_LUT_ENTRIES}xi8>, got "
            f"{table.shape} of {table.dtype}",
            op_id=op_id, stage=_STAGE,
        )
    if table.values is None:
        raise TosaImportError("tosa.table table operand must be a constant", op_id=op_id, stage=_STAGE)

    if not isinstance(result_type, TensorType):
        raise TosaImportError("tosa.table result must be a tensor", op_id=op_id, stage=_STAGE)
    _check_dtype(op_id, str(result_type.dtype))
    if result_type.shape != x.shape:
        raise TosaImportError(
            f"declared result type {result_type.shape} != input shape {x.shape}", op_id=op_id, stage=_STAGE
        )

    ctx.define(Tensor(id=result_name, shape=x.shape, dtype=str(result_type.dtype)))
    return Op(id=f"%{result_name}", kind="table", inputs=(x.id, table.id), outputs=(result_name,), attrs=None)


# ---------------------------------------------------------------------------
# The nearest-2x upsample idiom.
#
# TOSA has no upsample op that a real exporter reaches for. What YOLOv8n's
# TOSA actually contains -- all four of its `nn.Upsample(scale_factor=2)`
# layers, and zero `tosa.resize` -- is a five-op chain over a rank-5
# intermediate:
#
#     reshape -> tile -> reshape -> tile -> reshape
#
# (in the shipped NCHW artifact: 1x256x20x20 -> 1x256x20x1x20 -> tile
# [1,1,1,2,1] -> 1x256x40x20x1 -> tile [1,1,1,1,2] -> 1x256x40x40).
#
# This matcher does NOT pattern-match those shapes. Shape matching would
# be brittle in exactly the way that matters -- the same upsample written
# with the two tiles in the other order, or with the spatial dims in NHWC
# instead of NCHW, is the same computation and a different shape sequence,
# while a chain that differs by one transposed axis is a different
# computation and an identical-looking shape sequence. Instead the chain
# is *evaluated*: each tensor is tracked as the list of source element
# indices it holds (a reshape is the identity on that list, since
# row-major reshape does not move bytes; a tile is a modular gather), and
# the chain matches only if the final index list is exactly what
# nearest-neighbour replication would produce. That is a proof for the
# specific shapes in front of us, not a guess.
#
# Anything else built from `tosa.reshape`/`tosa.tile` is refused by
# `_reject_op` with its own diagnostic -- there is no partial credit here.
# ---------------------------------------------------------------------------

_CHAIN_OP_NAMES = ("tosa.reshape", "tosa.tile")

#: The only replication factor `OPCODE_UPSAMPLE` implements
#: (`cnn_accel_model.UPSAMPLE_FACTOR`; doc/cnn_accel_top_v2_arch.md
#: section 12 limitation 5). A 3x or 4x chain will simply not match, and
#: is then refused by name rather than lowered to the wrong instruction.
_UPSAMPLE_FACTOR = 2

#: Upper bound on the element count this matcher will evaluate a chain
#: over. The check is linear in the tensor's element count, and a chain
#: whose source is larger than this is not something the accelerator could
#: hold in one instruction anyway (`in_width`/`in_height` are 16-bit ISA
#: fields), so refusing to spend the time is not refusing a real program.
_MAX_CHAIN_ELEMENTS = 1 << 22


def _prod(shape: tuple[int, ...]) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def _tile_indices(indices: list[int], shape: tuple[int, ...], multiples: tuple[int, ...]) -> list[int]:
    """`tosa.tile`, applied to a list of source indices rather than to
    data: `out[i0, .., ik] = in[i0 % s0, .., ik % sk]`, row-major."""
    out_shape = tuple(s * m for s, m in zip(shape, multiples))
    strides = [1] * len(shape)
    for axis in range(len(shape) - 2, -1, -1):
        strides[axis] = strides[axis + 1] * shape[axis + 1]

    out: list[int] = []
    for flat in range(_prod(out_shape)):
        rest = flat
        source = 0
        for axis in range(len(out_shape) - 1, -1, -1):
            rest, index = divmod(rest, out_shape[axis])
            source += (index % shape[axis]) * strides[axis]
        out.append(indices[source])
    return out


def _nearest_upsample_indices(shape: tuple[int, int, int, int], factor: int) -> list[int]:
    """The source index of every element of `nearest_upsample(x, factor)`,
    for an NHWC `x` of `shape`, row-major -- i.e. what the chain's index
    list must equal, element for element, to be this upsample."""
    _n, h, w, c = shape
    out: list[int] = []
    for y in range(h * factor):
        row = (y // factor) * w
        for x in range(w * factor):
            base = (row + x // factor) * c
            out.extend(range(base, base + c))
    return out


def _shape_of(op: MlirOp) -> tuple[int, ...] | None:
    if not op.results:
        return None
    _name, ttype = op.results[0]
    return ttype.shape if isinstance(ttype, TensorType) else None


def _match_upsample_chains(
    func: MlirFunc,
) -> tuple[dict[str, tuple[str, int]], set[str], set[str]]:
    """Find every `reshape`/`tile` chain that computes a nearest-neighbour
    2x upsample.

    Returns `({final SSA name: (source SSA name, factor)}, {SSA names of
    the chain's interior ops}, {SSA names this matcher walked through
    without matching})`. `import_tosa` emits one GIR `upsample` where a
    chain ends and skips its interior; the third set is only used to pick
    a better diagnostic for what is left -- an op that was reached while
    following a `reshape`/`tile` chain gets told the chain did not compute
    a nearest-2x upsample, rather than being told it looks like a
    detection-head reshape.
    """
    use_count: dict[str, int] = {}
    for op in func.ops:
        for operand in op.operands:
            use_count[operand] = use_count.get(operand, 0) + 1

    producer: dict[str, MlirOp] = {op.results[0][0]: op for op in func.ops if op.results}
    consumers: dict[str, list[MlirOp]] = {}
    for op in func.ops:
        for operand in op.operands:
            consumers.setdefault(operand, []).append(op)

    chains: dict[str, tuple[str, int]] = {}
    interior: set[str] = set()
    attempted: set[str] = set()

    for start in func.ops:
        if start.name not in _CHAIN_OP_NAMES or not start.operands:
            continue
        source_name = start.operands[0]
        if source_name in interior:
            continue  # already inside a chain we matched
        source_producer = producer.get(source_name)
        if (
            source_producer is not None
            and source_producer.name in _CHAIN_OP_NAMES
            and source_name not in chains
        ):
            # Mid-chain: this op is not the root of its own chain. A source
            # that ENDS an already-matched chain is fine, though -- that is
            # a 4x upsample, written as two 2x chains back to back.
            continue

        source_shape: tuple[int, ...] | None = None
        if source_producer is not None:
            source_shape = _shape_of(source_producer)
        else:
            for name, ttype in zip(func.arg_names, func.arg_types):
                if name == source_name and isinstance(ttype, TensorType):
                    source_shape = ttype.shape
        if source_shape is None or len(source_shape) != 4 or source_shape[0] != 1:
            continue
        if _prod(source_shape) > _MAX_CHAIN_ELEMENTS:
            continue

        want = _nearest_upsample_indices(source_shape, _UPSAMPLE_FACTOR)
        want_shape = (
            1,
            source_shape[1] * _UPSAMPLE_FACTOR,
            source_shape[2] * _UPSAMPLE_FACTOR,
            source_shape[3],
        )

        shape: tuple[int, ...] = source_shape
        indices = list(range(_prod(source_shape)))
        current = start
        walked: list[str] = []
        matched = False
        while True:
            out_shape = _shape_of(current)
            if out_shape is None:
                break
            if current.name == "tosa.reshape":
                if _prod(out_shape) != _prod(shape):
                    break
                # Row-major reshape moves no data: the index list is
                # unchanged, only its interpretation.
                shape = out_shape
            else:  # tosa.tile
                if len(out_shape) != len(shape):
                    break
                if any(o % s or o < s for o, s in zip(out_shape, shape)):
                    break
                indices = _tile_indices(indices, shape, tuple(o // s for o, s in zip(out_shape, shape)))
                shape = out_shape

            name = current.results[0][0]
            if shape == want_shape and indices == want:
                chains[name] = (source_name, _UPSAMPLE_FACTOR)
                interior.update(walked)
                matched = True
                break
            walked.append(name)

            if use_count.get(name, 0) != 1:
                break  # a value read twice cannot be deleted with the chain
            (next_op,) = consumers[name]
            if next_op.name not in _CHAIN_OP_NAMES or next_op.operands[0] != name:
                break
            current = next_op

        if not matched:
            attempted.update(walked)
            attempted.add(start.results[0][0])

    return chains, interior, attempted


def _upsample_from_chain(
    op: MlirOp, ctx: _Ctx, source_name: str, factor: int
) -> Op:
    """Emit the GIR `upsample` a matched chain collapses to. `op` is the
    chain's LAST op, so its result name and type are the upsample's."""
    result_name, result_type = op.results[0]
    op_id = _op_id(op)
    x = ctx.data_operand(source_name, op_id, "upsample input")
    _check_dtype(op_id, x.dtype)
    if not isinstance(result_type, TensorType):
        raise TosaImportError("upsample chain result must be a tensor", op_id=op_id, stage=_STAGE)
    _check_dtype(op_id, str(result_type.dtype))
    if str(result_type.dtype) != x.dtype:
        raise TosaImportError(
            f"upsample chain changes dtype {x.dtype!r} -> {result_type.dtype!r}", op_id=op_id, stage=_STAGE
        )
    computed = upsample_output_shape(x.shape, factor)
    if result_type.shape != computed:
        raise TosaImportError(
            f"declared result type {result_type.shape} != computed {computed}", op_id=op_id, stage=_STAGE
        )
    ctx.define(Tensor(id=result_name, shape=computed, dtype=x.dtype))
    return Op(
        id=f"%{result_name}",
        kind="upsample",
        inputs=(x.id,),
        outputs=(result_name,),
        attrs=UpsampleAttrs(factor=factor),
    )


# ---------------------------------------------------------------------------
# The depth-to-space (pixel-shuffle / sub-pixel convolution) idiom.
#
# TOSA has no depth_to_space op either. What an exporter emits for
# `nn.PixelShuffle(r)` is a `reshape -> transpose -> reshape` chain over a
# rank-6 intermediate: split the channel axis into `[out_c, r, r]`,
# interleave the two new `r` axes with H and W, fold back down.
#
# Recognised exactly the way the upsample chain above is -- by EVALUATING
# the chain's index mapping, not by matching its shape sequence -- for the
# same reason: a rank-6 reshape/transpose chain is a very generic thing to
# write, and the difference between a pixel shuffle and an unrelated
# regrouping is entirely in which element lands where.
#
# What is new here, and has no analogue in `upsample`, is that there are
# TWO index mappings worth recognising, distinguished only by how the
# input channels are grouped:
#
#   plane-major    cin = (dy*r + dx)*out_c + c     <- what the hardware does
#   channel-major  cin = c*r**2 + dy*r + dx        <- what nn.PixelShuffle means
#
# Both are matched, and which one it was is recorded on the GIR op
# (`DepthToSpaceAttrs.channel_order`) rather than normalised away here:
# turning the channel-major one into the plane-major one means permuting
# the PRODUCING CONVOLUTION's output-channel rows, which is a graph
# rewrite over ops the frontend has not necessarily imported yet. That is
# `passes.depth_to_space_channels`' job; the frontend's job is to say,
# faithfully, which of the two the TOSA actually computes.
#
# (For `out_c == 1` the two mappings are literally the same function --
# `(dy*r+dx)*1 + 0 == 0*r**2 + dy*r + dx` -- so plane-major is tried
# first and such a chain is reported as needing no permutation at all.)
# ---------------------------------------------------------------------------

_DTS_CHAIN_OP_NAMES = ("tosa.reshape", "tosa.transpose")

#: Largest upscale factor this matcher will look for. Bounded because the
#: search tries every candidate factor whose square divides the channel
#: count; real sub-pixel convolutions are 2, 3 or 4.
_MAX_DTS_FACTOR = 8

#: The diagnostic for a `tosa.transpose`/`tosa.reshape` that was reached
#: while walking a depth-to-space-shaped chain that did not match. Kept
#: OUT of `_IDIOM_ONLY_OPS` on purpose: that table is consulted
#: unconditionally, and a `tosa.transpose` reached anywhere else really is
#: a host-side detection-head op and must keep saying so.
_DTS_IDIOM_MESSAGE = (
    "the accelerator's OPCODE_DEPTH_TO_SPACE is the pixel-shuffle step of a sub-pixel "
    "convolution -- it regroups factor**2 input channels into a factor-times-larger frame in "
    "both spatial dimensions -- and the only reshape/transpose chain this frontend lowers is "
    "one that spells exactly that out over an NHWC tensor (which is how nn.PixelShuffle is "
    "written once its channel axis is split into [out_channels, r, r]). This chain's index "
    "mapping was evaluated and is not a depth-to-space of any factor, in either the plane-major "
    "or the channel-major channel grouping, so there is no instruction for it"
)


def _transpose_indices(
    indices: list[int], shape: tuple[int, ...], perms: tuple[int, ...]
) -> list[int]:
    """`tosa.transpose`, applied to a list of source indices rather than
    to data: `out[o_0, .., o_k] = in[i_0, .., i_k]` with
    `i[perms[j]] = o[j]`, i.e. exactly `numpy.transpose(x, perms)`."""
    out_shape = tuple(shape[p] for p in perms)
    strides = [1] * len(shape)
    for axis in range(len(shape) - 2, -1, -1):
        strides[axis] = strides[axis + 1] * shape[axis + 1]

    out: list[int] = []
    for flat in range(_prod(out_shape)):
        rest = flat
        coords = [0] * len(out_shape)
        for axis in range(len(out_shape) - 1, -1, -1):
            rest, coords[axis] = divmod(rest, out_shape[axis])
        source = 0
        for j, p in enumerate(perms):
            source += coords[j] * strides[p]
        out.append(indices[source])
    return out


def _depth_to_space_indices(
    shape: tuple[int, int, int, int], factor: int, channel_major: bool
) -> list[int]:
    """The source index of every element of `depth_to_space(x, factor)`
    for an NHWC `x` of `shape`, row-major -- i.e. what a chain's index
    list must equal, element for element, to BE this pixel shuffle."""
    _n, h, w, cin = shape
    out_c = cin // (factor * factor)
    out: list[int] = []
    for y in range(h * factor):
        yi, dy = divmod(y, factor)
        for x in range(w * factor):
            xi, dx = divmod(x, factor)
            base = (yi * w + xi) * cin
            for c in range(out_c):
                if channel_major:
                    out.append(base + c * factor * factor + dy * factor + dx)
                else:
                    out.append(base + (dy * factor + dx) * out_c + c)
    return out


def _transpose_perms(
    op: MlirOp, const_defs: dict[str, MlirOp], resources: dict[str, bytes]
) -> tuple[int, ...] | None:
    """The permutation a `tosa.transpose` applies, in either of the two
    forms it is spelled in the wild: a `perms` ATTRIBUTE (TOSA v1.0, an
    `array<i32: ...>`), or -- in graphs exported against pre-v1.0 TOSA --
    a second, constant OPERAND. `None` when it cannot be determined, which
    makes the chain simply not match rather than match something guessed.
    """
    attr = op.attrs.get("perms")
    if isinstance(attr, DenseArrayAttr):
        return tuple(int(v) for v in attr.values)
    if isinstance(attr, DenseElementsAttr):
        try:
            return tuple(int(v) for v in attr.elements())
        except Exception:
            return None
    if len(op.operands) == 2:
        const = const_defs.get(op.operands[1])
        if const is None:
            return None
        values = const.attrs.get("values")
        try:
            if isinstance(values, DenseElementsAttr):
                return tuple(int(v) for v in values.elements())
            if isinstance(values, DenseResourceAttr):
                return tuple(int(v) for v in values.decode(resources))
        except Exception:
            return None
    return None


def _match_depth_to_space_chains(
    func: MlirFunc, resources: dict[str, bytes], skip: set[str]
) -> tuple[dict[str, tuple[str, int, str]], set[str], set[str]]:
    """Find every `reshape`/`transpose` chain that computes a depth-to-space.

    Returns `({final SSA name: (source SSA name, factor, channel_order)},
    {interior SSA names}, {SSA names walked without matching})` -- the
    same three-part shape `_match_upsample_chains` returns, and used the
    same way by `import_tosa`.

    `skip` is the set of SSA names the upsample matcher already claimed
    (its matched chain ends and their interiors), so the two matchers
    cannot both fold the same op.
    """
    use_count: dict[str, int] = {}
    for op in func.ops:
        for operand in op.operands:
            use_count[operand] = use_count.get(operand, 0) + 1

    producer: dict[str, MlirOp] = {op.results[0][0]: op for op in func.ops if op.results}
    consumers: dict[str, list[MlirOp]] = {}
    for op in func.ops:
        for operand in op.operands:
            consumers.setdefault(operand, []).append(op)
    const_defs: dict[str, MlirOp] = {
        op.results[0][0]: op for op in func.ops if op.name == "tosa.const"
    }

    chains: dict[str, tuple[str, int, str]] = {}
    interior: set[str] = set()
    attempted: set[str] = set()

    for start in func.ops:
        if start.name not in _DTS_CHAIN_OP_NAMES or not start.operands:
            continue
        if start.results and start.results[0][0] in skip:
            continue
        source_name = start.operands[0]
        if source_name in interior:
            continue  # already inside a chain we matched
        source_producer = producer.get(source_name)
        if (
            source_producer is not None
            and source_producer.name in _DTS_CHAIN_OP_NAMES
            and source_name not in chains
            and source_name not in skip
        ):
            continue  # mid-chain: not the root of its own chain

        source_shape: tuple[int, ...] | None = None
        if source_producer is not None:
            source_shape = _shape_of(source_producer)
        else:
            for name, ttype in zip(func.arg_names, func.arg_types):
                if name == source_name and isinstance(ttype, TensorType):
                    source_shape = ttype.shape
        if source_shape is None or len(source_shape) != 4 or source_shape[0] != 1:
            continue
        if _prod(source_shape) > _MAX_CHAIN_ELEMENTS:
            continue

        # Every depth-to-space this source COULD be, by factor and channel
        # grouping. Plane-major first so an out_channels == 1 chain (where
        # the two groupings are the same function) is reported as the one
        # that needs no permutation.
        cin = source_shape[3]
        wanted: list[tuple[tuple[int, ...], int, str, list[int]]] = []
        for factor in range(2, _MAX_DTS_FACTOR + 1):
            if factor * factor > cin or cin % (factor * factor):
                continue
            want_shape = (
                1,
                source_shape[1] * factor,
                source_shape[2] * factor,
                cin // (factor * factor),
            )
            for channel_major, order in ((False, PLANE_MAJOR), (True, CHANNEL_MAJOR)):
                wanted.append(
                    (want_shape, factor, order, _depth_to_space_indices(source_shape, factor, channel_major))
                )
        if not wanted:
            continue

        shape: tuple[int, ...] = source_shape
        indices = list(range(_prod(source_shape)))
        current = start
        walked: list[str] = []
        # Which chain ops were actually traversed. A walk that saw BOTH a
        # reshape and a transpose is depth-to-space-SHAPED, whatever it
        # turned out to compute, and its ops get the idiom diagnostic; a
        # walk that saw only one kind is not, and a bare `tosa.transpose`
        # there has to keep being told it is a host-side detection-head
        # op (`_reject_op`'s docstring spells out why that matters).
        kinds_seen: set[str] = set()
        matched = False
        while True:
            out_shape = _shape_of(current)
            if out_shape is None:
                break
            kinds_seen.add(current.name)
            if current.name == "tosa.reshape":
                if _prod(out_shape) != _prod(shape):
                    break
                # Row-major reshape moves no data: the index list is
                # unchanged, only its interpretation.
                shape = out_shape
            else:  # tosa.transpose
                perms = _transpose_perms(current, const_defs, resources)
                if perms is None or sorted(perms) != list(range(len(shape))):
                    break
                if out_shape != tuple(shape[p] for p in perms):
                    break
                indices = _transpose_indices(indices, shape, perms)
                shape = out_shape

            name = current.results[0][0]
            for want_shape, factor, order, want in wanted:
                if shape == want_shape and indices == want:
                    chains[name] = (source_name, factor, order)
                    interior.update(walked)
                    matched = True
                    break
            if matched:
                break
            walked.append(name)

            if use_count.get(name, 0) != 1:
                break  # a value read twice cannot be deleted with the chain
            (next_op,) = consumers[name]
            if next_op.name not in _DTS_CHAIN_OP_NAMES or next_op.operands[0] != name:
                break
            current = next_op

        if not matched and set(_DTS_CHAIN_OP_NAMES) <= kinds_seen:
            attempted.update(walked)
            attempted.add(start.results[0][0])

    return chains, interior, attempted


def _depth_to_space_from_chain(
    op: MlirOp, ctx: _Ctx, source_name: str, factor: int, channel_order: str
) -> Op:
    """Emit the GIR `depth_to_space` a matched chain collapses to. `op` is
    the chain's LAST op, so its result name and type are the shuffle's."""
    result_name, result_type = op.results[0]
    op_id = _op_id(op)
    x = ctx.data_operand(source_name, op_id, "depth_to_space input")
    _check_dtype(op_id, x.dtype)
    if not isinstance(result_type, TensorType):
        raise TosaImportError("depth_to_space chain result must be a tensor", op_id=op_id, stage=_STAGE)
    _check_dtype(op_id, str(result_type.dtype))
    if str(result_type.dtype) != x.dtype:
        raise TosaImportError(
            f"depth_to_space chain changes dtype {x.dtype!r} -> {result_type.dtype!r}",
            op_id=op_id, stage=_STAGE,
        )
    computed = depth_to_space_output_shape(x.shape, factor)
    if result_type.shape != computed:
        raise TosaImportError(
            f"declared result type {result_type.shape} != computed {computed}", op_id=op_id, stage=_STAGE
        )
    ctx.define(Tensor(id=result_name, shape=computed, dtype=x.dtype))
    return Op(
        id=f"%{result_name}",
        kind="depth_to_space",
        inputs=(x.id,),
        outputs=(result_name,),
        attrs=DepthToSpaceAttrs(factor=factor, channel_order=channel_order),
    )


def _import_concat(op: MlirOp, ctx: _Ctx) -> Op:
    """`tosa.concat` -> GIR `concat`.

    Imported for any axis; only the channel axis has a lowering, and
    `lower.to_hir._lower_concat` is where that is decided. Keeping the
    axis check out of the frontend means the diagnostic can talk about
    the DDR layout (`[C/T][H][W][T]` planes) that makes a channel concat
    free and any other axis a copy -- which is a target property, not a
    TOSA one.
    """
    result_name, result_type = op.results[0]
    op_id = _op_id(op)
    if not op.operands:
        raise TosaImportError("tosa.concat has no operands", op_id=op_id, stage=_STAGE)

    axis_attr = op.attrs.get("axis")
    if not isinstance(axis_attr, IntAttr):
        raise TosaImportError("missing or invalid 'axis' attribute", op_id=op_id, stage=_STAGE)

    parts = [ctx.data_operand(name, op_id, f"operand {i}") for i, name in enumerate(op.operands)]
    for part in parts:
        _check_dtype(op_id, part.dtype)
    if len(set(op.operands)) != len(op.operands):
        raise UnsupportedAttribute(
            "tosa.concat repeats an operand: one tensor cannot occupy two different channel "
            "ranges of the result at once",
            op_id=op_id, stage=_STAGE,
        )

    if not isinstance(result_type, TensorType):
        raise TosaImportError("tosa.concat result must be a tensor", op_id=op_id, stage=_STAGE)
    _check_dtype(op_id, str(result_type.dtype))

    ctx.define(Tensor(id=result_name, shape=result_type.shape, dtype=str(result_type.dtype)))
    return Op(
        id=f"%{result_name}",
        kind="concat",
        inputs=tuple(p.id for p in parts),
        outputs=(result_name,),
        attrs=ConcatAttrs(axis=int(axis_attr.value)),
    )


def _import_slice(op: MlirOp, ctx: _Ctx) -> Op:
    """`tosa.slice` -> GIR `slice`.

    `start` and `size` are `!tosa.shape<N>` OPERANDS (a `tosa.const_shape`
    each), not attributes -- the result type gives `size` away but says
    nothing about `start`, so the constant really has to be read.
    """
    result_name, result_type = op.results[0]
    op_id = _op_id(op)
    if len(op.operands) != 3:
        raise TosaImportError(
            f"tosa.slice expects 3 operands (input, start, size), got {len(op.operands)}",
            op_id=op_id, stage=_STAGE,
        )
    x_name, start_name, size_name = op.operands

    x = ctx.data_operand(x_name, op_id, "input")
    _check_dtype(op_id, x.dtype)
    start = ctx.shape_const(start_name, op_id, "slice start")
    size = ctx.shape_const(size_name, op_id, "slice size")
    if len(start) != len(x.shape) or len(size) != len(x.shape):
        raise TosaImportError(
            f"slice start {start} / size {size} must have {len(x.shape)} elements",
            op_id=op_id, stage=_STAGE,
        )

    if not isinstance(result_type, TensorType):
        raise TosaImportError("tosa.slice result must be a tensor", op_id=op_id, stage=_STAGE)
    _check_dtype(op_id, str(result_type.dtype))
    if result_type.shape != tuple(size):
        raise TosaImportError(
            f"declared result type {result_type.shape} != size {tuple(size)}", op_id=op_id, stage=_STAGE
        )

    ctx.define(Tensor(id=result_name, shape=tuple(size), dtype=str(result_type.dtype)))
    return Op(
        id=f"%{result_name}",
        kind="slice",
        inputs=(x.id,),
        outputs=(result_name,),
        attrs=SliceAttrs(start=tuple(start), size=tuple(size)),
    )


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
    "tosa.add": _import_add,
    "tosa.conv2d": _import_conv2d,
    "tosa.rescale": _import_rescale,
    "tosa.clamp": _import_clamp,
    "tosa.max_pool2d": _import_max_pool2d,
    "tosa.table": _import_table,
    "tosa.concat": _import_concat,
    "tosa.slice": _import_slice,
}

#: TOSA ops the accelerator has an opcode for, but whose TOSA semantics
#: that opcode does not reproduce. Kept apart from `_HOST_SIDE_HEAD_OPS`
#: (which the accelerator is not meant to run at all) and from the
#: catch-all "not implemented" case, because the right response is
#: different again: these need a numeric equivalence argument, not more
#: code. Emitting the opcode anyway would be silently wrong.
#: Ops that ARE lowered, but only as part of a larger idiom -- so a
#: standalone one is refused with a message about the idiom rather than a
#: bare "unsupported op". `tosa.reshape` is in `_HOST_SIDE_HEAD_OPS` too;
#: this table wins, because "it has to be part of an upsample chain" is
#: the more actionable of the two answers when a `tosa.tile` is present.
_IDIOM_ONLY_OPS = {
    "tosa.tile": (
        "the accelerator's OPCODE_UPSAMPLE is nearest-neighbour replication by exactly "
        f"{_UPSAMPLE_FACTOR}x in both spatial dimensions, and the only reshape/tile chain this "
        "frontend lowers is the one that spells exactly that out (which is how every "
        "nn.Upsample(scale_factor=2) in YOLOv8n's TOSA is written). This chain's index mapping "
        f"was evaluated and is not a nearest-{_UPSAMPLE_FACTOR}x replication of an NHWC tensor, "
        "so there is no instruction for it"
    ),
}

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


def _reject_op(op: MlirOp, *, idiom_candidate: bool = False, dts_candidate: bool = False) -> None:
    """Refuse `op` with a diagnostic that says *why* it is refused --
    host-side detection head vs simply not implemented -- and where it is.

    `idiom_candidate` marks an op the upsample matcher walked through
    without matching. For a `tosa.reshape` that is the difference between
    two very different pieces of advice ("split the graph, this is the
    detection head" vs "this chain is not a nearest-2x upsample"), so it
    picks the second. `dts_candidate` is the same thing for the
    depth-to-space matcher.

    The asymmetry between the two is deliberate. `tosa.tile` is in
    `_IDIOM_ONLY_OPS`, so a tile is refused with the upsample message
    wherever it appears -- there is no other reason for a tile to be in a
    graph this compiler accepts. `tosa.transpose` is NOT: a transpose is
    an ordinary host-side detection-head op, and only one reached WHILE
    walking a depth-to-space-shaped chain gets the idiom message. A
    transpose anywhere else must keep being told to split the graph."""
    op_id = _op_id(op)
    idiom = _IDIOM_ONLY_OPS.get(op.name)
    # `dts_candidate` is checked FIRST because it is the stricter signal:
    # it is only ever set for a chain that contained both a reshape and a
    # transpose, whereas `idiom_candidate` is set for any reshape the
    # upsample matcher walked. A `tosa.reshape` that is in both sets is
    # therefore the first op of a reshape/transpose chain, and the
    # depth-to-space message is the one that describes it. (A genuine
    # reshape/tile upsample chain never lands in `dts_candidate` at all:
    # its walk stops at the tile, having seen only a reshape.)
    if dts_candidate and op.name in _DTS_CHAIN_OP_NAMES:
        idiom = _DTS_IDIOM_MESSAGE
    elif idiom is None and idiom_candidate and op.name in _CHAIN_OP_NAMES:
        idiom = _IDIOM_ONLY_OPS["tosa.tile"]
    if idiom is not None:
        raise UnsupportedOp(f"{op.name} is not lowered on its own: {idiom}", op_id=op_id, stage=_STAGE)
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
    shape_defs: dict[str, MlirOp] = {
        op.results[0][0]: op for op in func.ops if op.name == "tosa.const_shape"
    }
    ctx = _Ctx(
        tensors=tensors, const_defs=const_defs, resources=module.resources, shape_defs=shape_defs
    )

    # A nearest-2x upsample arrives as a chain of `reshape`/`tile` ops, so
    # it cannot be imported one op at a time: the whole chain collapses to
    # a single GIR `upsample`. Matched up front, then the loop below emits
    # one op where a chain ENDS and skips its interior.
    upsample_chains, upsample_interior, upsample_attempted = _match_upsample_chains(func)
    # And the same for the pixel-shuffle idiom, which is a
    # reshape/TRANSPOSE chain. Run second, told which SSA names the
    # upsample matcher already claimed, so one chain can never be folded
    # twice; the two op-name sets overlap only in `tosa.reshape`.
    dts_chains, dts_interior, dts_attempted = _match_depth_to_space_chains(
        func, module.resources, skip=upsample_interior | set(upsample_chains)
    )

    ops: list[Op] = []
    outputs: tuple[str, ...] | None = None
    for op in func.ops:
        if op.name == "tosa.const":
            continue  # materialised lazily, only if referenced as data
        if op.name == "tosa.const_shape":
            # Structural, like `tosa.const`: it defines a `!tosa.shape<N>`
            # that only `reshape`/`tile` read, and both of those are
            # handled from their own declared result types. A shape const
            # whose consumer is refused is simply never reached.
            continue
        if op.name == "func.return":
            outputs = tuple(op.operands)
            continue

        result_name = op.results[0][0] if op.results else None
        if result_name in upsample_interior or result_name in dts_interior:
            continue  # folded into the op its chain ends at
        chain = upsample_chains.get(result_name) if result_name is not None else None
        dts_chain = dts_chains.get(result_name) if result_name is not None else None
        if chain is not None:
            gir_op = _upsample_from_chain(op, ctx, *chain)
        elif dts_chain is not None:
            gir_op = _depth_to_space_from_chain(op, ctx, *dts_chain)
        else:
            importer = _IMPORTERS.get(op.name)
            if importer is None:
                _reject_op(
                    op,
                    idiom_candidate=result_name in upsample_attempted,
                    dts_candidate=result_name in dts_attempted,
                )
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
