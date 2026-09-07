"""Dependency-free parser/printer for MLIR generic form (as printed by
`iree-opt --mlir-print-op-generic`).

Supports the small subset of the grammar needed to ingest TOSA IR emitted by
IREE: `builtin.module`, `func.func`, generic ops, and the attribute kinds used
by the conv/rescale/clamp fusion group. Anything outside that subset raises a
subclass of `MlirParseError`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


class MlirParseError(Exception):
    def __init__(self, message: str, line: int, col: int):
        super().__init__(f"{line}:{col}: {message}")
        self.line = line
        self.col = col


class GenericFormRequired(MlirParseError):
    pass


class UndefinedValue(MlirParseError):
    pass


class UnsupportedLiteral(MlirParseError):
    pass


class UnsupportedConstruct(MlirParseError):
    pass


# --------------------------------------------------------------------------
# Types
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ScalarType:
    name: str

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class TensorType:
    shape: tuple[int, ...]
    dtype: ScalarType

    def __str__(self) -> str:
        dims = "".join(f"{d}x" for d in self.shape)
        return f"tensor<{dims}{self.dtype}>"

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n


Type = Union[ScalarType, TensorType]


@dataclass(frozen=True)
class FunctionType:
    inputs: tuple[Type, ...]
    results: tuple[Type, ...]

    def __str__(self) -> str:
        ins = ", ".join(str(t) for t in self.inputs)
        if not self.results:
            res = "()"
        elif len(self.results) == 1:
            res = str(self.results[0])
        else:
            res = "(" + ", ".join(str(t) for t in self.results) + ")"
        return f"({ins}) -> {res}"


_INT_TYPE_RE = re.compile(r"^i\d+$")
_FLOAT_TYPE_RE = re.compile(r"^f\d+$")


def _is_scalar_type_name(name: str) -> bool:
    return name == "index" or name == "bf16" or bool(_INT_TYPE_RE.match(name)) or bool(_FLOAT_TYPE_RE.match(name))


def _is_float_type_name(name: str) -> bool:
    return name == "bf16" or bool(_FLOAT_TYPE_RE.match(name))


# --------------------------------------------------------------------------
# Attributes
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class IntAttr:
    value: int
    type: str

    def __str__(self) -> str:
        return f"{self.value} : {self.type}"


@dataclass(frozen=True)
class BoolAttr:
    value: bool

    def __str__(self) -> str:
        return "true" if self.value else "false"


@dataclass(frozen=True)
class TypeAttr:
    type: Type

    def __str__(self) -> str:
        return str(self.type)


@dataclass(frozen=True)
class DenseArrayAttr:
    elem_type: str
    values: tuple[int, ...]

    def __str__(self) -> str:
        if not self.values:
            return f"array<{self.elem_type}>"
        vals = ", ".join(str(v) for v in self.values)
        return f"array<{self.elem_type}: {vals}>"


def _nest_dense_values(values: tuple[int, ...], shape: tuple[int, ...]) -> str:
    """Format a flat, row-major list of dense element values with the
    shape-nested `[...]` bracketing `iree-opt` prints for rank > 1
    tensors, e.g. `tensor<2x2xi8>` -> `[[1, 2], [3, 4]]` rather than the
    flat `[1, 2, 3, 4]`."""
    if len(shape) <= 1:
        return "[" + ", ".join(str(v) for v in values) + "]"
    stride = 1
    for d in shape[1:]:
        stride *= d
    parts = [
        _nest_dense_values(values[i * stride : (i + 1) * stride], shape[1:]) for i in range(shape[0])
    ]
    return "[" + ", ".join(parts) + "]"


@dataclass(frozen=True)
class DenseElementsAttr:
    tensor_type: TensorType
    values: tuple[int, ...] | None
    splat: int | None

    def __str__(self) -> str:
        if self.splat is not None:
            body = str(self.splat)
        elif not self.values:
            body = ""
        else:
            body = _nest_dense_values(self.values, self.tensor_type.shape)
        return f"dense<{body}> : {self.tensor_type}"


@dataclass(frozen=True)
class EnumAttr:
    dialect: str
    kind: str
    value: str

    def __str__(self) -> str:
        return f"#{self.dialect}.{self.kind}<{self.value}>"


@dataclass(frozen=True)
class StringAttr:
    value: str

    def __str__(self) -> str:
        return f'"{self.value}"'


@dataclass(frozen=True)
class FunctionTypeAttr:
    inputs: tuple[Type, ...]
    results: tuple[Type, ...]

    def __str__(self) -> str:
        return str(FunctionType(self.inputs, self.results))


Attr = Union[
    IntAttr, BoolAttr, TypeAttr, DenseArrayAttr, DenseElementsAttr, EnumAttr, StringAttr, FunctionTypeAttr
]


# --------------------------------------------------------------------------
# IR model
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MlirOp:
    results: tuple[tuple[str, Type], ...]
    name: str
    operands: tuple[str, ...]
    attrs: dict[str, Attr]
    loc: tuple[int, int]


@dataclass
class MlirFunc:
    name: str
    arg_names: tuple[str, ...]
    arg_types: tuple[Type, ...]
    result_types: tuple[Type, ...]
    ops: list[MlirOp]


@dataclass
class MlirModule:
    funcs: list[MlirFunc]


# --------------------------------------------------------------------------
# Lexer
# --------------------------------------------------------------------------

_PUNCT = set("(){}<>,:=#[]")


@dataclass
class Token:
    kind: str
    text: str
    line: int
    col: int
    value: object = None
    is_float: bool = False


class Lexer:
    def __init__(self, text: str):
        self.text = text
        self.n = len(text)
        self.i = 0
        self.line = 1
        self.col = 1

    def _peek(self, off: int = 0) -> str:
        j = self.i + off
        return self.text[j] if j < self.n else ""

    def _advance(self) -> str:
        c = self.text[self.i]
        self.i += 1
        if c == "\n":
            self.line += 1
            self.col = 1
        else:
            self.col += 1
        return c

    def _skip_ws_comments(self) -> None:
        while self.i < self.n:
            c = self._peek()
            if c in " \t\r\n":
                self._advance()
            elif c == "/" and self._peek(1) == "/":
                while self.i < self.n and self._peek() != "\n":
                    self._advance()
            else:
                break

    def tokenize(self) -> list[Token]:
        tokens: list[Token] = []
        while True:
            self._skip_ws_comments()
            if self.i >= self.n:
                tokens.append(Token("EOF", "", self.line, self.col))
                break
            line, col = self.line, self.col
            c = self._peek()
            if c == '"':
                tokens.append(self._lex_string(line, col))
            elif c == "%":
                tokens.append(self._lex_sigil("%", "SSA", line, col))
            elif c == "^":
                tokens.append(self._lex_sigil("^", "BLOCK_LABEL", line, col))
            elif c == "@":
                tokens.append(self._lex_sigil("@", "SYMBOL", line, col))
            elif c.isdigit():
                tokens.append(self._lex_number(line, col))
            elif c == "-" and self._peek(1).isdigit():
                tokens.append(self._lex_number(line, col))
            elif c == "-" and self._peek(1) == ">":
                self._advance()
                self._advance()
                tokens.append(Token("ARROW", "->", line, col))
            elif c.isalpha() or c == "_":
                tokens.append(self._lex_ident(line, col))
            elif c in _PUNCT:
                self._advance()
                tokens.append(Token(c, c, line, col))
            elif c == "?":
                self._advance()
                tokens.append(Token("?", "?", line, col))
            else:
                raise MlirParseError(f"unexpected character {c!r}", line, col)
        return tokens

    def _lex_sigil(self, sigil: str, kind: str, line: int, col: int) -> Token:
        self._advance()
        start = self.i
        while self.i < self.n and (self._peek().isalnum() or self._peek() in "_$."):
            self._advance()
        name = self.text[start : self.i]
        if not name:
            raise MlirParseError(f"expected name after '{sigil}'", line, col)
        if sigil == "%" and self._peek() == ":" and self._peek(1).isdigit():
            self._advance()
            dstart = self.i
            while self.i < self.n and self._peek().isdigit():
                self._advance()
            count = int(self.text[dstart : self.i])
            return Token("SSA_MULTI", name, line, col, value=count)
        return Token(kind, name, line, col)

    def _lex_string(self, line: int, col: int) -> Token:
        self._advance()
        chars: list[str] = []
        while True:
            if self.i >= self.n or self._peek() == "\n":
                raise MlirParseError("unterminated string literal", line, col)
            c = self._peek()
            if c == '"':
                self._advance()
                break
            if c == "\\":
                self._advance()
                if self.i >= self.n:
                    raise MlirParseError("unterminated string literal", line, col)
                esc = self._advance()
                chars.append({"n": "\n", "t": "\t", '"': '"', "\\": "\\"}.get(esc, esc))
            else:
                chars.append(self._advance())
        return Token("STRING", "".join(chars), line, col)

    def _lex_ident(self, line: int, col: int) -> Token:
        start = self.i
        while self.i < self.n and (self._peek().isalnum() or self._peek() in "_$."):
            self._advance()
        return Token("IDENT", self.text[start : self.i], line, col)

    def _lex_number(self, line: int, col: int) -> Token:
        start = self.i
        if self._peek() == "-":
            self._advance()
        if (
            self._peek() == "0"
            and self._peek(1) in ("x", "X")
            and self._peek(2) in "0123456789abcdefABCDEF"
        ):
            # Only commit to hex-literal lexing when a hex digit actually
            # follows '0x'/'0X' -- otherwise this is the harmless '0'
            # dimension of a tensor shape immediately followed by an
            # 'x...'-prefixed shape/dtype suffix (e.g. `tensor<0xi8>`,
            # a zero-sized 1-D tensor), which must lex as NUMBER "0" then
            # a separate IDENT "xi8", not as a malformed hex literal.
            self._advance()
            self._advance()
            while self.i < self.n and self._peek() in "0123456789abcdefABCDEF":
                self._advance()
            text = self.text[start : self.i]
            return Token("NUMBER", text, line, col, value=int(text, 16), is_float=False)
        while self.i < self.n and self._peek().isdigit():
            self._advance()
        is_float = False
        if self._peek() == "." and self._peek(1).isdigit():
            is_float = True
            self._advance()
            while self.i < self.n and self._peek().isdigit():
                self._advance()
        if self._peek() in ("e", "E"):
            j = 1
            if self._peek(j) in ("+", "-"):
                j += 1
            if self._peek(j).isdigit():
                is_float = True
                self._advance()
                if self._peek() in ("+", "-"):
                    self._advance()
                while self.i < self.n and self._peek().isdigit():
                    self._advance()
        text = self.text[start : self.i]
        if is_float:
            return Token("NUMBER", text, line, col, value=None, is_float=True)
        return Token("NUMBER", text, line, col, value=int(text), is_float=False)


# --------------------------------------------------------------------------
# Raw parse tree (pre semantic checks)
# --------------------------------------------------------------------------


@dataclass
class RawBlock:
    label: str | None
    args: list[tuple[str, Type]]
    ops: list["RawOp"]


@dataclass
class RawRegion:
    blocks: list[RawBlock]


@dataclass
class RawOp:
    results: list[tuple[str, Type | None]]
    name: str
    operands: list[str]
    attrs: dict[str, Attr]
    region: RawRegion | None
    loc: tuple[int, int]


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------


class Parser:
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0

    def peek(self, off: int = 0) -> Token:
        j = self.pos + off
        if j >= len(self.tokens):
            return self.tokens[-1]
        return self.tokens[j]

    def advance(self) -> Token:
        tok = self.tokens[self.pos]
        if self.pos < len(self.tokens) - 1:
            self.pos += 1
        return tok

    def peek_op(self, char: str) -> bool:
        return self.peek().kind == char

    def expect_op(self, char: str) -> Token:
        tok = self.peek()
        if tok.kind != char:
            raise MlirParseError(f"expected '{char}', found {tok.kind!r} ({tok.text!r})", tok.line, tok.col)
        return self.advance()

    def expect_kind(self, kind: str) -> Token:
        tok = self.peek()
        if tok.kind != kind:
            raise MlirParseError(f"expected {kind}, found {tok.kind!r} ({tok.text!r})", tok.line, tok.col)
        return self.advance()

    def expect_eof(self) -> None:
        tok = self.peek()
        if tok.kind != "EOF":
            raise MlirParseError(f"unexpected trailing token {tok.text!r}", tok.line, tok.col)

    # -- types --------------------------------------------------------

    def parse_type(self) -> Type:
        tok = self.peek()
        if tok.kind == "IDENT" and tok.text == "tensor":
            return self._parse_tensor_type()
        if tok.kind == "IDENT" and _is_scalar_type_name(tok.text):
            self.advance()
            return ScalarType(tok.text)
        if tok.kind == "?":
            raise UnsupportedConstruct("dynamic dimensions are not supported", tok.line, tok.col)
        raise MlirParseError(f"expected type, found {tok.text!r}", tok.line, tok.col)

    def _parse_tensor_type(self) -> TensorType:
        self.advance()  # 'tensor'
        self.expect_op("<")
        first = self.peek()
        if first.kind == "?":
            raise UnsupportedConstruct("dynamic tensor dimensions are not supported", first.line, first.col)
        if first.kind != "NUMBER":
            raise MlirParseError("expected tensor dimension", first.line, first.col)
        if first.is_float:
            raise UnsupportedLiteral("float literal in tensor shape", first.line, first.col)
        self.advance()
        dims = [first.value]
        rest_tok = self.peek()
        if rest_tok.kind != "IDENT" or not rest_tok.text.startswith("x"):
            raise MlirParseError("malformed tensor shape", rest_tok.line, rest_tok.col)
        self.advance()
        parts = rest_tok.text[1:].split("x")
        *more_dims, dtype_name = parts
        for md in more_dims:
            if md == "?":
                raise UnsupportedConstruct("dynamic tensor dimensions are not supported", rest_tok.line, rest_tok.col)
            if md == "" or not md.lstrip("-").isdigit():
                raise MlirParseError("malformed tensor shape", rest_tok.line, rest_tok.col)
            dims.append(int(md))
        if not _is_scalar_type_name(dtype_name):
            raise MlirParseError(f"unknown tensor element type {dtype_name!r}", rest_tok.line, rest_tok.col)
        self.expect_op(">")
        return TensorType(shape=tuple(dims), dtype=ScalarType(dtype_name))

    def parse_function_type(self) -> FunctionType:
        self.expect_op("(")
        inputs: list[Type] = []
        if not self.peek_op(")"):
            inputs.append(self.parse_type())
            while self.peek_op(","):
                self.advance()
                inputs.append(self.parse_type())
        self.expect_op(")")
        self.expect_kind("ARROW")
        results = self._parse_result_types()
        return FunctionType(tuple(inputs), tuple(results))

    def _parse_result_types(self) -> list[Type]:
        if self.peek_op("("):
            self.advance()
            results: list[Type] = []
            if not self.peek_op(")"):
                results.append(self.parse_type())
                while self.peek_op(","):
                    self.advance()
                    results.append(self.parse_type())
            self.expect_op(")")
            return results
        return [self.parse_type()]

    # -- attributes -----------------------------------------------------

    def parse_attrs(self) -> dict[str, Attr]:
        attrs: dict[str, Attr] = {}
        if self.peek_op("}"):
            return attrs
        k, v = self._parse_attr_pair()
        attrs[k] = v
        while self.peek_op(","):
            self.advance()
            k, v = self._parse_attr_pair()
            attrs[k] = v
        return attrs

    def _parse_attr_pair(self) -> tuple[str, Attr]:
        key_tok = self.expect_kind("IDENT")
        self.expect_op("=")
        return key_tok.text, self._parse_attr_value()

    def _parse_attr_value(self) -> Attr:
        tok = self.peek()
        if tok.kind == "STRING":
            self.advance()
            return StringAttr(tok.text)
        if tok.kind == "IDENT" and tok.text in ("true", "false"):
            self.advance()
            return BoolAttr(tok.text == "true")
        if tok.kind == "IDENT" and tok.text == "dense":
            return self._parse_dense_attr()
        if tok.kind == "IDENT" and tok.text == "dense_resource":
            raise UnsupportedConstruct("dense_resource attributes are not supported", tok.line, tok.col)
        if tok.kind == "IDENT" and tok.text == "array":
            return self._parse_array_attr()
        if tok.kind == "#":
            return self._parse_enum_attr()
        if tok.kind == "(":
            ft = self.parse_function_type()
            return FunctionTypeAttr(ft.inputs, ft.results)
        if tok.kind == "NUMBER":
            return self._parse_int_attr()
        if tok.kind == "IDENT":
            return TypeAttr(self.parse_type())
        raise MlirParseError(f"unexpected token in attribute value: {tok.text!r}", tok.line, tok.col)

    def _parse_int_attr(self) -> IntAttr:
        tok = self.advance()
        if tok.is_float:
            raise UnsupportedLiteral("floating point literals are not supported", tok.line, tok.col)
        self.expect_op(":")
        t = self.parse_type()
        if not isinstance(t, ScalarType):
            raise MlirParseError("expected scalar type for integer literal", tok.line, tok.col)
        if _is_float_type_name(t.name):
            raise UnsupportedLiteral("floating point typed literals are not supported", tok.line, tok.col)
        return IntAttr(tok.value, t.name)

    def _parse_array_attr(self) -> DenseArrayAttr:
        self.advance()  # 'array'
        self.expect_op("<")
        et_tok = self.expect_kind("IDENT")
        values: list[int] = []
        if self.peek_op(":"):
            self.advance()
            values.append(self._parse_array_int())
            while self.peek_op(","):
                self.advance()
                values.append(self._parse_array_int())
        self.expect_op(">")
        return DenseArrayAttr(et_tok.text, tuple(values))

    def _parse_array_int(self) -> int:
        tok = self.advance()
        if tok.kind != "NUMBER" or tok.is_float:
            raise UnsupportedLiteral("expected integer literal in array attribute", tok.line, tok.col)
        return tok.value

    def _parse_dense_attr(self) -> DenseElementsAttr:
        self.advance()  # 'dense'
        self.expect_op("<")
        splat: int | None = None
        values: list[int] | None = None
        if self.peek_op("["):
            values = self._parse_dense_list()
        else:
            tok = self.advance()
            if tok.kind != "NUMBER":
                raise MlirParseError("expected dense value", tok.line, tok.col)
            if tok.is_float:
                raise UnsupportedLiteral("floating point dense values are not supported", tok.line, tok.col)
            splat = tok.value
        self.expect_op(">")
        self.expect_op(":")
        type_tok = self.peek()
        t = self.parse_type()
        if not isinstance(t, TensorType):
            raise MlirParseError("expected tensor type for dense attribute", type_tok.line, type_tok.col)
        if _is_float_type_name(t.dtype.name):
            raise UnsupportedLiteral("floating point dense attributes are not supported", type_tok.line, type_tok.col)
        if values is not None and len(values) != t.numel:
            raise MlirParseError(
                f"dense attribute has {len(values)} elements, expected {t.numel}", type_tok.line, type_tok.col
            )
        return DenseElementsAttr(t, tuple(values) if values is not None else None, splat)

    def _parse_dense_list(self) -> list[int]:
        self.expect_op("[")
        out: list[int] = []
        if not self.peek_op("]"):
            self._parse_dense_list_item(out)
            while self.peek_op(","):
                self.advance()
                self._parse_dense_list_item(out)
        self.expect_op("]")
        return out

    def _parse_dense_list_item(self, out: list[int]) -> None:
        if self.peek_op("["):
            out.extend(self._parse_dense_list())
            return
        tok = self.advance()
        if tok.kind != "NUMBER":
            raise MlirParseError("expected integer literal in dense list", tok.line, tok.col)
        if tok.is_float:
            raise UnsupportedLiteral("floating point literals are not supported", tok.line, tok.col)
        out.append(tok.value)

    def _parse_enum_attr(self) -> EnumAttr:
        self.advance()  # '#'
        ident_tok = self.expect_kind("IDENT")
        if "." not in ident_tok.text:
            raise MlirParseError("expected 'dialect.kind' for enum attribute", ident_tok.line, ident_tok.col)
        dialect, _, kind = ident_tok.text.rpartition(".")
        self.expect_op("<")
        value_tok = self.expect_kind("IDENT")
        self.expect_op(">")
        return EnumAttr(dialect, kind, value_tok.text)

    # -- ops / regions ----------------------------------------------------

    def parse_op(self) -> RawOp:
        result_names: list[str] = []
        if self.peek().kind == "SSA":
            tok = self.advance()
            result_names.append(tok.text)
            self.expect_op("=")
        elif self.peek().kind == "SSA_MULTI":
            tok = self.advance()
            raise UnsupportedConstruct(
                f"multi-result op '%{tok.text}:{tok.value}' is not supported", tok.line, tok.col
            )

        if self.peek().kind == "IDENT":
            tok = self.peek()
            raise GenericFormRequired(
                f"expected generic form (quoted op name), found bare identifier {tok.text!r}; "
                "custom/pretty op syntax is not supported",
                tok.line,
                tok.col,
            )
        name_tok = self.expect_kind("STRING")
        op_name = name_tok.text

        self.expect_op("(")
        operands: list[str] = []
        if not self.peek_op(")"):
            operands.append(self.expect_kind("SSA").text)
            while self.peek_op(","):
                self.advance()
                operands.append(self.expect_kind("SSA").text)
        self.expect_op(")")

        attrs: dict[str, Attr] = {}
        if self.peek_op("<"):
            self.advance()
            self.expect_op("{")
            attrs = self.parse_attrs()
            self.expect_op("}")
            self.expect_op(">")
        elif self.peek_op("{"):
            self.advance()
            attrs = self.parse_attrs()
            self.expect_op("}")

        region: RawRegion | None = None
        if self.peek_op("("):
            self.advance()
            region = self.parse_region()
            self.expect_op(")")

        self.expect_op(":")
        ftype = self.parse_function_type()

        results: list[tuple[str, Type | None]] = []
        if result_names:
            if len(ftype.results) != 1:
                raise MlirParseError(
                    "expected exactly one result type for single-result op", name_tok.line, name_tok.col
                )
            results = [(result_names[0], ftype.results[0])]

        return RawOp(results=results, name=op_name, operands=operands, attrs=attrs, region=region, loc=(name_tok.line, name_tok.col))

    def parse_region(self) -> RawRegion:
        self.expect_op("{")
        block = self.parse_block()
        self.expect_op("}")
        return RawRegion(blocks=[block])

    def parse_block(self) -> RawBlock:
        label: str | None = None
        args: list[tuple[str, Type]] = []
        if self.peek().kind == "BLOCK_LABEL":
            tok = self.advance()
            label = tok.text
            if self.peek_op("("):
                self.advance()
                if not self.peek_op(")"):
                    args.append(self._parse_block_arg())
                    while self.peek_op(","):
                        self.advance()
                        args.append(self._parse_block_arg())
                self.expect_op(")")
            self.expect_op(":")
        ops: list[RawOp] = []
        while not self.peek_op("}"):
            ops.append(self.parse_op())
        return RawBlock(label=label, args=args, ops=ops)

    def _parse_block_arg(self) -> tuple[str, Type]:
        tok = self.expect_kind("SSA")
        self.expect_op(":")
        t = self.parse_type()
        return (tok.text, t)


# --------------------------------------------------------------------------
# Semantic pass: raw tree -> MlirModule
# --------------------------------------------------------------------------


def _build_module(root: RawOp) -> MlirModule:
    if root.name != "builtin.module":
        raise UnsupportedConstruct(f"expected top-level 'builtin.module', found {root.name!r}", *root.loc)
    if root.region is None or len(root.region.blocks) != 1:
        raise MlirParseError("module is missing its body region", *root.loc)
    funcs = [_build_func(raw) for raw in root.region.blocks[0].ops]
    return MlirModule(funcs=funcs)


def _build_func(raw: RawOp) -> MlirFunc:
    if raw.name != "func.func":
        raise UnsupportedConstruct(f"unsupported top-level module op {raw.name!r}", *raw.loc)
    if raw.region is None or len(raw.region.blocks) != 1:
        raise MlirParseError("func.func is missing its body region", *raw.loc)
    ft_attr = raw.attrs.get("function_type")
    sym_attr = raw.attrs.get("sym_name")
    if not isinstance(ft_attr, FunctionTypeAttr):
        raise MlirParseError("func.func requires a 'function_type' attribute", *raw.loc)
    if not isinstance(sym_attr, StringAttr):
        raise MlirParseError("func.func requires a 'sym_name' attribute", *raw.loc)

    block = raw.region.blocks[0]
    arg_names = tuple(n for n, _ in block.args)
    arg_types = tuple(t for _, t in block.args)

    defined: set[str] = set(arg_names)
    ops: list[MlirOp] = []
    for raw_op in block.ops:
        for operand in raw_op.operands:
            if operand not in defined:
                raise UndefinedValue(f"use of undefined value '%{operand}'", *raw_op.loc)
        op_results = tuple((name, t) for name, t in raw_op.results)
        ops.append(
            MlirOp(
                results=op_results,
                name=raw_op.name,
                operands=tuple(raw_op.operands),
                attrs=raw_op.attrs,
                loc=raw_op.loc,
            )
        )
        for name, _ in op_results:
            if name in defined:
                raise MlirParseError(f"redefinition of value '%{name}'", *raw_op.loc)
            defined.add(name)

    return MlirFunc(
        name=sym_attr.value,
        arg_names=arg_names,
        arg_types=arg_types,
        result_types=ft_attr.results,
        ops=ops,
    )


# --------------------------------------------------------------------------
# Public API: parsing
# --------------------------------------------------------------------------


def parse_module(text: str) -> MlirModule:
    tokens = Lexer(text).tokenize()
    parser = Parser(tokens)
    root = parser.parse_op()
    parser.expect_eof()
    return _build_module(root)


def parse_file(path: str | Path) -> MlirModule:
    return parse_module(Path(path).read_text())


# --------------------------------------------------------------------------
# Public API: printing
# --------------------------------------------------------------------------


def _format_attrs(attrs: dict[str, Attr]) -> str:
    return ", ".join(f"{k} = {attrs[k]}" for k in sorted(attrs))


def _print_op(op: MlirOp, lines: list[str], indent: int, value_types: dict[str, Type]) -> None:
    pad = " " * indent
    prefix = f"%{op.results[0][0]} = " if op.results else ""
    operands_str = ", ".join(f"%{o}" for o in op.operands)
    attrs_part = f" <{{{_format_attrs(op.attrs)}}}>" if op.attrs else ""
    in_types = ", ".join(str(value_types[o]) for o in op.operands)
    out_types = str(op.results[0][1]) if op.results else "()"
    lines.append(f'{pad}{prefix}"{op.name}"({operands_str}){attrs_part} : ({in_types}) -> {out_types}')


def _print_func(func: MlirFunc, lines: list[str], indent: int) -> None:
    pad = " " * indent
    ft_attr = FunctionTypeAttr(func.arg_types, func.result_types)
    attrs = {"function_type": ft_attr, "sym_name": StringAttr(func.name)}
    lines.append(f'{pad}"func.func"() <{{{_format_attrs(attrs)}}}> ({{')
    args_str = ", ".join(f"%{n}: {t}" for n, t in zip(func.arg_names, func.arg_types))
    lines.append(f"{pad}^bb0({args_str}):")
    value_types: dict[str, Type] = dict(zip(func.arg_names, func.arg_types))
    for op in func.ops:
        _print_op(op, lines, indent + 2, value_types)
        for name, t in op.results:
            value_types[name] = t
    lines.append(f"{pad}}}) : () -> ()")


def print_generic(module: MlirModule) -> str:
    lines: list[str] = ['"builtin.module"() ({']
    for func in module.funcs:
        _print_func(func, lines, 2)
    lines.append("}) : () -> ()")
    return "\n".join(lines) + "\n"
