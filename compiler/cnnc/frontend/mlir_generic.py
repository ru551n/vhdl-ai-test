"""Dependency-free parser/printer for MLIR generic form (as printed by
`iree-opt --mlir-print-op-generic`).

Supports the small subset of the grammar needed to ingest TOSA IR emitted by
IREE: `builtin.module`, `func.func`, generic ops, and the attribute kinds used
by the conv/rescale/clamp fusion group. Anything outside that subset raises a
subclass of `MlirParseError`.

Deliberately *syntax*-permissive, semantics-strict: the parser's job is to
survive whatever real MLIR producers emit, and to let `frontend.tosa_import`
be the one place that says "this program is not compilable, because <op> at
<line:col> ...". Concretely, three things modern producers emit that this
parser therefore accepts rather than rejects:

* `!tosa.shape<N>` (TOSA-1.0 shape-typed operands: `tosa.const_shape`,
  `tosa.reshape`'s shape operand, `tosa.slice`'s start/size operands),
  modelled as `ShapeType`;
* floating-point literals (`1.0 : f32`, `dense<1.0> : tensor<4xf32>`),
  modelled as `FloatAttr` / a `DenseElementsAttr` whose `values` are
  `float`. A pre-quantization fp32 model must *parse* -- it is the import
  stage that reports "floating point dtype 'f32' is not supported", naming
  the op, rather than the lexer dying on the first literal;
* `dense_resource<key>` attributes plus the trailing
  `{-# dialect_resources: { builtin: { key: "0x..." } } #-}` block that
  torch-mlir emits by default for tensors above its inlining threshold.
  The blobs are parsed into `MlirModule.resources` and decoded on demand
  by `DenseResourceAttr.decode`, so a resource-carrying `tosa.const` is
  real data, not an unsupported construct.
"""

from __future__ import annotations

import re
import struct
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


@dataclass(frozen=True)
class ShapeType:
    """TOSA-1.0 `!tosa.shape<N>`: a rank-`N` list of `index` values used
    for the shape/start/size operands of `tosa.reshape`, `tosa.slice`,
    `tosa.tile` and friends, which in TOSA 0.x were plain attributes.
    Carries no element type -- the elements are always `index`."""

    rank: int

    def __str__(self) -> str:
        return f"!tosa.shape<{self.rank}>"


Type = Union[ScalarType, TensorType, ShapeType]


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


# One `<digits>x` dimension prefix of a tensor-shape suffix; see
# `Parser._parse_tensor_type` for why this is peeled rather than split.
_LEADING_DIM_RE = re.compile(r"^(\d+)x")

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
class FloatAttr:
    """A floating-point scalar literal (`1.0 : f32`). Never usable by the
    importer -- this compiler is integer-only -- but represented so that a
    pre-quantization model parses and is diagnosed at import."""

    value: float
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
    values: tuple[float, ...] | tuple[int, ...]

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
    """`dense<...> : tensor<...>` in any of its three printed forms: a
    splat (`dense<0>`), a nested value list (`dense<[[1, 2], [3, 4]]>`),
    or a raw hex blob (`dense<"0x7B7A...">`, which MLIR switches to above
    a size threshold). Exactly one of `splat`/`values`/`blob` is set;
    `elements()` normalises all three to a flat, row-major tuple."""

    tensor_type: TensorType
    values: tuple[float, ...] | tuple[int, ...] | None
    splat: float | int | None
    #: Raw little-endian element bytes for the `dense<"0x...">` form.
    #: Unlike a `dense_resource` blob these carry no alignment header.
    blob: bytes | None = None

    def __str__(self) -> str:
        if self.blob is not None:
            body = '"0x' + self.blob.hex().upper() + '"'
        elif self.splat is not None:
            body = str(self.splat)
        elif not self.values:
            body = ""
        else:
            body = _nest_dense_values(self.values, self.tensor_type.shape)
        return f"dense<{body}> : {self.tensor_type}"

    def elements(self) -> tuple[float, ...] | tuple[int, ...]:
        """Flat, row-major elements, whichever form this attribute took.
        Raises `ResourceDecodeError` for a blob whose length or element
        type this decoder cannot make sense of."""
        if self.blob is not None:
            return _decode_elements(self.blob, self.tensor_type, f"dense<...> : {self.tensor_type}")
        if self.splat is not None:
            return tuple([self.splat] * self.tensor_type.numel)
        return tuple(self.values) if self.values else ()


# `struct` format character + byte width per MLIR element type, for
# decoding a `dense_resource` blob. MLIR writes resources in the host's
# (little-endian on every platform this compiler runs on) in-memory
# representation; `i1` is one byte per element there, not a bit vector.
_RESOURCE_ELEM_FORMAT = {
    "i1": ("?", 1), "i8": ("b", 1), "i16": ("h", 2), "i32": ("i", 4), "i64": ("q", 8),
    "index": ("q", 8), "f32": ("f", 4), "f64": ("d", 8),
}

# Every MLIR resource blob is prefixed by its alignment as a little-endian
# uint32 (`0x04000000` = 4 in the artifacts here); the payload follows.
_RESOURCE_HEADER_BYTES = 4


class ResourceDecodeError(Exception):
    """A binary element blob could not be decoded (missing `dense_resource`
    key, element type this decoder does not know, or a blob whose length
    disagrees with the tensor type it is attached to)."""


def _decode_elements(payload: bytes, tensor_type: TensorType, what: str) -> tuple[float, ...] | tuple[int, ...]:
    """`payload` (raw little-endian elements, no header) as Python scalars.

    Raises `ResourceDecodeError` rather than returning something
    plausible-but-wrong: a silently mis-decoded weight blob is far worse
    than a refusal naming the blob."""
    dtype = tensor_type.dtype.name
    spec = _RESOURCE_ELEM_FORMAT.get(dtype)
    if spec is None:
        raise ResourceDecodeError(f"{what}: no decoder for element type {dtype!r}")
    fmt, width = spec
    numel = tensor_type.numel
    if len(payload) != numel * width:
        raise ResourceDecodeError(
            f"{what}: blob payload is {len(payload)} bytes, but {tensor_type} needs "
            f"{numel * width} ({numel} x {width}-byte {dtype})"
        )
    return tuple(struct.unpack(f"<{numel}{fmt}", payload))


@dataclass(frozen=True)
class DenseResourceAttr:
    """`dense_resource<key> : tensor<...>`: the tensor's elements live in
    the file-trailing `{-# dialect_resources ... #-}` block under `key`,
    as one hex blob. torch-mlir emits every tensor above its inlining
    threshold this way, so a real exported model is almost entirely made
    of these.

    The attribute itself carries only the key and the type; the blobs are
    on `MlirModule.resources` (one dict for the whole file), so decoding
    needs both -- see `decode`."""

    key: str
    tensor_type: TensorType

    def __str__(self) -> str:
        return f"dense_resource<{self.key}> : {self.tensor_type}"

    def decode(self, resources: dict[str, bytes]) -> tuple[float, ...] | tuple[int, ...]:
        """This resource's elements, row-major, as Python scalars.

        Raises `ResourceDecodeError` rather than returning something
        plausible-but-wrong: a silently mis-decoded weight blob is far
        worse than a refusal naming the key."""
        blob = resources.get(self.key)
        if blob is None:
            raise ResourceDecodeError(
                f"dense_resource key {self.key!r} is not in the file's dialect_resources block "
                f"(known keys: {sorted(resources)[:8]}{'...' if len(resources) > 8 else ''})"
            )
        return _decode_elements(blob[_RESOURCE_HEADER_BYTES:], self.tensor_type, f"dense_resource {self.key!r}")


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
    IntAttr, FloatAttr, BoolAttr, TypeAttr, DenseArrayAttr, DenseElementsAttr, DenseResourceAttr,
    EnumAttr, StringAttr, FunctionTypeAttr,
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
    #: `dense_resource` key -> raw blob bytes (alignment header included),
    #: from the file-trailing `{-# dialect_resources ... #-}` block. Empty
    #: for a file that inlines all of its constants.
    resources: dict[str, bytes] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Lexer
# --------------------------------------------------------------------------

_PUNCT = set("(){}<>,:=#[]")

# Delimiters of MLIR's file-scope metadata block (`{-# ... #-}`), which is
# where `dialect_resources` lives. Lexed like a comment -- see
# `Lexer._skip_ws_comments` -- with the raw text kept on the lexer so
# `parse_module` can pull the resource blobs out of it.
_METADATA_OPEN = "{-#"
_METADATA_CLOSE = "#-}"


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
        #: Raw text (delimiters excluded) of every `{-# ... #-}` file
        #: metadata block skipped while lexing, in file order.
        self.metadata_blocks: list[str] = []

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
            elif self.text.startswith(_METADATA_OPEN, self.i):
                self._skip_metadata_block()
            else:
                break

    def _skip_metadata_block(self) -> None:
        """Consume a `{-# ... #-}` file metadata block, keeping its body on
        `metadata_blocks`. Skipped like a comment because it is not part of
        the op grammar: it always sits *outside* the top-level op, and its
        body (an attribute dictionary of dialect resources) uses a syntax
        -- bare `key: "0x..."` entries -- the op parser has no production
        for. `parse_module` mines the body for resource blobs afterwards."""
        line, col = self.line, self.col
        start = self.i + len(_METADATA_OPEN)
        end = self.text.find(_METADATA_CLOSE, start)
        if end < 0:
            raise MlirParseError(f"unterminated '{_METADATA_OPEN}' file metadata block", line, col)
        self.metadata_blocks.append(self.text[start:end])
        while self.i < end + len(_METADATA_CLOSE):
            self._advance()

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
            elif c == "!":
                # A dialect type (`!tosa.shape<4>`). Lexed as one token so
                # `parse_type` never has to re-join '!' with the ident.
                tokens.append(self._lex_sigil("!", "DIALECT_TYPE", line, col))
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
            return Token("NUMBER", text, line, col, value=float(text), is_float=True)
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
        if tok.kind == "DIALECT_TYPE":
            return self._parse_dialect_type()
        if tok.kind == "?":
            raise UnsupportedConstruct("dynamic dimensions are not supported", tok.line, tok.col)
        raise MlirParseError(f"expected type, found {tok.text!r}", tok.line, tok.col)

    def _parse_dialect_type(self) -> Type:
        """`!tosa.shape<N>`; any other `!dialect.type` is out of subset."""
        tok = self.advance()
        if tok.text != "tosa.shape":
            raise UnsupportedConstruct(f"unsupported dialect type '!{tok.text}'", tok.line, tok.col)
        self.expect_op("<")
        rank_tok = self.expect_kind("NUMBER")
        if rank_tok.is_float:
            raise MlirParseError("!tosa.shape rank must be an integer", rank_tok.line, rank_tok.col)
        self.expect_op(">")
        return ShapeType(rank=rank_tok.value)

    def _parse_tensor_type(self) -> TensorType:
        self.advance()  # 'tensor'
        self.expect_op("<")
        first = self.peek()
        if first.kind == "?":
            raise UnsupportedConstruct("dynamic tensor dimensions are not supported", first.line, first.col)
        if first.kind == "IDENT" and _is_scalar_type_name(first.text):
            # Rank-0 tensor (`tensor<f32>`): no dimension list at all, just
            # the element type. TOSA emits these for scalar operands.
            self.advance()
            self.expect_op(">")
            return TensorType(shape=(), dtype=ScalarType(first.text))
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
        # `160x160xf32` lexes as one identifier, so the remaining dimensions
        # and the element type have to be split back apart here. Peel off
        # `<digits>x` prefixes one at a time rather than splitting on every
        # 'x': element type names may CONTAIN an 'x' -- `tensor<4xindex>`,
        # the type TOSA-1.0 gives `tosa.const_shape`'s value, splits into
        # ('inde', '') under the naive rule and was rejected as a malformed
        # shape.
        rest = rest_tok.text[1:]
        while True:
            m = _LEADING_DIM_RE.match(rest)
            if m is None:
                break
            dims.append(int(m.group(1)))
            rest = rest[m.end():]
        if rest.startswith("?"):
            raise UnsupportedConstruct("dynamic tensor dimensions are not supported", rest_tok.line, rest_tok.col)
        dtype_name = rest
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
            return self._parse_dense_resource_attr()
        if tok.kind == "IDENT" and tok.text == "array":
            return self._parse_array_attr()
        if tok.kind == "#":
            return self._parse_enum_attr()
        if tok.kind == "(":
            ft = self.parse_function_type()
            return FunctionTypeAttr(ft.inputs, ft.results)
        if tok.kind == "NUMBER":
            return self._parse_scalar_attr()
        if tok.kind in ("IDENT", "DIALECT_TYPE"):
            return TypeAttr(self.parse_type())
        raise MlirParseError(f"unexpected token in attribute value: {tok.text!r}", tok.line, tok.col)

    def _parse_scalar_attr(self) -> IntAttr | FloatAttr:
        """`<number> : <scalar type>`. A float literal (or an integer
        literal with a float type, which is how MLIR prints e.g. `0 :
        f32`) becomes a `FloatAttr`; the importer, not the parser, decides
        that this compiler cannot use one."""
        tok = self.advance()
        self.expect_op(":")
        t = self.parse_type()
        if not isinstance(t, ScalarType):
            raise MlirParseError("expected scalar type for numeric literal", tok.line, tok.col)
        if tok.is_float or _is_float_type_name(t.name):
            return FloatAttr(float(tok.value), t.name)
        return IntAttr(tok.value, t.name)

    def _parse_dense_resource_attr(self) -> DenseResourceAttr:
        self.advance()  # 'dense_resource'
        self.expect_op("<")
        key_tok = self.expect_kind("IDENT")
        self.expect_op(">")
        self.expect_op(":")
        type_tok = self.peek()
        t = self.parse_type()
        if not isinstance(t, TensorType):
            raise MlirParseError("expected tensor type for dense_resource attribute", type_tok.line, type_tok.col)
        return DenseResourceAttr(key=key_tok.text, tensor_type=t)

    def _parse_array_attr(self) -> DenseArrayAttr:
        self.advance()  # 'array'
        self.expect_op("<")
        et_tok = self.expect_kind("IDENT")
        values: list[float | int] = []
        if self.peek_op(":"):
            self.advance()
            values.append(self._parse_array_element())
            while self.peek_op(","):
                self.advance()
                values.append(self._parse_array_element())
        self.expect_op(">")
        return DenseArrayAttr(et_tok.text, tuple(values))

    def _parse_array_element(self) -> float | int:
        tok = self.advance()
        if tok.kind != "NUMBER":
            raise MlirParseError("expected numeric literal in array attribute", tok.line, tok.col)
        return tok.value

    def _parse_dense_attr(self) -> DenseElementsAttr:
        self.advance()  # 'dense'
        self.expect_op("<")
        splat: float | int | None = None
        values: list[float | int] | None = None
        blob: bytes | None = None
        if self.peek_op("["):
            values = self._parse_dense_list()
        elif self.peek().kind == "STRING":
            # `dense<"0x7B7A...">`: MLIR switches to a raw little-endian
            # hex blob above a size threshold, so any real (non-splat)
            # inlined weight tensor arrives in this form, not as a value
            # list. Kept as bytes and decoded on demand by `elements()`.
            tok = self.advance()
            if not tok.text.startswith("0x"):
                raise UnsupportedLiteral(
                    f"dense string literal {tok.text[:16]!r} is not a '0x...' hex blob", tok.line, tok.col
                )
            hex_text = tok.text[2:]
            if len(hex_text) % 2:
                raise MlirParseError("dense hex blob has an odd number of digits", tok.line, tok.col)
            try:
                blob = bytes.fromhex(hex_text)
            except ValueError as exc:
                raise MlirParseError(f"malformed dense hex blob: {exc}", tok.line, tok.col) from exc
        else:
            tok = self.advance()
            if tok.kind != "NUMBER":
                raise MlirParseError("expected dense value", tok.line, tok.col)
            splat = tok.value
        self.expect_op(">")
        self.expect_op(":")
        type_tok = self.peek()
        t = self.parse_type()
        if not isinstance(t, TensorType):
            raise MlirParseError("expected tensor type for dense attribute", type_tok.line, type_tok.col)
        if values is not None and len(values) != t.numel:
            raise MlirParseError(
                f"dense attribute has {len(values)} elements, expected {t.numel}", type_tok.line, type_tok.col
            )
        return DenseElementsAttr(t, tuple(values) if values is not None else None, splat, blob)

    def _parse_dense_list(self) -> list[float | int]:
        self.expect_op("[")
        out: list[float | int] = []
        if not self.peek_op("]"):
            self._parse_dense_list_item(out)
            while self.peek_op(","):
                self.advance()
                self._parse_dense_list_item(out)
        self.expect_op("]")
        return out

    def _parse_dense_list_item(self, out: list[float | int]) -> None:
        if self.peek_op("["):
            out.extend(self._parse_dense_list())
            return
        tok = self.advance()
        if tok.kind != "NUMBER":
            raise MlirParseError("expected numeric literal in dense list", tok.line, tok.col)
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


def _build_module(root: RawOp, resources: dict[str, bytes]) -> MlirModule:
    if root.name != "builtin.module":
        raise UnsupportedConstruct(f"expected top-level 'builtin.module', found {root.name!r}", *root.loc)
    if root.region is None or len(root.region.blocks) != 1:
        raise MlirParseError("module is missing its body region", *root.loc)
    funcs = [_build_func(raw) for raw in root.region.blocks[0].ops]
    return MlirModule(funcs=funcs, resources=resources)


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


# One `key: "0xABCD..."` entry of a `dialect_resources` block. Keys are
# bare identifiers (torch-mlir puts '.' in them, e.g.
# `torch_tensor_80_torch.float32_8`); the value is always a hex blob.
_RESOURCE_ENTRY_RE = re.compile(r'([A-Za-z_$][\w$.]*)\s*:\s*"0x([0-9A-Fa-f]*)"')


def parse_resources(block_text: str) -> dict[str, bytes]:
    """Resource blobs from the body of a `{-# ... #-}` metadata block.

    A regex rather than a grammar on purpose: the block is a nested
    attribute dictionary whose only content this compiler can use is the
    flat `key: "0x..."` leaves, and its outer structure (`dialect_resources
    { builtin { ... } }`) carries no information we act on. An entry whose
    key repeats across dialects would collide, so the *first* wins and
    later ones are ignored -- MLIR itself requires resource keys to be
    unique within a file, so a collision means a malformed file, not an
    ambiguity to resolve."""
    resources: dict[str, bytes] = {}
    for key, hex_text in _RESOURCE_ENTRY_RE.findall(block_text):
        if key in resources:
            continue
        resources[key] = bytes.fromhex(hex_text)
    return resources


def parse_module(text: str) -> MlirModule:
    lexer = Lexer(text)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    root = parser.parse_op()
    parser.expect_eof()
    resources: dict[str, bytes] = {}
    for block in lexer.metadata_blocks:
        resources.update(parse_resources(block))
    return _build_module(root, resources)


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
