"""Constraint-expression evaluator: no `eval`, a small hand-written
tokenizer + recursive-descent parser/evaluator instead.

Grammar::

    expr    := term (('+' | '-') term)*
    term    := factor (('*' | '/') factor)*
    factor  := INT | IDENT | 'ceil' '(' expr ')' | '(' expr ')' | '-' factor
    IDENT   := layer field name, e.g. in_width, in_channels, kernel_h, ...

`/` is exact integer division everywhere EXCEPT directly inside a
`ceil(...)` call's argument (and any sub-expression nested in it), where
it is true (fractional) division that `ceil` then rounds up. Outside
`ceil(...)`, a `/` that does not divide evenly is a `ConstraintEvalError`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Union

from .contract import Constraint


class ConstraintEvalError(Exception):
    """Raised for unknown identifiers, malformed tokens, or inexact
    division outside `ceil(...)`."""


@dataclass(frozen=True)
class ConstraintViolation:
    constraint: Constraint
    actual: int
    env: dict[str, int]


_SINGLE_CHAR_TOKENS = {"+", "-", "*", "/", "(", ")", ","}


def _tokenize(expr: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    i = 0
    n = len(expr)
    while i < n:
        ch = expr[i]
        if ch.isspace():
            i += 1
            continue
        if ch.isdigit():
            j = i
            while j < n and expr[j].isdigit():
                j += 1
            tokens.append(("INT", expr[i:j]))
            i = j
            continue
        if ch.isalpha() or ch == "_":
            j = i
            while j < n and (expr[j].isalnum() or expr[j] == "_"):
                j += 1
            tokens.append(("IDENT", expr[i:j]))
            i = j
            continue
        if ch in _SINGLE_CHAR_TOKENS:
            tokens.append((ch, ch))
            i += 1
            continue
        raise ConstraintEvalError(f"bad token {ch!r} at position {i} in {expr!r}")
    tokens.append(("EOF", ""))
    return tokens


Number = Union[int, Fraction]


class _Parser:
    def __init__(self, expr: str, env: dict[str, int]) -> None:
        self._expr = expr
        self._env = env
        self._tokens = _tokenize(expr)
        self._pos = 0

    def _peek(self) -> tuple[str, str]:
        return self._tokens[self._pos]

    def _advance(self) -> tuple[str, str]:
        tok = self._tokens[self._pos]
        self._pos += 1
        return tok

    def _expect(self, kind: str) -> tuple[str, str]:
        tok = self._peek()
        if tok[0] != kind:
            raise ConstraintEvalError(f"expected {kind!r}, got {tok[0]!r} in {self._expr!r}")
        return self._advance()

    def parse(self) -> int:
        value = self._expr_rule(in_ceil=False)
        self._expect("EOF")
        return _to_int(value, context=self._expr)

    def _expr_rule(self, in_ceil: bool) -> Number:
        value = self._term_rule(in_ceil)
        while self._peek()[0] in ("+", "-"):
            op = self._advance()[0]
            rhs = self._term_rule(in_ceil)
            value = value + rhs if op == "+" else value - rhs
        return value

    def _term_rule(self, in_ceil: bool) -> Number:
        value = self._factor_rule(in_ceil)
        while self._peek()[0] in ("*", "/"):
            op = self._advance()[0]
            rhs = self._factor_rule(in_ceil)
            if op == "*":
                value = value * rhs
            else:
                value = self._divide(value, rhs, in_ceil)
        return value

    def _divide(self, lhs: Number, rhs: Number, in_ceil: bool) -> Number:
        if in_ceil:
            return Fraction(lhs) / Fraction(rhs)
        if rhs == 0:
            raise ConstraintEvalError(f"division by zero in {self._expr!r}")
        if lhs % rhs != 0:
            raise ConstraintEvalError(
                f"inexact division {lhs} / {rhs} in {self._expr!r} (only exact outside ceil(...))"
            )
        return lhs // rhs

    def _factor_rule(self, in_ceil: bool) -> Number:
        kind, text = self._peek()
        if kind == "-":
            self._advance()
            return -self._factor_rule(in_ceil)
        if kind == "INT":
            self._advance()
            return int(text)
        if kind == "IDENT":
            if text == "ceil":
                self._advance()
                self._expect("(")
                inner = self._expr_rule(in_ceil=True)
                self._expect(")")
                return math.ceil(Fraction(inner))
            self._advance()
            if text not in self._env:
                raise ConstraintEvalError(f"unknown identifier {text!r} in {self._expr!r}")
            return self._env[text]
        if kind == "(":
            self._advance()
            value = self._expr_rule(in_ceil)
            self._expect(")")
            return value
        raise ConstraintEvalError(f"unexpected token {kind!r} in {self._expr!r}")


def _to_int(value: Number, *, context: str) -> int:
    if isinstance(value, Fraction):
        if value.denominator != 1:
            raise ConstraintEvalError(f"non-integer result {value} evaluating {context!r}")
        return int(value)
    return value


def evaluate(expr: str, env: dict[str, int]) -> int:
    return _Parser(expr, env).parse()


def check(constraint: Constraint, env: dict[str, int]) -> ConstraintViolation | None:
    actual = evaluate(constraint.expr, env)
    if constraint.kind == "max":
        if actual > constraint.value:
            return ConstraintViolation(constraint, actual, dict(env))
    elif constraint.kind == "min":
        if actual < constraint.value:
            return ConstraintViolation(constraint, actual, dict(env))
    elif constraint.kind == "divisible":
        if actual % constraint.by != 0:
            return ConstraintViolation(constraint, actual, dict(env))
    elif constraint.kind == "in_set":
        if actual not in constraint.values:
            return ConstraintViolation(constraint, actual, dict(env))
    else:  # pragma: no cover - Constraint.__post_init__ already rejects this
        raise ConstraintEvalError(f"unknown constraint kind {constraint.kind!r}")
    return None
