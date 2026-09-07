"""Target contract: frozen, JSON-round-trippable dataclasses describing
what a backend (e.g. `cnn_accel`) can execute. Nothing in this module
hard-codes accelerator properties -- values are supplied by callers
(typically `discover.py`, which derives them from the accelerator's own
source of truth) and merely validated/shaped here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

class TargetError(Exception):
    """Raised for malformed/invalid `Target` definitions, unknown target
    names, or JSON that does not describe a valid target."""


class CapabilityError(Exception):
    """Raised by later compiler stages when a lowering/legalization step
    requires a capability the `Target` does not provide."""


def _require(mapping: dict, key: str, context: str) -> Any:
    try:
        return mapping[key]
    except KeyError as exc:
        raise TargetError(f"missing field {key!r} in {context}") from exc


def _as_pairs(value: Any, context: str) -> tuple[tuple[int, int], ...]:
    try:
        return tuple((int(a), int(b)) for a, b in value)
    except (TypeError, ValueError) as exc:
        raise TargetError(f"{context} must be a list of [a, b] pairs") from exc


@dataclass(frozen=True)
class MemorySpace:
    name: str
    size_bytes: int
    align: int

    def __post_init__(self) -> None:
        if not self.name:
            raise TargetError("MemorySpace.name must be non-empty")
        if self.size_bytes <= 0:
            raise TargetError("MemorySpace.size_bytes must be positive")
        if self.align <= 0 or (self.align & (self.align - 1)) != 0:
            raise TargetError("MemorySpace.align must be a positive power of two")

    def to_dict(self) -> dict:
        return {"name": self.name, "size_bytes": self.size_bytes, "align": self.align}

    @classmethod
    def from_dict(cls, data: dict) -> "MemorySpace":
        return cls(
            name=_require(data, "name", "MemorySpace"),
            size_bytes=_require(data, "size_bytes", "MemorySpace"),
            align=_require(data, "align", "MemorySpace"),
        )


@dataclass(frozen=True)
class Memory:
    spaces: dict[str, MemorySpace]
    activation_layout: str
    weight_layout: str
    bias_format: str

    def __post_init__(self) -> None:
        if not self.spaces:
            raise TargetError("Memory.spaces must be non-empty")
        for name, space in self.spaces.items():
            if not isinstance(space, MemorySpace):
                raise TargetError(f"Memory.spaces[{name!r}] must be a MemorySpace")
        for field_name in ("activation_layout", "weight_layout", "bias_format"):
            if not getattr(self, field_name):
                raise TargetError(f"Memory.{field_name} must be non-empty")

    def to_dict(self) -> dict:
        return {
            "spaces": {name: space.to_dict() for name, space in self.spaces.items()},
            "activation_layout": self.activation_layout,
            "weight_layout": self.weight_layout,
            "bias_format": self.bias_format,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Memory":
        raw_spaces = _require(data, "spaces", "Memory")
        spaces = {name: MemorySpace.from_dict(v) for name, v in raw_spaces.items()}
        return cls(
            spaces=spaces,
            activation_layout=_require(data, "activation_layout", "Memory"),
            weight_layout=_require(data, "weight_layout", "Memory"),
            bias_format=_require(data, "bias_format", "Memory"),
        )


ROUNDING_MODES = ("half_up", "half_even")


@dataclass(frozen=True)
class RescaleCaps:
    multiplier_bits: int
    multiplier_signed: bool
    implicit_shift: int
    shift_min: int
    shift_max: int
    rounding: str
    per_channel: bool
    input_zp: bool
    output_zp: bool

    def __post_init__(self) -> None:
        if self.multiplier_bits <= 0:
            raise TargetError("RescaleCaps.multiplier_bits must be positive")
        if self.rounding not in ROUNDING_MODES:
            raise TargetError(f"RescaleCaps.rounding invalid: {self.rounding!r}")
        if self.shift_min > self.shift_max:
            raise TargetError("RescaleCaps.shift_min must be <= shift_max")

    def to_dict(self) -> dict:
        return {
            "multiplier_bits": self.multiplier_bits,
            "multiplier_signed": self.multiplier_signed,
            "implicit_shift": self.implicit_shift,
            "shift_min": self.shift_min,
            "shift_max": self.shift_max,
            "rounding": self.rounding,
            "per_channel": self.per_channel,
            "input_zp": self.input_zp,
            "output_zp": self.output_zp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RescaleCaps":
        return cls(
            multiplier_bits=_require(data, "multiplier_bits", "RescaleCaps"),
            multiplier_signed=_require(data, "multiplier_signed", "RescaleCaps"),
            implicit_shift=_require(data, "implicit_shift", "RescaleCaps"),
            shift_min=_require(data, "shift_min", "RescaleCaps"),
            shift_max=_require(data, "shift_max", "RescaleCaps"),
            rounding=_require(data, "rounding", "RescaleCaps"),
            per_channel=_require(data, "per_channel", "RescaleCaps"),
            input_zp=_require(data, "input_zp", "RescaleCaps"),
            output_zp=_require(data, "output_zp", "RescaleCaps"),
        )


@dataclass(frozen=True)
class Epilogue:
    bias: bool
    rescale: RescaleCaps
    clamp_ranges: Any  # list[tuple[int, int]] | Literal["any"]

    def __post_init__(self) -> None:
        if not isinstance(self.rescale, RescaleCaps):
            raise TargetError("Epilogue.rescale must be a RescaleCaps")
        if self.clamp_ranges != "any":
            if not isinstance(self.clamp_ranges, (list, tuple)):
                raise TargetError("Epilogue.clamp_ranges must be 'any' or a list of (lo, hi) pairs")
            for pair in self.clamp_ranges:
                lo, hi = pair
                if lo > hi:
                    raise TargetError(f"Epilogue.clamp_ranges pair {pair!r} has lo > hi")

    def to_dict(self) -> dict:
        clamp = self.clamp_ranges if self.clamp_ranges == "any" else [list(p) for p in self.clamp_ranges]
        return {"bias": self.bias, "rescale": self.rescale.to_dict(), "clamp_ranges": clamp}

    @classmethod
    def from_dict(cls, data: dict) -> "Epilogue":
        clamp = _require(data, "clamp_ranges", "Epilogue")
        if clamp != "any":
            clamp = _as_pairs(clamp, "Epilogue.clamp_ranges")
        return cls(
            bias=_require(data, "bias", "Epilogue"),
            rescale=RescaleCaps.from_dict(_require(data, "rescale", "Epilogue")),
            clamp_ranges=clamp,
        )


CONSTRAINT_KINDS = ("max", "min", "divisible", "in_set")


@dataclass(frozen=True)
class Constraint:
    kind: str
    expr: str
    source: str
    value: int | None = None
    by: int | None = None
    values: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.kind not in CONSTRAINT_KINDS:
            raise TargetError(f"Constraint.kind invalid: {self.kind!r}")
        if not self.expr:
            raise TargetError("Constraint.expr must be non-empty")
        if not self.source:
            raise TargetError("Constraint.source must be non-empty")
        if self.kind in ("max", "min") and self.value is None:
            raise TargetError(f"Constraint.value required for kind={self.kind!r}")
        if self.kind == "divisible" and self.by is None:
            raise TargetError("Constraint.by required for kind='divisible'")
        if self.kind == "in_set" and not self.values:
            raise TargetError("Constraint.values required for kind='in_set'")

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "expr": self.expr,
            "source": self.source,
            "value": self.value,
            "by": self.by,
            "values": list(self.values) if self.values is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Constraint":
        values = data.get("values")
        return cls(
            kind=_require(data, "kind", "Constraint"),
            expr=_require(data, "expr", "Constraint"),
            source=_require(data, "source", "Constraint"),
            value=data.get("value"),
            by=data.get("by"),
            values=tuple(values) if values is not None else None,
        )


@dataclass(frozen=True)
class InternalTiling:
    cin: int
    cout: int

    def __post_init__(self) -> None:
        if self.cin <= 0 or self.cout <= 0:
            raise TargetError("InternalTiling.cin/cout must be positive")

    def to_dict(self) -> dict:
        return {"cin": self.cin, "cout": self.cout}

    @classmethod
    def from_dict(cls, data: dict) -> "InternalTiling":
        return cls(cin=_require(data, "cin", "InternalTiling"), cout=_require(data, "cout", "InternalTiling"))


@dataclass(frozen=True)
class Unit:
    name: str
    ops: tuple[str, ...]
    dtypes: dict[str, str]
    kernels: Any  # tuple[tuple[int, int], ...] | Literal["any"]
    strides: Any  # tuple[tuple[int, int], ...] | Literal["any"]
    dilation: tuple[tuple[int, int], ...]
    padding: str
    batch: int
    internal_tiling: InternalTiling
    constraints: tuple[Constraint, ...]
    epilogue: Epilogue
    isa_version: str
    partial_sum_io: bool

    def __post_init__(self) -> None:
        if not self.name:
            raise TargetError("Unit.name must be non-empty")
        if not self.ops:
            raise TargetError("Unit.ops must be non-empty")
        if not isinstance(self.internal_tiling, InternalTiling):
            raise TargetError("Unit.internal_tiling must be an InternalTiling")
        if not isinstance(self.epilogue, Epilogue):
            raise TargetError("Unit.epilogue must be an Epilogue")
        for constraint in self.constraints:
            if not isinstance(constraint, Constraint):
                raise TargetError("Unit.constraints entries must be Constraint")
        if self.batch <= 0:
            raise TargetError("Unit.batch must be positive")

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "ops": list(self.ops),
            "dtypes": dict(self.dtypes),
            "kernels": self.kernels if self.kernels == "any" else [list(k) for k in self.kernels],
            "strides": self.strides if self.strides == "any" else [list(s) for s in self.strides],
            "dilation": [list(d) for d in self.dilation],
            "padding": self.padding,
            "batch": self.batch,
            "internal_tiling": self.internal_tiling.to_dict(),
            "constraints": [c.to_dict() for c in self.constraints],
            "epilogue": self.epilogue.to_dict(),
            "isa_version": self.isa_version,
            "partial_sum_io": self.partial_sum_io,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Unit":
        kernels = _require(data, "kernels", "Unit")
        if kernels != "any":
            kernels = _as_pairs(kernels, "Unit.kernels")
        strides = _require(data, "strides", "Unit")
        if strides != "any":
            strides = _as_pairs(strides, "Unit.strides")
        return cls(
            name=_require(data, "name", "Unit"),
            ops=tuple(_require(data, "ops", "Unit")),
            dtypes=dict(_require(data, "dtypes", "Unit")),
            kernels=kernels,
            strides=strides,
            dilation=_as_pairs(_require(data, "dilation", "Unit"), "Unit.dilation"),
            padding=_require(data, "padding", "Unit"),
            batch=_require(data, "batch", "Unit"),
            internal_tiling=InternalTiling.from_dict(_require(data, "internal_tiling", "Unit")),
            constraints=tuple(Constraint.from_dict(c) for c in _require(data, "constraints", "Unit")),
            epilogue=Epilogue.from_dict(_require(data, "epilogue", "Unit")),
            isa_version=_require(data, "isa_version", "Unit"),
            partial_sum_io=_require(data, "partial_sum_io", "Unit"),
        )


@dataclass(frozen=True)
class IsaInfo:
    instr_word_bytes: int
    opcodes: dict[str, int]
    flags: dict[str, int]
    fields: dict[str, tuple[int, int, bool]]

    def __post_init__(self) -> None:
        if self.instr_word_bytes <= 0:
            raise TargetError("IsaInfo.instr_word_bytes must be positive")
        if not self.opcodes:
            raise TargetError("IsaInfo.opcodes must be non-empty")
        if not self.fields:
            raise TargetError("IsaInfo.fields must be non-empty")

    def to_dict(self) -> dict:
        return {
            "instr_word_bytes": self.instr_word_bytes,
            "opcodes": dict(self.opcodes),
            "flags": dict(self.flags),
            "fields": {name: list(spec) for name, spec in self.fields.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "IsaInfo":
        raw_fields = _require(data, "fields", "IsaInfo")
        fields = {name: tuple(spec) for name, spec in raw_fields.items()}
        return cls(
            instr_word_bytes=_require(data, "instr_word_bytes", "IsaInfo"),
            opcodes=dict(_require(data, "opcodes", "IsaInfo")),
            flags=dict(_require(data, "flags", "IsaInfo")),
            fields=fields,
        )


@dataclass(frozen=True)
class Target:
    name: str
    program_model: str
    memory: Memory
    units: tuple[Unit, ...]
    isa: IsaInfo
    provenance: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.name:
            raise TargetError("Target.name must be non-empty")
        if not self.program_model:
            raise TargetError("Target.program_model must be non-empty")
        if not isinstance(self.memory, Memory):
            raise TargetError("Target.memory must be a Memory")
        if not self.units:
            raise TargetError("Target.units must be non-empty")
        for unit in self.units:
            if not isinstance(unit, Unit):
                raise TargetError("Target.units entries must be Unit")
        if not isinstance(self.isa, IsaInfo):
            raise TargetError("Target.isa must be an IsaInfo")

    def unit(self, name: str) -> Unit:
        for candidate in self.units:
            if candidate.name == name:
                return candidate
        raise TargetError(f"Target {self.name!r} has no unit named {name!r}")

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "program_model": self.program_model,
            "memory": self.memory.to_dict(),
            "units": [u.to_dict() for u in self.units],
            "isa": self.isa.to_dict(),
            "provenance": self.provenance,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "Target":
        if not isinstance(data, dict):
            raise TargetError("Target JSON root must be an object")
        return cls(
            name=_require(data, "name", "Target"),
            program_model=_require(data, "program_model", "Target"),
            memory=Memory.from_dict(_require(data, "memory", "Target")),
            units=tuple(Unit.from_dict(u) for u in _require(data, "units", "Target")),
            isa=IsaInfo.from_dict(_require(data, "isa", "Target")),
            provenance=dict(_require(data, "provenance", "Target")),
        )

    @classmethod
    def from_json(cls, text: str) -> "Target":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise TargetError(f"invalid JSON: {exc}") from exc
        return cls.from_dict(data)
