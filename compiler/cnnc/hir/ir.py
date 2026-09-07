"""Hardware IR (HIR): the HOW (doc/tosa_compiler_plan.md §2.2).

A single `HirModule` is progressively filled by three passes that each
return a new module: `to_hir` (stage='mapped') assigns buffers/ops/units
to a GIR graph, `schedule` (stage='scheduled') assigns `HirOp.seq`, and
`memplan` (stage='planned') assigns `Buffer.addr`/`HirModule.memory_size`.
`HirModule.stage` tells `verify.verify_hir` which fields must already be
populated (see also §7 for `BufferView`/`TileInfo`, unused until M13, and
§8 for the memory model the planner implements). Every node is a frozen
dataclass; passes return new values rather than mutating existing ones,
mirroring `gir.ir`.
"""

from __future__ import annotations

import dataclasses
from types import MappingProxyType

LAYOUTS = ("HWC", "OHWI", "I32_VEC", "SCALE_TABLE", "PROGRAM")
ROLES = ("input", "output", "const", "intermediate", "program")
STAGES = ("mapped", "scheduled", "planned")

# HIR op kinds don't always match the target Unit.ops vocabulary 1:1 --
# `conv_layer` (a fused conv+bias+requant+relu HIR op) maps to the
# `conv2d` capability a Unit advertises.
KIND_TO_OP = {"conv_layer": "conv2d"}


@dataclasses.dataclass(frozen=True)
class Buffer:
    id: str
    space: str
    size_bytes: int
    align: int
    role: str
    layout: str
    shape: tuple[int, ...]
    dtype: str
    addr: int | None = None
    data: bytes | None = None  # const/program payload
    gir_tensor: str | None = None

    @property
    def end(self) -> int:
        if self.addr is None:
            raise ValueError(f"buffer {self.id!r} has no addr")
        return self.addr + self.size_bytes


@dataclasses.dataclass(frozen=True)
class BufferView:
    """A tiled sub-range of a `Buffer` (§7, unused before M13)."""

    buffer: str
    offset_elems: tuple[int, ...]
    shape: tuple[int, ...]
    contiguous: bool


@dataclasses.dataclass(frozen=True)
class TileInfo:
    """Tiling metadata attached to an `HirOp` (§7, unused before M13)."""

    axis: str
    index: int
    count: int
    halo_before: int
    halo_after: int
    first: bool
    last: bool


@dataclasses.dataclass(frozen=True)
class HirOp:
    id: str
    unit: str
    kind: str
    params: MappingProxyType  # str -> int | str | bool, scalar only (deterministic dump)
    reads: tuple[str, ...]  # buffer ids
    writes: tuple[str, ...]  # buffer ids
    deps: tuple[str, ...]  # op ids
    seq: int | None = None
    tile: TileInfo | None = None
    gir_op: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.params, MappingProxyType):
            object.__setattr__(self, "params", MappingProxyType(dict(self.params)))

    def replace(self, **kwargs) -> "HirOp":
        return dataclasses.replace(self, **kwargs)


@dataclasses.dataclass(frozen=True)
class HirModule:
    target_name: str
    ops: tuple[HirOp, ...]
    buffers: MappingProxyType  # buffer id -> Buffer
    entry_inputs: tuple[str, ...]  # buffer ids
    entry_outputs: tuple[str, ...]  # buffer ids
    program: str | None = None  # buffer id
    memory_size: int | None = None
    stage: str = "mapped"
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.buffers, MappingProxyType):
            object.__setattr__(self, "buffers", MappingProxyType(dict(self.buffers)))

    def buffer(self, buffer_id: str) -> Buffer:
        return self.buffers[buffer_id]

    def op(self, op_id: str) -> HirOp:
        for candidate in self.ops:
            if candidate.id == op_id:
                return candidate
        raise KeyError(f"unknown HIR op {op_id!r}")

    def replace(self, **kwargs) -> "HirModule":
        return dataclasses.replace(self, **kwargs)

    def with_buffers(self, updates: dict) -> "HirModule":
        """Return a new module with `buffers` merged with `updates`. The
        original module (and its `buffers` mapping) is left unchanged."""
        merged = dict(self.buffers)
        merged.update(updates)
        return self.replace(buffers=merged)

    def with_ops(self, ops) -> "HirModule":
        return self.replace(ops=tuple(ops))

    def writer_of(self, buffer_id: str) -> HirOp | None:
        for op in self.ops:
            if buffer_id in op.writes:
                return op
        return None

    def readers_of(self, buffer_id: str) -> tuple[HirOp, ...]:
        return tuple(op for op in self.ops if buffer_id in op.reads)
