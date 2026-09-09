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

LAYOUTS = ("HWC", "PLANES", "OHWI", "TILED_OHWI", "I32_VEC", "I32_TILED", "SCALE_TABLE", "ACT_LUT", "PROGRAM")
ROLES = ("input", "output", "const", "intermediate", "program")
STAGES = ("mapped", "scheduled", "planned")
ALIAS_KINDS = ("view", "part")

# HIR op kinds don't always match the target Unit.ops vocabulary 1:1 --
# `conv_layer` (a fused conv+bias+requant+relu HIR op) maps to the `conv2d`
# capability a Unit advertises, and `max_pool` to `max_pool2d` -- the HIR
# kind names the instruction shape, the capability names what the unit can
# do, and the two only coincide by accident.
KIND_TO_OP = {"conv_layer": "conv2d", "max_pool": "max_pool2d", "act": "table"}


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
    # --- buffer views (§7) ---------------------------------------------
    # A buffer with `alias_parent` set owns no storage of its own: it IS a
    # channel-plane range of another buffer, at
    # `parent.addr + alias_plane_offset * plane_channels * H * W`. Both
    # halves of `tosa.concat`/`tosa.slice` lower to this and emit no
    # instruction at all -- channel concatenation of S6 plane-tiled
    # activations is address arithmetic, not data movement, exactly as
    # `accel_v2.model.Tensor.alias_parent`/`alias_plane_offset` treat it.
    #
    # `alias_kind` says which DIRECTION the aliasing runs, and the two are
    # not interchangeable:
    #
    #   "view"  -- a read-only window into a parent somebody else writes
    #              (a `tosa.slice` result). The parent is written normally;
    #              this buffer is never written.
    #   "part"  -- a piece the PRODUCER is redirected to write into (a
    #              `tosa.concat` operand). The parent is never written
    #              directly; its parts write it between them, which is why
    #              the concat itself costs nothing.
    #
    # `plane_channels` is not stored here because it is a target property,
    # not a buffer one; callers that need bytes pass it to
    # `alias_byte_offset`.
    alias_parent: str | None = None
    alias_plane_offset: int = 0
    alias_kind: str | None = None  # None | "view" | "part"

    def __post_init__(self) -> None:
        if (self.alias_parent is None) != (self.alias_kind is None):
            raise ValueError(
                f"buffer {self.id!r}: alias_parent and alias_kind must be set together "
                f"(alias_parent={self.alias_parent!r}, alias_kind={self.alias_kind!r})"
            )
        if self.alias_kind is not None and self.alias_kind not in ALIAS_KINDS:
            raise ValueError(f"buffer {self.id!r}: alias_kind {self.alias_kind!r} not in {ALIAS_KINDS}")
        if self.alias_plane_offset < 0:
            raise ValueError(f"buffer {self.id!r}: alias_plane_offset {self.alias_plane_offset} is negative")

    @property
    def end(self) -> int:
        if self.addr is None:
            raise ValueError(f"buffer {self.id!r} has no addr")
        return self.addr + self.size_bytes

    def alias_byte_offset(self, plane_channels: int) -> int:
        """This view's byte offset from its parent's address: whole
        channel planes, each `plane_channels * H * W` bytes (decision S6's
        `[C/T][H][W][T]` layout, whose planes are contiguous and equal in
        size -- which is the entire reason a channel concat can be free)."""
        if self.alias_parent is None:
            return 0
        _n, h, w, _c = self.shape
        return self.alias_plane_offset * plane_channels * h * w


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

    def alias_root(self, buffer_id: str) -> str:
        """The buffer that actually owns the storage `buffer_id` lives in:
        follow `alias_parent` to the end. A buffer that is not a view is
        its own root."""
        seen = {buffer_id}
        current = buffer_id
        while True:
            parent = self.buffers[current].alias_parent
            if parent is None:
                return current
            if parent in seen:
                raise ValueError(f"alias cycle through buffer {buffer_id!r}")
            seen.add(parent)
            current = parent

    def alias_children(self, buffer_id: str) -> tuple[Buffer, ...]:
        """Every buffer whose `alias_parent` is `buffer_id`, in id order."""
        return tuple(
            buf for _, buf in sorted(self.buffers.items()) if buf.alias_parent == buffer_id
        )

    def storage_dependencies(self, buffer_id: str) -> tuple[str, ...]:
        """Every buffer whose writer must finish before `buffer_id` can be
        read -- `buffer_id` itself plus the aliases that share its bytes.

        Two directions, and they are not symmetric:

        * ANCESTORS. A view is filled by whoever writes the buffer it
          looks into, so reading it waits for that buffer's writer. (An
          ancestor that is itself assembled from parts has no writer of
          its own and so contributes nothing here -- correctly: reading
          one half of a concat must not wait for the other half.)
        * `part` DESCENDANTS. A concat result is filled by its parts
          between them, so reading it waits for all of them.

        Without this, an op reading a `tosa.slice` view would look
        dependency-free -- nothing writes the view's id -- and the
        scheduler would be free to hoist it above the op that actually
        produces the bytes.
        """
        result = {buffer_id}

        current = buffer_id
        while True:
            parent = self.buffers[current].alias_parent
            if parent is None or parent in result:
                break
            result.add(parent)
            current = parent

        stack = [buffer_id]
        while stack:
            for child in self.alias_children(stack.pop()):
                if child.alias_kind == "part" and child.id not in result:
                    result.add(child.id)
                    stack.append(child.id)

        return tuple(sorted(result))

    def writer_of(self, buffer_id: str) -> HirOp | None:
        for op in self.ops:
            if buffer_id in op.writes:
                return op
        return None

    def readers_of(self, buffer_id: str) -> tuple[HirOp, ...]:
        return tuple(op for op in self.ops if buffer_id in op.reads)
