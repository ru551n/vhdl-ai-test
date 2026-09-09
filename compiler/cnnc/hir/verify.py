"""HIR verifier (doc/tosa_compiler_plan.md §11, "HIR (mapped/scheduled/
planned)" rows; memory-planning invariants from §8).

`verify_hir` checks the invariants for `module.stage` and every earlier
stage (a 'planned' module must also satisfy 'mapped'/'scheduled'). Every
violation raises a typed exception naming the offending op or buffer id.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cnnc.errors import CompilerError
from cnnc.target.contract import TargetError

from .ir import KIND_TO_OP, LAYOUTS, ROLES, STAGES, HirModule, HirOp

if TYPE_CHECKING:
    from cnnc.target.contract import Target

_NEVER_WRITTEN_ROLES = ("input", "const", "program")
_WRITTEN_ONCE_ROLES = ("intermediate", "output")


class HirVerifyError(CompilerError):
    """Raised when an `HirModule` violates a HIR invariant."""


class MemoryPlanError(HirVerifyError):
    """Raised for stage='planned' violations of the §8 memory model."""


def _fail(cls, id_: str, message: str) -> None:
    raise cls(message, op_id=id_, stage="verify")


def verify_hir(module: HirModule, target: "Target | None" = None) -> None:
    if module.stage not in STAGES:
        _fail(HirVerifyError, module.target_name, f"unknown HirModule.stage {module.stage!r}, expected one of {STAGES}")
    _verify_ids_unique(module)
    _verify_mapped(module, target)
    if module.stage in ("scheduled", "planned"):
        _verify_scheduled(module)
    if module.stage == "planned":
        _verify_planned(module, target)


def _verify_ids_unique(module: HirModule) -> None:
    seen: set[str] = set()
    for op in module.ops:
        if op.id in seen:
            _fail(HirVerifyError, op.id, "duplicate op id")
        seen.add(op.id)
    for buffer_id, buf in module.buffers.items():
        if buf.id != buffer_id:
            _fail(HirVerifyError, buffer_id, f"buffer key {buffer_id!r} does not match Buffer.id {buf.id!r}")


def _verify_op_capability(op: HirOp, target: "Target") -> None:
    if op.kind == "halt":
        if op.unit != "sequencer":
            _fail(HirVerifyError, op.id, f"halt op unit {op.unit!r} must be 'sequencer'")
        return
    try:
        unit = target.unit(op.unit)
    except TargetError as exc:
        _fail(HirVerifyError, op.id, str(exc))
        return
    expected_op = KIND_TO_OP.get(op.kind, op.kind)
    if expected_op not in unit.ops:
        _fail(
            HirVerifyError,
            op.id,
            f"unit {op.unit!r} has no capability {expected_op!r} for kind {op.kind!r} (ops={unit.ops})",
        )


def _verify_mapped(module: HirModule, target: "Target | None") -> None:
    op_ids = {op.id for op in module.ops}
    writers: dict[str, list[str]] = {bid: [] for bid in module.buffers}

    for op in module.ops:
        for bid in op.reads:
            if bid not in module.buffers:
                _fail(HirVerifyError, op.id, f"reads unknown buffer {bid!r}")
        for bid in op.writes:
            if bid not in module.buffers:
                _fail(HirVerifyError, op.id, f"writes unknown buffer {bid!r}")
            else:
                writers[bid].append(op.id)
        for dep in op.deps:
            if dep not in op_ids:
                _fail(HirVerifyError, op.id, f"depends on unknown op {dep!r}")

    for bid, buf in module.buffers.items():
        if buf.role not in ROLES:
            _fail(HirVerifyError, bid, f"buffer role {buf.role!r} not in {ROLES}")
        if buf.layout not in LAYOUTS:
            _fail(HirVerifyError, bid, f"buffer layout {buf.layout!r} not in {LAYOUTS}")
        if buf.size_bytes <= 0:
            _fail(HirVerifyError, bid, f"buffer size_bytes {buf.size_bytes} must be positive")
        if buf.align <= 0 or (buf.align & (buf.align - 1)) != 0:
            _fail(HirVerifyError, bid, f"buffer align {buf.align} must be a positive power of two")
        if target is not None and buf.space not in target.memory.spaces:
            _fail(HirVerifyError, bid, f"buffer space {buf.space!r} not in target memory spaces {sorted(target.memory.spaces)}")

        buf_writers = writers[bid]
        if buf.alias_parent is not None:
            if buf.alias_parent not in module.buffers:
                _fail(HirVerifyError, bid, f"alias_parent {buf.alias_parent!r} is not a known buffer")
            elif buf.alias_kind == "view" and buf_writers:
                _fail(
                    HirVerifyError, bid,
                    f"'view' alias must never be written (it is a read-only window into "
                    f"{buf.alias_parent!r}), written by {buf_writers}",
                )
            elif buf.alias_kind == "part":
                # A part is normally written by its own producer. The
                # exception is a nested concat -- a part that is itself
                # assembled from parts -- which is written by those.
                nested = any(c.alias_kind == "part" for c in module.alias_children(bid))
                if nested and buf_writers:
                    _fail(
                        HirVerifyError, bid,
                        f"'part' alias is itself assembled from parts and must not also be written "
                        f"directly, written by {buf_writers}",
                    )
                elif not nested and len(buf_writers) != 1:
                    _fail(
                        HirVerifyError, bid,
                        f"'part' alias must be written by exactly one op (its producer writes it in "
                        f"place inside {buf.alias_parent!r}), written by {buf_writers}",
                    )
            continue
        if any(child.alias_kind == "part" for child in module.alias_children(bid)):
            # A concat result: its parts write it between them, so it has
            # no writer of its own and the "written exactly once" rule
            # below would misfire.
            if buf_writers:
                _fail(
                    HirVerifyError, bid,
                    f"buffer is assembled from 'part' aliases and must not also be written "
                    f"directly, written by {buf_writers}",
                )
            if buf.role not in _WRITTEN_ONCE_ROLES:
                _fail(
                    HirVerifyError, bid,
                    f"buffer assembled from 'part' aliases has role {buf.role!r}, "
                    f"expected one of {_WRITTEN_ONCE_ROLES}",
                )
            if buf.role == "output" and bid not in module.entry_outputs:
                _fail(HirVerifyError, bid, "output buffer not listed in entry_outputs")
            continue
        if buf.role in _NEVER_WRITTEN_ROLES:
            if buf_writers:
                _fail(HirVerifyError, bid, f"{buf.role} buffer must never be written, written by {buf_writers}")
            if buf.role == "const":
                if buf.data is None:
                    _fail(HirVerifyError, bid, "const buffer has no data")
                elif len(buf.data) != buf.size_bytes:
                    _fail(HirVerifyError, bid, f"const buffer data length {len(buf.data)} != size_bytes {buf.size_bytes}")
        elif buf.role in _WRITTEN_ONCE_ROLES:
            if len(buf_writers) != 1:
                _fail(HirVerifyError, bid, f"buffer must be written by exactly one op, written by {buf_writers}")
            if buf.role == "output" and bid not in module.entry_outputs:
                _fail(HirVerifyError, bid, "output buffer not listed in entry_outputs")

    for bid in module.entry_inputs:
        buf = module.buffers.get(bid)
        if buf is None:
            _fail(HirVerifyError, bid, "entry_inputs references unknown buffer")
        elif buf.role != "input":
            _fail(HirVerifyError, bid, f"entry_inputs buffer has role {buf.role!r}, expected 'input'")
    for bid in module.entry_outputs:
        buf = module.buffers.get(bid)
        if buf is None:
            _fail(HirVerifyError, bid, "entry_outputs references unknown buffer")
        elif buf.role != "output":
            _fail(HirVerifyError, bid, f"entry_outputs buffer has role {buf.role!r}, expected 'output'")

    for op in module.ops:
        if target is not None:
            _verify_op_capability(op, target)
        for bid in op.reads:
            buf = module.buffers.get(bid)
            if buf is None:
                continue  # already reported above
            if bid in module.entry_inputs or buf.role == "const":
                continue
            if writers.get(bid):
                continue
            if buf.alias_parent is not None or module.alias_children(bid):
                # An alias is filled by whoever writes the storage it
                # shares -- its parent, or its own parts -- not by an op
                # naming this id.
                continue
            _fail(HirVerifyError, op.id, f"reads buffer {bid!r} that is neither an entry input, a const, nor written by any op")


def _verify_scheduled(module: HirModule) -> None:
    n = len(module.ops)
    seqs: dict[str, int] = {}
    for op in module.ops:
        if op.seq is None:
            _fail(HirVerifyError, op.id, "scheduled stage requires every op to have a seq")
        seqs[op.id] = op.seq

    if sorted(seqs.values()) != list(range(n)):
        _fail(HirVerifyError, module.target_name, f"op seqs {sorted(seqs.values())} are not a permutation of 0..{n - 1}")

    for op in module.ops:
        for dep in op.deps:
            if not (seqs[dep] < seqs[op.id]):
                _fail(HirVerifyError, op.id, f"dep {dep!r} has seq {seqs[dep]} >= own seq {seqs[op.id]}")

    writer_seq: dict[str, int] = {}
    for op in module.ops:
        for bid in op.writes:
            writer_seq[bid] = seqs[op.id]

    for op in module.ops:
        for bid in op.reads:
            buf = module.buffers.get(bid)
            if buf is None or buf.role not in ("intermediate", "output"):
                continue
            # An alias has no writer of its own; what has to be finished
            # before it can be read is everything that writes the storage
            # it shares (`HirModule.storage_dependencies`).
            candidates = [writer_seq.get(source) for source in module.storage_dependencies(bid)]
            w_seq = max((s for s in candidates if s is not None), default=None)
            if w_seq is not None and seqs[op.id] <= w_seq:
                _fail(HirVerifyError, op.id, f"reads buffer {bid!r} at seq {seqs[op.id]} not after its writer's seq {w_seq}")


def _lifetimes(module: HirModule) -> dict:
    """Per-buffer `(start, end)` in `seq` units.

    Every access to an alias is charged to the buffer that OWNS the
    storage (`alias_root`): a concat part being written is the parent
    coming to life, and a slice view being read is the parent still being
    needed. Getting this wrong would let the planner reuse a parent's
    bytes while one of its views was still live."""
    neg_inf, pos_inf = float("-inf"), float("inf")
    writer_seq: dict[str, int] = {}
    reader_seqs: dict[str, list[int]] = {bid: [] for bid in module.buffers}
    for op in module.ops:
        for bid in op.writes:
            root = module.alias_root(bid)
            writer_seq[bid] = op.seq
            if root != bid:
                # A part's write starts the parent's lifetime too; the
                # earliest such write is the one that matters.
                writer_seq[root] = min(writer_seq.get(root, op.seq), op.seq)
        for bid in op.reads:
            reader_seqs.setdefault(bid, []).append(op.seq)
            root = module.alias_root(bid)
            if root != bid:
                reader_seqs.setdefault(root, []).append(op.seq)

    lifetimes: dict[str, tuple[float, float]] = {}
    for bid, buf in module.buffers.items():
        if buf.role in ("const", "program"):
            lifetimes[bid] = (neg_inf, pos_inf)
        elif buf.role == "input":
            reads = reader_seqs.get(bid, [])
            lifetimes[bid] = (-1, max(reads) if reads else -1)
        elif buf.role == "output":
            lifetimes[bid] = (writer_seq.get(bid, neg_inf), pos_inf)
        else:  # intermediate
            w = writer_seq.get(bid, neg_inf)
            reads = reader_seqs.get(bid, [])
            lifetimes[bid] = (w, max(reads) if reads else w)
    return lifetimes


def _verify_aliases_planned(module: HirModule, target: "Target | None") -> None:
    """Every view sits exactly where its plane offset says, and inside its
    parent; sibling `part`s do not overlap each other.

    This is the check that makes a concat *provably* free rather than
    hopefully free: if the producer of a part were pointed anywhere but
    `parent.addr + offset`, the concat result would be assembled out of
    the wrong bytes, and no value-level test of the parts alone would
    notice (they would each be individually correct)."""
    if target is None:
        return
    plane_channels = target.memory.activation_plane_channels

    for bid, buf in module.buffers.items():
        if buf.alias_parent is None:
            continue
        parent = module.buffers.get(buf.alias_parent)
        if parent is None or parent.addr is None or buf.addr is None:
            continue
        expected = parent.addr + buf.alias_byte_offset(plane_channels)
        if buf.addr != expected:
            _fail(
                MemoryPlanError, bid,
                f"alias addr 0x{buf.addr:x} != parent {buf.alias_parent!r} addr 0x{parent.addr:x} + "
                f"plane offset {buf.alias_plane_offset} (0x{expected:x})",
            )
        if buf.addr < parent.addr or buf.addr + buf.size_bytes > parent.addr + parent.size_bytes:
            _fail(
                MemoryPlanError, bid,
                f"alias range [0x{buf.addr:x},0x{buf.addr + buf.size_bytes:x}) is not contained in "
                f"parent {buf.alias_parent!r} [0x{parent.addr:x},0x{parent.addr + parent.size_bytes:x})",
            )

    for bid in module.buffers:
        parts = [c for c in module.alias_children(bid) if c.alias_kind == "part"]
        for i, first in enumerate(parts):
            for second in parts[i + 1 :]:
                if first.addr is None or second.addr is None:
                    continue
                if first.addr < second.addr + second.size_bytes and second.addr < first.addr + first.size_bytes:
                    _fail(
                        MemoryPlanError, first.id,
                        f"concat part overlaps sibling part {second.id!r} inside {bid!r} "
                        f"([0x{first.addr:x},0x{first.addr + first.size_bytes:x}) vs "
                        f"[0x{second.addr:x},0x{second.addr + second.size_bytes:x}))",
                    )


def _verify_planned(module: HirModule, target: "Target | None") -> None:
    for bid, buf in module.buffers.items():
        if buf.addr is None:
            _fail(MemoryPlanError, bid, "planned stage requires every buffer to have an addr")
            continue
        if buf.alias_parent is not None:
            # A view's address is its parent's plus a plane offset; the
            # parent carries the alignment, and demanding it again here
            # would forbid perfectly legal offsets into an aligned buffer.
            continue
        if buf.addr % buf.align != 0:
            _fail(MemoryPlanError, bid, f"addr 0x{buf.addr:x} not aligned to {buf.align}")
        if target is not None:
            space = target.memory.spaces.get(buf.space)
            if space is not None and buf.addr + buf.size_bytes > space.size_bytes:
                _fail(
                    MemoryPlanError,
                    bid,
                    f"buffer end 0x{buf.addr + buf.size_bytes:x} exceeds space {buf.space!r} size 0x{space.size_bytes:x}",
                )

    _verify_aliases_planned(module, target)

    lifetimes = _lifetimes(module)
    ids = list(module.buffers)
    for i, bid in enumerate(ids):
        buf = module.buffers[bid]
        for other_id in ids[i + 1 :]:
            other = module.buffers[other_id]
            if buf.space != other.space:
                continue
            if module.alias_root(bid) == module.alias_root(other_id):
                # Two views of one buffer (or a view and its parent) share
                # storage on purpose. `_verify_aliases_planned` above is
                # what checks that sharing is the sharing that was meant;
                # the live-range rule below is about buffers that were
                # supposed to be independent.
                continue
            a0, a1 = buf.addr, buf.addr + buf.size_bytes
            b0, b1 = other.addr, other.addr + other.size_bytes
            if not (a0 < b1 and b0 < a1):
                continue  # address ranges don't overlap
            lt_a, lt_b = lifetimes[bid], lifetimes[other_id]
            if not (lt_a[1] < lt_b[0] or lt_b[1] < lt_a[0]):
                _fail(
                    MemoryPlanError,
                    bid,
                    f"overlaps address range [0x{a0:x},0x{a1:x}) with buffer {other_id!r} "
                    f"[0x{b0:x},0x{b1:x}) while their lifetimes intersect",
                )

    # No op may read and write overlapping bytes. Before buffer views this
    # was impossible by construction (every buffer had its own range);
    # with views it is not -- a convolution can legitimately read one
    # channel range of a concat result and write another, and reading and
    # writing the SAME range would corrupt its own input mid-instruction
    # (the hardware streams, it does not snapshot).
    for op in module.ops:
        for read_id in op.reads:
            read_buf = module.buffers.get(read_id)
            if read_buf is None or read_buf.addr is None:
                continue
            for write_id in op.writes:
                write_buf = module.buffers.get(write_id)
                if write_buf is None or write_buf.addr is None or write_buf.space != read_buf.space:
                    continue
                if read_buf.addr < write_buf.addr + write_buf.size_bytes and (
                    write_buf.addr < read_buf.addr + read_buf.size_bytes
                ):
                    _fail(
                        MemoryPlanError, op.id,
                        f"reads {read_id!r} [0x{read_buf.addr:x},0x{read_buf.addr + read_buf.size_bytes:x}) "
                        f"and writes {write_id!r} [0x{write_buf.addr:x},"
                        f"0x{write_buf.addr + write_buf.size_bytes:x}), which overlap",
                    )

    max_end = max((buf.addr + buf.size_bytes for buf in module.buffers.values() if buf.addr is not None), default=0)
    if module.memory_size is None or module.memory_size < max_end:
        _fail(MemoryPlanError, module.target_name, f"memory_size {module.memory_size} < required {max_end}")

    if module.program is None:
        _fail(MemoryPlanError, module.target_name, "planned stage requires program to be set")
        return
    prog = module.buffers.get(module.program)
    if prog is None:
        _fail(MemoryPlanError, module.program, "program buffer id not found in buffers")
    elif prog.role != "program":
        _fail(MemoryPlanError, module.program, f"program buffer role {prog.role!r} != 'program'")
    elif prog.data is None or len(prog.data) != prog.size_bytes:
        _fail(MemoryPlanError, module.program, "program buffer data missing or length mismatch")
