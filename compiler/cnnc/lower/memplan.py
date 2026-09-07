"""DDR memory planner (doc/tosa_compiler_plan.md §8, §11 row 07, §13 M7).

`plan_memory` turns a `stage='scheduled'` `HirModule` into `stage='planned'`
by assigning every `Buffer.addr` and `HirModule.memory_size`, in one
target memory space (asserted to be the only one `target.memory` has --
`cnn_accel` streams everything from DDR; an SRAM-resident target adds a
space, not a redesign, per §8).

Region order, all A-aligned (`A = space.align`):

1. program (fixed size, HALT-terminated placeholder blob of zero bytes --
   the real descriptor stream is M8's `emit.py` job)
2. constants, packed contiguously in module (insertion) order
3. entry inputs, then entry outputs, pinned in module order
4. intermediates, interval first-fit with reuse over a free list, sorted
   by (lifetime start, id); a block is only reused once its previous
   occupant's lifetime ended *strictly* before the new one starts
   (`end < start`), matching the HIR verifier's overlap rule in
   `hir.verify._verify_planned`.

Policy, not a hardware requirement: buffers smaller than 4 KiB are placed
so they never straddle a 4 KiB address boundary (the DMA splits bursts at
4 KiB anyway; keeping small buffers inside one burst window is a
conservative simplification, not something the AXI master demands).
Buffers >= 4 KiB are only A-aligned.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from cnnc.hir.ir import Buffer, HirModule
from cnnc.hir.verify import verify_hir

if TYPE_CHECKING:
    from cnnc.target.contract import Target

_PAGE = 4096


def _align_up(value: int, align: int) -> int:
    return (value + align - 1) // align * align


def _place(cursor: int, size: int, align: int) -> int:
    """Return an aligned address for `size` bytes starting at/after
    `cursor`, bumped to the next 4 KiB boundary if a sub-4 KiB buffer
    would otherwise straddle one (policy, see module docstring)."""
    addr = _align_up(cursor, align)
    if size <= _PAGE:
        start_page, end_page = addr // _PAGE, (addr + size - 1) // _PAGE
        if start_page != end_page:
            addr = _align_up(addr, _PAGE)
    return addr


def lifetimes(module: HirModule) -> dict[str, tuple[int, int]]:
    """`[writer.seq, max(reader.seq)]` (closed) for every `intermediate`
    buffer -- the interval the planner's free-list reuse operates on.
    Requires a scheduled module (every op has `seq`)."""
    writer_seq: dict[str, int] = {}
    reader_seqs: dict[str, list[int]] = {}
    for op in module.ops:
        for buffer_id in op.writes:
            writer_seq[buffer_id] = op.seq
        for buffer_id in op.reads:
            reader_seqs.setdefault(buffer_id, []).append(op.seq)

    result: dict[str, tuple[int, int]] = {}
    for buffer_id, buf in module.buffers.items():
        if buf.role != "intermediate":
            continue
        start = writer_seq.get(buffer_id)
        reads = reader_seqs.get(buffer_id, [])
        end = max(reads) if reads else start
        result[buffer_id] = (start, end)
    return result


def memory_map(module: HirModule) -> str:
    """Human-readable table, sorted by address, of every planned buffer."""
    rows = sorted(
        module.buffers.values(),
        key=lambda buf: (buf.addr if buf.addr is not None else -1, buf.id),
    )
    lines = [f"{'addr':>10} {'end':>10} {'size':>8} {'role':<12} id"]
    for buf in rows:
        addr = buf.addr if buf.addr is not None else -1
        end = addr + buf.size_bytes if buf.addr is not None else -1
        lines.append(f"0x{addr:08x} 0x{end:08x} {buf.size_bytes:8d} {buf.role:<12} {buf.id}")
    return "\n".join(lines)


def plan_memory(
    module: HirModule,
    target: "Target",
    *,
    program_size_bytes: int | None = None,
    base: int = 0,
) -> HirModule:
    assert len(target.memory.spaces) == 1, f"expected exactly one memory space, got {sorted(target.memory.spaces)}"
    space_name, space = next(iter(target.memory.spaces.items()))
    align = space.align

    placements: dict[str, Buffer] = {}
    cursor = base

    # 1. program region.
    existing_program = module.buffers.get(module.program) if module.program else None
    if existing_program is not None and existing_program.role == "program":
        size = existing_program.size_bytes
        addr = _place(cursor, size, align)
        data = existing_program.data if existing_program.data is not None else bytes(size)
        program_buffer = dataclasses.replace(existing_program, addr=addr, data=data)
        program_id = existing_program.id
    else:
        n_ops = sum(1 for op in module.ops if op.kind != "halt")
        size = program_size_bytes if program_size_bytes is not None else target.isa.instr_word_bytes * (n_ops + 1)
        addr = _place(cursor, size, align)
        program_id = "%program"
        program_buffer = Buffer(
            id=program_id, space=space_name, size_bytes=size, align=align, role="program",
            layout="PROGRAM", shape=(size,), dtype="u8", addr=addr, data=bytes(size),
        )
    placements[program_id] = program_buffer
    cursor = addr + size

    # 2. constants, insertion order.
    for buffer_id, buf in module.buffers.items():
        if buf.role != "const":
            continue
        addr = _place(cursor, buf.size_bytes, align)
        placements[buffer_id] = dataclasses.replace(buf, addr=addr)
        cursor = addr + buf.size_bytes

    # 3. entry inputs, then entry outputs, pinned in module order.
    for buffer_id in (*module.entry_inputs, *module.entry_outputs):
        buf = module.buffers[buffer_id]
        addr = _place(cursor, buf.size_bytes, align)
        placements[buffer_id] = dataclasses.replace(buf, addr=addr)
        cursor = addr + buf.size_bytes

    # 4. intermediates: interval first-fit with reuse over a free list.
    lts = lifetimes(module)
    intermediate_ids = sorted(lts, key=lambda bid: (lts[bid][0], bid))
    free_blocks: list[dict] = []  # {"addr", "size", "end_seq"}
    high_water = cursor
    for buffer_id in intermediate_ids:
        buf = module.buffers[buffer_id]
        start, end = lts[buffer_id]
        reused = None
        for block in free_blocks:
            if block["end_seq"] < start and block["size"] >= buf.size_bytes:
                reused = block
                break
        if reused is not None:
            addr = reused["addr"]
            reused["end_seq"] = end
        else:
            addr = _place(high_water, buf.size_bytes, align)
            high_water = addr + buf.size_bytes
            free_blocks.append({"addr": addr, "size": buf.size_bytes, "end_seq": end})
        placements[buffer_id] = dataclasses.replace(buf, addr=addr)
    cursor = max(cursor, high_water)

    module = module.with_buffers(placements)
    memory_size = _align_up(
        max((buf.addr + buf.size_bytes for buf in module.buffers.values() if buf.addr is not None), default=0),
        align,
    )
    module = module.replace(program=program_id, memory_size=memory_size, stage="planned")

    verify_hir(module, target)
    return module
