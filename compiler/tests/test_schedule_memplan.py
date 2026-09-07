"""M7 tests: `cnnc.lower.schedule` (topological scheduling) and
`cnnc.lower.memplan` (§8 DDR memory planner).

doc/tosa_compiler_plan.md §2.2 (HIR shape), §8 (memory model, this
milestone's spec), §11 rows 06/07, §13 M7.
"""

from __future__ import annotations

import dataclasses

import pytest

from cnnc.hir.ir import Buffer, HirModule, HirOp
from cnnc.hir.printer import print_hir
from cnnc.hir.verify import MemoryPlanError, verify_hir
from cnnc.lower.memplan import lifetimes, memory_map, plan_memory
from cnnc.lower.schedule import ScheduleError, schedule
from cnnc.target.load import load_target

TARGET = load_target("cnn_accel")


def _buf(id_: str, role: str, size: int, *, layout: str = "HWC", shape=None, dtype: str = "i8", data: bytes | None = None) -> Buffer:
    return Buffer(
        id=id_, space="ddr", size_bytes=size, align=64, role=role, layout=layout,
        shape=shape or (size,), dtype=dtype, data=data,
    )


def _op(id_: str, reads: tuple, writes: tuple, *, deps: tuple = ()) -> HirOp:
    return HirOp(id=id_, unit="conv_engine", kind="conv_layer", params={}, reads=reads, writes=writes, deps=deps)


# ---------------------------------------------------------------------------
# schedule
# ---------------------------------------------------------------------------


def _three_op_chain() -> HirModule:
    """`%in -> #A -> %t1 -> #B -> %t2 -> #C -> %out`, ops given to
    `HirModule.ops` in reverse (`#C, #B, #A`) to exercise reordering."""
    buffers = {
        "%in": _buf("%in", "input", 64),
        "%t1": _buf("%t1", "intermediate", 64),
        "%t2": _buf("%t2", "intermediate", 64),
        "%out": _buf("%out", "output", 64),
    }
    op_a = _op("#A", ("%in",), ("%t1",))
    op_b = _op("#B", ("%t1",), ("%t2",))
    op_c = _op("#C", ("%t2",), ("%out",))
    return HirModule(
        target_name="t", ops=(op_c, op_b, op_a), buffers=buffers, entry_inputs=("%in",),
        entry_outputs=("%out",), stage="mapped",
    )


def test_schedule_reorders_reverse_input_topologically():
    scheduled = schedule(_three_op_chain())
    assert [op.id for op in scheduled.ops] == ["#A", "#B", "#C"]
    assert [op.seq for op in scheduled.ops] == [0, 1, 2]


def test_schedule_fills_implied_deps():
    scheduled = schedule(_three_op_chain())
    by_id = {op.id: op for op in scheduled.ops}
    assert by_id["#A"].deps == ()
    assert by_id["#B"].deps == ("#A",)
    assert by_id["#C"].deps == ("#B",)


def test_schedule_respects_explicit_dep():
    # %x/%y share no buffer (both only *read* %in), so the implied graph
    # alone allows either order; the explicit dep must still be honored.
    buffers = {
        "%in": _buf("%in", "input", 64),
        "%x": _buf("%x", "output", 64),
        "%y": _buf("%y", "output", 64),
    }
    op_y = _op("#Y", ("%in",), ("%y",), deps=("#X",))
    op_x = _op("#X", ("%in",), ("%x",))
    module = HirModule(
        target_name="t", ops=(op_y, op_x), buffers=buffers, entry_inputs=("%in",),
        entry_outputs=("%x", "%y"), stage="mapped",
    )
    scheduled = schedule(module)
    seqs = {op.id: op.seq for op in scheduled.ops}
    assert seqs["#X"] < seqs["#Y"]


def test_schedule_cycle_raises_schedule_error_naming_ops():
    buffers = {
        "%a": _buf("%a", "intermediate", 64),
        "%b": _buf("%b", "intermediate", 64),
    }
    op_a = _op("#A", ("%b",), ("%a",))
    op_b = _op("#B", ("%a",), ("%b",))
    module = HirModule(
        target_name="t", ops=(op_a, op_b), buffers=buffers, entry_inputs=(), entry_outputs=(), stage="mapped",
    )
    with pytest.raises(ScheduleError) as exc_info:
        schedule(module)
    assert "#A" in str(exc_info.value) and "#B" in str(exc_info.value)


def test_schedule_deterministic():
    module = _three_op_chain()
    first = schedule(module)
    second = schedule(module)
    assert print_hir(first) == print_hir(second)


# ---------------------------------------------------------------------------
# memplan
# ---------------------------------------------------------------------------


def _single_layer_module() -> HirModule:
    """Mirrors the §2.2 example shapes: 8x8x4 in, 3x3 conv, 8 out channels."""
    buffers = {
        "%arg0": _buf("%arg0", "input", 256, shape=(1, 8, 8, 4)),
        "%w": _buf("%w", "const", 288, layout="OHWI", shape=(8, 3, 3, 4), data=bytes(288)),
        "%b": _buf("%b", "const", 32, layout="I32_VEC", shape=(8,), dtype="i32", data=bytes(32)),
        "%y": _buf("%y", "output", 512, shape=(1, 8, 8, 8)),
    }
    op = _op("#0", ("%arg0", "%w", "%b"), ("%y",))
    return HirModule(
        target_name="cnn_accel_v1", ops=(op,), buffers=buffers, entry_inputs=("%arg0",),
        entry_outputs=("%y",), stage="mapped",
    )


def _chain_module(length: int) -> HirModule:
    """`length` conv layers, `%in -> ... -> %out`, with intermediates
    `%t1 .. %t{length-1}` in between (e.g. `length=3` -> `%t1`, `%t2`)."""
    buffers = {"%in": _buf("%in", "input", 512), "%out": _buf("%out", "output", 512)}
    ops = []
    prev = "%in"
    for i in range(length):
        last = i == length - 1
        out_id = "%out" if last else f"%t{i + 1}"
        if not last:
            buffers[out_id] = _buf(out_id, "intermediate", 512)
        ops.append(_op(f"#{i}", (prev,), (out_id,)))
        prev = out_id
    return HirModule(
        target_name="t", ops=tuple(ops), buffers=buffers, entry_inputs=("%in",),
        entry_outputs=("%out",), stage="mapped",
    )


def test_memplan_single_layer_program_consts_io():
    planned = plan_memory(schedule(_single_layer_module()), TARGET)
    prog = planned.buffer(planned.program)
    assert prog.role == "program"
    assert prog.size_bytes == 128  # 64-byte instr word x (1 conv op + HALT)
    assert prog.addr == 0
    for bid in ("%w", "%b", "%arg0", "%y"):
        assert planned.buffer(bid).addr % 64 == 0
    verify_hir(planned, TARGET)


def test_lifetimes_reports_writer_and_last_reader_seq():
    scheduled = schedule(_chain_module(3))
    assert lifetimes(scheduled) == {"%t1": (0, 1), "%t2": (1, 2)}


def test_memplan_three_layers_no_reuse_when_lifetimes_touch():
    # %t1 lives [0,1], %t2 lives [1,2]: %t2's write is at the same seq as
    # %t1's last read, so 1 < 1 is false -> reuse is not allowed.
    planned = plan_memory(schedule(_chain_module(3)), TARGET)
    assert planned.buffer("%t1").addr != planned.buffer("%t2").addr
    verify_hir(planned, TARGET)


def test_memplan_four_layers_reuses_freed_block():
    # %t1 lives [0,1], %t2 lives [1,2], %t3 lives [2,3]: %t3's write (seq 2)
    # is strictly after %t1's last read (seq 1) -> %t1's block is reused.
    planned = plan_memory(schedule(_chain_module(4)), TARGET)
    assert planned.buffer("%t3").addr == planned.buffer("%t1").addr
    assert planned.buffer("%t2").addr != planned.buffer("%t1").addr
    verify_hir(planned, TARGET)


def test_memplan_small_buffer_does_not_straddle_4kib_boundary():
    buffers = {"%t": _buf("%t", "intermediate", 100), "%out": _buf("%out", "output", 64)}
    op1 = _op("#1", (), ("%t",))
    op2 = _op("#2", ("%t",), ("%out",))
    module = HirModule(
        target_name="t", ops=(op1, op2), buffers=buffers, entry_inputs=(), entry_outputs=("%out",), stage="mapped",
    )
    # program (8064 B) + %out (64 B) land the cursor at 0x1fc0; an
    # unaligned-to-4KiB placement of the 100 B %t would straddle 0x2000.
    planned = plan_memory(schedule(module), TARGET, program_size_bytes=8064, base=0)
    addr = planned.buffer("%t").addr
    assert addr == 0x2000
    assert addr // 4096 == (addr + 99) // 4096
    verify_hir(planned, TARGET)


def test_memplan_space_too_small_raises():
    space = next(iter(TARGET.memory.spaces.values()))
    tiny_space = dataclasses.replace(space, size_bytes=256)
    tiny_memory = dataclasses.replace(TARGET.memory, spaces={space.name: tiny_space})
    tiny_target = dataclasses.replace(TARGET, memory=tiny_memory)
    with pytest.raises(MemoryPlanError):
        plan_memory(schedule(_single_layer_module()), tiny_target)


def test_memplan_verifier_catches_misaligned_addr():
    planned = plan_memory(schedule(_single_layer_module()), TARGET)
    bad = dataclasses.replace(planned.buffer("%w"), addr=planned.buffer("%w").addr + 1)
    module = planned.with_buffers({"%w": bad})
    with pytest.raises(MemoryPlanError):
        verify_hir(module, TARGET)


def test_memplan_verifier_catches_overlap_with_program():
    planned = plan_memory(schedule(_single_layer_module()), TARGET)
    bad = dataclasses.replace(planned.buffer("%w"), addr=0)  # program lives at addr 0
    module = planned.with_buffers({"%w": bad})
    with pytest.raises(MemoryPlanError):
        verify_hir(module, TARGET)


def test_memplan_verifier_catches_overlapping_live_intermediates():
    planned = plan_memory(schedule(_chain_module(3)), TARGET)
    bad = dataclasses.replace(planned.buffer("%t2"), addr=planned.buffer("%t1").addr)
    module = planned.with_buffers({"%t2": bad})
    with pytest.raises(MemoryPlanError):
        verify_hir(module, TARGET)


def test_memplan_verifier_catches_buffer_beyond_space_size():
    planned = plan_memory(schedule(_single_layer_module()), TARGET)
    space = next(iter(TARGET.memory.spaces.values()))
    bad = dataclasses.replace(planned.buffer("%y"), addr=space.size_bytes - 10)
    module = planned.with_buffers({"%y": bad})
    with pytest.raises(MemoryPlanError):
        verify_hir(module, TARGET)


def test_memplan_deterministic():
    scheduled = schedule(_chain_module(4))
    first = plan_memory(scheduled, TARGET)
    second = plan_memory(scheduled, TARGET)
    assert print_hir(first) == print_hir(second)


def test_memory_map_is_sorted_and_readable():
    planned = plan_memory(schedule(_single_layer_module()), TARGET)
    lines = memory_map(planned).splitlines()
    assert lines[0].split()[-1] == "id"
    addrs = [int(line.split()[0], 16) for line in lines[1:]]
    assert addrs == sorted(addrs)
