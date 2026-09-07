"""M6 tests: `cnnc.hir` (Hardware IR dataclasses, verifier, printer/to_json).

doc/tosa_compiler_plan.md §2.2 (HIR shape), §7 (BufferView/TileInfo, unused
before M13), §8 (memory model + planner verifier), §11 rows 05/06/07.
"""

from __future__ import annotations

import copy
import dataclasses
import json

import pytest

from cnnc.hir.ir import Buffer, HirModule, HirOp
from cnnc.hir.printer import print_hir, to_json
from cnnc.hir.verify import HirVerifyError, MemoryPlanError, verify_hir
from cnnc.target import load_target

TARGET = load_target("cnn_accel")

_CONV_PARAMS = {
    "in_channels": 4,
    "in_height": 8,
    "in_width": 8,
    "kernel_h": 3,
    "kernel_w": 3,
    "out_channels": 8,
    "pad_top": 1,
    "pad_bottom": 1,
    "pad_left": 1,
    "pad_right": 1,
    "stride_h": 1,
    "stride_w": 1,
    "bias_en": True,
    "requant_en": True,
    "requant_scale": 1073741824,
    "requant_shift": 23,
    "relu_en": True,
}


def _example_buffers() -> dict:
    return {
        "%arg0": Buffer(
            id="%arg0", space="ddr", size_bytes=256, align=64, role="input",
            layout="HWC", shape=(1, 8, 8, 4), dtype="i8",
        ),
        "%0": Buffer(
            id="%0", space="ddr", size_bytes=288, align=64, role="const",
            layout="OHWI", shape=(8, 3, 3, 4), dtype="i8", data=bytes(288),
        ),
        "%1": Buffer(
            id="%1", space="ddr", size_bytes=32, align=64, role="const",
            layout="I32_VEC", shape=(8,), dtype="i32", data=bytes(32),
        ),
        "%10": Buffer(
            id="%10", space="ddr", size_bytes=512, align=64, role="output",
            layout="HWC", shape=(1, 8, 8, 8), dtype="i8",
        ),
        "%program": Buffer(
            id="%program", space="ddr", size_bytes=128, align=64, role="program",
            layout="PROGRAM", shape=(128,), dtype="i8",
        ),
    }


def _example_ops() -> tuple:
    conv = HirOp(
        id="#0", unit="conv_engine", kind="conv_layer", params=_CONV_PARAMS,
        reads=("%arg0", "%0", "%1"), writes=("%10",), deps=(), gir_op="%10",
    )
    halt = HirOp(id="#1", unit="sequencer", kind="halt", params={}, reads=(), writes=(), deps=("#0",))
    return (conv, halt)


def _mapped_module() -> HirModule:
    return HirModule(
        target_name="cnn_accel_v1", ops=_example_ops(), buffers=_example_buffers(),
        entry_inputs=("%arg0",), entry_outputs=("%10",), stage="mapped",
    )


def _scheduled_module() -> HirModule:
    ops = tuple(op.replace(seq=i) for i, op in enumerate(_example_ops()))
    return HirModule(
        target_name="cnn_accel_v1", ops=ops, buffers=_example_buffers(),
        entry_inputs=("%arg0",), entry_outputs=("%10",), stage="scheduled",
    )


def _planned_module() -> HirModule:
    ops = tuple(op.replace(seq=i) for i, op in enumerate(_example_ops()))
    addrs = {"%arg0": 0x10000, "%0": 0x1000, "%1": 0x1140, "%10": 0x20000, "%program": 0}
    buffers = {}
    for bid, buf in _example_buffers().items():
        data = bytes(buf.size_bytes) if bid == "%program" else buf.data
        buffers[bid] = dataclasses.replace(buf, addr=addrs[bid], data=data)
    return HirModule(
        target_name="cnn_accel_v1", ops=ops, buffers=buffers, entry_inputs=("%arg0",),
        entry_outputs=("%10",), program="%program", memory_size=0x20000 + 512, stage="planned",
    )


# ---------------------------------------------------------------------------
# Positive: mapped/scheduled/planned all verify (with the real target).
# ---------------------------------------------------------------------------


def test_mapped_module_verifies():
    verify_hir(_mapped_module(), TARGET)


def test_scheduled_module_verifies():
    verify_hir(_scheduled_module(), TARGET)


def test_planned_module_verifies():
    verify_hir(_planned_module(), TARGET)


def test_verify_without_target_skips_capability_checks():
    verify_hir(_planned_module())


# ---------------------------------------------------------------------------
# Mapped-stage negatives.
# ---------------------------------------------------------------------------


def test_read_of_nonexistent_buffer():
    op = HirOp(id="#0", unit="conv_engine", kind="conv_layer", params={}, reads=("%ghost",), writes=(), deps=())
    module = HirModule(target_name="t", ops=(op,), buffers={}, entry_inputs=(), entry_outputs=())
    with pytest.raises(HirVerifyError, match="#0") as exc_info:
        verify_hir(module)
    assert "%ghost" in str(exc_info.value)


def test_intermediate_written_twice():
    in_buf = Buffer(id="%in", space="ddr", size_bytes=64, align=64, role="input", layout="HWC", shape=(1, 4, 4, 4), dtype="i8")
    mid_buf = Buffer(id="%mid", space="ddr", size_bytes=64, align=64, role="intermediate", layout="HWC", shape=(1, 4, 4, 4), dtype="i8")
    op_a = HirOp(id="#a", unit="conv_engine", kind="conv_layer", params={}, reads=("%in",), writes=("%mid",), deps=())
    op_b = HirOp(id="#b", unit="conv_engine", kind="conv_layer", params={}, reads=("%in",), writes=("%mid",), deps=())
    module = HirModule(
        target_name="t", ops=(op_a, op_b), buffers={"%in": in_buf, "%mid": mid_buf},
        entry_inputs=("%in",), entry_outputs=(),
    )
    with pytest.raises(HirVerifyError) as exc_info:
        verify_hir(module)
    assert "%mid" in str(exc_info.value)


def test_input_written():
    in_buf = Buffer(id="%in", space="ddr", size_bytes=64, align=64, role="input", layout="HWC", shape=(1, 4, 4, 4), dtype="i8")
    op = HirOp(id="#0", unit="conv_engine", kind="conv_layer", params={}, reads=(), writes=("%in",), deps=())
    module = HirModule(target_name="t", ops=(op,), buffers={"%in": in_buf}, entry_inputs=("%in",), entry_outputs=())
    with pytest.raises(HirVerifyError) as exc_info:
        verify_hir(module)
    assert "%in" in str(exc_info.value)


def test_const_without_data():
    c = Buffer(id="%c", space="ddr", size_bytes=16, align=64, role="const", layout="I32_VEC", shape=(4,), dtype="i32")
    module = HirModule(target_name="t", ops=(), buffers={"%c": c}, entry_inputs=(), entry_outputs=())
    with pytest.raises(HirVerifyError) as exc_info:
        verify_hir(module)
    assert "%c" in str(exc_info.value)


def test_const_data_length_mismatch():
    c = Buffer(id="%c", space="ddr", size_bytes=16, align=64, role="const", layout="I32_VEC", shape=(4,), dtype="i32", data=bytes(4))
    module = HirModule(target_name="t", ops=(), buffers={"%c": c}, entry_inputs=(), entry_outputs=())
    with pytest.raises(HirVerifyError) as exc_info:
        verify_hir(module)
    assert "%c" in str(exc_info.value)


def test_dep_on_unknown_op():
    op = HirOp(id="#0", unit="conv_engine", kind="conv_layer", params={}, reads=(), writes=(), deps=("#missing",))
    module = HirModule(target_name="t", ops=(op,), buffers={}, entry_inputs=(), entry_outputs=())
    with pytest.raises(HirVerifyError) as exc_info:
        verify_hir(module)
    assert "#0" in str(exc_info.value) and "#missing" in str(exc_info.value)


def test_capability_unknown_unit():
    module = _mapped_module()
    bad_op = module.op("#0").replace(unit="bogus_unit")
    module = module.with_ops((bad_op, module.op("#1")))
    with pytest.raises(HirVerifyError) as exc_info:
        verify_hir(module, TARGET)
    assert "#0" in str(exc_info.value)


def test_capability_kind_not_supported():
    module = _mapped_module()
    bad_op = module.op("#0").replace(kind="pool2d")
    module = module.with_ops((bad_op, module.op("#1")))
    with pytest.raises(HirVerifyError) as exc_info:
        verify_hir(module, TARGET)
    assert "#0" in str(exc_info.value)


def test_space_not_in_target():
    module = _mapped_module()
    bad_buf = dataclasses.replace(module.buffer("%arg0"), space="sram")
    module = module.with_buffers({"%arg0": bad_buf})
    with pytest.raises(HirVerifyError) as exc_info:
        verify_hir(module, TARGET)
    assert "%arg0" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Scheduled-stage negatives.
# ---------------------------------------------------------------------------


def test_scheduled_missing_seq():
    module = _scheduled_module()
    ops = (module.op("#0").replace(seq=None), module.op("#1"))
    module = module.with_ops(ops)
    with pytest.raises(HirVerifyError) as exc_info:
        verify_hir(module)
    assert "#0" in str(exc_info.value)


def test_scheduled_dep_with_larger_seq():
    module = _scheduled_module()
    # #1 already depends on #0; force #0's seq above #1's to break the order.
    ops = (module.op("#0").replace(seq=1), module.op("#1").replace(seq=0))
    module = module.with_ops(ops)
    with pytest.raises(HirVerifyError) as exc_info:
        verify_hir(module)
    assert "#1" in str(exc_info.value)


def test_scheduled_reader_before_writer():
    in_buf = Buffer(id="%in", space="ddr", size_bytes=64, align=64, role="input", layout="HWC", shape=(1, 4, 4, 4), dtype="i8")
    mid_buf = Buffer(id="%mid", space="ddr", size_bytes=64, align=64, role="intermediate", layout="HWC", shape=(1, 4, 4, 4), dtype="i8")
    reader = HirOp(id="#reader", unit="conv_engine", kind="conv_layer", params={}, reads=("%mid",), writes=(), deps=(), seq=0)
    writer = HirOp(id="#writer", unit="conv_engine", kind="conv_layer", params={}, reads=("%in",), writes=("%mid",), deps=(), seq=1)
    module = HirModule(
        target_name="t", ops=(reader, writer), buffers={"%in": in_buf, "%mid": mid_buf},
        entry_inputs=("%in",), entry_outputs=(), stage="scheduled",
    )
    with pytest.raises(HirVerifyError) as exc_info:
        verify_hir(module)
    assert "#reader" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Planned-stage negatives.
# ---------------------------------------------------------------------------


def test_planned_misaligned_addr():
    module = _planned_module()
    bad_buf = dataclasses.replace(module.buffer("%arg0"), addr=0x10001)
    module = module.with_buffers({"%arg0": bad_buf})
    with pytest.raises(MemoryPlanError) as exc_info:
        verify_hir(module, TARGET)
    assert "%arg0" in str(exc_info.value)


def test_planned_addr_beyond_space_size():
    module = _planned_module()
    ddr_size = TARGET.memory.spaces["ddr"].size_bytes
    bad_addr = ((ddr_size - 64) // 64) * 64  # 64-aligned, but +size_bytes overflows
    bad_buf = dataclasses.replace(module.buffer("%arg0"), addr=bad_addr)
    module = module.with_buffers({"%arg0": bad_buf})
    with pytest.raises(MemoryPlanError) as exc_info:
        verify_hir(module, TARGET)
    assert "%arg0" in str(exc_info.value)


def _two_intermediates_module(write_a: int, read_a: int, write_b: int, read_b: int, addr_a: int, addr_b: int) -> HirModule:
    """Two independent intermediate buffers %a (written at seq `write_a`,
    read at seq `read_a`) and %b (written at seq `write_b`, read at seq
    `read_b`), placed at `addr_a`/`addr_b` -- used to probe the
    overlap-vs-lifetime rule. `write_a/read_a/write_b/read_b` plus one
    trailing op must be a permutation of 0..4."""
    in_buf = Buffer(id="%in", space="ddr", size_bytes=64, align=64, role="input", layout="HWC", shape=(1, 4, 4, 4), dtype="i8", addr=0x10000)
    a_buf = Buffer(id="%a", space="ddr", size_bytes=64, align=64, role="intermediate", layout="HWC", shape=(1, 4, 4, 4), dtype="i8", addr=addr_a)
    b_buf = Buffer(id="%b", space="ddr", size_bytes=64, align=64, role="intermediate", layout="HWC", shape=(1, 4, 4, 4), dtype="i8", addr=addr_b)
    sink = Buffer(id="%out", space="ddr", size_bytes=64, align=64, role="output", layout="HWC", shape=(1, 4, 4, 4), dtype="i8", addr=0x20000)
    prog = Buffer(id="%program", space="ddr", size_bytes=16, align=64, role="program", layout="PROGRAM", shape=(16,), dtype="i8", addr=0x30000, data=bytes(16))
    used = {write_a, read_a, write_b, read_b}
    final_seq = next(s for s in range(5) if s not in used)
    write_a_op = HirOp(id="#write_a", unit="conv_engine", kind="conv_layer", params={}, reads=("%in",), writes=("%a",), deps=(), seq=write_a)
    read_a_op = HirOp(id="#read_a", unit="conv_engine", kind="conv_layer", params={}, reads=("%a",), writes=(), deps=(), seq=read_a)
    write_b_op = HirOp(id="#write_b", unit="conv_engine", kind="conv_layer", params={}, reads=("%in",), writes=("%b",), deps=(), seq=write_b)
    read_b_op = HirOp(id="#read_b", unit="conv_engine", kind="conv_layer", params={}, reads=("%b",), writes=(), deps=(), seq=read_b)
    final = HirOp(id="#final", unit="conv_engine", kind="conv_layer", params={}, reads=("%in",), writes=("%out",), deps=(), seq=final_seq)
    ops = (write_a_op, read_a_op, write_b_op, read_b_op, final)
    return HirModule(
        target_name="t", ops=ops,
        buffers={"%in": in_buf, "%a": a_buf, "%b": b_buf, "%out": sink, "%program": prog},
        entry_inputs=("%in",), entry_outputs=("%out",),
        program="%program", memory_size=0x30000 + 16, stage="planned",
    )


def test_planned_overlapping_addresses_overlapping_lifetimes():
    # %a lives [0,2], %b lives [1,3]: intervals intersect -> same address rejected.
    module = _two_intermediates_module(write_a=0, read_a=2, write_b=1, read_b=3, addr_a=0x1000, addr_b=0x1000)
    with pytest.raises(MemoryPlanError) as exc_info:
        verify_hir(module)
    assert "%a" in str(exc_info.value) or "%b" in str(exc_info.value)


def test_planned_overlapping_addresses_disjoint_lifetimes_accepted():
    # %a lives [0,1], %b lives [2,3]: disjoint -> sharing the address is fine.
    module = _two_intermediates_module(write_a=0, read_a=1, write_b=2, read_b=3, addr_a=0x1000, addr_b=0x1000)
    verify_hir(module)


def test_planned_program_overlapping_data_buffer():
    module = _planned_module()
    bad_const = dataclasses.replace(module.buffer("%0"), addr=0)  # collides with %program at addr 0
    module = module.with_buffers({"%0": bad_const})
    with pytest.raises(MemoryPlanError) as exc_info:
        verify_hir(module)
    assert "%0" in str(exc_info.value) or "%program" in str(exc_info.value)


def test_planned_memory_size_too_small():
    module = _planned_module()
    module = module.replace(memory_size=100)
    with pytest.raises(MemoryPlanError):
        verify_hir(module)


def test_planned_requires_program():
    module = _planned_module().replace(program=None)
    with pytest.raises(MemoryPlanError):
        verify_hir(module)


def test_planned_program_data_length_mismatch():
    module = _planned_module()
    bad_prog = dataclasses.replace(module.buffer("%program"), data=bytes(4))
    module = module.with_buffers({"%program": bad_prog})
    with pytest.raises(MemoryPlanError) as exc_info:
        verify_hir(module)
    assert "%program" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Printer.
# ---------------------------------------------------------------------------

_GOLDEN = (
    "hir.module target=cnn_accel_v1 stage=planned memory_size=131584\n"
    "buffers:\n"
    "  %arg0 ddr HWC     1x8x8x4xi8 256 B align 64 addr 0x00010000 role input\n"
    "  %0    ddr OHWI    8x3x3x4xi8 288 B align 64 addr 0x00001000 role const\n"
    "  %1    ddr I32_VEC 8xi32       32 B align 64 addr 0x00001140 role const\n"
    "  %10   ddr HWC     1x8x8x8xi8 512 B align 64 addr 0x00020000 role output\n"
    "ops:\n"
    "  #0 seq 0 unit conv_engine kind conv_layer gir %10\n"
    "     reads [%arg0 %0 %1] writes [%10] deps []\n"
    "     params {bias_en=1 in_channels=4 in_height=8 in_width=8 kernel_h=3 kernel_w=3 "
    "out_channels=8 pad_bottom=1 pad_left=1 pad_right=1 pad_top=1 relu_en=1 requant_en=1 "
    "requant_scale=1073741824 requant_shift=23 stride_h=1 stride_w=1}\n"
    "  #1 seq 1 unit sequencer kind halt gir -\n"
    "     reads [] writes [] deps [#0]\n"
    "     params {}\n"
    "program: %program 128 B addr 0x00000000"
)


def test_print_hir_golden():
    assert print_hir(_planned_module()) == _GOLDEN


def test_print_hir_addr_none():
    text = print_hir(_mapped_module())
    assert "addr -" in text
    assert "memory_size=-" in text
    assert "program: -" in text
    assert "seq -" in text


def test_print_hir_notes():
    module = _mapped_module().replace(notes=("hint one", "hint two"))
    text = print_hir(module)
    assert text.endswith("note: hint one\nnote: hint two")


def test_to_json_deterministic_and_dumpable():
    module = _planned_module()
    d1 = to_json(module)
    d2 = to_json(module)
    assert d1 == d2
    text1 = json.dumps(d1, sort_keys=True)
    text2 = json.dumps(d2, sort_keys=True)
    assert text1 == text2
    # `data` is replaced by a sha256 + length, never raw bytes.
    const_buf = d1["buffers"]["%0"]["data"]
    assert set(const_buf) == {"sha256", "length"}
    assert const_buf["length"] == 288
    assert isinstance(const_buf["sha256"], str) and len(const_buf["sha256"]) == 64


def test_to_json_roundtrips_through_json_module():
    module = _planned_module()
    text = json.dumps(to_json(module))
    reloaded = json.loads(text)
    assert reloaded["target_name"] == "cnn_accel_v1"
    assert reloaded["stage"] == "planned"


# ---------------------------------------------------------------------------
# HirModule helpers.
# ---------------------------------------------------------------------------


def test_with_buffers_does_not_mutate_original():
    module = _mapped_module()
    original_buf = module.buffer("%0")
    new_buf = dataclasses.replace(original_buf, addr=0x9999)
    updated = module.with_buffers({"%0": new_buf})

    assert module.buffer("%0") is original_buf
    assert module.buffer("%0").addr is None
    assert updated.buffer("%0").addr == 0x9999
    assert updated is not module


def test_writer_and_readers_of():
    module = _mapped_module()
    writer = module.writer_of("%10")
    assert writer is not None and writer.id == "#0"
    assert module.writer_of("%arg0") is None
    readers = module.readers_of("%arg0")
    assert [op.id for op in readers] == ["#0"]


def test_op_and_buffer_lookup():
    module = _mapped_module()
    assert module.op("#0").kind == "conv_layer"
    with pytest.raises(KeyError):
        module.op("#missing")
    with pytest.raises(KeyError):
        module.buffer("%missing")
