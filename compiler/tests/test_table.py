"""`tosa.table` -> `ACT` (ISA v2.0), end to end.

This is how a general activation reaches the hardware. SiLU -- YOLOv8n's
activation everywhere -- has no instruction and never will; what it has is
256 answers, and `OPCODE_ACT` looks them up.

The one thing that can go silently wrong here is the *index order*: TOSA
indexes an int8 TABLE by `value - type_min` (entry 0 answers -128) and the
hardware indexes by the raw byte (entry 0 answers 0). Same 256 answers,
rotated by 128. `lower.layout.pack_act_lut` is the single place that
rotation happens; `test_act_lut_image_answers_every_int8_input` checks it
against the golden model's own indexing, and the mutation test at the end
shows that dropping it is not a crash but a wrong answer on every element.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from cnnc.backend.cnn_accel_v1 import decode_program, run_program
from cnnc.driver import compile_tosa
from cnnc.errors import (
    CapabilityError,
    TosaImportError,
    UnsupportedAttribute,
)
from cnnc.frontend.mlir_generic import parse_module
from cnnc.frontend.tosa_import import import_tosa
from cnnc.gir import interp
from cnnc.lower.layout import ACT_LUT_ENTRIES, pack_act_lut
from cnnc.target.contract import Target

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# fixture name -> input shape.
FIXTURES = {
    # One SiLU LUT; C=12 is not a multiple of the 8-channel DDR plane.
    "table_silu": (1, 4, 6, 12),
    # The same LUT twice: a two-instruction chain through an intermediate,
    # with one shared const.
    "table_twice": (1, 8, 8, 8),
}


def _compile(name: str, target, tmp_path=None):
    out = None if tmp_path is None else tmp_path / "out"
    return compile_tosa(FIXTURES_DIR / f"{name}.mlir", target, out_dir=out)


def _seed_input(shape: tuple[int, ...], seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(-128, 128, size=shape).astype(np.int8)


def _act_ops(result):
    return [op for op in result.hir_planned.ops if op.kind == "act"]


def _text(name: str) -> str:
    return (FIXTURES_DIR / f"{name}.mlir").read_text()


def _without_elementwise_unit(target: Target) -> Target:
    data = target.to_dict()
    data["units"] = [u for u in data["units"] if "table" not in u["ops"]]
    return Target.from_dict(data)


# ---------------------------------------------------------------------------
# The LUT image: the one place TOSA's index order becomes the hardware's
# ---------------------------------------------------------------------------


def test_act_lut_image_answers_every_int8_input():
    """For every one of the 256 int8 inputs, the packed image looked up
    the hardware's way must give what TOSA's table says. Checked against
    `cnn_accel_model.act_lut` itself, not against a restatement of the
    rotation."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "modules" / "cnn_accel"))
    import cnn_accel_model  # noqa: E402

    # A table with 256 distinct answers, so no coincidence can hide a
    # mis-rotation: entry i (answering input i - 128) returns i - 128 + 1
    # wrapped into int8.
    tosa_table = tuple(((i - 128 + 1 + 128) % 256) - 128 for i in range(ACT_LUT_ENTRIES))
    raw = pack_act_lut(tosa_table)
    hw_lut = [b - 256 if b >= 128 else b for b in raw]

    for value in range(-128, 128):
        assert cnn_accel_model.act_lut([value], hw_lut) == [tosa_table[value + 128]]


def test_pack_act_lut_rejects_a_table_that_is_not_256_entries():
    with pytest.raises(ValueError, match="256 entries"):
        pack_act_lut(tuple(range(-128, 127)))


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------


def test_import_keeps_the_lut_as_a_tensor_operand():
    graph = import_tosa(parse_module(_text("table_silu")))
    op = next(op for op in graph.ops if op.kind == "table")
    assert len(op.inputs) == 2
    table = graph.tensor(op.inputs[1])
    assert table.shape == (ACT_LUT_ENTRIES,)
    assert table.dtype == "i8"
    assert len(table.values) == ACT_LUT_ENTRIES


def test_import_refuses_an_int16_style_table_input():
    with pytest.raises(UnsupportedAttribute) as exc:
        import_tosa(parse_module(_text("table_silu").replace("x12xi8>", "x12xi32>")))
    assert "int8" in str(exc.value) or "i8" in str(exc.value)


def test_import_refuses_a_table_of_the_wrong_length():
    text = (
        '"builtin.module"() ({\n'
        '  "func.func"() <{function_type = (tensor<1x2x2x8xi8>) -> tensor<1x2x2x8xi8>, sym_name = "main"}> ({\n'
        "  ^bb0(%arg0: tensor<1x2x2x8xi8>):\n"
        '    %lut = "tosa.const"() <{values = dense<1> : tensor<128xi8>}> : () -> tensor<128xi8>\n'
        '    %0 = "tosa.table"(%arg0, %lut) : (tensor<1x2x2x8xi8>, tensor<128xi8>) -> tensor<1x2x2x8xi8>\n'
        '    "func.return"(%0) : (tensor<1x2x2x8xi8>) -> ()\n'
        "  }) : () -> ()\n"
        "}) : () -> ()\n"
    )
    with pytest.raises(UnsupportedAttribute) as exc:
        import_tosa(parse_module(text))
    assert "tensor<256xi8>" in str(exc.value)


def test_import_refuses_a_runtime_computed_table():
    """The accelerator loads its LUT from a constant DDR address, so a
    table that is not a `tosa.const` has no lowering."""
    text = (
        '"builtin.module"() ({\n'
        '  "func.func"() <{function_type = (tensor<1x2x2x8xi8>, tensor<256xi8>) -> tensor<1x2x2x8xi8>, '
        'sym_name = "main"}> ({\n'
        "  ^bb0(%arg0: tensor<1x2x2x8xi8>, %arg1: tensor<256xi8>):\n"
        '    %0 = "tosa.table"(%arg0, %arg1) : (tensor<1x2x2x8xi8>, tensor<256xi8>) -> tensor<1x2x2x8xi8>\n'
        '    "func.return"(%0) : (tensor<1x2x2x8xi8>) -> ()\n'
        "  }) : () -> ()\n"
        "}) : () -> ()\n"
    )
    with pytest.raises(TosaImportError) as exc:
        import_tosa(parse_module(text))
    assert "must be a constant" in str(exc.value)


# ---------------------------------------------------------------------------
# Lower + emit
# ---------------------------------------------------------------------------


def test_table_lowers_onto_the_elementwise_unit(target, tmp_path):
    result = _compile("table_silu", target, tmp_path)
    (op,) = _act_ops(result)
    assert op.unit == "elementwise_engine"
    assert dict(op.params) == {
        "in_width": 6, "in_height": 4, "in_channels": 12, "act_lut_en": True,
    }
    assert len(op.reads) == 2  # input + LUT
    lut = result.hir_planned.buffer(op.reads[1])
    assert (lut.role, lut.layout, lut.size_bytes) == ("const", "ACT_LUT", ACT_LUT_ENTRIES)


def test_act_descriptor_points_weight_addr_at_the_lut(target, tmp_path):
    result = _compile("table_silu", target, tmp_path)
    planned = result.hir_planned
    program_addr = planned.buffer(planned.program).addr
    desc = decode_program(result.program.program_bytes, target, program_addr=program_addr)[0]
    (op,) = _act_ops(result)
    in_buf, lut_buf = (planned.buffer(bid) for bid in op.reads)

    assert desc.opcode == target.isa.opcodes["ACT"]
    assert desc.in_addr == in_buf.addr
    assert desc.out_addr == planned.buffer(op.writes[0]).addr
    # Ambiguity #3: the LUT rides in the "compile-time side table" slot.
    assert desc.weight_addr == lut_buf.addr
    assert desc.xfer_bytes == in_buf.size_bytes
    assert (desc.bias_addr, desc.scale_addr) == (0, 0)
    flags = {name for name, bit in target.isa.flags.items() if (desc.flags >> bit) & 1}
    assert flags == {"ACT_LUT_EN"}


def test_a_shared_lut_is_placed_once(target, tmp_path):
    """`table_twice`'s two ACTs read one `tosa.const`; it must become one
    buffer at one address, not two copies."""
    result = _compile("table_twice", target, tmp_path)
    first, second = _act_ops(result)
    assert first.reads[1] == second.reads[1]
    luts = [b for b in result.hir_planned.buffers.values() if b.layout == "ACT_LUT"]
    assert len(luts) == 1


def test_table_needs_a_unit_that_implements_it(target):
    from cnnc.lower.to_hir import to_hir

    graph = import_tosa(parse_module(_text("table_silu")))
    with pytest.raises(CapabilityError) as exc:
        to_hir(graph, _without_elementwise_unit(target))
    assert "table" in str(exc.value)


def test_table_needs_isa_v20(target):
    from cnnc.lower.to_hir import to_hir

    graph = import_tosa(parse_module(_text("table_silu")))
    data = target.to_dict()
    for unit in data["units"]:
        unit["isa_version"] = "1.2"
    with pytest.raises(CapabilityError) as exc:
        to_hir(graph, Target.from_dict(data))
    assert "2.0" in str(exc.value)


# ---------------------------------------------------------------------------
# End to end: TOSA reference == emitted program on the golden model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(FIXTURES))
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_table_program_matches_tosa_reference(name, seed, target, tmp_path):
    result = _compile(name, target, tmp_path)
    graph = result.imported_graph
    x = _seed_input(FIXTURES[name], seed)
    expected = interp.run(graph, {graph.inputs[0]: x})
    actual = run_program(result.program, {graph.inputs[0]: x})
    assert set(expected) == set(actual)
    for tid, want in expected.items():
        np.testing.assert_array_equal(actual[tid], want, err_msg=f"{name} seed {seed} tensor %{tid}")


def test_every_int8_input_value_goes_through_the_lut(target, tmp_path):
    """An exhaustive input: all 256 int8 codes, so every LUT entry is
    exercised at least once rather than whichever ~200 a random seed
    happens to hit."""
    result = _compile("table_twice", target, tmp_path)
    graph = result.imported_graph
    x = np.arange(-128, 128, dtype=np.int8).reshape(1, 8, 8, 4)
    x = np.concatenate([x, x], axis=3)  # 1x8x8x8
    expected = interp.run(graph, {graph.inputs[0]: x})[graph.outputs[0]]
    actual = run_program(result.program, {graph.inputs[0]: x})[graph.outputs[0]]
    np.testing.assert_array_equal(actual, expected)


# ---------------------------------------------------------------------------
# Mutation: break the index rotation, prove the equality test notices
# ---------------------------------------------------------------------------


def test_mutation_unrotated_lut_breaks_the_equality_test(monkeypatch, target, tmp_path):
    """Writing TOSA's table order into DDR verbatim is not a crash -- the
    LUT is still 256 valid bytes -- it is a wrong activation for every
    input. The end-to-end test must catch it."""
    from cnnc.lower import to_hir as to_hir_mod

    monkeypatch.setattr(
        to_hir_mod, "pack_act_lut", lambda values: bytes(int(v) & 0xFF for v in values)
    )
    result = _compile("table_silu", target, tmp_path)
    graph = result.imported_graph
    x = _seed_input(FIXTURES["table_silu"], 1)
    expected = interp.run(graph, {graph.inputs[0]: x})
    actual = run_program(result.program, {graph.inputs[0]: x})
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(actual[graph.outputs[0]], expected[graph.outputs[0]])


def test_mutation_pointing_act_at_no_lut_breaks_the_equality_test(monkeypatch, target, tmp_path):
    """If W3 stops carrying the LUT address, ACT reads 256 bytes of
    whatever is at address 0 as its table."""
    from cnnc.backend.cnn_accel_v1 import emit

    real = emit._build_act_descriptor

    def broken(*args, **kwargs):
        return dataclasses.replace(real(*args, **kwargs), weight_addr=0)

    monkeypatch.setattr(emit, "_build_act_descriptor", broken)
    result = _compile("table_silu", target, tmp_path)
    graph = result.imported_graph
    x = _seed_input(FIXTURES["table_silu"], 1)
    expected = interp.run(graph, {graph.inputs[0]: x})
    actual = run_program(result.program, {graph.inputs[0]: x})
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(actual[graph.outputs[0]], expected[graph.outputs[0]])
