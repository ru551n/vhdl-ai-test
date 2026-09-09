"""The nearest-2x upsample idiom -> `UPSAMPLE` (ISA v2.0), end to end.

TOSA has no upsample op. What a real exporter emits -- and what all four
of YOLOv8n's `nn.Upsample(scale_factor=2)` layers are in the shipped TOSA
artifact, with zero `tosa.resize` anywhere -- is a five-op chain:

    reshape -> tile -> reshape -> tile -> reshape

over a rank-5 intermediate. `frontend.tosa_import._match_upsample_chains`
recognises it by *evaluating* the chain's index mapping, not by matching
its shapes, and these tests are mostly about that distinction:

* `test_matches_the_chain_however_the_two_axes_are_ordered` -- the same
  upsample with W replicated before H is a different shape sequence and
  the same computation, and must match.
* `test_refuses_a_tile_that_is_not_a_nearest_upsample` and
  `test_refuses_a_3x_chain` -- chains that *look* like the idiom but
  compute something else, or something the hardware cannot do, are
  refused by name rather than lowered to a 2x replicate.

Everything then runs all the way down, `gir.interp` against the emitted
program on the RTL golden model, as in `test_pool.py`/`test_add.py`.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from cnnc.backend.cnn_accel_v1 import decode_program, run_program
from cnnc.driver import compile_tosa
from cnnc.errors import CapabilityError, UnsupportedOp
from cnnc.frontend.mlir_generic import parse_module
from cnnc.frontend.tosa_import import import_tosa
from cnnc.gir import interp
from cnnc.gir.ir import UpsampleAttrs
from cnnc.target.contract import Target

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# fixture name -> input shape.
FIXTURES = {
    # Non-square, odd spatial dims: a swapped-axis match cannot pass by
    # symmetry.
    "upsample2x": (1, 3, 5, 8),
    # Two chained 2x upsamples: a two-instruction program, and the shape
    # YOLOv8n's neck reaches by upsampling twice.
    "upsample4x": (1, 2, 3, 12),
}


def _compile(name: str, target, tmp_path=None):
    out = None if tmp_path is None else tmp_path / "out"
    return compile_tosa(FIXTURES_DIR / f"{name}.mlir", target, out_dir=out)


def _seed_input(shape: tuple[int, ...], seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(-128, 128, size=shape).astype(np.int8)


def _text(name: str) -> str:
    return (FIXTURES_DIR / f"{name}.mlir").read_text()


def _upsample_ops(result):
    return [op for op in result.hir_planned.ops if op.kind == "upsample"]


def _nearest2x(x: np.ndarray) -> np.ndarray:
    return np.repeat(np.repeat(x, 2, axis=1), 2, axis=2)


def _module(body: str, in_dims: str, out_dims: str) -> str:
    return (
        '"builtin.module"() ({\n'
        f'  "func.func"() <{{function_type = (tensor<{in_dims}xi8>) -> tensor<{out_dims}xi8>, '
        'sym_name = "main"}> ({\n'
        f"  ^bb0(%arg0: tensor<{in_dims}xi8>):\n"
        f"{body}"
        f'    "func.return"(%out) : (tensor<{out_dims}xi8>) -> ()\n'
        "  }) : () -> ()\n"
        "}) : () -> ()\n"
    )


def _shape_const(name: str, values: list[int]) -> str:
    dims = ", ".join(str(v) for v in values)
    return (
        f'    {name} = "tosa.const_shape"() <{{values = dense<[{dims}]> : '
        f"tensor<{len(values)}xindex>}}> : () -> !tosa.shape<{len(values)}>\n"
    )


# ---------------------------------------------------------------------------
# Recognising the chain
# ---------------------------------------------------------------------------


def test_import_collapses_the_five_op_chain_to_one_upsample():
    graph = import_tosa(parse_module(_text("upsample2x")))
    ops = [op for op in graph.ops if op.kind != "const"]
    assert [op.kind for op in ops] == ["upsample"]
    assert ops[0].attrs == UpsampleAttrs(factor=2)
    assert ops[0].inputs == ("arg0",)
    assert graph.tensor(ops[0].outputs[0]).shape == (1, 6, 10, 8)


def test_import_collapses_two_chained_upsamples_separately():
    """A 4x upsample is two 2x chains back to back; the second one's
    source is the first one's LAST op, which must still count as a chain
    root."""
    graph = import_tosa(parse_module(_text("upsample4x")))
    ops = [op for op in graph.ops if op.kind != "const"]
    assert [op.kind for op in ops] == ["upsample", "upsample"]
    assert ops[1].inputs == (ops[0].outputs[0],)


def test_matches_the_chain_however_the_two_axes_are_ordered():
    """W replicated before H: a different shape sequence, the same
    computation. A shape-matching recogniser would miss this; an
    index-evaluating one cannot."""
    body = (
        _shape_const("%s0", [1, 2, 3, 1, 8])
        + _shape_const("%s1", [1, 1, 1, 2, 1])
        + _shape_const("%s2", [1, 2, 1, 6, 8])
        + _shape_const("%s3", [1, 1, 2, 1, 1])
        + _shape_const("%s4", [1, 4, 6, 8])
        + '    %0 = "tosa.reshape"(%arg0, %s0) : (tensor<1x2x3x8xi8>, !tosa.shape<5>) -> tensor<1x2x3x1x8xi8>\n'
        + '    %1 = "tosa.tile"(%0, %s1) : (tensor<1x2x3x1x8xi8>, !tosa.shape<5>) -> tensor<1x2x3x2x8xi8>\n'
        + '    %2 = "tosa.reshape"(%1, %s2) : (tensor<1x2x3x2x8xi8>, !tosa.shape<5>) -> tensor<1x2x1x6x8xi8>\n'
        + '    %3 = "tosa.tile"(%2, %s3) : (tensor<1x2x1x6x8xi8>, !tosa.shape<5>) -> tensor<1x2x2x6x8xi8>\n'
        + '    %out = "tosa.reshape"(%3, %s4) : (tensor<1x2x2x6x8xi8>, !tosa.shape<4>) -> tensor<1x4x6x8xi8>\n'
    )
    graph = import_tosa(parse_module(_module(body, "1x2x3x8", "1x4x6x8")))
    ops = [op for op in graph.ops if op.kind != "const"]
    assert [op.kind for op in ops] == ["upsample"]

    # And it really is the same function.
    x = np.arange(1 * 2 * 3 * 8, dtype=np.int8).reshape(1, 2, 3, 8)
    got = interp.run(graph, {"arg0": x})[graph.outputs[0]]
    np.testing.assert_array_equal(got, _nearest2x(x))


def test_refuses_a_tile_that_is_not_a_nearest_upsample():
    """A single tile that repeats the whole tensor (`[1,2,2,1]`) has the
    right output SHAPE for a 2x upsample and the wrong contents: it
    interleaves whole rows instead of replicating each one. The index
    evaluation is what tells the two apart."""
    body = (
        _shape_const("%s0", [1, 2, 2, 1])
        + '    %out = "tosa.tile"(%arg0, %s0) : (tensor<1x2x3x8xi8>, !tosa.shape<4>) -> tensor<1x4x6x8xi8>\n'
    )
    with pytest.raises(UnsupportedOp) as exc:
        import_tosa(parse_module(_module(body, "1x2x3x8", "1x4x6x8")))
    assert "nearest-2x replication" in str(exc.value)


def test_refuses_a_3x_chain():
    """3x nearest upsample is a perfectly good chain that this hardware
    cannot do: `OPCODE_UPSAMPLE` has no factor field."""
    body = (
        _shape_const("%s0", [1, 2, 1, 3, 8])
        + _shape_const("%s1", [1, 1, 3, 1, 1])
        + _shape_const("%s2", [1, 6, 3, 1, 8])
        + _shape_const("%s3", [1, 1, 1, 3, 1])
        + _shape_const("%s4", [1, 6, 9, 8])
        + '    %0 = "tosa.reshape"(%arg0, %s0) : (tensor<1x2x3x8xi8>, !tosa.shape<5>) -> tensor<1x2x1x3x8xi8>\n'
        + '    %1 = "tosa.tile"(%0, %s1) : (tensor<1x2x1x3x8xi8>, !tosa.shape<5>) -> tensor<1x2x3x3x8xi8>\n'
        + '    %2 = "tosa.reshape"(%1, %s2) : (tensor<1x2x3x3x8xi8>, !tosa.shape<5>) -> tensor<1x6x3x1x8xi8>\n'
        + '    %3 = "tosa.tile"(%2, %s3) : (tensor<1x6x3x1x8xi8>, !tosa.shape<5>) -> tensor<1x6x3x3x8xi8>\n'
        + '    %out = "tosa.reshape"(%3, %s4) : (tensor<1x6x3x3x8xi8>, !tosa.shape<4>) -> tensor<1x6x9x8xi8>\n'
    )
    with pytest.raises(UnsupportedOp) as exc:
        import_tosa(parse_module(_module(body, "1x2x3x8", "1x6x9x8")))
    assert "2x" in str(exc.value)


def test_refuses_an_nchw_chain():
    """YOLOv8n's shipped artifact is NCHW, and its chain replicates axes 2
    and 3 -- which in the NHWC layout GIR uses are W and C, not H and W.
    That is not this compiler's upsample, and it must not be mistaken for
    one just because the op sequence looks familiar."""
    body = (
        _shape_const("%s0", [1, 8, 2, 1, 3])
        + _shape_const("%s1", [1, 1, 1, 2, 1])
        + _shape_const("%s2", [1, 8, 4, 3, 1])
        + _shape_const("%s3", [1, 1, 1, 1, 2])
        + _shape_const("%s4", [1, 8, 4, 6])
        + '    %0 = "tosa.reshape"(%arg0, %s0) : (tensor<1x8x2x3xi8>, !tosa.shape<5>) -> tensor<1x8x2x1x3xi8>\n'
        + '    %1 = "tosa.tile"(%0, %s1) : (tensor<1x8x2x1x3xi8>, !tosa.shape<5>) -> tensor<1x8x2x2x3xi8>\n'
        + '    %2 = "tosa.reshape"(%1, %s2) : (tensor<1x8x2x2x3xi8>, !tosa.shape<5>) -> tensor<1x8x4x3x1xi8>\n'
        + '    %3 = "tosa.tile"(%2, %s3) : (tensor<1x8x4x3x1xi8>, !tosa.shape<5>) -> tensor<1x8x4x3x2xi8>\n'
        + '    %out = "tosa.reshape"(%3, %s4) : (tensor<1x8x4x3x2xi8>, !tosa.shape<4>) -> tensor<1x8x4x6xi8>\n'
    )
    with pytest.raises(UnsupportedOp):
        import_tosa(parse_module(_module(body, "1x8x2x3", "1x8x4x6")))


# ---------------------------------------------------------------------------
# Lower + emit
# ---------------------------------------------------------------------------


def test_upsample_lowers_onto_the_elementwise_unit(target, tmp_path):
    result = _compile("upsample2x", target, tmp_path)
    (op,) = _upsample_ops(result)
    assert op.unit == "elementwise_engine"
    # The INPUT geometry: the output is implied by the opcode's fixed factor.
    assert dict(op.params) == {"in_width": 5, "in_height": 3, "in_channels": 8}
    assert len(op.reads) == 1


def test_upsample_descriptor_carries_only_the_input_geometry(target, tmp_path):
    result = _compile("upsample2x", target, tmp_path)
    planned = result.hir_planned
    program_addr = planned.buffer(planned.program).addr
    desc = decode_program(result.program.program_bytes, target, program_addr=program_addr)[0]
    (op,) = _upsample_ops(result)

    assert desc.opcode == target.isa.opcodes["UPSAMPLE"]
    assert desc.in_addr == planned.buffer(op.reads[0]).addr
    assert desc.out_addr == planned.buffer(op.writes[0]).addr
    assert (desc.in_width, desc.in_height, desc.in_channels) == (5, 3, 8)
    assert desc.flags == 0  # no epilogue of any kind
    assert (desc.weight_addr, desc.bias_addr, desc.scale_addr, desc.xfer_bytes) == (0, 0, 0, 0)
    assert (desc.requant_scale, desc.requant_shift) == (0, 0)


def test_the_output_buffer_is_sized_for_the_upsampled_shape(target, tmp_path):
    result = _compile("upsample2x", target, tmp_path)
    (op,) = _upsample_ops(result)
    in_buf = result.hir_planned.buffer(op.reads[0])
    out_buf = result.hir_planned.buffer(op.writes[0])
    assert out_buf.shape == (1, 6, 10, 8)
    assert out_buf.size_bytes == in_buf.size_bytes * 4


def test_upsample_factor_is_read_from_the_golden_model(target):
    """`UPSAMPLE` has no factor field, so the compiler must take the
    hardware's factor from the hardware, not from a literal of its own."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "modules" / "cnn_accel"))
    import cnn_accel_model  # noqa: E402

    assert target.unit("elementwise_engine").upsample_factor == cnn_accel_model.UPSAMPLE_FACTOR


def test_a_factor_the_hardware_does_not_have_is_refused(target):
    """Reached through a hand-built graph rather than through the frontend
    (which only ever matches 2x), because the lowering must refuse it on
    its own -- it is the only thing standing between a 3x GIR op and a
    descriptor that would quietly do 2x."""
    from cnnc.lower.to_hir import to_hir

    graph = import_tosa(parse_module(_text("upsample2x")))
    op = next(o for o in graph.ops if o.kind == "upsample")
    y = graph.tensor(op.outputs[0])
    broken = graph.replace(
        ops=tuple(
            dataclasses.replace(o, attrs=UpsampleAttrs(factor=3)) if o is op else o for o in graph.ops
        ),
        # Keep the declared shape consistent with factor 3 so the GIR
        # verifier is not what rejects it -- the point is the lowering.
        tensors={
            tid: (dataclasses.replace(t, shape=(1, 9, 15, 8)) if tid == y.id else t)
            for tid, t in graph.tensors.items()
        },
    )
    with pytest.raises(CapabilityError) as exc:
        to_hir(broken, target)
    assert "factor 3" in str(exc.value)


def test_upsample_needs_a_unit_that_implements_it(target):
    from cnnc.lower.to_hir import to_hir

    graph = import_tosa(parse_module(_text("upsample2x")))
    data = target.to_dict()
    data["units"] = [u for u in data["units"] if "upsample" not in u["ops"]]
    for unit in data["units"]:
        unit["upsample_factor"] = None
    with pytest.raises(CapabilityError) as exc:
        to_hir(graph, Target.from_dict(data))
    assert "upsample" in str(exc.value)


# ---------------------------------------------------------------------------
# End to end: TOSA reference == emitted program on the golden model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(FIXTURES))
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_upsample_program_matches_tosa_reference(name, seed, target, tmp_path):
    result = _compile(name, target, tmp_path)
    graph = result.imported_graph
    x = _seed_input(FIXTURES[name], seed)
    expected = interp.run(graph, {graph.inputs[0]: x})
    actual = run_program(result.program, {graph.inputs[0]: x})
    assert set(expected) == set(actual)
    for tid, want in expected.items():
        np.testing.assert_array_equal(actual[tid], want, err_msg=f"{name} seed {seed} tensor %{tid}")


def test_upsample_result_is_literally_the_replicated_input(target, tmp_path):
    """Independently of the interpreter: the emitted program's output must
    equal `numpy.repeat` twice over. A distinct value per input pixel, so
    any transposed or off-by-one replication shows up."""
    result = _compile("upsample2x", target, tmp_path)
    graph = result.imported_graph
    x = (np.arange(1 * 3 * 5 * 8) % 251 - 125).astype(np.int8).reshape(1, 3, 5, 8)
    actual = run_program(result.program, {graph.inputs[0]: x})[graph.outputs[0]]
    np.testing.assert_array_equal(actual, _nearest2x(x))


# ---------------------------------------------------------------------------
# Mutation: break the lowering, prove the equality test notices
# ---------------------------------------------------------------------------


def test_mutation_swapping_the_descriptor_geometry_breaks_the_equality_test(
    monkeypatch, target, tmp_path
):
    """`in_width`/`in_height` are not interchangeable: the fixture is
    3x5, so transposing them makes UPSAMPLE walk the ifmap in the wrong
    order. (This is why the fixture is deliberately non-square.)"""
    from cnnc.backend.cnn_accel_v1 import emit

    real = emit._build_upsample_descriptor

    def broken(*args, **kwargs):
        desc = real(*args, **kwargs)
        return dataclasses.replace(desc, in_width=desc.in_height, in_height=desc.in_width)

    monkeypatch.setattr(emit, "_build_upsample_descriptor", broken)
    result = _compile("upsample2x", target, tmp_path)
    graph = result.imported_graph
    x = _seed_input(FIXTURES["upsample2x"], 1)
    expected = interp.run(graph, {graph.inputs[0]: x})
    actual = run_program(result.program, {graph.inputs[0]: x})
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(actual[graph.outputs[0]], expected[graph.outputs[0]])
