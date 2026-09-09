"""`tosa.add` -> `ADD` (ISA v2.0), end to end.

The residual shortcut is the first *two-input* op this compiler lowers, and
the first that uses ISA v2.0 at all, so these tests cover the two things
that are genuinely new:

* **The second source address rides in W15 `xfer_bytes`.** The descriptor
  has no third address field, so `accel_v2/program.py` ratified W15 as
  `src1_addr` for ADD and `cnn_accel_model.run_layer` executes it that
  way. `test_add_descriptor_points_xfer_bytes_at_the_second_operand` pins
  the emitter to that, and the mutation test below shows what breaks when
  it drifts.
* **One rescale pair for two operands.** `OPCODE_ADD` carries a single
  `(requant_scale, requant_shift)` and applies it to each operand
  *before* the sum. `passes.fuse._fold_add_rescales` is what puts a real
  quantized residual add into that shape, and it is only allowed to when
  the fold is provably value-preserving --
  `test_folding_the_operand_rescales_does_not_change_any_value` is the
  check that it is.

As with `test_pool.py`, every fixture is run all the way down: the TOSA
reference (`gir.interp`) and the emitted program executed on the RTL
golden model (`backend.cnn_accel_v1.run_program`) must agree exactly.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from cnnc.backend.cnn_accel_v1 import decode_program, run_program
from cnnc.driver import compile_tosa
from cnnc.errors import CapabilityError, UnsupportedAttribute, VerifyError
from cnnc.frontend.mlir_generic import parse_module
from cnnc.frontend.tosa_import import import_tosa
from cnnc.gir import interp
from cnnc.gir.ir import AddAttrs
from cnnc.passes import PassContext, default_pipeline, run_pipeline
from cnnc.target.contract import Target

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# fixture name -> {graph input name: shape}.
FIXTURES = {
    # No rescale at all: the identity `(2**15, 15)` pair, i.e. a plain
    # saturating int8 add.
    "add_plain": {"arg0": (1, 8, 8, 8), "arg1": (1, 8, 8, 8)},
    # The quantized idiom: both operands rescaled by 0.5 first. C=12 is
    # deliberately not a multiple of the 8-channel DDR plane, so the
    # padding lanes are exercised too.
    "add_rescaled": {"arg0": (1, 4, 6, 12), "arg1": (1, 4, 6, 12)},
    # A real shortcut shape: conv+rescale+clamp on one side, a graph input
    # on the other, joined by a rescaled add.
    "conv_add": {"arg0": (1, 8, 8, 4), "arg1": (1, 8, 8, 8)},
}


def _compile(name: str, target, tmp_path=None):
    out = None if tmp_path is None else tmp_path / "out"
    return compile_tosa(FIXTURES_DIR / f"{name}.mlir", target, out_dir=out)


def _seed_inputs(name: str, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    return {
        tid: rng.integers(-128, 128, size=shape).astype(np.int8)
        for tid, shape in FIXTURES[name].items()
    }


def _add_op(result):
    return next(op for op in result.hir_planned.ops if op.kind == "add")


def _fuse(name: str, target):
    graph = import_tosa(parse_module((FIXTURES_DIR / f"{name}.mlir").read_text()))
    return graph, run_pipeline(graph, default_pipeline(target), PassContext(target=target), first_index=2)


def _without_elementwise_unit(target: Target) -> Target:
    """`target` with no unit advertising `add` -- an ISA v1.x accelerator,
    which has no such opcode at all."""
    data = target.to_dict()
    data["units"] = [u for u in data["units"] if "add" not in u["ops"]]
    return Target.from_dict(data)


def _pre_v2_target(target: Target) -> Target:
    """`target` with the elementwise unit demoted to ISA v1.2: the opcode
    is advertised, but the revision that defined it is not."""
    data = target.to_dict()
    for unit in data["units"]:
        unit["isa_version"] = "1.2"
    return Target.from_dict(data)


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------


def test_import_builds_an_add_with_the_identity_rescale():
    graph = import_tosa(parse_module((FIXTURES_DIR / "add_plain.mlir").read_text()))
    op = next(op for op in graph.ops if op.kind == "add")
    # apply_scale_32(v, 2**15, 15) == v for every v.
    assert op.attrs == AddAttrs(multiplier=1 << 15, shift=15)
    assert op.inputs == ("arg0", "arg1")
    assert all(interp.apply_scale_32(v, 1 << 15, 15, "SINGLE_ROUND") == v for v in range(-128, 128))


def test_import_refuses_an_i32_add_and_says_what_to_do_instead():
    text = (FIXTURES_DIR / "add_plain.mlir").read_text().replace("xi8>", "xi32>")
    with pytest.raises(UnsupportedAttribute) as exc:
        import_tosa(parse_module(text))
    assert "i32" in str(exc.value)
    assert "quantize the add itself to int8" in str(exc.value)


def test_import_refuses_a_broadcasting_add():
    text = (
        '"builtin.module"() ({\n'
        '  "func.func"() <{function_type = (tensor<1x8x8x8xi8>) -> tensor<1x8x8x8xi8>, sym_name = "main"}> ({\n'
        "  ^bb0(%arg0: tensor<1x8x8x8xi8>):\n"
        '    %b = "tosa.const"() <{values = dense<3> : tensor<1x1x1x8xi8>}> : () -> tensor<1x1x1x8xi8>\n'
        '    %0 = "tosa.add"(%arg0, %b) : (tensor<1x8x8x8xi8>, tensor<1x1x1x8xi8>) -> tensor<1x8x8x8xi8>\n'
        '    "func.return"(%0) : (tensor<1x8x8x8xi8>) -> ()\n'
        "  }) : () -> ()\n"
        "}) : () -> ()\n"
    )
    with pytest.raises(UnsupportedAttribute) as exc:
        import_tosa(parse_module(text))
    assert "broadcasting" in str(exc.value)


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------


def test_verify_rejects_an_add_whose_operand_rescale_can_leave_int8():
    """Scale > 1 is what makes the two-rescales-then-add fold inexact (a
    TOSA `rescale i8->i8` would clamp each operand first, the hardware
    would not), so the GIR verifier refuses it however the op was built."""
    from cnnc.gir.verify import verify

    graph = import_tosa(parse_module((FIXTURES_DIR / "add_plain.mlir").read_text()))
    op = next(op for op in graph.ops if op.kind == "add")
    broken = graph.replace(
        ops=tuple(
            dataclasses.replace(o, attrs=AddAttrs(multiplier=(1 << 16) + 1, shift=15)) if o is op else o
            for o in graph.ops
        )
    )
    with pytest.raises(VerifyError) as exc:
        verify(broken)
    assert "scale > 1" in str(exc.value)


# ---------------------------------------------------------------------------
# Fuse: folding the two per-operand rescales into the add
# ---------------------------------------------------------------------------


def test_fuse_folds_two_identical_operand_rescales_into_the_add(target):
    _graph, fused = _fuse("add_rescaled", target)
    ops = [op for op in fused.ops if op.kind != "const"]
    assert [op.kind for op in ops] == ["add"]  # both rescales absorbed
    assert ops[0].attrs == AddAttrs(multiplier=1 << 30, shift=31)
    assert ops[0].inputs == ("arg0", "arg1")


def test_folding_the_operand_rescales_does_not_change_any_value(target):
    """The fold is a re-grouping, not a re-quantization: `gir.interp` must
    give identical results before and after. This is the whole
    justification for `_add_rescale_foldable`'s `multiplier <= 2**shift`
    bound, so it is checked rather than asserted in a comment."""
    graph, fused = _fuse("add_rescaled", target)
    for seed in (0, 1, 2, 3):
        inputs = _seed_inputs("add_rescaled", seed)
        before = interp.run(graph, inputs)
        after = interp.run(fused, inputs)
        np.testing.assert_array_equal(
            after[fused.outputs[0]], before[graph.outputs[0]], err_msg=f"seed {seed}"
        )


def test_fuse_leaves_mismatched_operand_rescales_alone(target):
    """One `(requant_scale, requant_shift)` pair cannot express two
    different rescales, so the fold declines rather than picking one."""
    text = (FIXTURES_DIR / "add_rescaled.mlir").read_text().replace(
        '%5 = "tosa.rescale"(%arg1, %0, %1,',
        '%5 = "tosa.rescale"(%arg1, %0, %20,',
    ).replace(
        '    %4 = "tosa.rescale"',
        '    %20 = "tosa.const"() <{values = dense<30> : tensor<1xi8>}> : () -> tensor<1xi8>\n'
        '    %4 = "tosa.rescale"',
    )
    graph = import_tosa(parse_module(text))
    fused = run_pipeline(graph, default_pipeline(target), PassContext(target=target), first_index=2)
    assert sorted(op.kind for op in fused.ops if op.kind != "const") == ["add", "rescale", "rescale"]


def test_fuse_leaves_a_scale_above_one_alone(target):
    """`multiplier > 2**shift` is exactly the case the per-operand int8
    clamp becomes observable in, so it must NOT be folded."""
    text = (FIXTURES_DIR / "add_rescaled.mlir").read_text().replace(
        "dense<31> : tensor<1xi8>", "dense<29> : tensor<1xi8>"
    )  # 2**30 / 2**29 == 2.0
    graph = import_tosa(parse_module(text))
    fused = run_pipeline(graph, default_pipeline(target), PassContext(target=target), first_index=2)
    assert sorted(op.kind for op in fused.ops if op.kind != "const") == ["add", "rescale", "rescale"]


def test_fuse_does_not_steal_a_convolutions_own_epilogue_rescale(target):
    """Regression: a conv's i32 -> i8 epilogue rescale matches every
    *structural* condition of the operand fold. Folding it would leave a
    bare `conv2d` behind (which `to_hir` rejects), so the fold is gated on
    an i8 input dtype -- i.e. on it really being an operand rescale."""
    text = (FIXTURES_DIR / "conv_add.mlir").read_text()
    # Feed the conv's epilogue rescale straight into the add, with no
    # clamp and no second rescale in between.
    text = text.replace(
        '%12 = "tosa.rescale"(%10, %5, %11, %8, %8)',
        '%12 = "tosa.rescale"(%9, %5, %11, %8, %8)',
    )
    graph = import_tosa(parse_module(text))
    fused = run_pipeline(graph, default_pipeline(target), PassContext(target=target), first_index=2)
    kinds = sorted(op.kind for op in fused.ops if op.kind != "const")
    assert "fused_conv" in kinds  # the conv kept its epilogue
    assert kinds.count("rescale") == 0  # the two i8->i8 operand rescales still folded


# ---------------------------------------------------------------------------
# Lower + emit
# ---------------------------------------------------------------------------


def test_add_lowers_onto_its_own_unit(target, tmp_path):
    result = _compile("add_rescaled", target, tmp_path)
    op = _add_op(result)
    assert op.unit == "elementwise_engine"
    assert dict(op.params) == {
        "in_width": 6, "in_height": 4, "in_channels": 12,
        "requant_en": True,
        "requant_scale": 1 << 30,
        # The descriptor shift is the TOSA shift minus the Q15 implicit
        # shift the hardware applies anyway.
        "requant_shift": 31 - target.unit("elementwise_engine").epilogue.rescale.implicit_shift,
    }
    assert len(op.reads) == 2  # two ifmaps, no constants at all
    assert len(op.writes) == 1


def test_add_descriptor_points_xfer_bytes_at_the_second_operand(target, tmp_path):
    result = _compile("add_rescaled", target, tmp_path)
    planned = result.hir_planned
    program_addr = planned.buffer(planned.program).addr
    desc = decode_program(result.program.program_bytes, target, program_addr=program_addr)[0]

    op = _add_op(result)
    src0, src1 = (planned.buffer(bid) for bid in op.reads)
    assert desc.opcode == target.isa.opcodes["ADD"]
    assert desc.in_addr == src0.addr
    assert desc.xfer_bytes == src1.addr  # W15 is src1's ADDRESS, not a byte count
    assert desc.out_addr == planned.buffer(op.writes[0]).addr
    assert (desc.in_width, desc.in_height, desc.in_channels) == (6, 4, 12)
    flags = {name for name, bit in target.isa.flags.items() if (desc.flags >> bit) & 1}
    # REQUANT_EN only: ADD has no bias, no padding, no clamp bounds and no
    # per-channel table.
    assert flags == {"REQUANT_EN"}
    # And no weight/bias/scale table.
    assert (desc.weight_addr, desc.bias_addr, desc.scale_addr) == (0, 0, 0)


def test_add_needs_a_unit_that_implements_it(target):
    _graph, fused = _fuse("add_plain", target)
    from cnnc.lower.to_hir import to_hir

    with pytest.raises(CapabilityError) as exc:
        to_hir(fused, _without_elementwise_unit(target))
    assert "add" in str(exc.value)


def test_add_needs_isa_v20(target):
    _graph, fused = _fuse("add_plain", target)
    from cnnc.lower.to_hir import to_hir

    with pytest.raises(CapabilityError) as exc:
        to_hir(fused, _pre_v2_target(target))
    assert "2.0" in str(exc.value)


# ---------------------------------------------------------------------------
# End to end: TOSA reference == emitted program on the golden model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(FIXTURES))
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_add_program_matches_tosa_reference(name, seed, target, tmp_path):
    result = _compile(name, target, tmp_path)
    graph = result.imported_graph
    inputs = _seed_inputs(name, seed)
    expected = interp.run(graph, inputs)
    actual = run_program(result.program, inputs)
    assert set(expected) == set(actual)
    for tid, want in expected.items():
        np.testing.assert_array_equal(actual[tid], want, err_msg=f"{name} seed {seed} tensor %{tid}")


def test_add_saturates_rather_than_wrapping(target, tmp_path):
    """Two large same-sign operands: the exact sum leaves int8 and the
    hardware saturates. Pinned separately from the random seeds because
    the random inputs rarely reach the boundary, and a wrapping
    implementation would look right on them."""
    result = _compile("add_plain", target, tmp_path)
    graph = result.imported_graph
    inputs = {
        "arg0": np.full((1, 8, 8, 8), 100, dtype=np.int8),
        "arg1": np.full((1, 8, 8, 8), 100, dtype=np.int8),
    }
    actual = run_program(result.program, inputs)[graph.outputs[0]]
    np.testing.assert_array_equal(actual, np.full((1, 8, 8, 8), 127, dtype=np.int8))


# ---------------------------------------------------------------------------
# Mutation tests: break the lowering, prove the equality test notices
# ---------------------------------------------------------------------------


def _run_equality(name: str, target, tmp_path, seed: int = 1):
    result = _compile(name, target, tmp_path)
    inputs = _seed_inputs(name, seed)
    expected = interp.run(result.imported_graph, inputs)
    actual = run_program(result.program, inputs)
    for tid, want in expected.items():
        np.testing.assert_array_equal(actual[tid], want)


def test_mutation_dropping_xfer_bytes_breaks_the_equality_test(monkeypatch, target, tmp_path):
    """If W15 stops carrying src1's address, ADD reads whatever is at
    address 0 as its second operand. The end-to-end test must fail."""
    from cnnc.backend.cnn_accel_v1 import emit

    real = emit._build_add_descriptor

    def broken(*args, **kwargs):
        return dataclasses.replace(real(*args, **kwargs), xfer_bytes=0)

    monkeypatch.setattr(emit, "_build_add_descriptor", broken)
    with pytest.raises(AssertionError):
        _run_equality("add_rescaled", target, tmp_path)


def test_mutation_forgetting_the_implicit_shift_breaks_the_equality_test(monkeypatch, target, tmp_path):
    """`requant_shift` is the TOSA shift MINUS the hardware's Q15 implicit
    shift. Emitting the TOSA shift verbatim scales the operands by
    2**-15 too much."""
    from cnnc.backend.cnn_accel_v1 import emit

    real = emit._build_add_descriptor

    def broken(op, *args, **kwargs):
        desc = real(op, *args, **kwargs)
        return dataclasses.replace(desc, requant_shift=desc.requant_shift + 15)

    monkeypatch.setattr(emit, "_build_add_descriptor", broken)
    with pytest.raises(AssertionError):
        _run_equality("add_rescaled", target, tmp_path)


def test_mutation_folding_a_scale_above_one_would_change_the_result(target):
    """Why `_add_rescale_foldable` refuses `multiplier > 2**shift`, shown
    rather than asserted: fold the two rescales of a scale-2 graph by hand
    and the folded graph computes something different from the original.

    TOSA clamps each rescaled operand to int8 before adding; the hardware
    (and therefore the folded `add`) does not, and saturates once at the
    end instead. At scale 2 an operand above 63 leaves int8, and the two
    stop agreeing. No monkeypatching: the point is a numeric fact about
    the fold, not about any one function's plumbing."""
    from cnnc.gir.ir import Graph, Op

    text = (FIXTURES_DIR / "add_rescaled.mlir").read_text().replace(
        "dense<31> : tensor<1xi8>", "dense<29> : tensor<1xi8>"
    )
    graph = import_tosa(parse_module(text))
    add = next(op for op in graph.ops if op.kind == "add")
    rescales = [graph.producer(tid) for tid in add.inputs]

    hand_folded = graph.replace(
        ops=tuple(
            Op(id=add.id, kind="add", inputs=tuple(r.inputs[0] for r in rescales),
               outputs=add.outputs, attrs=AddAttrs(multiplier=1 << 30, shift=29))
            if op is add else op
            for op in graph.ops
            if op not in rescales
        ),
        tensors={tid: t for tid, t in graph.tensors.items() if tid not in {r.outputs[0] for r in rescales}},
    )
    assert isinstance(hand_folded, Graph)

    # Opposite signs, so the per-operand clamp is not hidden by the final
    # saturation (at +100/+100 both spellings saturate to 127 and agree).
    inputs = {
        "arg0": np.full((1, 4, 6, 12), 100, dtype=np.int8),
        "arg1": np.full((1, 4, 6, 12), -100, dtype=np.int8),
    }
    before = interp.run(graph, inputs)[graph.outputs[0]]
    after = interp.run(hand_folded, inputs)[hand_folded.outputs[0]]
    # TOSA: clamp(200) + clamp(-200) = 127 + (-128) = -1.
    # Folded: sat_i8(200 + (-200)) = 0.
    np.testing.assert_array_equal(before, np.full((1, 4, 6, 12), -1, dtype=np.int8))
    np.testing.assert_array_equal(after, np.full((1, 4, 6, 12), 0, dtype=np.int8))
