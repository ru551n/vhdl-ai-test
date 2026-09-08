"""`tosa.max_pool2d` -> `POOL_MAX`, end to end.

Pooling is the first non-convolution op this compiler lowers, so these
tests deliberately go all the way down rather than stopping at "the HIR
looks plausible": for every fixture the TOSA reference (`gir.interp`) and
the emitted program executed on the RTL golden model
(`backend.cnn_accel_v1.run_program`) must agree byte for byte.

The pinned case is YOLOv8n's SPPF block -- 5x5 kernel, stride 1, padding 2
-- because it is the one that only works at all with ISA v2.1's
`pad_value`: TOSA pads MAX_POOL2D with the int8 minimum, and a hardware
that could only pad with a literal 0 would let the padding win the max on
every border output of a mostly-negative activation tensor. If `pad_value`
ever stops reaching the descriptor, `test_sppf_padded_pool_matches_model`
fails rather than silently drifting at the borders.

Average pooling is NOT lowered; `test_avg_pool2d_is_refused_with_reasons`
pins the refusal and the two reasons for it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from conftest import v10_target

from cnnc.backend.cnn_accel_v1 import decode_program, run_program
from cnnc.driver import compile_tosa
from cnnc.errors import CapabilityError, UnsupportedOp, VerifyError
from cnnc.frontend.mlir_generic import parse_module
from cnnc.frontend.tosa_import import import_tosa
from cnnc.gir import interp
from cnnc.gir.ir import PoolAttrs, pool2d_output_shape

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# fixture name -> input shape.
FIXTURES = {
    "maxpool_sppf": (1, 8, 8, 8),       # 5x5 / stride 1 / pad 2, the SPPF shape
    "maxpool_stride2": (1, 8, 8, 8),    # 2x2 / stride 2, no padding
    "conv_maxpool": (1, 8, 8, 4),       # conv+rescale+clamp followed by a pool
}


def _compile(name: str, target, tmp_path):
    return compile_tosa(FIXTURES_DIR / f"{name}.mlir", target, out_dir=tmp_path / "out")


def _seed_input(shape: tuple[int, ...], seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(-128, 128, size=shape).astype(np.int8)


def _pool_op(result):
    return next(op for op in result.hir_planned.ops if op.kind == "max_pool")


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------


def test_import_builds_a_pool_op_with_type_minimum_padding():
    graph = import_tosa(parse_module((FIXTURES_DIR / "maxpool_sppf.mlir").read_text()))
    op = next(op for op in graph.ops if op.kind == "pool")
    assert op.attrs == PoolAttrs(mode="max", kernel=(5, 5), stride=(1, 1), pad=(2, 2, 2, 2), pad_value=-128)
    # 8 + 2 + 2 - 5 + 1 == 8: SPPF keeps its spatial size.
    assert graph.tensor(op.outputs[0]).shape == (1, 8, 8, 8)


def test_pool_output_shape_formula():
    attrs = PoolAttrs(mode="max", kernel=(3, 3), stride=(2, 2), pad=(1, 0, 1, 0), pad_value=-128)
    assert pool2d_output_shape((1, 8, 8, 16), attrs) == (1, 4, 4, 16)


def test_import_rejects_declared_shape_that_disagrees_with_the_formula():
    text = (FIXTURES_DIR / "maxpool_stride2.mlir").read_text().replace("1x4x4x8xi8", "1x5x5x8xi8")
    from cnnc.errors import TosaImportError

    with pytest.raises(TosaImportError) as exc:
        import_tosa(parse_module(text))
    assert "computed" in str(exc.value)


def test_avg_pool2d_is_refused_with_reasons():
    text = (
        '"builtin.module"() ({\n'
        '  "func.func"() <{function_type = (tensor<1x8x8x8xi8>) -> tensor<1x4x4x8xi8>, sym_name = "main"}> ({\n'
        "  ^bb0(%arg0: tensor<1x8x8x8xi8>):\n"
        '    %z = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>\n'
        '    %0 = "tosa.avg_pool2d"(%arg0, %z, %z) <{acc_type = i32, kernel = array<i64: 2, 2>, '
        'pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>}> : '
        "(tensor<1x8x8x8xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x4x4x8xi8>\n"
        '    "func.return"(%0) : (tensor<1x4x4x8xi8>) -> ()\n'
        "  }) : () -> ()\n"
        "}) : () -> ()\n"
    )
    with pytest.raises(UnsupportedOp) as exc:
        import_tosa(parse_module(text))
    message = str(exc.value)
    assert "count-include-pad" in message  # reason 1: which taps are counted
    assert "reciprocal_scale" in message  # reason 2: how the division rounds


# ---------------------------------------------------------------------------
# Verify: the GIR invariant that keeps max-pool padding honest
# ---------------------------------------------------------------------------


def test_verify_rejects_max_pool_padded_with_anything_but_the_type_minimum():
    import dataclasses

    from cnnc.gir.verify import verify

    graph = import_tosa(parse_module((FIXTURES_DIR / "maxpool_sppf.mlir").read_text()))
    op = next(op for op in graph.ops if op.kind == "pool")
    broken = graph.replace(
        ops=tuple(
            dataclasses.replace(o, attrs=dataclasses.replace(o.attrs, pad_value=0)) if o is op else o
            for o in graph.ops
        )
    )
    with pytest.raises(VerifyError) as exc:
        verify(broken)
    assert "pad_value" in str(exc.value)


# ---------------------------------------------------------------------------
# Lower + emit
# ---------------------------------------------------------------------------


def test_pool_lowers_onto_its_own_unit(target, tmp_path):
    result = _compile("maxpool_sppf", target, tmp_path)
    op = _pool_op(result)
    assert op.unit == "pool_engine"
    assert dict(op.params) == {
        "in_width": 8, "in_height": 8, "in_channels": 8,
        "pool_kernel_h": 5, "pool_kernel_w": 5, "pool_stride_h": 1, "pool_stride_w": 1,
        "pad_top": 2, "pad_bottom": 2, "pad_left": 2, "pad_right": 2,
        "pad_en": True, "pad_value": -128,
    }
    # No weights, no bias, no scale table: the pool reads only its input.
    assert len(op.reads) == 1


def test_pool_descriptor_carries_pad_value_and_pool_fields(target, tmp_path):
    result = _compile("maxpool_sppf", target, tmp_path)
    program_addr = result.hir_planned.buffer(result.hir_planned.program).addr
    descriptors = decode_program(result.program.program_bytes, target, program_addr=program_addr)
    desc = descriptors[0]
    assert desc.opcode == target.isa.opcodes["POOL_MAX"]
    assert (desc.pool_kernel_h, desc.pool_kernel_w) == (5, 5)
    assert (desc.pool_stride_h, desc.pool_stride_w) == (1, 1)
    assert (desc.pad_top, desc.pad_bottom, desc.pad_left, desc.pad_right) == (2, 2, 2, 2)
    assert desc.pad_value == -128
    flags = {name for name, bit in target.isa.flags.items() if (desc.flags >> bit) & 1}
    # PAD_EN only: BIAS_EN/REQUANT_EN would switch on an epilogue that
    # `cnn_accel_model.pool_max` deliberately bypasses.
    assert flags == {"PAD_EN"}


def test_unpadded_pool_clears_pad_en_and_pad_value(target, tmp_path):
    result = _compile("maxpool_stride2", target, tmp_path)
    program_addr = result.hir_planned.buffer(result.hir_planned.program).addr
    desc = decode_program(result.program.program_bytes, target, program_addr=program_addr)[0]
    assert desc.flags == 0
    assert desc.pad_value == -128  # still written: the field is not conditional


def test_padded_pool_needs_isa_v21(target, tmp_path):
    """On a pre-`pad_value` ISA a padded max pool is refused, not
    approximated with zero padding."""
    graph = import_tosa(parse_module((FIXTURES_DIR / "maxpool_sppf.mlir").read_text()))
    from cnnc.lower.to_hir import to_hir

    old = v10_target(target)
    data = old.to_dict()
    for unit in data["units"]:
        unit["isa_version"] = "1.2"
    from cnnc.target.contract import Target

    with pytest.raises(CapabilityError) as exc:
        to_hir(graph, Target.from_dict(data))
    assert "pad_value" in str(exc.value)


def test_pool_kernel_beyond_max_pool_kernel_size_is_rejected(target, tmp_path):
    text = (FIXTURES_DIR / "maxpool_sppf.mlir").read_text().replace(
        "kernel = array<i64: 5, 5>", "kernel = array<i64: 7, 7>"
    ).replace("pad = array<i64: 2, 2, 2, 2>", "pad = array<i64: 3, 3, 3, 3>")
    graph = import_tosa(parse_module(text))
    from cnnc.lower.to_hir import to_hir

    with pytest.raises(CapabilityError) as exc:
        to_hir(graph, target)
    assert "kernel" in str(exc.value)


# ---------------------------------------------------------------------------
# End to end: TOSA reference == emitted program on the golden model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(FIXTURES))
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_pool_program_matches_tosa_reference(name, seed, target, tmp_path):
    result = _compile(name, target, tmp_path)
    graph = result.imported_graph
    x = _seed_input(FIXTURES[name], seed)
    expected = interp.run(graph, {graph.inputs[0]: x})
    actual = run_program(result.program, {graph.inputs[0]: x})
    assert set(expected) == set(actual)
    for tid, want in expected.items():
        np.testing.assert_array_equal(actual[tid], want, err_msg=f"{name} seed {seed} tensor %{tid}")


def test_sppf_padded_pool_matches_model(target, tmp_path):
    """The SPPF case singled out: an all-negative input, where zero
    padding (the pre-v2.1 behaviour) would win the max at every border and
    produce zeros instead of the real maxima."""
    result = _compile("maxpool_sppf", target, tmp_path)
    graph = result.imported_graph
    x = np.full((1, 8, 8, 8), -7, dtype=np.int8)
    actual = run_program(result.program, {graph.inputs[0]: x})[graph.outputs[0]]
    np.testing.assert_array_equal(actual, np.full((1, 8, 8, 8), -7, dtype=np.int8))
