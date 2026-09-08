"""Conv `in_zp != 0`: ISA v2.1 `pad_value` + the `bias -= in_zp * sum(w)`
fold (doc/tosa_compiler_plan.md extension 5).

`lower.to_hir` used to reject any non-zero conv zero-point outright
("padding is literal 0 in HW"). It is now lowered exactly, and "exactly"
is the whole point of this module: the two halves of the lowering are
individually wrong and only jointly correct, so every test here is
either an end-to-end `interp == run_program` equality or a demonstration
that breaking ONE half breaks the result.

Why the pair works (see `lower.to_hir._zero_point_bias_delta` for the
derivation):

    TOSA:  acc[o] = sum_window (x_pad - in_zp) * w,  x_pad padded with in_zp
    HW:    hw[o]  = sum_window  x_pad      * w,      x_pad padded with pad_value

    pad_value = in_zp  =>  acc[o] = hw[o] - in_zp * sum_kernel(w[o])

and that correction is the same at every output position -- interior and
border alike -- which is what makes it a compile-time bias fold.

`in_zp = -128` is not an arbitrary sample: it is YOLOv8n's activation
zero-point, the case that motivated the ISA field.
"""

from __future__ import annotations

import dataclasses
import struct

import numpy as np
import pytest
from test_to_hir import _build_mlir

from cnnc.backend.cnn_accel_v1 import run_program
from cnnc.backend.cnn_accel_v1.emit import Program, encode_descriptor
from cnnc.driver import compile_tosa
from cnnc.gir import interp

IN_SHAPE = (1, 8, 8, 4)
# The builder's weights are a splat of 1 over a 3x3x4 kernel, so
# sum(w[o]) == 36 for every output channel.
KERNEL_WEIGHT_SUM = 3 * 3 * 4


#: `shift=37` (i.e. `acc >> 7` after the Q15 multiplier) and the identity
#: clamp are chosen so the epilogue does NOT saturate for any of the
#: zero-points exercised here: with `in_zp = -128` the shifted input spans
#: 0..255 and 36 taps accumulate to ~9180, which a smaller shift would
#: clip to 127 everywhere -- making every comparison below pass
#: vacuously, including the two that are supposed to detect a broken
#: lowering.
_SHIFT = (37,)
_IDENTITY_CLAMP = (-128, 127)


def _compile(target, tmp_path, **kwargs):
    kwargs.setdefault("clamp", _IDENTITY_CLAMP)
    text = _build_mlir(shift=_SHIFT, **kwargs)
    src = tmp_path / "zp.mlir"
    src.write_text(text)
    return compile_tosa(src, target, out_dir=tmp_path / "out")


def _seed_input(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(-128, 128, size=IN_SHAPE).astype(np.int8)


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("in_zp", [0, 7, -128, 127])
@pytest.mark.parametrize("pad", [(1, 1, 1, 1), (0, 0, 0, 0)])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_zero_point_conv_matches_tosa_reference(in_zp, pad, seed, target, tmp_path):
    result = _compile(target, tmp_path, conv_in_zp=in_zp, pad=pad)
    graph = result.imported_graph
    x = _seed_input(seed)
    expected = interp.run(graph, {graph.inputs[0]: x})
    actual = run_program(result.program, {graph.inputs[0]: x})
    for tid, want in expected.items():
        np.testing.assert_array_equal(actual[tid], want, err_msg=f"in_zp={in_zp} pad={pad} seed={seed}")


# ---------------------------------------------------------------------------
# Each half of the lowering, on its own
# ---------------------------------------------------------------------------


def test_bias_is_folded_by_in_zp_times_the_weight_sum(target, tmp_path):
    result = _compile(target, tmp_path, conv_in_zp=-128, bias=(100,) * 8)
    op = result.hir_planned.ops[0]
    bias_buf = result.hir_planned.buffer(op.reads[2])
    folded = struct.unpack(f"<{bias_buf.size_bytes // 4}i", bias_buf.data)
    assert set(folded[:8]) == {100 - (-128) * KERNEL_WEIGHT_SUM}


def test_pad_value_reaches_the_descriptor(target, tmp_path):
    result = _compile(target, tmp_path, conv_in_zp=-128)
    assert result.program.descriptors[0].pad_value == -128


def test_zero_zero_point_leaves_pad_value_and_bias_untouched(target, tmp_path):
    result = _compile(target, tmp_path, conv_in_zp=0, bias=(100,) * 8)
    op = result.hir_planned.ops[0]
    assert "pad_value" not in op.params
    assert result.program.descriptors[0].pad_value == 0
    bias_buf = result.hir_planned.buffer(op.reads[2])
    folded = struct.unpack(f"<{bias_buf.size_bytes // 4}i", bias_buf.data)
    assert set(folded[:8]) == {100}


def _rewritten_program(result, target, **desc_overrides) -> Program:
    """`result.program` with the first descriptor's fields overridden and
    the instruction stream re-encoded -- the surgical way to ask "what
    would the hardware compute if only ONE half of the lowering had
    landed?"."""
    descriptors = list(result.program.descriptors)
    descriptors[0] = dataclasses.replace(descriptors[0], **desc_overrides)
    program_bytes = b"".join(encode_descriptor(d, target) for d in descriptors)
    return dataclasses.replace(
        result.program, program_bytes=program_bytes, descriptors=tuple(descriptors)
    )


def test_pad_value_without_the_bias_fold_is_wrong(target, tmp_path):
    """The failure mode the ISA v2.1 note warns about: setting
    `pad_value` and stopping there. Every output is off by
    `in_zp * sum(w)` before requantization, so the result differs
    everywhere -- not just at the border."""
    result = _compile(target, tmp_path, conv_in_zp=-128)
    graph = result.imported_graph
    x = _seed_input(1)
    expected = interp.run(graph, {graph.inputs[0]: x})[graph.outputs[0]]

    # Un-fold the bias back to its TOSA value, keeping pad_value = in_zp.
    op = result.hir_planned.ops[0]
    bias_id = op.reads[2]
    unfolded = dataclasses.replace(result.hir_planned.buffer(bias_id), data=bytes(32))
    broken = result.hir_planned.with_buffers({bias_id: unfolded})
    from cnnc.backend.cnn_accel_v1 import emit_program

    actual = run_program(emit_program(broken, target), {graph.inputs[0]: x})[graph.outputs[0]]
    assert not np.array_equal(actual, expected)


def test_bias_fold_without_pad_value_is_wrong_at_the_border(target, tmp_path):
    """The mirror-image failure: fold the bias but keep padding with a
    literal 0. Interior outputs stay correct (no padded taps), so this is
    exactly the kind of bug that survives a spot check and corrupts every
    border pixel of every padded layer."""
    result = _compile(target, tmp_path, conv_in_zp=-128, pad=(1, 1, 1, 1))
    graph = result.imported_graph
    x = _seed_input(1)
    expected = interp.run(graph, {graph.inputs[0]: x})[graph.outputs[0]]

    broken = _rewritten_program(result, target, pad_value=0)
    actual = run_program(broken, {graph.inputs[0]: x})[graph.outputs[0]]
    assert not np.array_equal(actual, expected)
    # ... and the damage really is confined to the border.
    np.testing.assert_array_equal(actual[:, 1:-1, 1:-1, :], expected[:, 1:-1, 1:-1, :])
