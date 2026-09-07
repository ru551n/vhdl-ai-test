"""M3 tests: `cnnc.gir.interp` vs IREE (doc/tosa_compiler_plan.md §10 item 3,
§13 M3: "IREE comparison (random seed 0 input)"). Skipped when IREE is not
installed next to the interpreter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cnnc.frontend.mlir_generic import parse_module
from cnnc.frontend.tosa_import import import_tosa, load_tosa_file
from cnnc.gir import interp
from cnnc.gir.interp import run
from cnnc.testing.iree_oracle import iree_available, run_iree

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "conv_rescale_clamp.mlir"

pytestmark = pytest.mark.iree

if not iree_available():
    pytest.skip("iree-compile/iree-run-module not found next to the interpreter", allow_module_level=True)


def _seed0_input() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(-128, 128, size=(1, 8, 8, 4)).astype(np.int8)


def test_fixture_iree_matches_interp(tmp_path):
    graph = load_tosa_file(FIXTURE_PATH)
    x = _seed0_input()

    interp_out = run(graph, {"arg0": x})["10"]
    iree_out = run_iree(FIXTURE_PATH, [x], tmp_path)[0]

    assert iree_out.dtype == np.int8
    assert iree_out.shape == interp_out.shape
    np.testing.assert_array_equal(iree_out, interp_out)


def _tie_provoking_mlir_text() -> str:
    """Textual substitution on the fixture (doc/tosa_compiler_plan.md §11
    style "mutate the fixture" negative-test pattern): multiplier stays
    `2**30` but shift becomes 31 (an exact 0.5 scale at the tie boundary,
    vs. the fixture's shift=38) and bias becomes 1 (odd), so a run with an
    all-ones 3x3x4 kernel produces an odd accumulator -- and hence a
    round-half-up tie -- at roughly half the output pixels."""
    text = FIXTURE_PATH.read_text()
    bias_old, bias_new = "dense<0> : tensor<8xi32>", "dense<1> : tensor<8xi32>"
    shift_old, shift_new = "dense<38> : tensor<1xi8>", "dense<31> : tensor<1xi8>"
    assert text.count(bias_old) == 1
    assert text.count(shift_old) == 1
    return text.replace(bias_old, bias_new).replace(shift_old, shift_new)


def test_tie_provoking_variant_iree_matches_interp(tmp_path):
    tie_text = _tie_provoking_mlir_text()
    tie_path = tmp_path / "tie.mlir"
    tie_path.write_text(tie_text)

    graph = import_tosa(parse_module(tie_text))
    x = _seed0_input()

    conv_op = graph.producer("4")
    w = np.asarray(graph.tensor("0").values, dtype=np.int64).reshape(graph.tensor("0").shape)
    b = np.asarray(graph.tensor("1").values, dtype=np.int64).reshape(graph.tensor("1").shape)
    acc = interp._conv2d_int64(x, w, b, conv_op.attrs, conv_op.id)
    tie_pixel_count = int(np.sum(acc % 2 != 0))
    assert tie_pixel_count > 0, "test setup failed to provoke any rescale ties"

    interp_out = run(graph, {"arg0": x})["10"]
    iree_out = run_iree(tie_path, [x], tmp_path)[0]

    if not np.array_equal(iree_out, interp_out):
        diff = np.argwhere(iree_out != interp_out)
        details = [(tuple(idx), int(iree_out[tuple(idx)]), int(interp_out[tuple(idx)])) for idx in diff[:20]]
        pytest.fail(
            f"IREE and interp disagree at {len(diff)}/{iree_out.size} positions "
            f"(index, iree, interp), first 20: {details}"
        )
