"""M9 tests: multi-layer fixtures (doc/tosa_compiler_plan.md M9,
~line 602) -- `two_layer.mlir` (16->32->32 channels, 3x3 s1 then 3x3
s2) and `first_layer_cin3.mlir` (Cin=3, 3x3 s2, exercises D11 partial
input-tile zero padding).

Both fixtures were produced by `tests/fixtures/gen_fixtures.py`
(deterministic, non-splat int8 weights/int32 bias; re-running it
reproduces the two `.mlir` files byte-identically) and their four
golden dumps -- `.gir.txt`, `.fused.gir.txt`, `.hir.txt`,
`.program.txt` -- were captured from a real `compile_tosa()` run (not
hand-written), mirroring `conv_rescale_clamp`'s golden set
(`test_tosa_import.py`, `test_fuse.py`, `test_to_hir.py`,
`test_backend.py`).

ACCEPTANCE (M9): e2e byte-exact for both fixtures (`run_program` ==
`gir.interp`, and == IREE for seed 0); planner reuse observed in the
dump -- see `test_two_layer_memplan_no_reuse_with_only_two_layers`
below for why two layers in particular *cannot* show it, and what
would.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cnnc.backend.cnn_accel_v1 import decode_program, print_program, run_program
from cnnc.driver import CompileResult, compile_tosa
from cnnc.gir import interp
from cnnc.gir.printer import print_gir
from cnnc.hir.printer import print_hir
from cnnc.lower.memplan import lifetimes
from cnnc.testing.iree_oracle import iree_available, run_iree

FIXTURES_DIR = Path(__file__).parent / "fixtures"
GOLDEN_DIR = Path(__file__).parent / "golden"

FIXTURES = {
    "two_layer": (1, 8, 8, 16),
    "first_layer_cin3": (1, 8, 8, 3),
}


def _compile(name: str, target, tmp_path) -> CompileResult:
    return compile_tosa(FIXTURES_DIR / f"{name}.mlir", target, out_dir=tmp_path / "out")


def _seed_input(shape: tuple[int, ...], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(-128, 128, size=shape).astype(np.int8)


# ---------------------------------------------------------------------------
# Golden dumps (gir / fused gir / hir / program), all 4 forms x 2 fixtures.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_fixture_matches_golden_dumps(name, target, tmp_path):
    result = _compile(name, target, tmp_path)

    assert print_gir(result.imported_graph) == (GOLDEN_DIR / f"{name}.gir.txt").read_text()
    assert print_gir(result.fused_graph) == (GOLDEN_DIR / f"{name}.fused.gir.txt").read_text()
    assert print_hir(result.hir_mapped) == (GOLDEN_DIR / f"{name}.hir.txt").read_text()

    program_addr = result.hir_planned.buffer(result.hir_planned.program).addr
    descriptors = decode_program(result.program.program_bytes, target, program_addr=program_addr)
    text = print_program(descriptors, target, program_addr=program_addr)
    assert text == (GOLDEN_DIR / f"{name}.program.txt").read_text()


# ---------------------------------------------------------------------------
# End-to-end: run_program == interp for 3 seeds, both fixtures; plus the
# non-degenerate-output assertion (>= 16 distinct values, not saturated).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(FIXTURES))
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_e2e_run_program_matches_interp(name, seed, target, tmp_path):
    result = _compile(name, target, tmp_path)
    x = _seed_input(FIXTURES[name], seed)
    out_id = result.fused_graph.outputs[0]

    interp_out = interp.run(result.fused_graph, {"arg0": x})[out_id]
    run_out = run_program(result.program, {"arg0": x})[out_id]

    assert run_out.dtype == np.int8
    assert run_out.shape == interp_out.shape
    np.testing.assert_array_equal(run_out, interp_out)

    # Non-degenerate: real datapath behavior, not a saturated/constant plane.
    distinct = np.unique(interp_out)
    assert len(distinct) >= 16, f"{name} seed={seed}: only {len(distinct)} distinct output values"
    saturated = np.count_nonzero((interp_out == interp_out.min()) | (interp_out == interp_out.max()))
    assert saturated < interp_out.size, f"{name} seed={seed}: output is entirely at its own min/max"


# ---------------------------------------------------------------------------
# End-to-end oracle cross-check: run_program == interp == IREE, seed 0.
# ---------------------------------------------------------------------------


@pytest.mark.iree
@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_e2e_run_program_matches_interp_and_iree(name, target, tmp_path):
    if not iree_available():
        pytest.skip("iree-compile/iree-run-module not found next to the interpreter")

    result = _compile(name, target, tmp_path)
    x = _seed_input(FIXTURES[name], seed=0)
    out_id = result.fused_graph.outputs[0]

    interp_out = interp.run(result.fused_graph, {"arg0": x})[out_id]
    run_out = run_program(result.program, {"arg0": x})[out_id]
    iree_out = run_iree(FIXTURES_DIR / f"{name}.mlir", [x], tmp_path / "iree_out")[0]

    np.testing.assert_array_equal(run_out, interp_out)
    np.testing.assert_array_equal(iree_out, interp_out)


# ---------------------------------------------------------------------------
# Memory planner: two_layer has exactly one fused_conv->fused_conv edge, so
# exactly one `intermediate`-role buffer -- there is nothing else it could
# ever be reused against. `lower/memplan.py`'s free-list reuse only
# considers `intermediate`-role buffers (inputs/outputs/consts/program are
# placed once and never freed, see `plan_memory`'s stages 1-3 vs. 4), and
# `test_schedule_memplan.py::test_memplan_three_layers_no_reuse_when_lifetimes_touch`
# already shows even a *second* intermediate isn't enough on its own: with
# 3 ops (2 intermediates) %t1's lifetime end (its last reader's seq) equals
# %t2's lifetime start (its writer's seq), and the planner requires the new
# buffer's start to be *strictly after* a freed block's end, so they still
# don't overlap in address. Reuse first appears at 4 ops / 3 intermediates
# (`test_memplan_four_layers_reuses_freed_block`), where %t3's write (seq 2)
# is strictly after %t1's last read (seq 1).
#
# So: two_layer's memplan dump (tests/golden/two_layer.hir.txt covers the
# pre-schedule/plan `hir_mapped` stage; see the 07_memplan dump under
# `dump_after_all=True` for placed addresses) shows NO address reuse, and
# it is architecturally impossible for it to show any with only 2 fused
# layers. A *3*-layer variant would still not show it (per the cited unit
# test); a 4-layer variant would be the minimum. Per the M9 task, we do
# not add a third (or fourth) fixture just to force this -- the reuse
# mechanism itself is already covered by `test_schedule_memplan.py`.
# ---------------------------------------------------------------------------


def test_two_layer_memplan_no_reuse_with_only_two_layers(target, tmp_path):
    result = _compile("two_layer", target, tmp_path)

    intermediates = {bid: buf for bid, buf in result.hir_planned.buffers.items() if buf.role == "intermediate"}
    assert len(intermediates) == 1, (
        "two_layer has exactly one fused_conv->fused_conv edge, hence exactly one "
        "intermediate buffer -- there is nothing for the planner to reuse it against"
    )
    lts = lifetimes(result.hir_scheduled)
    assert set(lts) == set(intermediates)

    # Every buffer (program, consts, input, the one intermediate, output)
    # occupies a distinct, non-overlapping address range -- i.e. no reuse
    # happened (there was nothing to reuse against).
    placed = sorted(
        (buf.addr, buf.addr + buf.size_bytes, bid) for bid, buf in result.hir_planned.buffers.items()
    )
    for (a_start, a_end, a_id), (b_start, b_end, b_id) in zip(placed, placed[1:]):
        assert a_end <= b_start, f"unexpected overlap between {a_id} and {b_id}"
