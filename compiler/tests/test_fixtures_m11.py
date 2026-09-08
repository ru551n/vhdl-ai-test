"""M11 tests: `output_zp` + general clamp lowering onto the ISA v1.1 (H1)
epilogue fields (doc/tosa_compiler_plan.md §5 extension 1, M11).

Fixtures (both produced by `tests/fixtures/gen_fixtures.py`, byte-
reproducible):

* `out_zp_relu.mlir` -- `rescale.output_zp = -128`, identity clamp
  `[-128,127]` (the asymmetric-output quantized-ReLU idiom the v1.0
  `RELU_EN` flag cannot express). Lowered as `output_offset=-128,
  CLAMP_EN, [-128,127]`.
* `clamp_5_100.mlir` -- general clamp `[5,100]`, `output_zp = 0`. Lowered
  as `CLAMP_EN, [5,100]`.

Semantics being pinned (TOSA vs HW, doc §6): TOSA `rescale` yields
`clamp_i8(s + out_zp)`, a following `clamp` narrows within int8; the HW
epilogue (`cnn_accel_model.bias_requantize_relu`, RTL
`cnn_accel_bias_requant`) computes `clamp(s + output_offset, lo, hi)` with
`[lo,hi]` inside int8 -- the exact composition. `gir.interp` is the TOSA
side, `run_program` the HW side, IREE the external oracle.

ACCEPTANCE (M11): e2e byte-exact `interp == run_program == IREE` for both
fixtures x 3 seeds; the MVP fixture (`conv_rescale_clamp`) is emitted as
`CLAMP_EN,[0,127]` and is behaviourally bit-identical to the v1.0
`RELU_EN=1` encoding (the program bytes differ ONLY in the flags byte and
W13); an `isa_version 1.0` target + `out_zp != 0` is a `CapabilityError`.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
from conftest import v10_target

from cnnc.backend.cnn_accel_v1 import decode_program, print_program, run_program
from cnnc.backend.cnn_accel_v1.emit import encode_descriptor
from cnnc.backend.cnn_accel_v1.vectors import write_conv_core_vectors
from cnnc.driver import CompileResult, compile_tosa
from cnnc.errors import CapabilityError
from cnnc.gir import interp
from cnnc.gir.printer import print_gir
from cnnc.hir.printer import print_hir
from cnnc.testing.iree_oracle import iree_available, run_iree

FIXTURES_DIR = Path(__file__).parent / "fixtures"
GOLDEN_DIR = Path(__file__).parent / "golden"

# name -> (input shape, expected (output_offset, clamp_min, clamp_max))
FIXTURES = {
    "out_zp_relu": ((1, 8, 8, 8), (-128, -128, 127)),
    "clamp_5_100": ((1, 8, 8, 8), (0, 5, 100)),
}
MVP_FIXTURE = "conv_rescale_clamp"
MVP_SHAPE = (1, 8, 8, 4)


def _compile(name: str, target, tmp_path) -> CompileResult:
    return compile_tosa(FIXTURES_DIR / f"{name}.mlir", target, out_dir=tmp_path / "out")


def _seed_input(shape: tuple[int, ...], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(-128, 128, size=shape).astype(np.int8)


def _flag_names(flags: int, target) -> set[str]:
    return {name for name, bit in target.isa.flags.items() if (flags >> bit) & 1}


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


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_fixture_descriptor_carries_w13_epilogue(name, target, tmp_path):
    result = _compile(name, target, tmp_path)
    _shape, (offset, lo, hi) = FIXTURES[name]

    # The importer kept out_zp, fuse folded the clamp (also the identity
    # one: normalize keeps it on a `clamp_ranges: "any"` target).
    fused = [op for op in result.fused_graph.ops if op.kind == "fused_conv"]
    assert len(fused) == 1
    assert fused[0].attrs.rescale.out_zp == offset
    assert fused[0].attrs.clamp is not None
    assert (fused[0].attrs.clamp.min, fused[0].attrs.clamp.max) == (lo, hi)
    assert not any(op.kind in ("rescale", "clamp") for op in result.fused_graph.ops)

    params = result.hir_mapped.ops[0].params
    assert params["relu_en"] is False and params["clamp_en"] is True
    assert (params["output_offset"], params["clamp_min"], params["clamp_max"]) == (offset, lo, hi)

    desc = result.program.descriptors[0]
    assert (desc.output_offset, desc.clamp_min, desc.clamp_max) == (offset, lo, hi)
    assert _flag_names(desc.flags, target) == {"CLAMP_EN", "BIAS_EN", "REQUANT_EN", "PAD_EN"}


# ---------------------------------------------------------------------------
# End-to-end: run_program == interp for 3 seeds, both fixtures, and the
# epilogue actually bites (offset floor / both clamp bounds hit).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(FIXTURES))
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_e2e_run_program_matches_interp(name, seed, target, tmp_path):
    result = _compile(name, target, tmp_path)
    shape, (_offset, lo, hi) = FIXTURES[name]
    x = _seed_input(shape, seed)
    out_id = result.fused_graph.outputs[0]

    interp_out = interp.run(result.fused_graph, {"arg0": x})[out_id]
    run_out = run_program(result.program, {"arg0": x})[out_id]

    assert run_out.dtype == np.int8
    assert run_out.shape == interp_out.shape
    np.testing.assert_array_equal(run_out, interp_out)

    # Non-degenerate and the new fields matter: many distinct values, the
    # lower bound (out_zp floor / clamp_min) is hit, output stays in [lo,hi].
    assert len(np.unique(interp_out)) >= 16
    assert interp_out.min() == lo, f"{name} seed={seed}: lower bound {lo} never reached"
    assert interp_out.max() <= hi
    if name == "clamp_5_100":
        assert interp_out.max() == hi, f"seed={seed}: clamp_max {hi} never reached"
    saturated = np.count_nonzero((interp_out == lo) | (interp_out == hi))
    assert saturated < interp_out.size


# ---------------------------------------------------------------------------
# External oracle: run_program == interp == IREE, both fixtures x 3 seeds.
# This is the check that IREE's `tosa.rescale` output_zp/clamp semantics
# match the plan's assumption (§6) -- not just our two CPU-side oracles.
# ---------------------------------------------------------------------------


@pytest.mark.iree
@pytest.mark.parametrize("name", sorted(FIXTURES))
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_e2e_run_program_matches_interp_and_iree(name, seed, target, tmp_path):
    if not iree_available():
        pytest.skip("iree-compile/iree-run-module not found next to the interpreter")

    result = _compile(name, target, tmp_path)
    shape, _ = FIXTURES[name]
    x = _seed_input(shape, seed)
    out_id = result.fused_graph.outputs[0]

    interp_out = interp.run(result.fused_graph, {"arg0": x})[out_id]
    run_out = run_program(result.program, {"arg0": x})[out_id]
    iree_out = run_iree(FIXTURES_DIR / f"{name}.mlir", [x], tmp_path / f"iree_{seed}")[0]

    np.testing.assert_array_equal(run_out, interp_out)
    np.testing.assert_array_equal(iree_out, interp_out)


# ---------------------------------------------------------------------------
# Rejections: ISA v1.0 target has no output_offset / general clamp.
# ---------------------------------------------------------------------------


def test_out_zp_rejected_on_isa_v10_target(target, tmp_path):
    with pytest.raises(CapabilityError) as exc_info:
        _compile("out_zp_relu", v10_target(target), tmp_path)
    # FusePass leaves the out_zp rescale standalone on `output_zp: false`,
    # so the rejection names the un-fused conv2d; either way it is loud.
    assert exc_info.value.stage == "to_hir"


def test_general_clamp_rejected_on_isa_v10_target(target, tmp_path):
    with pytest.raises(CapabilityError) as exc_info:
        _compile("clamp_5_100", v10_target(target), tmp_path)
    assert exc_info.value.stage == "to_hir"
    assert exc_info.value.constraint == "clamp"


# ---------------------------------------------------------------------------
# MVP fixture: CLAMP_EN,[0,127] is behaviourally identical to RELU_EN=1.
# ---------------------------------------------------------------------------


def _legacy_relu_program(result: CompileResult, target):
    """Re-encode `result.program` with the ISA v1.0 ReLU epilogue: RELU_EN
    set, CLAMP_EN clear, W13 zero. Everything else (addresses, constants,
    manifest) is untouched, so `run_program` executes the same layer with
    the legacy epilogue encoding."""
    relu_bit = target.isa.flags["RELU_EN"]
    clamp_bit = target.isa.flags["CLAMP_EN"]
    conv_opcode = target.isa.opcodes["CONV2D"]
    legacy_descs = []
    for desc in result.program.descriptors:
        if desc.opcode != conv_opcode:
            legacy_descs.append(desc)
            continue
        assert (desc.flags >> clamp_bit) & 1 and (desc.clamp_min, desc.clamp_max) == (0, 127)
        flags = (desc.flags & ~(1 << clamp_bit)) | (1 << relu_bit)
        legacy_descs.append(dataclasses.replace(desc, flags=flags, output_offset=0, clamp_min=0, clamp_max=0))
    legacy_bytes = b"".join(encode_descriptor(d, target) for d in legacy_descs)
    return dataclasses.replace(result.program, program_bytes=legacy_bytes, descriptors=tuple(legacy_descs))


def test_mvp_fixture_emitted_as_clamp_en_0_127(target, tmp_path):
    result = _compile(MVP_FIXTURE, target, tmp_path)
    desc = result.program.descriptors[0]
    assert _flag_names(desc.flags, target) == {"CLAMP_EN", "BIAS_EN", "REQUANT_EN", "PAD_EN"}
    assert (desc.output_offset, desc.clamp_min, desc.clamp_max) == (0, 0, 127)


def test_mvp_clamp_en_encoding_bit_identical_in_behaviour_to_relu_en(target, tmp_path):
    result = _compile(MVP_FIXTURE, target, tmp_path)
    legacy = _legacy_relu_program(result, target)

    # Byte encoding differs ONLY in the flags field and the W13 epilogue
    # fields (output_offset/clamp_min/clamp_max); every other byte is equal.
    fields = target.isa.fields
    allowed = set()
    for name in ("flags", "output_offset", "clamp_min", "clamp_max"):
        offset, width, _signed = fields[name]
        allowed.update(range(offset, offset + width))
    word = target.isa.instr_word_bytes
    new_bytes, old_bytes = result.program.program_bytes, legacy.program_bytes
    assert len(new_bytes) == len(old_bytes)
    differing = {i for i, (a, b) in enumerate(zip(new_bytes, old_bytes)) if a != b}
    assert differing, "the two encodings should not be byte-identical"
    assert all((i % word) in allowed for i in differing), sorted(differing)
    assert _flag_names(legacy.descriptors[0].flags, target) == {"RELU_EN", "BIAS_EN", "REQUANT_EN", "PAD_EN"}

    # ... and the golden model produces identical outputs for both, i.e.
    # CLAMP_EN,[0,127] == RELU_EN in behaviour for every seed.
    for seed in (0, 1, 2):
        x = _seed_input(MVP_SHAPE, seed)
        new_out = run_program(result.program, {"arg0": x})["10"]
        old_out = run_program(legacy, {"arg0": x})["10"]
        interp_out = interp.run(result.fused_graph, {"arg0": x})["10"]
        np.testing.assert_array_equal(new_out, old_out)
        np.testing.assert_array_equal(new_out, interp_out)


# ---------------------------------------------------------------------------
# RTL vectors hook (M10) keeps working: desc.txt carries the W13 fields and
# the CLAMP_EN flag bit for tb_cnn_accel_conv_core's reader.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_conv_core_vectors_desc_carries_w13_fields(name, target, tmp_path):
    result = _compile(name, target, tmp_path)
    shape, (offset, lo, hi) = FIXTURES[name]
    inputs = {"arg0": _seed_input(shape, 0)}
    out_dir = tmp_path / "vectors"
    vectors = write_conv_core_vectors(result.program, inputs, out_dir, target=target, graph=result.fused_graph)
    assert vectors.written == ["op0"] and vectors.skipped == []

    desc_fields = {}
    for line in (out_dir / "op0" / "desc.txt").read_text().splitlines():
        key, _, value = line.partition(" ")
        desc_fields[key] = int(value)
    assert (desc_fields["output_offset"], desc_fields["clamp_min"], desc_fields["clamp_max"]) == (offset, lo, hi)
    assert (desc_fields["flags"] >> target.isa.flags["CLAMP_EN"]) & 1 == 1
    assert (desc_fields["flags"] >> target.isa.flags["RELU_EN"]) & 1 == 0

    expected = np.array([int(v) for v in (out_dir / "op0" / "expected.txt").read_text().split()], dtype=np.int8)
    assert expected.min() == lo and expected.max() <= hi
