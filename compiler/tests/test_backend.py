"""M8 tests: `cnnc.backend.cnn_accel_v1` (`emit`/`decode`/`run`) and
`cnnc.driver.compile_tosa`, doc/tosa_compiler_plan.md §2.3, §5, §10, §11
row 08, §13 M8.

MVP acceptance (§13 M8): the fixture compiles; `run_program` output ==
`gir.interp` output == IREE output (byte-exact) for 3 seeds; decoded
descriptors match the emitted ones and the HIR params; `09_program.txt`
shows `CONV2D` + `HALT`. HW milestone H0 (half-up rounding) has landed
on the real target, so these run directly against `cnn_accel_v1`
(the `target` fixture from `tests/conftest.py`), no rounding-gate
workaround needed.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from cnnc.backend.cnn_accel_v1 import decode_program, emit_program, print_program, run_program
from cnnc.driver import compile_tosa
from cnnc.errors import CapabilityError, CompilerError
from cnnc.gir import interp
from cnnc.testing.iree_oracle import iree_available, run_iree

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = Path(__file__).parent / "fixtures" / "conv_rescale_clamp.mlir"
GOLDEN_PROGRAM_PATH = Path(__file__).parent / "golden" / "conv_rescale_clamp.program.txt"


def _compile(target, tmp_path, *, dump_after_all: bool = False):
    return compile_tosa(FIXTURE_PATH, target, out_dir=tmp_path / "out", dump_after_all=dump_after_all)


def _seed_input(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(-128, 128, size=(1, 8, 8, 4)).astype(np.int8)


def _load_module_from(root: Path, name: str):
    """`importlib`-fresh-load one accelerator module by name (no
    `sys.path` mutation), mirroring `test_target_contract.py`'s
    `_load_accel_module` and `backend.cnn_accel_v1.run._import_fresh`."""
    spec = importlib.util.spec_from_file_location(name, root / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# emit_program / decode_program: golden dump + round trip
# ---------------------------------------------------------------------------


def test_fixture_matches_golden_program(target, tmp_path):
    result = _compile(target, tmp_path)
    program_addr = result.hir_planned.buffer(result.hir_planned.program).addr
    descriptors = decode_program(result.program.program_bytes, target, program_addr=program_addr)
    text = print_program(descriptors, target, program_addr=program_addr)
    assert text == GOLDEN_PROGRAM_PATH.read_text()


def test_decode_round_trip_matches_emitted_descriptors(target, tmp_path):
    result = _compile(target, tmp_path)
    program_addr = result.hir_planned.buffer(result.hir_planned.program).addr
    decoded = decode_program(result.program.program_bytes, target, program_addr=program_addr)
    assert decoded == result.program.descriptors


def test_decode_cross_checks_against_cnn_accel_model(target, tmp_path):
    """The compiler's own `decode_descriptor` and the RTL golden model's
    `decode_instruction` must agree field-for-field on the exact bytes
    `emit_program` produced (same `accel_root` the target was discovered
    from, so this is a same-checkout comparison)."""
    result = _compile(target, tmp_path)
    accel_root = Path(target.provenance["accel_root"])
    _load_module_from(accel_root, "cnn_accel_constants")
    model = _load_module_from(accel_root, "cnn_accel_model")

    program_addr = result.hir_planned.buffer(result.hir_planned.program).addr
    ours = decode_program(result.program.program_bytes, target, program_addr=program_addr)

    word = target.isa.instr_word_bytes
    for i, our_desc in enumerate(ours):
        chunk = result.program.program_bytes[i * word : (i + 1) * word]
        model_desc = model.decode_instruction(chunk)
        for field_name in ("opcode", "flags", "in_addr", "out_addr", "weight_addr", "bias_addr",
                           "in_width", "in_height", "in_channels", "out_channels",
                           "kernel_h", "kernel_w", "stride_h", "stride_w",
                           "pad_top", "pad_bottom", "pad_left", "pad_right",
                           "requant_scale", "requant_shift", "next_instr_addr"):
            assert getattr(our_desc, field_name) == getattr(model_desc, field_name), (
                f"descriptor {i} field {field_name!r}: "
                f"ours={getattr(our_desc, field_name)} model={getattr(model_desc, field_name)}"
            )


def test_descriptor_params_match_hir(target, tmp_path):
    result = _compile(target, tmp_path)
    program_addr = result.hir_planned.buffer(result.hir_planned.program).addr
    conv_desc = decode_program(result.program.program_bytes, target, program_addr=program_addr)[0]

    assert conv_desc.requant_scale == 1073741824
    assert conv_desc.requant_shift == 23
    assert (conv_desc.in_width, conv_desc.in_height, conv_desc.in_channels) == (8, 8, 4)
    assert conv_desc.out_channels == 8
    assert (conv_desc.kernel_h, conv_desc.kernel_w) == (3, 3)
    assert (conv_desc.stride_h, conv_desc.stride_w) == (1, 1)
    assert (conv_desc.pad_top, conv_desc.pad_bottom, conv_desc.pad_left, conv_desc.pad_right) == (1, 1, 1, 1)

    # ISA v1.1 (M11) epilogue encoding of the fixture's ReLU: CLAMP_EN with
    # [0,127] in W13, RELU_EN clear (see test_fixtures_m11.py for the proof
    # that this is behaviourally identical to the v1.0 RELU_EN encoding).
    set_flags = {name for name, bit in target.isa.flags.items() if (conv_desc.flags >> bit) & 1}
    assert set_flags == {"CLAMP_EN", "BIAS_EN", "REQUANT_EN", "PAD_EN"}
    assert (conv_desc.output_offset, conv_desc.clamp_min, conv_desc.clamp_max) == (0, 0, 127)


def test_program_bytes_end_with_halt(target, tmp_path):
    result = _compile(target, tmp_path)
    program_addr = result.hir_planned.buffer(result.hir_planned.program).addr
    descriptors = decode_program(result.program.program_bytes, target, program_addr=program_addr)
    opcode_name = {v: k for k, v in target.isa.opcodes.items()}
    assert [opcode_name[d.opcode] for d in descriptors] == ["CONV2D", "HALT"]


# ---------------------------------------------------------------------------
# End-to-end: run_program == interp == IREE, 3 seeds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_e2e_run_program_matches_interp(target, tmp_path, seed):
    result = _compile(target, tmp_path)
    x = _seed_input(seed)

    interp_out = interp.run(result.fused_graph, {"arg0": x})["10"]
    run_out = run_program(result.program, {"arg0": x})["10"]

    assert run_out.dtype == np.int8
    assert run_out.shape == interp_out.shape
    np.testing.assert_array_equal(run_out, interp_out)


@pytest.mark.iree
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_e2e_run_program_matches_interp_and_iree(target, tmp_path, seed):
    if not iree_available():
        pytest.skip("iree-compile/iree-run-module not found next to the interpreter")

    result = _compile(target, tmp_path)
    x = _seed_input(seed)

    interp_out = interp.run(result.fused_graph, {"arg0": x})["10"]
    run_out = run_program(result.program, {"arg0": x})["10"]
    iree_out = run_iree(FIXTURE_PATH, [x], tmp_path / f"iree_{seed}")[0]

    np.testing.assert_array_equal(run_out, interp_out)
    np.testing.assert_array_equal(iree_out, interp_out)


# ---------------------------------------------------------------------------
# emit_program error paths
# ---------------------------------------------------------------------------


def test_emit_requires_planned_stage(target, tmp_path):
    result = _compile(target, tmp_path)
    with pytest.raises(CompilerError, match="planned"):
        emit_program(result.hir_scheduled, target)


def test_emit_rejects_field_overflow(target, tmp_path):
    result = _compile(target, tmp_path)
    op = result.hir_planned.ops[0]
    bad_op = op.replace(params={**op.params, "stride_h": 1000})  # ISA field is 1 byte
    bad_module = result.hir_planned.with_ops([bad_op])
    with pytest.raises(CompilerError, match="stride_h"):
        emit_program(bad_module, target)


def test_emit_rejects_unknown_param(target, tmp_path):
    result = _compile(target, tmp_path)
    op = result.hir_planned.ops[0]
    bad_op = op.replace(params={**op.params, "mystery_param": 1})
    bad_module = result.hir_planned.with_ops([bad_op])
    with pytest.raises(CapabilityError, match="mystery_param"):
        emit_program(bad_module, target)


def test_emit_rejects_missing_program_buffer(target, tmp_path):
    result = _compile(target, tmp_path)
    no_program = result.hir_planned.replace(program=None)
    with pytest.raises(CompilerError, match="program"):
        emit_program(no_program, target)


# ---------------------------------------------------------------------------
# run_program error paths
# ---------------------------------------------------------------------------


def test_run_program_rejects_missing_input(target, tmp_path):
    result = _compile(target, tmp_path)
    with pytest.raises(CompilerError, match="missing input"):
        run_program(result.program, {})


def test_run_program_rejects_wrong_shape(target, tmp_path):
    result = _compile(target, tmp_path)
    bad_input = np.zeros((1, 4, 4, 4), dtype=np.int8)
    with pytest.raises(CompilerError, match="shape"):
        run_program(result.program, {"arg0": bad_input})
