"""M10 tests: `cnnc.backend.cnn_accel_v1.vectors.write_conv_core_vectors`
(doc/tosa_compiler_plan.md M10, ~line 613).

ACCEPTANCE (M10): a `tb_cnn_accel_conv_core`-shaped case directory per
`conv_layer` op, `desc.txt` parsing back to the real `cnn_accel_model.
LayerDesc` field-for-field, `weights_packed.txt`/`bias.txt` matching
`cnn_accel_model.pack_weights_for_hw`/the GIR bias const exactly,
`input.txt`/`expected.txt` matching the seeded input / `gir.interp.run`
output, and `two_layer`'s layer-1 `expected.txt` == layer-2 `input.txt`
(the two cases really are consecutive layers of one program, not two
independent shapes).
"""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from cnnc.backend.cnn_accel_v1 import decode_program, write_conv_core_vectors
from cnnc.driver import compile_tosa
from cnnc.gir import interp

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _load_module_from(root: Path, name: str):
    """`importlib`-fresh-load one accelerator module by name, mirroring
    `test_backend.py`'s identical helper."""
    spec = importlib.util.spec_from_file_location(name, root / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _seed_input(shape: tuple[int, ...], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(-128, 128, size=shape).astype(np.int8)


def _read_int_lines(path: Path) -> list[int]:
    return [int(line) for line in path.read_text().splitlines() if line]


# ---------------------------------------------------------------------------
# Single-layer fixtures: one case, checked field-for-field / byte-for-byte
# against the golden model and gir.interp.run.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,shape",
    [
        ("conv_rescale_clamp", (1, 8, 8, 4)),
        ("first_layer_cin3", (1, 8, 8, 3)),
    ],
)
def test_single_layer_case_matches_golden_model(name, shape, target, tmp_path):
    result = compile_tosa(FIXTURES_DIR / f"{name}.mlir", target)
    x = _seed_input(shape, seed=0)
    inputs = {"arg0": x}

    out_dir = tmp_path / "vectors"
    vectors = write_conv_core_vectors(result.program, inputs, out_dir, target=target, graph=result.fused_graph)

    assert vectors.written == ["op0"]
    assert vectors.skipped == []
    assert (out_dir / "cases.txt").read_text() == "op0\n"

    case_dir = out_dir / "op0"
    for filename in ("desc.txt", "weights_packed.txt", "bias.txt", "input.txt", "expected.txt"):
        assert (case_dir / filename).is_file(), f"missing {filename}"

    accel_root = Path(target.provenance["accel_root"])
    _load_module_from(accel_root, "cnn_accel_constants")
    model = _load_module_from(accel_root, "cnn_accel_model")

    desc_fields: dict[str, int] = {}
    for line in (case_dir / "desc.txt").read_text().splitlines():
        key, _, value = line.partition(" ")
        desc_fields[key] = int(value)

    layer_desc_field_names = [f.name for f in dataclasses.fields(model.LayerDesc)]
    # desc.txt has every LayerDesc field, plus exactly the two D10 extras.
    assert set(desc_fields) == set(layer_desc_field_names) | {"tile_channels", "pe_rows"}

    written_desc = model.LayerDesc(**{k: desc_fields[k] for k in layer_desc_field_names})

    program_addr = result.hir_planned.buffer(result.hir_planned.program).addr
    decoded = decode_program(result.program.program_bytes, target, program_addr=program_addr)[0]
    for field_name in layer_desc_field_names:
        assert getattr(written_desc, field_name) == getattr(decoded, field_name), field_name

    unit_name = result.program.manifest["ops"][0]["unit"]
    tiling = target.unit(unit_name).internal_tiling
    assert desc_fields["tile_channels"] == tiling.cin
    assert desc_fields["pe_rows"] == tiling.cout

    conv_op = next(op for op in result.fused_graph.ops if op.kind == "fused_conv")
    _, weight_tensor_id, bias_tensor_id = conv_op.inputs
    logical_weights = list(result.fused_graph.tensor(weight_tensor_id).values)
    logical_bias = list(result.fused_graph.tensor(bias_tensor_id).values)

    expected_packed = model.pack_weights_for_hw(
        logical_weights, written_desc, desc_fields["tile_channels"], desc_fields["pe_rows"]
    )
    assert _read_int_lines(case_dir / "weights_packed.txt") == expected_packed
    assert _read_int_lines(case_dir / "bias.txt") == logical_bias
    assert _read_int_lines(case_dir / "input.txt") == x.reshape(-1).tolist()

    out_id = result.fused_graph.outputs[0]
    interp_out = interp.run(result.fused_graph, inputs)[out_id]
    assert _read_int_lines(case_dir / "expected.txt") == interp_out.reshape(-1).tolist()


# ---------------------------------------------------------------------------
# two_layer: max_out_channels skips both (out_channels=32 > 8); without the
# limit, both are written and chain (layer 1's output == layer 2's input).
# ---------------------------------------------------------------------------


def test_two_layer_max_out_channels_skips_both(target, tmp_path):
    result = compile_tosa(FIXTURES_DIR / "two_layer.mlir", target)
    inputs = {"arg0": _seed_input((1, 8, 8, 16), seed=0)}

    out_dir = tmp_path / "vectors"
    vectors = write_conv_core_vectors(
        result.program, inputs, out_dir, target=target, graph=result.fused_graph, max_out_channels=8,
    )

    assert vectors.written == []
    assert [name for name, _ in vectors.skipped] == ["op0", "op1"]
    assert all("out_channels" in reason for _, reason in vectors.skipped)
    assert (out_dir / "cases.txt").read_text() == ""
    assert not (out_dir / "op0").exists()
    assert not (out_dir / "op1").exists()


def test_two_layer_without_limit_writes_two_chained_cases(target, tmp_path):
    result = compile_tosa(FIXTURES_DIR / "two_layer.mlir", target)
    inputs = {"arg0": _seed_input((1, 8, 8, 16), seed=0)}

    out_dir = tmp_path / "vectors"
    vectors = write_conv_core_vectors(result.program, inputs, out_dir, target=target, graph=result.fused_graph)

    assert vectors.written == ["op0", "op1"]
    assert vectors.skipped == []
    assert (out_dir / "cases.txt").read_text() == "op0\nop1\n"

    layer1_expected = _read_int_lines(out_dir / "op0" / "expected.txt")
    layer2_input = _read_int_lines(out_dir / "op1" / "input.txt")
    assert layer1_expected == layer2_input


# ---------------------------------------------------------------------------
# case_prefix.
# ---------------------------------------------------------------------------


def test_case_prefix_is_prepended(target, tmp_path):
    result = compile_tosa(FIXTURES_DIR / "conv_rescale_clamp.mlir", target)
    inputs = {"arg0": _seed_input((1, 8, 8, 4), seed=0)}

    out_dir = tmp_path / "vectors"
    vectors = write_conv_core_vectors(
        result.program, inputs, out_dir, target=target, graph=result.fused_graph, case_prefix="crc_",
    )

    assert vectors.written == ["crc_op0"]
    assert (out_dir / "cases.txt").read_text() == "crc_op0\n"
    assert (out_dir / "crc_op0" / "desc.txt").is_file()
