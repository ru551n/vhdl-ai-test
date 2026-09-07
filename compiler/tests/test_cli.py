"""M8 tests: `python -m cnnc compile`/`run` CLI smoke tests
(doc/tosa_compiler_plan.md §11, §13 M8: "CLI smoke test producing all 9
dumps"). Runs the real `python -m cnnc ...` entry point in a subprocess
(not just `cnnc.cli.main` in-process) so it also exercises `__main__.py`
and process-level argument parsing/exit codes."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from cnnc.gir import interp
from cnnc.frontend.tosa_import import load_tosa_file
from cnnc.testing.accel_variants import load_half_up_target

COMPILER_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = Path(__file__).parent / "fixtures" / "conv_rescale_clamp.mlir"

_DUMP_STEMS = (
    "00_mlir.txt",
    "01_gir.txt", "01_gir.json",
    "02_normalize.txt", "02_normalize.json",
    "03_legalize_rescale.txt", "03_legalize_rescale.json",
    "04_fuse.txt", "04_fuse.json",
    "05_hir.txt", "05_hir.json",
    "06_sched.txt", "06_sched.json",
    "07_memplan.txt", "07_memplan.json",
    "08_program.txt",
)
_ARTIFACTS = ("program.bin", "constants.bin", "manifest.json")


def _env() -> dict:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(COMPILER_ROOT) if not existing else f"{COMPILER_ROOT}{os.pathsep}{existing}"
    return env


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "cnnc", *args], capture_output=True, text=True, env=_env(),
    )


@pytest.fixture
def half_up_target_json(tmp_path):
    target = load_half_up_target(tmp_path / "accel")
    path = tmp_path / "target.json"
    path.write_text(target.to_json())
    return path


def test_cli_compile_writes_all_dumps_and_artifacts(half_up_target_json, tmp_path):
    out_dir = tmp_path / "out"
    result = _run_cli(
        "compile", str(FIXTURE_PATH), "--target", str(half_up_target_json), "--out", str(out_dir),
        "--dump-after-all",
    )
    assert result.returncode == 0, result.stderr
    assert "compiled" in result.stdout

    names = {p.name for p in out_dir.iterdir()}
    for stem in _DUMP_STEMS:
        assert stem in names, f"missing dump {stem}; got {sorted(names)}"
    for artifact in _ARTIFACTS:
        assert artifact in names, f"missing artifact {artifact}; got {sorted(names)}"


def test_cli_compile_without_dump_after_all_only_writes_artifacts(half_up_target_json, tmp_path):
    out_dir = tmp_path / "out"
    result = _run_cli("compile", str(FIXTURE_PATH), "--target", str(half_up_target_json), "--out", str(out_dir))
    assert result.returncode == 0, result.stderr

    names = {p.name for p in out_dir.iterdir()}
    assert names == set(_ARTIFACTS)


def test_cli_run_matches_interp(half_up_target_json, tmp_path):
    out_dir = tmp_path / "out"
    compile_result = _run_cli(
        "compile", str(FIXTURE_PATH), "--target", str(half_up_target_json), "--out", str(out_dir)
    )
    assert compile_result.returncode == 0, compile_result.stderr

    rng = np.random.default_rng(0)
    x = rng.integers(-128, 128, size=(1, 8, 8, 4)).astype(np.int8)
    input_path = tmp_path / "arg0.npy"
    np.save(input_path, x)

    run_out_dir = tmp_path / "run_out"
    run_result = _run_cli(
        "run", str(out_dir), "--input", f"arg0={input_path}", "--output", str(run_out_dir),
    )
    assert run_result.returncode == 0, run_result.stderr
    assert "ran" in run_result.stdout

    output_path = run_out_dir / "10.npy"
    assert output_path.is_file()
    cli_out = np.load(output_path)

    graph = load_tosa_file(FIXTURE_PATH)
    interp_out = interp.run(graph, {"arg0": x})["10"]
    np.testing.assert_array_equal(cli_out, interp_out)


def test_cli_compile_missing_file_reports_error(half_up_target_json, tmp_path):
    result = _run_cli(
        "compile", str(tmp_path / "does_not_exist.mlir"), "--target", str(half_up_target_json),
        "--out", str(tmp_path / "out"),
    )
    assert result.returncode == 1
    assert "error:" in result.stderr


def test_cli_run_missing_out_dir_reports_error(tmp_path):
    result = _run_cli("run", str(tmp_path / "does_not_exist"), "--output", str(tmp_path / "run_out"))
    assert result.returncode == 1
    assert "error:" in result.stderr


def test_cli_target_dump_still_works():
    result = subprocess.run(
        [sys.executable, "-m", "cnnc.target", "dump", "cnn_accel"],
        capture_output=True, text=True, env=_env(),
    )
    assert result.returncode == 0, result.stderr
    assert '"name": "cnn_accel_v1"' in result.stdout
