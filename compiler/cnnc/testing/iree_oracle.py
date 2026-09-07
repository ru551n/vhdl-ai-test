"""External oracle: compile+run TOSA `.mlir` with IREE (doc/tosa_compiler_plan.md
§10 item 3). `numpy` is used here and in `cnnc.gir.interp` only.

Compiler flags: `iree-compile` in this environment accepts the newer
`--iree-hal-target-device=local --iree-hal-local-target-device-backends=llvm-cpu`
pair (verified by actually running it against `conv_rescale_clamp.mlir`); the
older `--iree-hal-target-backends=llvm-cpu` also works in this install and is
kept as a fallback for other IREE versions. `iree-run-module` takes
`--device=local-task`, `--module=<vmfb>`, `--function=<name>`, and
`--input=@x.npy` / `--output=@y.npy` (verified against `iree-run-module
--help` and by actually running it); signed int8 tensors round-trip through
`numpy.save`/`numpy.load` as `dtype=int8` with no extra handling needed.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

_PRIMARY_COMPILE_FLAGS = (
    "--iree-hal-target-device=local",
    "--iree-hal-local-target-device-backends=llvm-cpu",
)
_FALLBACK_COMPILE_FLAGS = ("--iree-hal-target-backends=llvm-cpu",)


def _tool_path(name: str) -> Path:
    return Path(sys.executable).parent / name


def iree_available() -> bool:
    """`True` iff `iree-compile` and `iree-run-module` sit next to the
    running interpreter (i.e. `.venv-compiler/bin/`)     and are executable."""
    return all(os.access(_tool_path(name), os.X_OK) for name in ("iree-compile", "iree-run-module"))


def _count_return_operands(mlir_text: str) -> int:
    """Number of `func.return` operands (i.e. graph outputs); return-value
    SSA operands are plain `%name` references, never nested parens, so a
    plain split on the operand list is sufficient for the generic-form
    fixtures this compiler consumes."""
    match = re.search(r'"func\.return"\(([^)]*)\)', mlir_text)
    if match is None:
        raise ValueError("no 'func.return' found in mlir text")
    operands = [o for o in match.group(1).split(",") if o.strip()]
    return len(operands)


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def _compile(mlir_path: Path, vmfb_path: Path) -> None:
    iree_compile = str(_tool_path("iree-compile"))
    primary = [iree_compile, str(mlir_path), "-o", str(vmfb_path), *_PRIMARY_COMPILE_FLAGS]
    result = _run(primary)
    if result.returncode == 0 and vmfb_path.exists():
        return
    fallback = [iree_compile, str(mlir_path), "-o", str(vmfb_path), *_FALLBACK_COMPILE_FLAGS]
    fallback_result = _run(fallback)
    if fallback_result.returncode == 0 and vmfb_path.exists():
        return
    raise RuntimeError(
        "iree-compile failed with both flag sets:\n"
        f"primary ({' '.join(primary)}):\n{result.stderr}\n"
        f"fallback ({' '.join(fallback)}):\n{fallback_result.stderr}"
    )


def run_iree(mlir_path: Path, inputs: list[np.ndarray], work_dir: Path, function: str = "main") -> list[np.ndarray]:
    """Compile `mlir_path` with `iree-compile` and execute it with
    `iree-run-module`, feeding `inputs` (written as signed-`int8` `.npy`
    files) and returning the outputs (read back with `numpy.load`)."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    mlir_path = Path(mlir_path)

    vmfb_path = work_dir / "out.vmfb"
    _compile(mlir_path, vmfb_path)

    input_paths = []
    for i, arr in enumerate(inputs):
        path = work_dir / f"input{i}.npy"
        np.save(path, np.asarray(arr, dtype=np.int8))
        input_paths.append(path)

    num_outputs = _count_return_operands(mlir_path.read_text())
    output_paths = [work_dir / f"out{i}.npy" for i in range(num_outputs)]

    iree_run_module = str(_tool_path("iree-run-module"))
    cmd = [
        iree_run_module,
        "--device=local-task",
        f"--module={vmfb_path}",
        f"--function={function}",
        *(f"--input=@{p}" for p in input_paths),
        *(f"--output=@{p}" for p in output_paths),
    ]
    result = _run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"iree-run-module failed ({' '.join(cmd)}):\n{result.stderr}")

    return [np.load(p) for p in output_paths]
