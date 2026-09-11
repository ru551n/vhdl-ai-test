"""End-to-end TOSA -> `cnn_accel_v1` compile driver (doc/tosa_compiler_plan.md
§11 table, §13 M8).

`compile_tosa` runs every stage in order -- parse, import,
depth_to_space_pad, depth_to_space_channels, normalize,
legalize_rescale, fuse, to_hir,
schedule, plan_memory, emit -- and, when `dump_after_all=True`, writes the
numbered `NN_<stage>.txt`/`.json` dumps §11 specifies (`00_mlir` ..
`10_program`; `02`-`06` are written by `passes.framework.run_pipeline`
itself, keyed by each pass's own `.name`, so the post-pass indices move
whenever the pipeline gains or loses a pass -- as they did when
`PermuteDepthToSpaceChannelsPass` and then
`PadDepthToSpaceChannelsPass` were added).
`program.bin`/`constants.bin`/`manifest.json` are written to `out_dir`
whenever one is given, independent of `dump_after_all` -- those are the
compiler's actual output artifacts, not debug dumps.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from cnnc.backend.cnn_accel_v1 import Program, decode_program, emit_program, print_program
from cnnc.errors import CompilerError
from cnnc.frontend.mlir_generic import MlirModule, parse_file, print_generic
from cnnc.frontend.tosa_import import import_tosa
from cnnc.gir.ir import Graph
from cnnc.gir.printer import print_gir
from cnnc.gir.printer import to_json as gir_to_json
from cnnc.hir.ir import HirModule
from cnnc.hir.printer import print_hir
from cnnc.hir.printer import to_json as hir_to_json
from cnnc.lower.memplan import plan_memory
from cnnc.lower.schedule import schedule
from cnnc.lower.to_hir import to_hir
from cnnc.passes import PassContext, default_pipeline, run_pipeline
from cnnc.target.contract import Target

_STAGE = "driver"


@dataclasses.dataclass(frozen=True)
class CompileResult:
    """Every stage's output, so tests/CLI callers can inspect intermediate
    state without re-running the pipeline."""

    mlir_module: MlirModule
    imported_graph: Graph
    fused_graph: Graph
    hir_mapped: HirModule
    hir_scheduled: HirModule
    hir_planned: HirModule
    program: Program
    target: Target
    dump_dir: Path | None


def _write_text(path: Path, text: str) -> None:
    path.write_text(text)


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def compile_tosa(
    mlir_path: str | Path,
    target: Target,
    *,
    out_dir: str | Path | None = None,
    dump_after_all: bool = False,
) -> CompileResult:
    """Compile one TOSA generic-form `.mlir` file to a `cnn_accel_v1`
    `Program`. `out_dir`, when given, always receives `program.bin`,
    `constants.bin` and `manifest.json`; `dump_after_all` additionally
    writes the `NN_<stage>` dumps (requires `out_dir`)."""
    if dump_after_all and out_dir is None:
        raise CompilerError("dump_after_all=True requires an out_dir", stage=_STAGE)

    out_path = Path(out_dir) if out_dir is not None else None
    if out_path is not None:
        out_path.mkdir(parents=True, exist_ok=True)
    dump_dir = out_path if dump_after_all else None

    mlir_source = Path(mlir_path)
    if not mlir_source.is_file():
        raise CompilerError(f"mlir file not found: {mlir_source}", stage=_STAGE)
    mlir_module = parse_file(mlir_source)
    if dump_dir is not None:
        _write_text(dump_dir / "00_mlir.txt", print_generic(mlir_module))

    imported_graph = import_tosa(mlir_module)
    if dump_dir is not None:
        _write_text(dump_dir / "01_gir.txt", print_gir(imported_graph))
        _write_json(dump_dir / "01_gir.json", gir_to_json(imported_graph))

    ctx = PassContext(target=target, dump_dir=dump_dir)
    fused_graph = run_pipeline(imported_graph, default_pipeline(target), ctx, first_index=2)

    hir_mapped = to_hir(fused_graph, target)
    if dump_dir is not None:
        _write_text(dump_dir / "07_hir.txt", print_hir(hir_mapped))
        _write_json(dump_dir / "07_hir.json", hir_to_json(hir_mapped))

    hir_scheduled = schedule(hir_mapped)
    if dump_dir is not None:
        _write_text(dump_dir / "08_sched.txt", print_hir(hir_scheduled))
        _write_json(dump_dir / "08_sched.json", hir_to_json(hir_scheduled))

    hir_planned = plan_memory(hir_scheduled, target)
    if dump_dir is not None:
        _write_text(dump_dir / "09_memplan.txt", print_hir(hir_planned))
        _write_json(dump_dir / "09_memplan.json", hir_to_json(hir_planned))

    program = emit_program(hir_planned, target)
    if dump_dir is not None:
        program_addr = hir_planned.buffer(hir_planned.program).addr
        descriptors = decode_program(program.program_bytes, target, program_addr=program_addr)
        _write_text(dump_dir / "10_program.txt", print_program(descriptors, target, program_addr=program_addr))

    if out_path is not None:
        (out_path / "program.bin").write_bytes(program.program_bytes)
        (out_path / "constants.bin").write_bytes(program.constants_bytes)
        _write_json(out_path / "manifest.json", program.manifest)

    return CompileResult(
        mlir_module=mlir_module,
        imported_graph=imported_graph,
        fused_graph=fused_graph,
        hir_mapped=hir_mapped,
        hir_scheduled=hir_scheduled,
        hir_planned=hir_planned,
        program=program,
        target=target,
        dump_dir=dump_dir,
    )
