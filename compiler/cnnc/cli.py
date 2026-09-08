"""`cnnc` command-line interface (doc/tosa_compiler_plan.md §11, §13 M8):

    python -m cnnc compile <x.mlir> --target cnn_accel_v1 --out <dir> [--dump-after-all]
    python -m cnnc run <out_dir> --input <name>=<file.npy> [--input ...] --output <dir> [--accel-root <path>]
    python -m cnnc vectors <x.mlir> --target cnn_accel_v1 --out <dir> [--seed N] [--max-out-channels N] [--case-prefix P]

`compile` drives `cnnc.driver.compile_tosa`; `run` reloads the
`manifest.json`/`program.bin`/`constants.bin` a prior `compile` wrote and
executes them with `cnnc.backend.cnn_accel_v1.run_program`. `vectors`
compiles `<x.mlir>` in memory (no `program.bin`/etc. written) and calls
`cnnc.backend.cnn_accel_v1.write_conv_core_vectors` with seeded random
int8 input(s), writing one `tb_cnn_accel_conv_core`-shaped case directory
per `conv_layer` op (doc/tosa_compiler_plan.md M10). Every typed
`CompilerError` (and target-loading `TargetError`) is reported as a single
``error: ...`` line on stderr with exit code 1; success prints a one-line
summary and exits 0.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from cnnc.backend.cnn_accel_v1 import Program, run_program, write_conv_core_vectors
from cnnc.driver import compile_tosa
from cnnc.errors import CompilerError
from cnnc.target.contract import TargetError
from cnnc.target.load import load_target


def _parse_input_arg(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise CompilerError(f"--input must be NAME=FILE.npy, got {raw!r}", stage="cli")
    name, _, path = raw.partition("=")
    if not name:
        raise CompilerError(f"--input must be NAME=FILE.npy, got {raw!r}", stage="cli")
    return name, Path(path)


def _cmd_compile(args: argparse.Namespace) -> int:
    target = load_target(args.target)
    result = compile_tosa(args.mlir, target, out_dir=args.out, dump_after_all=args.dump_after_all)
    n_ops = len(result.hir_planned.ops)
    print(
        f"compiled {args.mlir} -> {args.out} "
        f"({n_ops} op(s), {len(result.program.program_bytes)} program byte(s), "
        f"{len(result.program.constants_bytes)} constants byte(s))"
    )
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    for name in ("manifest.json", "program.bin", "constants.bin"):
        if not (out_dir / name).is_file():
            raise CompilerError(f"{name} not found under {out_dir} (not a 'cnnc compile' output dir?)", stage="cli")
    manifest = json.loads((out_dir / "manifest.json").read_text())
    program_bytes = (out_dir / "program.bin").read_bytes()
    constants_bytes = (out_dir / "constants.bin").read_bytes()
    program = Program(
        program_bytes=program_bytes, constants_bytes=constants_bytes, manifest=manifest, descriptors=()
    )

    inputs: dict[str, np.ndarray] = {}
    for raw in args.input:
        name, path = _parse_input_arg(raw)
        if not path.is_file():
            raise CompilerError(f"--input {raw!r}: file not found: {path}", stage="cli")
        inputs[name] = np.load(path)

    outputs = run_program(program, inputs, accel_root=args.accel_root)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, arr in outputs.items():
        np.save(output_dir / f"{name}.npy", arr)

    summary = ", ".join(f"{name}: shape={arr.shape} dtype={arr.dtype}" for name, arr in sorted(outputs.items()))
    print(f"ran {out_dir} -> {output_dir} ({summary})")
    return 0


def _cmd_vectors(args: argparse.Namespace) -> int:
    target = load_target(args.target)
    result = compile_tosa(args.mlir, target)

    rng = np.random.default_rng(args.seed)
    input_bufs = sorted(
        (b for b in result.program.manifest["buffers"] if b["role"] == "input"),
        key=lambda b: b["gir_tensor"],
    )
    inputs = {
        b["gir_tensor"]: rng.integers(-128, 128, size=tuple(b["shape"])).astype(np.int8) for b in input_bufs
    }

    vectors = write_conv_core_vectors(
        result.program,
        inputs,
        args.out,
        target=target,
        graph=result.fused_graph,
        max_out_channels=args.max_out_channels,
        case_prefix=args.case_prefix,
    )

    print(f"wrote {len(vectors.written)} case(s) to {args.out}: {', '.join(vectors.written)}")
    if vectors.skipped:
        print(f"skipped {len(vectors.skipped)} case(s):")
        for name, reason in vectors.skipped:
            print(f"  {name}: {reason}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cnnc")
    sub = parser.add_subparsers(dest="command", required=True)

    p_compile = sub.add_parser("compile", help="compile a TOSA .mlir file to a cnn_accel_v1 program")
    p_compile.add_argument("mlir", help="path to a TOSA generic-form .mlir file")
    p_compile.add_argument("--target", required=True, help="target name (e.g. cnn_accel_v1) or path to a target .json")
    p_compile.add_argument("--out", required=True, help="output directory (program.bin/constants.bin/manifest.json)")
    p_compile.add_argument("--dump-after-all", action="store_true", help="write NN_<stage>.txt/.json dumps to --out")
    p_compile.set_defaults(func=_cmd_compile)

    p_run = sub.add_parser("run", help="run a compiled program against the cnn_accel_model golden model")
    p_run.add_argument("out_dir", help="a directory previously written by 'cnnc compile'")
    p_run.add_argument(
        "--input", action="append", default=[], metavar="NAME=FILE.npy",
        help="graph input tensor id (Buffer.gir_tensor) mapped to a .npy file; repeatable",
    )
    p_run.add_argument("--output", required=True, help="directory to write output <name>.npy files into")
    p_run.add_argument("--accel-root", default=None, help="override the accelerator source root recorded in the manifest")
    p_run.set_defaults(func=_cmd_run)

    p_vectors = sub.add_parser(
        "vectors", help="write tb_cnn_accel_conv_core-shaped per-layer test vectors from a compiled TOSA program"
    )
    p_vectors.add_argument("mlir", help="path to a TOSA generic-form .mlir file")
    p_vectors.add_argument("--target", required=True, help="target name (e.g. cnn_accel_v1) or path to a target .json")
    p_vectors.add_argument("--out", required=True, help="output directory (one case subdirectory per conv_layer op, plus cases.txt)")
    p_vectors.add_argument(
        "--seed", type=int, default=0,
        help="np.random.default_rng(seed).integers(-128, 128, ...) seed for the graph's own entry input(s)",
    )
    p_vectors.add_argument(
        "--max-out-channels", type=int, default=None,
        help="skip (not write) any conv_layer op whose out_channels exceeds this (conv_core single-output-tile limitation)",
    )
    p_vectors.add_argument("--case-prefix", default="", help="prefix prepended to every case directory name")
    p_vectors.set_defaults(func=_cmd_vectors)

    return parser


def main(argv: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (CompilerError, TargetError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
