"""Execute an emitted `cnn_accel_v1` `Program` against the RTL golden model
(doc/tosa_compiler_plan.md §10 item 2, §11 row 08, §13 M8).

This is the *only* module in `backend/cnn_accel_v1` that imports
`cnn_accel_model`: `emit.py`/`decode.py` derive every byte offset/width
from `target.isa` so they never need it, but actually *running* a program
needs the model's numeric semantics (`run_program`, `build_memory_image`),
not just its byte layout. The model (and the `cnn_accel_constants` it
imports) is loaded fresh from the path recorded in the manifest's
`target.provenance.accel_root` (or an explicit `accel_root` override) and
checked against `target.provenance.files`'s sha256 hashes first, so a
stale/hand-edited accelerator checkout fails loudly instead of silently
producing a result from the wrong golden model.

`prepare_memory_image`/`read_activation_buffer` are factored out of
`run_program` so `vectors.write_conv_core_vectors` (M10) can read every
per-layer activation buffer's content out of the same memory image,
built with the exact same provenance/shape checks, instead of running
the model a second time or duplicating any of this module's logic.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

from cnnc.errors import CompilerError
from cnnc.lower.layout import pack_activation_planes, unpack_activation_planes

from .emit import Program

_STAGE = "run"
_CONSTANTS_MODULE_NAME = "cnn_accel_constants"
_MODEL_MODULE_NAME = "cnn_accel_model"
_MODEL_FILES = ("cnn_accel_constants.py", "cnn_accel_model.py")

_NP_DTYPE = {"i8": np.int8, "i32": np.int32}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _import_fresh(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise CompilerError(f"could not build an import spec for {path}", stage=_STAGE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(name, None)
        raise CompilerError(f"failed importing {path}: {exc}", stage=_STAGE) from exc
    return module


def _load_model(accel_root: Path, provenance_files: dict) -> types.ModuleType:
    """Import `cnn_accel_model.py` (and the `cnn_accel_constants.py` it
    depends on) fresh from `accel_root`, verifying each file's sha256
    against the manifest's recorded provenance first. `cnn_accel_model.py`
    does `from cnn_accel_constants import ...`; registering the constants
    module under `sys.modules["cnn_accel_constants"]` *before* importing
    the model (rather than mutating `sys.path`) is what makes that
    resolve to the exact file just hashed, mirroring
    `target.discover._import_fresh`."""
    for filename in _MODEL_FILES:
        path = accel_root / filename
        if not path.is_file():
            raise CompilerError(f"{filename} not found under {accel_root}", stage=_STAGE)
        expected = provenance_files.get(filename)
        actual = _sha256(path)
        if expected is not None and actual != expected:
            raise CompilerError(
                f"{filename} sha256 mismatch at {path}: manifest provenance expects "
                f"{expected}, found {actual} (stale/hand-edited accelerator checkout?)",
                stage=_STAGE,
            )

    saved = {name: sys.modules.get(name) for name in (_CONSTANTS_MODULE_NAME, _MODEL_MODULE_NAME)}
    try:
        _import_fresh(_CONSTANTS_MODULE_NAME, accel_root / "cnn_accel_constants.py")
        return _import_fresh(_MODEL_MODULE_NAME, accel_root / "cnn_accel_model.py")
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _buffers_by_role(manifest: dict, role: str) -> list[dict]:
    return [b for b in manifest["buffers"] if b["role"] == role]


def _encode_input(name: str, arr: np.ndarray, buf: dict, plane_channels: int) -> bytes:
    dtype = _NP_DTYPE.get(buf["dtype"])
    if dtype is None:
        raise CompilerError(f"input {name!r}: unsupported buffer dtype {buf['dtype']!r}", stage=_STAGE)
    arr = np.asarray(arr)
    expected_shape = tuple(buf["shape"])
    if arr.shape != expected_shape:
        raise CompilerError(
            f"input {name!r}: shape {arr.shape} != expected {expected_shape}", stage=_STAGE
        )
    if arr.dtype != dtype:
        raise CompilerError(
            f"input {name!r}: dtype {arr.dtype} != expected {np.dtype(dtype)}", stage=_STAGE
        )
    if buf["layout"] != "PLANES":
        raise CompilerError(f"input {name!r}: unsupported activation layout {buf['layout']!r}", stage=_STAGE)
    data = pack_activation_planes(arr, plane_channels)
    if len(data) != buf["size_bytes"]:
        raise CompilerError(
            f"input {name!r}: encoded size {len(data)} != buffer size_bytes {buf['size_bytes']}",
            stage=_STAGE,
        )
    return data


def read_activation_buffer(memory: bytearray, buf: dict, plane_channels: int) -> np.ndarray:
    """Decode one `PLANES`-layout activation buffer's current contents out
    of `memory`. Works for ANY buffer role (`input`/`intermediate`/
    `output`) -- `run_program` below only ever calls this for
    role='output' buffers, but `vectors.write_conv_core_vectors` needs
    every activation buffer's contents (every layer's input AND output,
    not just the graph's own entry/exit), which is why this is a plain
    function of `buf` rather than something baked into `run_program`'s
    loop."""
    dtype = _NP_DTYPE.get(buf["dtype"])
    if dtype is None:
        raise CompilerError(f"buffer {buf['id']!r}: unsupported buffer dtype {buf['dtype']!r}", stage=_STAGE)
    if buf["layout"] != "PLANES":
        raise CompilerError(f"buffer {buf['id']!r}: unsupported activation layout {buf['layout']!r}", stage=_STAGE)
    addr, size = buf["addr"], buf["size_bytes"]
    raw = bytes(memory[addr : addr + size])
    return unpack_activation_planes(raw, tuple(buf["shape"]), dtype, plane_channels)


def prepare_memory_image(
    program: Program,
    inputs: dict,
    *,
    accel_root: str | Path | None = None,
) -> tuple[types.ModuleType, dict, bytearray, int]:
    """Shared setup for `run_program` and `vectors.write_conv_core_vectors`
    (the latter needs every activation buffer's contents, not just the
    graph's own outputs, so it cannot just call `run_program`): validate
    the manifest, load+provenance-check the golden model (`_load_model`),
    and build the DDR memory image with the program, every const buffer
    and every encoded input already written in. Returns `(model,
    manifest, memory, program_addr)`; the caller still has to call
    `model.run_program(memory, program_addr, ...)` themselves -- this
    function never executes anything, so both callers share exactly the
    same provenance/sha256/shape-checking logic up to (but not including)
    execution."""
    manifest = program.manifest
    if manifest.get("format_version") != 1:
        raise CompilerError(f"unsupported manifest format_version {manifest.get('format_version')!r}", stage=_STAGE)

    provenance = manifest["target"]["provenance"]
    root = Path(accel_root) if accel_root is not None else Path(provenance["accel_root"])
    model = _load_model(root, provenance.get("files", {}))

    program_info = manifest["memory"]["program"]
    plane_channels = manifest["memory"]["activation_plane_channels"]
    if len(program.program_bytes) != program_info["size_bytes"]:
        raise CompilerError(
            f"program_bytes length {len(program.program_bytes)} != manifest size_bytes "
            f"{program_info['size_bytes']}",
            stage=_STAGE,
        )

    chunks: dict[int, bytes] = {program_info["addr"]: program.program_bytes}

    for buf in _buffers_by_role(manifest, "const"):
        offset = buf["constants_offset"]
        data = program.constants_bytes[offset : offset + buf["size_bytes"]]
        if len(data) != buf["size_bytes"]:
            raise CompilerError(
                f"const buffer {buf['id']!r}: constants_bytes slice too short "
                f"({len(data)} < {buf['size_bytes']})",
                stage=_STAGE,
            )
        chunks[buf["addr"]] = data

    input_bufs = {b["gir_tensor"]: b for b in _buffers_by_role(manifest, "input")}
    missing = sorted(set(input_bufs) - set(inputs))
    extra = sorted(set(inputs) - set(input_bufs))
    if missing:
        raise CompilerError(f"missing input(s): {missing}", stage=_STAGE)
    if extra:
        raise CompilerError(f"unexpected input(s) (not a graph entry input): {extra}", stage=_STAGE)
    for name, buf in input_bufs.items():
        chunks[buf["addr"]] = _encode_input(name, inputs[name], buf, plane_channels)

    memory = model.build_memory_image(chunks, size=manifest["memory"]["size_bytes"])
    return model, manifest, memory, program_info["addr"]


def run_program(
    program: Program,
    inputs: dict,
    *,
    accel_root: str | Path | None = None,
    max_instructions: int = 1000,
) -> dict:
    """Run `program` (from `emit.emit_program`, or reconstructed from a
    saved manifest/`program.bin`/`constants.bin`) against
    `cnn_accel_model.run_program`. `inputs`/the return value are keyed by
    GIR tensor id (`Buffer.gir_tensor`), matching `gir.interp.run` and
    `testing.iree_oracle.run_iree`, so all three oracles are directly
    comparable (doc/tosa_compiler_plan.md §10)."""
    model, manifest, memory, program_addr = prepare_memory_image(program, inputs, accel_root=accel_root)
    model.run_program(memory, program_addr, max_instructions=max_instructions)

    plane_channels = manifest["memory"]["activation_plane_channels"]
    outputs = {}
    for buf in _buffers_by_role(manifest, "output"):
        outputs[buf["gir_tensor"]] = read_activation_buffer(memory, buf, plane_channels)
    return outputs
