"""Per-layer stimulus/expected test-vector export for
`tb_cnn_accel_conv_core.vhd` (doc/tosa_compiler_plan.md M10, ~line 613).

`write_conv_core_vectors` writes one case directory per `conv_layer` op
in an emitted `Program`, in exactly the shape `run_case` in
`tb_cnn_accel_conv_core.vhd` reads (`desc.txt`/`weights_packed.txt`/
`bias.txt`/`input.txt`/`expected.txt`, doc/cnn_accel_test_vectors.md),
plus a `cases.txt` listing them in program order -- the same case-
directory contract `modules/cnn_accel/generate_vectors.py` already
produces from the golden model directly, now produced from a real
compiled program instead.

Every byte written here comes from the compiler's OWN emitted artifacts:
`weights_packed.txt`/`bias.txt` are sliced straight out of
`program.constants_bytes` (already tile-packed by
`lower.layout.pack_weights_tiled`/`pack_bias_tiled` at `to_hir` time --
see that module's docstring for why this is bit-identical to
`cnn_accel_model.pack_weights_for_hw`/`pack_bias_for_hw`, and
`tests/test_layout.py` for the pin), and `input.txt`/`expected.txt` come
from running the compiler's OWN emitted program against
`cnn_accel_model.run_program` (`backend.cnn_accel_v1.run.
prepare_memory_image`/`read_activation_buffer`, factored out of
`run_program` so this module never re-derives that provenance/shape
logic). `desc.txt`'s field list is read off the freshly-imported model's
own `LayerDesc` dataclass (`dataclasses.fields`), never hard-coded here,
so a `LayerDesc` field rename/reorder cannot silently desync this
module's output from what `tb_cnn_accel_conv_core.vhd` reads.

ISA v1.2 (H2): a descriptor with `PER_CHANNEL_EN` set additionally gets a
`scale_table_packed.txt` -- the padded `pack_scale_table_for_hw` image the
model read from DDR at `scale_addr`, decoded with the model's own
`unpack_scale_table_from_hw` into `generate_vectors._write_scale_table_
packed`'s two-records-per-entry format (doc/cnn_accel_test_vectors.md).

Before any of this is trusted enough to write to disk, the compiler's
two independent CPU-side oracles for `program`/`inputs` -- the emitted
`Program` run through `cnn_accel_model.run_program` and the un-lowered
`graph` run through `gir.interp.run` -- are cross-checked byte-exact;
a mismatch raises `CompilerError` rather than ever writing a vector a
disagreement could have produced (this is the same invariant
`test_backend.py`/`test_fixtures_m9.py` already check per-test, just
enforced here too since this module's caller may not).

Layers whose `out_channels` exceeds `max_out_channels` are recorded in
`.skipped`, not written: `cnn_accel_conv_core` (and this module's own
sibling `tb_cnn_accel_conv_core.vhd`) supports only a single output-
channel tile (`out_channels <= g_pe_rows`) -- layer-level output-channel
tiling (re-streaming the whole ifmap once per output-channel tile) is a
future milestone, see that testbench's own header comment and
`flow_status.md`.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from cnnc.errors import CompilerError
from cnnc.gir import interp

from .emit import Program
from .run import prepare_memory_image, read_activation_buffer, run_program

if TYPE_CHECKING:
    from cnnc.gir.ir import Graph
    from cnnc.target.contract import Target

_STAGE = "vectors"


@dataclasses.dataclass(frozen=True)
class VectorsResult:
    """`written`: case names actually written, in program order (the same
    order recorded in `cases.txt`). `skipped`: `(name, reason)` pairs for
    `conv_layer` ops that exceeded `max_out_channels` -- not written, and
    not listed in `cases.txt`."""

    written: list[str]
    skipped: list[tuple[str, str]]


def _write_int_lines(path: Path, values) -> None:
    path.write_text("".join(f"{int(v)}\n" for v in values))


def _sanitize_op_id(op_id: str) -> str:
    """`'#0'` -> `'op0'` -- HIR op ids are always `'#<index>'`
    (`lower.to_hir.to_hir`'s own `op_id = f"#{idx}"`); `'#'` is not a
    filesystem-friendly directory-name character, so this strips it and
    prefixes `'op'` instead of inventing a second, independent op-naming
    scheme."""
    return "op" + op_id.lstrip("#")


def write_conv_core_vectors(
    program: Program,
    inputs: dict[str, np.ndarray],
    out_dir: str | Path,
    *,
    target: "Target",
    graph: "Graph",
    accel_root: str | Path | None = None,
    max_out_channels: int | None = None,
    case_prefix: str = "",
) -> VectorsResult:
    """Write one `tb_cnn_accel_conv_core`-shaped case directory per
    `conv_layer` op in `program` (program order, from
    `program.manifest["ops"]`/`program.descriptors`) under `out_dir`,
    plus a `cases.txt` listing every case actually written. `graph` is
    the un-lowered GIR graph `program` was emitted from (e.g.
    `driver.compile_tosa(...).fused_graph`) -- used only for the
    interp-vs-model cross-check described in this module's docstring,
    never for byte content."""
    manifest = program.manifest
    if manifest.get("format_version") != 1:
        raise CompilerError(f"unsupported manifest format_version {manifest.get('format_version')!r}", stage=_STAGE)

    # Cross-oracle guard (module docstring): never write a vector a
    # model/interp disagreement could have produced.
    interp_out = interp.run(graph, inputs)
    run_out = run_program(program, inputs, accel_root=accel_root)
    for tensor_id, run_arr in run_out.items():
        interp_arr = interp_out.get(tensor_id)
        if interp_arr is None:
            raise CompilerError(
                f"program output {tensor_id!r} has no corresponding gir.interp.run output "
                "-- graph/program mismatch, refusing to write vectors",
                stage=_STAGE,
            )
        if run_arr.shape != interp_arr.shape or not np.array_equal(run_arr, interp_arr):
            raise CompilerError(
                f"cnn_accel_model.run_program and gir.interp.run disagree on output {tensor_id!r} "
                "-- refusing to write vectors from a model/interp disagreement",
                stage=_STAGE,
            )

    model, manifest, memory, program_addr = prepare_memory_image(program, inputs, accel_root=accel_root)
    model.run_program(memory, program_addr, max_instructions=1000)

    plane_channels = manifest["memory"]["activation_plane_channels"]
    buffers_by_addr = {buf["addr"]: buf for buf in manifest["buffers"]}
    layer_desc_fields = [f.name for f in dataclasses.fields(model.LayerDesc)]
    per_channel_bit = target.isa.flags.get("PER_CHANNEL_EN")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    skipped: list[tuple[str, str]] = []

    # `program.descriptors` has one more entry than `manifest["ops"]`
    # (the trailing HALT, appended by `emit_program` after every op's own
    # descriptor) -- `zip` stops at the shorter of the two, dropping HALT
    # for us. Both lists are built by the same `enumerate(ops_sorted)`
    # loop in `emit.emit_program`, so index i in one is index i in the
    # other by construction.
    for op_entry, desc in zip(manifest["ops"], program.descriptors):
        out_channels = op_entry["params"]["out_channels"]
        name = f"{case_prefix}{_sanitize_op_id(op_entry['id'])}"

        if max_out_channels is not None and out_channels > max_out_channels:
            skipped.append((
                name,
                f"out_channels={out_channels} > max_out_channels={max_out_channels}: "
                "cnn_accel_conv_core supports only a single output-channel tile "
                "(layer-level output-channel tiling is a future milestone)",
            ))
            continue

        unit = target.unit(op_entry["unit"])
        tile_channels = unit.internal_tiling.cin
        pe_rows = unit.internal_tiling.cout

        in_buf = buffers_by_addr[desc.in_addr]
        out_buf = buffers_by_addr[desc.out_addr]
        weight_buf = buffers_by_addr[desc.weight_addr]
        bias_buf = buffers_by_addr[desc.bias_addr]

        case_dir = out_path / name
        case_dir.mkdir(parents=True, exist_ok=True)

        desc_lines = [f"{field_name} {getattr(desc, field_name)}\n" for field_name in layer_desc_fields]
        # D10 (generate_vectors.py's own `_write_desc`): tile_channels/
        # pe_rows are host-compiler-time packing parameters, not
        # LayerDesc/ISA fields, appended after the standard fields.
        desc_lines.append(f"tile_channels {tile_channels}\n")
        desc_lines.append(f"pe_rows {pe_rows}\n")
        (case_dir / "desc.txt").write_text("".join(desc_lines))

        w_offset = weight_buf["constants_offset"]
        w_data = program.constants_bytes[w_offset : w_offset + weight_buf["size_bytes"]]
        _write_int_lines(case_dir / "weights_packed.txt", np.frombuffer(w_data, dtype=np.int8))

        # `bias.txt` is LOGICAL/unpadded (`out_channels` values, doc/
        # cnn_accel_test_vectors.md) -- `max_out_channels` guarantees a
        # single output-channel tile here, so the first `out_channels`
        # int32 LE values of the (possibly zero-padded) const buffer are
        # exactly the real ones (`lower.layout.pack_bias_tiled`'s own
        # zero-padding always comes after them within one tile).
        b_offset = bias_buf["constants_offset"]
        b_data = program.constants_bytes[b_offset : b_offset + bias_buf["size_bytes"]]
        bias_all = np.frombuffer(b_data, dtype="<i4")
        _write_int_lines(case_dir / "bias.txt", bias_all[:out_channels])

        # ISA v1.2 (H2, module docstring): the per-channel table the model
        # itself consumed, read back out of the DDR image at scale_addr
        # (padded to whole pe_rows tiles, `packed_scale_table_bytes`).
        if per_channel_bit is not None and (desc.flags >> per_channel_bit) & 1:
            layer_desc = model.LayerDesc(**{name: getattr(desc, name) for name in layer_desc_fields})
            n_bytes = model.packed_scale_table_bytes(layer_desc, pe_rows)
            table = model.unpack_scale_table_from_hw(
                bytes(memory[desc.scale_addr : desc.scale_addr + n_bytes]),
                n_bytes // model.SCALE_TABLE_ENTRY_BYTES,
            )
            _write_int_lines(
                case_dir / "scale_table_packed.txt",
                [v for multiplier, shift in table for v in (multiplier, shift)],
            )

        in_arr = read_activation_buffer(memory, in_buf, plane_channels)
        out_arr = read_activation_buffer(memory, out_buf, plane_channels)
        _write_int_lines(case_dir / "input.txt", in_arr.reshape(-1))
        _write_int_lines(case_dir / "expected.txt", out_arr.reshape(-1))

        written.append(name)

    (out_path / "cases.txt").write_text("".join(f"{n}\n" for n in written))
    return VectorsResult(written=written, skipped=skipped)
