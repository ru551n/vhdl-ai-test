"""Deterministic text/JSON dumps of an `HirModule` (doc/tosa_compiler_plan.md
§2.2, §11 "HIR (mapped/scheduled/planned)" dump rows). Both `print_hir` and
`to_json` only read the module -- neither mutates nor verifies it."""

from __future__ import annotations

import hashlib

from .ir import Buffer, HirModule, HirOp, TileInfo


def _shape_dtype(shape: tuple[int, ...], dtype: str) -> str:
    return "x".join(str(d) for d in shape) + "x" + dtype


def _fmt_addr(addr: int | None) -> str:
    return f"0x{addr:08x}" if addr is not None else "-"


def _format_buffers(module: HirModule) -> list[str]:
    # The program buffer gets its own dedicated `program:` line, not a
    # `buffers:` row (doc/tosa_compiler_plan.md §2.2 dump style).
    bufs = [b for b in module.buffers.values() if b.role != "program"]
    if not bufs:
        return []
    shape_dtypes = [_shape_dtype(b.shape, b.dtype) for b in bufs]
    id_w = max(len(b.id) for b in bufs)
    space_w = max(len(b.space) for b in bufs)
    layout_w = max(len(b.layout) for b in bufs)
    sd_w = max(len(s) for s in shape_dtypes)
    size_w = max(len(str(b.size_bytes)) for b in bufs)
    lines = []
    for b, sd in zip(bufs, shape_dtypes):
        # A buffer view owns no bytes; saying so on its own line is what
        # keeps a dump readable when two buffers share an address range.
        alias = (
            f" {b.alias_kind} of {b.alias_parent} +{b.alias_plane_offset} planes"
            if b.alias_parent is not None
            else ""
        )
        lines.append(
            f"  {b.id.ljust(id_w)} {b.space.ljust(space_w)} {b.layout.ljust(layout_w)} "
            f"{sd.ljust(sd_w)} {str(b.size_bytes).rjust(size_w)} B align {b.align} "
            f"addr {_fmt_addr(b.addr)} role {b.role}{alias}"
        )
    return lines


def _fmt_param_value(value) -> str:
    if isinstance(value, bool):
        return str(int(value))
    return str(value)


def _fmt_params(params) -> str:
    body = " ".join(f"{k}={_fmt_param_value(params[k])}" for k in sorted(params))
    return "{" + body + "}"


def _format_ops(module: HirModule) -> list[str]:
    lines = []
    for idx, op in enumerate(module.ops):
        seq = str(op.seq) if op.seq is not None else "-"
        gir = op.gir_op if op.gir_op is not None else "-"
        lines.append(f"  #{idx} seq {seq} unit {op.unit} kind {op.kind} gir {gir}")
        lines.append(f"     reads [{' '.join(op.reads)}] writes [{' '.join(op.writes)}] deps [{' '.join(op.deps)}]")
        lines.append(f"     params {_fmt_params(op.params)}")
    return lines


def _format_program(module: HirModule) -> str:
    if module.program is None:
        return "program: -"
    buf = module.buffers.get(module.program)
    if buf is None:
        return f"program: {module.program} ?"
    return f"program: {buf.id} {buf.size_bytes} B addr {_fmt_addr(buf.addr)}"


def print_hir(module: HirModule) -> str:
    mem = str(module.memory_size) if module.memory_size is not None else "-"
    lines = [f"hir.module target={module.target_name} stage={module.stage} memory_size={mem}"]
    lines.append("buffers:")
    lines.extend(_format_buffers(module))
    lines.append("ops:")
    lines.extend(_format_ops(module))
    lines.append(_format_program(module))
    for note in module.notes:
        lines.append(f"note: {note}")
    return "\n".join(lines)


def _data_to_json(data: bytes | None) -> dict | None:
    if data is None:
        return None
    return {"sha256": hashlib.sha256(data).hexdigest(), "length": len(data)}


def _buffer_to_json(buf: Buffer) -> dict:
    return {
        "id": buf.id,
        "space": buf.space,
        "size_bytes": buf.size_bytes,
        "align": buf.align,
        "role": buf.role,
        "layout": buf.layout,
        "shape": list(buf.shape),
        "dtype": buf.dtype,
        "addr": buf.addr,
        "data": _data_to_json(buf.data),
        "gir_tensor": buf.gir_tensor,
        "alias_parent": buf.alias_parent,
        "alias_plane_offset": buf.alias_plane_offset,
        "alias_kind": buf.alias_kind,
    }


def _tile_to_json(tile: TileInfo | None) -> dict | None:
    if tile is None:
        return None
    return {
        "axis": tile.axis,
        "index": tile.index,
        "count": tile.count,
        "halo_before": tile.halo_before,
        "halo_after": tile.halo_after,
        "first": tile.first,
        "last": tile.last,
    }


def _op_to_json(op: HirOp) -> dict:
    return {
        "id": op.id,
        "unit": op.unit,
        "kind": op.kind,
        "params": dict(op.params),
        "reads": list(op.reads),
        "writes": list(op.writes),
        "deps": list(op.deps),
        "seq": op.seq,
        "tile": _tile_to_json(op.tile),
        "gir_op": op.gir_op,
    }


def to_json(module: HirModule) -> dict:
    return {
        "target_name": module.target_name,
        "stage": module.stage,
        "memory_size": module.memory_size,
        "program": module.program,
        "entry_inputs": list(module.entry_inputs),
        "entry_outputs": list(module.entry_outputs),
        "buffers": {bid: _buffer_to_json(b) for bid, b in module.buffers.items()},
        "ops": [_op_to_json(op) for op in module.ops],
        "notes": list(module.notes),
    }
