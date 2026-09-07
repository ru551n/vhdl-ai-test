"""HIR -> `cnn_accel_v1` program emitter (doc/tosa_compiler_plan.md §2.3,
§5, §11 row 08, §13 M8).

`emit_program` turns a `stage='planned'` `HirModule` (every `Buffer.addr`
assigned by `lower.memplan`) into a `Program`: a `program_bytes` 64-byte
`LayerDesc`-shaped instruction stream, a `constants_bytes` blob (weights/
bias, packed in the order the planner placed them), and a JSON-serialisable
`manifest` recording every address, size and per-op provenance link back to
GIR. `Descriptor` mirrors `cnn_accel_model.LayerDesc` field-for-field, but
this module never imports `cnn_accel_model` -- every byte offset/width/
signedness comes from `target.isa.fields` (itself derived from
`cnn_accel_constants.ISA_LAYOUT` by `target.discover`), so a stale/hand-
edited target JSON can never produce a descriptor the RTL misreads: any
`HirOp.params` key with no corresponding ISA field or flag bit for the
target's `isa_version` raises `CapabilityError` naming the field, and any
value that does not fit its field's width raises `CompilerError` naming
the field (see `decode.py`/`run.py` for the read side and the one place
that *does* import `cnn_accel_model`, to execute the emitted program).
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import TYPE_CHECKING

import cnnc
from cnnc.errors import CapabilityError, CompilerError
from cnnc.hir.ir import HirModule, HirOp

if TYPE_CHECKING:
    from cnnc.target.contract import Target

_STAGE = "emit"

# `HirOp.params` keys that become a `flags` bit, mapped to the `IsaInfo.flags`
# name they correspond to (doc/tosa_compiler_plan.md §5, cnn_accel_constants.FLAGS).
_FLAG_PARAM_TO_FLAG_NAME = {
    "relu_en": "RELU_EN",
    "bias_en": "BIAS_EN",
    "requant_en": "REQUANT_EN",
    "pad_en": "PAD_EN",
    "clamp_en": "CLAMP_EN",
    "per_channel_en": "PER_CHANNEL_EN",
}

# `HirOp.params` keys copied verbatim onto a same-named `Descriptor`/ISA field.
_CORE_FIELD_PARAMS = (
    "in_width", "in_height", "in_channels", "out_channels",
    "kernel_h", "kernel_w", "stride_h", "stride_w",
    "pad_top", "pad_bottom", "pad_left", "pad_right",
    "requant_scale", "requant_shift",
)

_WEIGHT_LAYOUT_NOTE = (
    '"weight_ddr_layout": "OHWI (cnn_accel_model.run_program convention; RTL-native '
    'packing per pack_weights_for_hw is deferred to M10)"'
)


@dataclasses.dataclass(frozen=True)
class Descriptor:
    """One 64-byte instruction, field-for-field identical to
    `cnn_accel_model.LayerDesc` (same names, same defaults) so a decoded
    `Descriptor` and a hand-built `LayerDesc(**same kwargs)` compare equal
    field-by-field without any translation."""

    opcode: int
    flags: int = 0
    in_addr: int = 0
    out_addr: int = 0
    weight_addr: int = 0
    bias_addr: int = 0
    in_width: int = 0
    in_height: int = 0
    in_channels: int = 0
    out_channels: int = 0
    kernel_h: int = 1
    kernel_w: int = 1
    stride_h: int = 1
    stride_w: int = 1
    pad_top: int = 0
    pad_bottom: int = 0
    pad_left: int = 0
    pad_right: int = 0
    requant_scale: int = 0
    requant_shift: int = 0
    pool_kernel_h: int = 1
    pool_kernel_w: int = 1
    pool_stride_h: int = 1
    pool_stride_w: int = 1
    next_instr_addr: int = 0


@dataclasses.dataclass(frozen=True)
class Program:
    program_bytes: bytes
    constants_bytes: bytes
    manifest: dict
    descriptors: tuple[Descriptor, ...]


def _check_range(name: str, value: int, width_bytes: int, signed: bool, *, op_id: str | None) -> None:
    if signed:
        lo, hi = -(2 ** (8 * width_bytes - 1)), 2 ** (8 * width_bytes - 1) - 1
    else:
        lo, hi = 0, 2 ** (8 * width_bytes) - 1
    if not (lo <= value <= hi):
        raise CompilerError(
            f"field {name!r} value {value} does not fit in {width_bytes} byte(s) "
            f"({'signed' if signed else 'unsigned'} range [{lo}, {hi}])",
            op_id=op_id, stage=_STAGE,
        )


def encode_descriptor(desc: Descriptor, target: "Target", *, op_id: str | None = None) -> bytes:
    """Encode `desc` using `target.isa.fields` only (no RTL import).
    Reserved bytes (absent from `target.isa.fields`) stay zero."""
    fields = target.isa.fields
    buf = bytearray(target.isa.instr_word_bytes)
    for f in dataclasses.fields(desc):
        name = f.name
        value = getattr(desc, name)
        spec = fields.get(name)
        if spec is None:
            raise CapabilityError(
                f"field {name!r} not in target ISA (isa_version-gated); refusing to emit",
                op_id=op_id, stage=_STAGE, constraint=name,
            )
        offset, width, signed = spec
        _check_range(name, value, width, signed, op_id=op_id)
        buf[offset:offset + width] = int(value).to_bytes(width, "little", signed=signed)
    return bytes(buf)


def _assemble_flags(op: HirOp, target: "Target") -> int:
    flags = 0
    for key, flag_name in _FLAG_PARAM_TO_FLAG_NAME.items():
        value = op.params.get(key)
        if not value:
            continue
        bit = target.isa.flags.get(flag_name)
        if bit is None:
            raise CapabilityError(
                f"flag {flag_name!r} (param {key!r}={value!r}) not defined by target ISA",
                op_id=op.id, stage=_STAGE, constraint=flag_name,
            )
        flags |= 1 << bit
    return flags


def _reject_unknown_params(op: HirOp, isa_version: str) -> None:
    handled = set(_CORE_FIELD_PARAMS) | set(_FLAG_PARAM_TO_FLAG_NAME)
    for key in op.params:
        if key not in handled:
            raise CapabilityError(
                f"param {key!r} has no cnn_accel_v1 ISA v{isa_version} field/flag mapping",
                op_id=op.id, stage=_STAGE, constraint=key,
            )


def _halt_descriptor(target: "Target") -> Descriptor:
    opcode = target.isa.opcodes.get("HALT")
    if opcode is None:
        raise CapabilityError("target ISA has no HALT opcode", stage=_STAGE)
    return Descriptor(
        opcode=opcode, flags=0,
        in_addr=0, out_addr=0, weight_addr=0, bias_addr=0,
        in_width=0, in_height=0, in_channels=0, out_channels=0,
        kernel_h=0, kernel_w=0, stride_h=0, stride_w=0,
        pad_top=0, pad_bottom=0, pad_left=0, pad_right=0,
        requant_scale=0, requant_shift=0,
        pool_kernel_h=0, pool_kernel_w=0, pool_stride_h=0, pool_stride_w=0,
        next_instr_addr=0,
    )


def _build_conv_descriptor(
    op: HirOp, module: HirModule, target: "Target", program_addr: int, index: int, isa_version: str
) -> Descriptor:
    _reject_unknown_params(op, isa_version)

    if len(op.reads) != 3:
        raise CompilerError(
            f"conv_layer op must read exactly 3 buffers (input, weight, bias), got {len(op.reads)}",
            op_id=op.id, stage=_STAGE,
        )
    if len(op.writes) != 1:
        raise CompilerError(
            f"conv_layer op must write exactly 1 buffer, got {len(op.writes)}", op_id=op.id, stage=_STAGE,
        )
    in_buf, w_buf, b_buf = (module.buffer(bid) for bid in op.reads)
    out_buf = module.buffer(op.writes[0])
    for label, buf in (("in", in_buf), ("weight", w_buf), ("bias", b_buf), ("out", out_buf)):
        if buf.addr is None:
            raise CompilerError(f"buffer {buf.id!r} ({label}) has no addr", op_id=op.id, stage=_STAGE)

    opcode = target.isa.opcodes.get("CONV2D")
    if opcode is None:
        raise CapabilityError("target ISA has no CONV2D opcode", op_id=op.id, stage=_STAGE)

    params = op.params
    for key in _CORE_FIELD_PARAMS:
        if key not in params:
            raise CompilerError(f"conv_layer op missing required param {key!r}", op_id=op.id, stage=_STAGE)

    return Descriptor(
        opcode=opcode,
        flags=_assemble_flags(op, target),
        in_addr=in_buf.addr,
        out_addr=out_buf.addr,
        weight_addr=w_buf.addr,
        bias_addr=b_buf.addr,
        in_width=params["in_width"],
        in_height=params["in_height"],
        in_channels=params["in_channels"],
        out_channels=params["out_channels"],
        kernel_h=params["kernel_h"],
        kernel_w=params["kernel_w"],
        stride_h=params["stride_h"],
        stride_w=params["stride_w"],
        pad_top=params["pad_top"],
        pad_bottom=params["pad_bottom"],
        pad_left=params["pad_left"],
        pad_right=params["pad_right"],
        requant_scale=params["requant_scale"],
        requant_shift=params["requant_shift"],
        next_instr_addr=program_addr + (index + 1) * target.isa.instr_word_bytes,
    )


def _isa_version(target: "Target", ops: list) -> str:
    versions = {target.unit(op.unit).isa_version for op in ops}
    if not versions:
        return target.units[0].isa_version
    if len(versions) > 1:
        raise CompilerError(f"program mixes ISA versions across units: {sorted(versions)}", stage=_STAGE)
    return next(iter(versions))


def _build_manifest(
    module: HirModule,
    target: "Target",
    program_buf,
    ops_sorted: list,
    const_bufs: list,
    const_offsets: dict,
    isa_version: str,
) -> dict:
    buffers_sorted = sorted(module.buffers.values(), key=lambda b: (b.addr, b.id))
    buffer_entries = []
    for buf in buffers_sorted:
        entry = {
            "id": buf.id,
            "role": buf.role,
            "layout": buf.layout,
            "shape": list(buf.shape),
            "dtype": buf.dtype,
            "addr": buf.addr,
            "size_bytes": buf.size_bytes,
            "gir_tensor": buf.gir_tensor,
        }
        if buf.role == "const":
            entry["constants_offset"] = const_offsets[buf.id]
            entry["sha256"] = hashlib.sha256(buf.data).hexdigest()
        buffer_entries.append(entry)

    op_entries = []
    for index, op in enumerate(ops_sorted):
        op_entries.append({
            "id": op.id,
            "seq": op.seq,
            "unit": op.unit,
            "kind": op.kind,
            "gir_op": op.gir_op,
            "descriptor_index": index,
            "descriptor_addr": program_buf.addr + index * target.isa.instr_word_bytes,
            "params": dict(op.params),
        })

    return {
        "format_version": 1,
        "compiler": {"name": "cnnc", "version": cnnc.__version__},
        "target": {
            "name": target.name,
            "isa_version": isa_version,
            "provenance": target.provenance,
        },
        "memory": {
            "space": program_buf.space,
            "size_bytes": module.memory_size,
            "program": {
                "addr": program_buf.addr,
                "size_bytes": program_buf.size_bytes,
                "base_offset_field_note": (
                    "addresses are absolute DDR today; field names addr/offset/base are "
                    "chosen so a relocatable base+offset format can be added later without "
                    "renaming fields (doc/tosa_compiler_plan.md §14 item 5)"
                ),
            },
        },
        "buffers": buffer_entries,
        "inputs": list(module.entry_inputs),
        "outputs": list(module.entry_outputs),
        "ops": op_entries,
        "notes": list(module.notes) + [_WEIGHT_LAYOUT_NOTE],
    }


def emit_program(module: HirModule, target: "Target") -> Program:
    if module.stage != "planned":
        raise CompilerError(f"emit_program requires a stage='planned' module, got {module.stage!r}", stage=_STAGE)
    for bid, buf in module.buffers.items():
        if buf.addr is None:
            raise CompilerError(f"buffer {bid!r} has no addr; module is not fully planned", stage=_STAGE)
    if module.program is None:
        raise CompilerError("planned module has no program buffer", stage=_STAGE)
    program_buf = module.buffer(module.program)

    ops_sorted = sorted(module.ops, key=lambda op: op.seq if op.seq is not None else -1)
    for op in ops_sorted:
        if op.seq is None:
            raise CompilerError(f"op {op.id!r} has no seq; module is not scheduled", op_id=op.id, stage=_STAGE)

    isa_version = _isa_version(target, ops_sorted)

    descriptors: list[Descriptor] = []
    op_ids: list[str | None] = []
    for index, op in enumerate(ops_sorted):
        if op.kind != "conv_layer":
            raise CapabilityError(
                f"HIR op kind {op.kind!r} has no cnn_accel_v1 instruction encoding",
                op_id=op.id, stage=_STAGE, constraint=op.kind,
            )
        descriptors.append(_build_conv_descriptor(op, module, target, program_buf.addr, index, isa_version))
        op_ids.append(op.id)
    descriptors.append(_halt_descriptor(target))
    op_ids.append(None)

    program_bytes = b"".join(
        encode_descriptor(desc, target, op_id=op_id) for desc, op_id in zip(descriptors, op_ids)
    )
    if len(program_bytes) != program_buf.size_bytes:
        raise CompilerError(
            f"emitted program length {len(program_bytes)} != planned program size {program_buf.size_bytes}",
            stage=_STAGE,
        )

    const_bufs = sorted((b for b in module.buffers.values() if b.role == "const"), key=lambda b: (b.addr, b.id))
    const_offsets: dict[str, int] = {}
    constants_chunks: list[bytes] = []
    offset = 0
    for buf in const_bufs:
        if buf.data is None:
            raise CompilerError(f"const buffer {buf.id!r} has no data", stage=_STAGE)
        data = buf.data if len(buf.data) == buf.size_bytes else buf.data.ljust(buf.size_bytes, b"\x00")
        const_offsets[buf.id] = offset
        constants_chunks.append(data)
        offset += len(data)
    constants_bytes = b"".join(constants_chunks)

    manifest = _build_manifest(module, target, program_buf, ops_sorted, const_bufs, const_offsets, isa_version)

    return Program(
        program_bytes=program_bytes,
        constants_bytes=constants_bytes,
        manifest=manifest,
        descriptors=tuple(descriptors),
    )
