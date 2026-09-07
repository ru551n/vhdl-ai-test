"""Program decode + human-readable dump (doc/tosa_compiler_plan.md §11
row 08, §13 M8).

`decode_program` is `encode_descriptor`'s exact inverse, fetching one
64-byte `Descriptor` at a time and following its own `next_instr_addr`
(mirroring `cnn_accel_model.run_program`'s fetch loop) until `HALT`,
using only `target.isa` -- this module never imports `cnn_accel_model`.
`print_program` renders the `08_program.txt` dump.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from cnnc.errors import CompilerError

from .emit import Descriptor

if TYPE_CHECKING:
    from cnnc.target.contract import Target

_STAGE = "decode"


def decode_descriptor(data: bytes, target: "Target") -> Descriptor:
    """Inverse of `emit.encode_descriptor`: read every `target.isa.fields`
    entry out of a 64-byte instruction word."""
    if len(data) != target.isa.instr_word_bytes:
        raise CompilerError(
            f"expected {target.isa.instr_word_bytes} bytes, got {len(data)}", stage=_STAGE,
        )
    fields = target.isa.fields
    kwargs = {}
    for f in dataclasses.fields(Descriptor):
        spec = fields.get(f.name)
        if spec is None:
            raise CompilerError(f"ISA field {f.name!r} missing from target {target.name!r}", stage=_STAGE)
        offset, width, signed = spec
        kwargs[f.name] = int.from_bytes(data[offset:offset + width], "little", signed=signed)
    return Descriptor(**kwargs)


def decode_program(
    program_bytes: bytes, target: "Target", program_addr: int = 0, *, max_instructions: int = 1000
) -> tuple[Descriptor, ...]:
    """Follow the `next_instr_addr` chain starting at `program_addr`
    (`program_bytes[0]` corresponds to DDR address `program_addr`) until a
    `HALT` opcode is decoded. Raises `CompilerError` on an out-of-range PC,
    a `next_instr_addr` loop, or a chain that never reaches `HALT`."""
    word = target.isa.instr_word_bytes
    halt_opcode = target.isa.opcodes.get("HALT")
    if halt_opcode is None:
        raise CompilerError("target ISA has no HALT opcode", stage=_STAGE)

    descriptors: list[Descriptor] = []
    seen: set[int] = set()
    pc = program_addr
    for _ in range(max_instructions + 1):
        if pc in seen:
            raise CompilerError(f"decode_program: next_instr_addr loop detected at 0x{pc:08x}", stage=_STAGE)
        seen.add(pc)
        rel = pc - program_addr
        if rel < 0 or rel + word > len(program_bytes):
            raise CompilerError(
                f"decode_program: pc 0x{pc:08x} outside program_bytes range "
                f"[0x{program_addr:08x}, 0x{program_addr + len(program_bytes):08x})",
                stage=_STAGE,
            )
        desc = decode_descriptor(program_bytes[rel:rel + word], target)
        descriptors.append(desc)
        if desc.opcode == halt_opcode:
            return tuple(descriptors)
        pc = desc.next_instr_addr
    raise CompilerError(
        f"decode_program: exceeded max_instructions={max_instructions} without HALT "
        "(check next_instr_addr chain)",
        stage=_STAGE,
    )


def _flag_names(flags: int, target: "Target") -> list[str]:
    return [name for name, bit in sorted(target.isa.flags.items(), key=lambda kv: kv[1]) if (flags >> bit) & 1]


def print_program(descriptors: tuple[Descriptor, ...], target: "Target", *, program_addr: int = 0) -> str:
    """Deterministic text dump matching doc/tosa_compiler_plan.md §11's
    `08_program.txt` row: one line per decoded descriptor."""
    opcode_name = {v: k for k, v in target.isa.opcodes.items()}
    word = target.isa.instr_word_bytes
    lines = []
    addr = program_addr
    for i, desc in enumerate(descriptors):
        name = opcode_name.get(desc.opcode, f"0x{desc.opcode:02x}")
        if name == "HALT":
            lines.append(f"[{i}] @0x{addr:08x} HALT")
            addr += word
            continue
        flags = _flag_names(desc.flags, target)
        parts = [
            f"[{i}] @0x{addr:08x} {name}",
            "flags={" + ",".join(flags) + "}",
            f"in_addr=0x{desc.in_addr:08x}",
            f"out_addr=0x{desc.out_addr:08x}",
            f"weight_addr=0x{desc.weight_addr:08x}",
            f"bias_addr=0x{desc.bias_addr:08x}",
            f"in={desc.in_width}x{desc.in_height}x{desc.in_channels}",
            f"out_channels={desc.out_channels}",
            f"kernel={desc.kernel_h}x{desc.kernel_w}",
            f"stride={desc.stride_h}x{desc.stride_w}",
            f"pad={desc.pad_top}/{desc.pad_bottom}/{desc.pad_left}/{desc.pad_right}",
            f"requant_scale={desc.requant_scale}",
            f"requant_shift={desc.requant_shift}",
            f"next=0x{desc.next_instr_addr:08x}",
        ]
        lines.append(" ".join(parts))
        addr += word
    return "\n".join(lines)
