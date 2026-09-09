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
    "act_lut_en": "ACT_LUT_EN",
}

# `HirOp.params` keys copied verbatim onto a same-named `Descriptor`/ISA field.
_CORE_FIELD_PARAMS = (
    "in_width", "in_height", "in_channels", "out_channels",
    "kernel_h", "kernel_w", "stride_h", "stride_w",
    "pad_top", "pad_bottom", "pad_left", "pad_right",
    "requant_scale", "requant_shift",
)

# ISA v1.1 (H1) W13 epilogue params (doc/tosa_compiler_plan.md §5 ext. 1,
# M11): optional in `HirOp.params` -- `lower.to_hir` sets them only for
# `isa_version >= 1.1` units -- and copied verbatim like the core fields.
# On a v1.0 target they have no ISA field, so `encode_descriptor` rejects
# any nonzero value (a v1.0 `to_hir` never produces one).
_W13_FIELD_PARAMS = ("output_offset", "clamp_min", "clamp_max")

# `HirOp.params` keys a `max_pool` op sets, likewise copied verbatim onto
# the same-named ISA field. Disjoint from `_CORE_FIELD_PARAMS` except for
# the shared geometry/padding fields: pooling has no out_channels, no
# weights and no requant, and uses the separate `pool_*` W11 fields rather
# than the convolution's `kernel_*`/`stride_*`.
_POOL_FIELD_PARAMS = (
    "in_width", "in_height", "in_channels",
    "pool_kernel_h", "pool_kernel_w", "pool_stride_h", "pool_stride_w",
    "pad_top", "pad_bottom", "pad_left", "pad_right",
)

# `HirOp.params` keys an ISA v2.0 `add` op sets. A subset of
# `_CORE_FIELD_PARAMS`: ADD shares the geometry and the requant pair, and
# has nothing else -- no out_channels, no kernel/stride, no padding.
_ADD_FIELD_PARAMS = (
    "in_width", "in_height", "in_channels", "requant_scale", "requant_shift",
)

# `HirOp.params` keys an ISA v2.0 `act` op sets: the geometry only. The
# LUT's address is a buffer address (W3 `weight_addr`), and the enable is
# a flag, so neither is a param here.
_ACT_FIELD_PARAMS = ("in_width", "in_height", "in_channels")

# `HirOp.params` keys an ISA v2.0 `upsample` op sets: the INPUT geometry
# only. The output geometry is implied (`UPSAMPLE_FACTOR` in both spatial
# dimensions) and has no descriptor field, which is why the factor is
# checked at lowering time rather than encoded here.
_UPSAMPLE_FIELD_PARAMS = ("in_width", "in_height", "in_channels")

# ISA v2.1 W10 byte 41, opcode-agnostic (see `Descriptor.pad_value`).
_PAD_VALUE_PARAM = "pad_value"

_LAYOUT_NOTE = (
    '"ddr_layouts": "activations are S6 channel-tiled planes (memory.activation_plane_channels); '
    'weights/bias are the D10/D11 tile-major images of cnn_accel_model.pack_weights_for_hw/'
    'pack_bias_for_hw (tiled by the unit\'s internal_tiling.cin x cout)"'
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
    # ISA v1.1 (HW milestone H1) epilogue fields, W13, written by the M11
    # lowering for `isa_version >= 1.1` units (with `CLAMP_EN` in `flags`).
    # On an `isa_version 1.0` target they have no ISA field and
    # `encode_descriptor` accepts them only as 0 (= the reserved bytes they
    # occupy there).
    output_offset: int = 0
    clamp_min: int = 0
    clamp_max: int = 0
    # ISA v1.2 (HW milestone H2) per-channel requant table address, W14.
    # Meaningful only with FLAG_PER_CHANNEL_EN; always 0 until a lowering
    # writes the table (same v1.0/v1.1 reserved-zero rule as W13 above).
    scale_addr: int = 0
    # ISA v2.0, W15: byte count for the data-movement/elementwise family
    # (`COPY`/`ACT`), and the SECOND source operand's address for `ADD`
    # (`accel_v2/program.py`'s "ambiguity #2", which
    # `cnn_accel_model.run_layer` follows). 0 for every v1.2 opcode, which
    # ignores the field, and 0 is also what a pre-v2.0 target -- where
    # these bytes are reserved -- requires.
    xfer_bytes: int = 0
    # ISA v2.1, W10 byte 41: the int8 value padded taps take. Kept
    # field-for-field identical to `cnn_accel_model.LayerDesc` (see this
    # class' docstring, and `vectors.py`, which walks `LayerDesc`'s fields
    # to write `desc.txt`), and opcode-agnostic in the hardware: `POOL_*`
    # and `CONV2D`/`DWCONV2D`/`FC` all fill their padded taps from it.
    #
    # Two lowerings write it, for the same underlying reason -- TOSA's
    # padding value is not 0 -- and both are only correct as a pair with
    # something else:
    #
    #   * `max_pool`: TOSA pads MAX_POOL2D with the element type's minimum
    #     (-128 for int8) so a padded tap can never win the max. Paired
    #     with `gir.verify._verify_pool`, which refuses any other value.
    #   * `conv_layer`: TOSA pads with the input zero-point and then
    #     subtracts it, so `pad_value = in_zp` is paired with the
    #     `bias'[o] = bias[o] - in_zp * sum(w[o])` fold that
    #     `lower.to_hir._zero_point_bias_delta` derives. Either half alone
    #     is silently wrong (`tests/test_conv_zero_point.py` demonstrates
    #     both failure modes).
    #
    # On a pre-v2.1 target byte 41 is not an ISA field at all, so
    # `encode_descriptor` accepts it only as 0 -- and `lower.to_hir`
    # refuses the lowerings that would need it, naming `pad_value`.
    pad_value: int = 0


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
    Reserved bytes (absent from `target.isa.fields`) stay zero. A
    `Descriptor` field the target's ISA version does not define (e.g. the
    v1.1 W13 fields on a v1.0 target) is accepted only when it is 0 --
    which is exactly what those reserved bytes must hold -- and otherwise
    raises `CapabilityError` naming the field."""
    fields = target.isa.fields
    buf = bytearray(target.isa.instr_word_bytes)
    for f in dataclasses.fields(desc):
        name = f.name
        value = getattr(desc, name)
        spec = fields.get(name)
        if spec is None and value == 0:
            continue
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
    handled = (
        set(_CORE_FIELD_PARAMS)
        | set(_W13_FIELD_PARAMS)
        | set(_POOL_FIELD_PARAMS)
        | set(_ADD_FIELD_PARAMS)
        | set(_ACT_FIELD_PARAMS)
        | set(_UPSAMPLE_FIELD_PARAMS)
        | set(_FLAG_PARAM_TO_FLAG_NAME)
        | {_PAD_VALUE_PARAM}
    )
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

    params = op.params
    per_channel = bool(params.get("per_channel_en"))
    # ISA v1.2 (M12): a per-channel `conv_layer` reads a 4th buffer, the
    # SCALE_TABLE const, which becomes W14 `scale_addr`.
    n_reads = 4 if per_channel else 3
    if len(op.reads) != n_reads:
        what = "input, weight, bias, scale table" if per_channel else "input, weight, bias"
        raise CompilerError(
            f"conv_layer op must read exactly {n_reads} buffers ({what}), got {len(op.reads)}",
            op_id=op.id, stage=_STAGE,
        )
    if len(op.writes) != 1:
        raise CompilerError(
            f"conv_layer op must write exactly 1 buffer, got {len(op.writes)}", op_id=op.id, stage=_STAGE,
        )
    in_buf, w_buf, b_buf = (module.buffer(bid) for bid in op.reads[:3])
    scale_buf = module.buffer(op.reads[3]) if per_channel else None
    out_buf = module.buffer(op.writes[0])
    addressed = [("in", in_buf), ("weight", w_buf), ("bias", b_buf), ("out", out_buf)]
    if scale_buf is not None:
        if scale_buf.layout != "SCALE_TABLE" or scale_buf.role != "const":
            raise CompilerError(
                f"per-channel conv_layer's 4th read {scale_buf.id!r} must be a const SCALE_TABLE buffer, "
                f"got role {scale_buf.role!r} layout {scale_buf.layout!r}",
                op_id=op.id, stage=_STAGE,
            )
        addressed.append(("scale table", scale_buf))
    for label, buf in addressed:
        if buf.addr is None:
            raise CompilerError(f"buffer {buf.id!r} ({label}) has no addr", op_id=op.id, stage=_STAGE)

    opcode = target.isa.opcodes.get("CONV2D")
    if opcode is None:
        raise CapabilityError("target ISA has no CONV2D opcode", op_id=op.id, stage=_STAGE)

    for key in _CORE_FIELD_PARAMS:
        if key not in params:
            raise CompilerError(f"conv_layer op missing required param {key!r}", op_id=op.id, stage=_STAGE)
    if params.get("clamp_en") and params.get("clamp_min", 0) > params.get("clamp_max", 0):
        # Same rule as `cnn_accel_model.encode_instruction`: an empty clamp
        # range has no defined HW result; fail here, naming the op.
        raise CompilerError(
            f"clamp_min {params['clamp_min']} > clamp_max {params['clamp_max']} (empty clamp range)",
            op_id=op.id, stage=_STAGE,
        )

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
        scale_addr=scale_buf.addr if scale_buf is not None else 0,
        # ISA v2.1: the value padded taps take. `lower.to_hir` sets it only
        # for a conv whose TOSA `in_zp` is non-zero, and *only together
        # with* the matching `bias -= in_zp * sum(w)` fold -- see
        # `_zero_point_bias_delta`. 0 (the default) is the correct pad for
        # every zero-point-free convolution and is what a pre-v2.1 target,
        # which has no such ISA field, requires.
        pad_value=int(params.get(_PAD_VALUE_PARAM, 0)),
        **{key: params.get(key, 0) for key in _W13_FIELD_PARAMS},
    )


def _build_pool_descriptor(
    op: HirOp, module: HirModule, target: "Target", program_addr: int, index: int, isa_version: str
) -> Descriptor:
    """One `POOL_MAX` instruction. Reads one buffer and writes one; there
    are no weights, no bias and no scale table, so `weight_addr`/
    `bias_addr`/`scale_addr` stay 0 and no flag but `PAD_EN` is set (the
    `bias_en`/`requant_en` a convolution always sets would enable an
    epilogue `cnn_accel_model.pool_max` deliberately bypasses)."""
    _reject_unknown_params(op, isa_version)

    params = op.params
    if len(op.reads) != 1:
        raise CompilerError(
            f"max_pool op must read exactly 1 buffer (input), got {len(op.reads)}", op_id=op.id, stage=_STAGE
        )
    if len(op.writes) != 1:
        raise CompilerError(
            f"max_pool op must write exactly 1 buffer, got {len(op.writes)}", op_id=op.id, stage=_STAGE
        )
    in_buf = module.buffer(op.reads[0])
    out_buf = module.buffer(op.writes[0])
    for label, buf in (("in", in_buf), ("out", out_buf)):
        if buf.addr is None:
            raise CompilerError(f"buffer {buf.id!r} ({label}) has no addr", op_id=op.id, stage=_STAGE)

    opcode = target.isa.opcodes.get("POOL_MAX")
    if opcode is None:
        raise CapabilityError("target ISA has no POOL_MAX opcode", op_id=op.id, stage=_STAGE)

    for key in _POOL_FIELD_PARAMS:
        if key not in params:
            raise CompilerError(f"max_pool op missing required param {key!r}", op_id=op.id, stage=_STAGE)

    return Descriptor(
        opcode=opcode,
        flags=_assemble_flags(op, target),
        in_addr=in_buf.addr,
        out_addr=out_buf.addr,
        # A pooled tensor keeps its channel count; `out_channels` is a
        # convolution field the pool datapath does not read, so leaving it
        # 0 (rather than echoing in_channels) says "not applicable".
        next_instr_addr=program_addr + (index + 1) * target.isa.instr_word_bytes,
        pad_value=int(params.get(_PAD_VALUE_PARAM, 0)),
        **{key: params[key] for key in _POOL_FIELD_PARAMS},
    )


def _build_add_descriptor(
    op: HirOp, module: HirModule, target: "Target", program_addr: int, index: int, isa_version: str
) -> Descriptor:
    """One ISA v2.0 `ADD` instruction: two ifmaps of one geometry, one
    ofmap, one shared `(requant_scale, requant_shift)` pair.

    The second operand's ADDRESS goes in W15 `xfer_bytes`, not a byte
    count -- the descriptor has only `in_addr`/`out_addr`/`weight_addr`/
    `bias_addr` for addresses and ADD needs a third source slot. That is
    the reading `accel_v2/program.py` ratified ("ambiguity #2") and what
    `cnn_accel_model.run_layer` executes, so all three agree by
    construction rather than by comment. `REQUANT_EN` is the only flag:
    ADD has no bias, no padding, no clamp bounds and no per-channel
    table."""
    _reject_unknown_params(op, isa_version)

    params = op.params
    if len(op.reads) != 2:
        raise CompilerError(
            f"add op must read exactly 2 buffers (src0, src1), got {len(op.reads)}", op_id=op.id, stage=_STAGE
        )
    if len(op.writes) != 1:
        raise CompilerError(
            f"add op must write exactly 1 buffer, got {len(op.writes)}", op_id=op.id, stage=_STAGE
        )
    a_buf, b_buf = (module.buffer(bid) for bid in op.reads)
    out_buf = module.buffer(op.writes[0])
    for label, buf in (("src0", a_buf), ("src1", b_buf), ("out", out_buf)):
        if buf.addr is None:
            raise CompilerError(f"buffer {buf.id!r} ({label}) has no addr", op_id=op.id, stage=_STAGE)

    opcode = target.isa.opcodes.get("ADD")
    if opcode is None:
        raise CapabilityError("target ISA has no ADD opcode", op_id=op.id, stage=_STAGE)
    if "xfer_bytes" not in target.isa.fields:
        # Without W15 there is nowhere to put src1's address, so the
        # instruction cannot be encoded at all -- say that, rather than
        # emitting an ADD that reads whatever happens to be at address 0.
        raise CapabilityError(
            "target ISA has no 'xfer_bytes' field (ISA v2.0 W15), which is where ADD's second "
            "source address rides; refusing to emit",
            op_id=op.id, stage=_STAGE, constraint="xfer_bytes",
        )

    for key in _ADD_FIELD_PARAMS:
        if key not in params:
            raise CompilerError(f"add op missing required param {key!r}", op_id=op.id, stage=_STAGE)

    return Descriptor(
        opcode=opcode,
        flags=_assemble_flags(op, target),
        in_addr=a_buf.addr,
        out_addr=out_buf.addr,
        xfer_bytes=b_buf.addr,
        next_instr_addr=program_addr + (index + 1) * target.isa.instr_word_bytes,
        **{key: params[key] for key in _ADD_FIELD_PARAMS},
    )


def _build_act_descriptor(
    op: HirOp, module: HirModule, target: "Target", program_addr: int, index: int, isa_version: str
) -> Descriptor:
    """One ISA v2.0 `ACT` instruction: one ifmap, one ofmap, and the
    256-byte int8 -> int8 LUT.

    The LUT's address rides in W3 `weight_addr` -- section 5.1 names no
    field for it, and `weight_addr` is the slot whose role already is
    "the compile-time side table for this instruction" (it plays the same
    part for CONV2D). That is `accel_v2/program.py`'s "ambiguity #3", and
    `cnn_accel_model.run_layer` reads the LUT from the same place.
    `xfer_bytes` carries the ifmap's byte length, matching what
    `accel_v2/program.py` emits, even though the executor derives the
    length from the geometry."""
    _reject_unknown_params(op, isa_version)

    params = op.params
    if len(op.reads) != 2:
        raise CompilerError(
            f"act op must read exactly 2 buffers (input, LUT), got {len(op.reads)}", op_id=op.id, stage=_STAGE
        )
    if len(op.writes) != 1:
        raise CompilerError(
            f"act op must write exactly 1 buffer, got {len(op.writes)}", op_id=op.id, stage=_STAGE
        )
    in_buf, lut_buf = (module.buffer(bid) for bid in op.reads)
    out_buf = module.buffer(op.writes[0])
    if lut_buf.layout != "ACT_LUT" or lut_buf.role != "const":
        raise CompilerError(
            f"act op's 2nd read {lut_buf.id!r} must be a const ACT_LUT buffer, got role "
            f"{lut_buf.role!r} layout {lut_buf.layout!r}",
            op_id=op.id, stage=_STAGE,
        )
    for label, buf in (("in", in_buf), ("lut", lut_buf), ("out", out_buf)):
        if buf.addr is None:
            raise CompilerError(f"buffer {buf.id!r} ({label}) has no addr", op_id=op.id, stage=_STAGE)

    opcode = target.isa.opcodes.get("ACT")
    if opcode is None:
        raise CapabilityError("target ISA has no ACT opcode", op_id=op.id, stage=_STAGE)

    for key in _ACT_FIELD_PARAMS:
        if key not in params:
            raise CompilerError(f"act op missing required param {key!r}", op_id=op.id, stage=_STAGE)

    return Descriptor(
        opcode=opcode,
        flags=_assemble_flags(op, target),
        in_addr=in_buf.addr,
        out_addr=out_buf.addr,
        weight_addr=lut_buf.addr,
        xfer_bytes=in_buf.size_bytes,
        next_instr_addr=program_addr + (index + 1) * target.isa.instr_word_bytes,
        **{key: params[key] for key in _ACT_FIELD_PARAMS},
    )


def _build_upsample_descriptor(
    op: HirOp, module: HirModule, target: "Target", program_addr: int, index: int, isa_version: str
) -> Descriptor:
    """One ISA v2.0 `UPSAMPLE` instruction: one ifmap in, one ofmap
    `UPSAMPLE_FACTOR` times larger in both spatial dimensions.

    Everything is implied by the input geometry -- no flags, no weights,
    no requant, and no factor field (the hardware has exactly one factor,
    which `lower.to_hir` is what checks). `xfer_bytes` stays 0: this is a
    shaped opcode, not a byte mover, so W15 has no role here
    (`accel_v2/program.py` emits it the same way)."""
    _reject_unknown_params(op, isa_version)

    params = op.params
    if len(op.reads) != 1:
        raise CompilerError(
            f"upsample op must read exactly 1 buffer (input), got {len(op.reads)}", op_id=op.id, stage=_STAGE
        )
    if len(op.writes) != 1:
        raise CompilerError(
            f"upsample op must write exactly 1 buffer, got {len(op.writes)}", op_id=op.id, stage=_STAGE
        )
    in_buf = module.buffer(op.reads[0])
    out_buf = module.buffer(op.writes[0])
    for label, buf in (("in", in_buf), ("out", out_buf)):
        if buf.addr is None:
            raise CompilerError(f"buffer {buf.id!r} ({label}) has no addr", op_id=op.id, stage=_STAGE)

    opcode = target.isa.opcodes.get("UPSAMPLE")
    if opcode is None:
        raise CapabilityError("target ISA has no UPSAMPLE opcode", op_id=op.id, stage=_STAGE)

    for key in _UPSAMPLE_FIELD_PARAMS:
        if key not in params:
            raise CompilerError(f"upsample op missing required param {key!r}", op_id=op.id, stage=_STAGE)

    return Descriptor(
        opcode=opcode,
        flags=_assemble_flags(op, target),
        in_addr=in_buf.addr,
        out_addr=out_buf.addr,
        next_instr_addr=program_addr + (index + 1) * target.isa.instr_word_bytes,
        **{key: params[key] for key in _UPSAMPLE_FIELD_PARAMS},
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
        if buf.alias_parent is not None:
            # A buffer view owns no bytes of its own: it is a channel-plane
            # range of `alias_parent` (`tosa.concat`/`tosa.slice`). Recorded
            # so a reader of the manifest can see that two buffers sharing
            # an address range is intentional, not a planning bug.
            entry["alias"] = {
                "parent": buf.alias_parent,
                "plane_offset": buf.alias_plane_offset,
                "kind": buf.alias_kind,
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
            "activation_layout": target.memory.activation_layout,
            "activation_plane_channels": target.memory.activation_plane_channels,
            "weight_layout": target.memory.weight_layout,
            "bias_format": target.memory.bias_format,
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
        "notes": list(module.notes) + [_LAYOUT_NOTE],
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
    builders = {
        "conv_layer": _build_conv_descriptor,
        "max_pool": _build_pool_descriptor,
        "add": _build_add_descriptor,
        "act": _build_act_descriptor,
        "upsample": _build_upsample_descriptor,
    }
    for index, op in enumerate(ops_sorted):
        builder = builders.get(op.kind)
        if builder is None:
            raise CapabilityError(
                f"HIR op kind {op.kind!r} has no cnn_accel_v1 instruction encoding "
                f"(encodable kinds: {', '.join(sorted(builders))})",
                op_id=op.id, stage=_STAGE, constraint=op.kind,
            )
        descriptors.append(builder(op, module, target, program_buf.addr, index, isa_version))
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
