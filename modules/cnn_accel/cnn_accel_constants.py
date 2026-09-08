"""Single Python source of truth for every `modules/cnn_accel/` constant that
used to be hand-duplicated across VHDL, the golden model, and
`module_cnn_accel.py` -- accelerator HW properties, ISA opcodes/flags, and
the 64-byte instruction word's byte-offset table.

This is the file a human edits to change a constant. Two consumers read it:

- `module_cnn_accel.py`'s `registers_hook()` and `cnn_accel_isa_generator.py`
  turn it into generated VHDL packages (`regs_src/cnn_accel_regs_pkg.vhd`,
  `regs_src/cnn_accel_isa_pkg.vhd`, ...) via `hdl-registers`. See
  `doc/cnn_accel_csr_req.md` ("Register map") and `doc/cnn_accel_arch.md`
  ("Instruction Set (v1)") for the specifications this module implements.
- `cnn_accel_model.py` (the golden model) derives its `OPCODE_*`/`FLAG_*`/
  `OFF_*` names from the same tables, so the model's ISA encoder and the
  generated VHDL package are byte-for-byte identical by construction
  instead of by hand-maintained comment ("cnn_accel_model.py's encoder
  must agree with these byte-for-byte" -- this file is what now makes
  that structural rather than aspirational).

Deliberately pure stdlib, no `hdl_registers` import: the golden model
(run under system `python3`/pytest, no venv guarantees) must be able to
import this cheaply, and hdl-registers is not a golden-model dependency.
"""

from __future__ import annotations

import math
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Accelerator HW properties.
#
# Mirrors the reference hardware configuration in module_cnn_accel.py
# (`_PE_ROWS`, `_WEIGHT_BUFFER_DEPTH`, ...), which imports these names
# rather than redefining them -- see that file's own comment for where
# each number comes from (doc/cnn_accel_tiled_dataflow_proposal.md section
# 7 and doc/cnn_accel_weight_buffer.md's depth-sizing note). Kept here,
# not there, so they are also available as generated `hdl-registers`
# constants (`cnn_accel_regs_pkg.vhd`) for RTL to reference directly
# instead of via a generic that some instantiation site might get wrong.
# ---------------------------------------------------------------------------

# PE_ROWS is THE single scaling knob (flow_status.md S1-S7): output-channel
# lanes, i.e. the number of output channels computed per pass over the
# ifmap. 8 is the shipped default (~30 fps on the 320x240 target, see
# doc/cnn_accel_sizing_proposal.md); 16 is the 60 fps point that CI proves
# in parallel (netlist builds + tb_cnn_accel_conv_core config in
# module_cnn_accel.py, vectors generated per config at test time) WITHOUT
# changing the default. PE_COLS and TILE_CHANNELS are fixed forever.
PE_ROWS = 8
# Every legal value must divide every layer's out_channels in the target
# backbone (layer 1 has 16), and must keep 8*PE_ROWS <= axi_stream_data_sz
# (128) for cnn_accel_bias_requant's int8 output beat -- both of which cap
# the set at exactly these two. Asserted here and in cnn_accel_conv_core.vhd.
PE_ROWS_LEGAL = (8, 16)
PE_ROWS_SCALED = 16
PE_COLS = 8
TILE_CHANNELS = 8  # = PE_COLS, see doc/cnn_accel_tiled_dataflow_proposal.md section 3.

assert PE_ROWS in PE_ROWS_LEGAL, f"PE_ROWS={PE_ROWS} not in {PE_ROWS_LEGAL}"
assert PE_ROWS_SCALED in PE_ROWS_LEGAL, f"PE_ROWS_SCALED={PE_ROWS_SCALED} not in {PE_ROWS_LEGAL}"
assert PE_ROWS_SCALED != PE_ROWS, "PE_ROWS_SCALED must be a second, distinct legal point"
# ACTIVATION_PLANE_CHANNELS is `T` from decision S6: activations live in DDR
# as channel-tiled planes `[C/T][H][W][T]`, so one pixel occupies exactly T
# contiguous int8 bytes and every plane's byte address and byte length is a
# multiple of T. Numerically equal to TILE_CHANNELS, and deliberately a
# SEPARATE name: TILE_CHANNELS is a datapath property (how many input
# channels the PE array consumes per beat), T is a memory-layout property
# (how the host and the DMA engines agree to pack DDR). They happen to
# coincide at 8 and are both fixed forever, but code that means "the DDR
# layout" must not read the datapath constant, or a future decoupling would
# silently corrupt addresses rather than fail to compile.
#
# The alignment consequence is load-bearing, not cosmetic. Both
# cnn_accel_ofmap_dma (via hdl-modules' dma_axi_write_simple) and
# cnn_accel_axi_read_dma require req.addr and req.length to be multiples of
# g_axi_data_width/8; a violation does not error, it simply never completes
# (dma_done never fires) and silently hangs the layer. Because every
# activation request is a whole plane, addr/length are always multiples of
# ACTIVATION_PLANE_CHANNELS bytes -- so any AXI bus of at most that many
# bytes per beat is aligned unconditionally, with no runtime check and no
# host-side contract. That is the bound MAX_AXI_DATA_WIDTH encodes, and both
# DMA entities assert it at elaboration.
ACTIVATION_PLANE_CHANNELS = 8
MAX_AXI_DATA_WIDTH = 8 * ACTIVATION_PLANE_CHANNELS

assert ACTIVATION_PLANE_CHANNELS == TILE_CHANNELS, (
    "T and TILE_CHANNELS are conceptually distinct but must coincide while "
    "cnn_accel_window_gen consumes exactly one plane per beat"
)
assert ACTIVATION_PLANE_CHANNELS & (ACTIVATION_PLANE_CHANNELS - 1) == 0, (
    "T must be a power of two: plane indexing is a shift-and-add, and the "
    "MAX_AXI_DATA_WIDTH alignment argument assumes an AXI-legal width"
)

MAX_KERNEL_SIZE = 3
MAX_ROW_TILE_WORDS = 512
WEIGHT_BUFFER_DEPTH = 288
BIAS_BUFFER_DEPTH = 8
ACCUM_WIDTH = 32

# ISA v2.0 (doc/cnn_accel_top_v2_arch.md sections 5 and 8): the version the
# host reads back from CSR.HW_INFO2.ISA_VERSION, and the single source of
# truth `accel_v2/isa.py`'s `ISA_VERSION` derives from -- previously that
# module restated the `0x0200` literal itself, which this constant
# replaces. Major.minor packed as `(major << 8) | minor`.
ISA_VERSION = 0x0200

# ISA v1.2 (doc/tosa_compiler_plan.md section 5, extension 2 / HW milestone
# H2): per-channel requantization table in DDR at `scale_addr`, one entry
# per output channel (zero-padded to whole PE_ROWS tiles exactly like the
# bias image, see cnn_accel_model.pack_scale_table_for_hw):
#
#   byte 0..3  multiplier, int32 little-endian (same Q15 semantics as
#              requant_scale)
#   byte 4     shift, uint8 (same semantics as requant_shift)
#   byte 5..7  zero, must be 0
#
# 8 bytes = one 64-bit AXI beat per channel, so the table streams through
# the same weight/bias DMA path as the bias image with no repacking. The
# accelerator keeps SCALE_BUFFER_ENTRY_BITS (32 + 8 = 40) of each entry in
# `cnn_accel_weight_buffer`'s `scale_buffer` (PE_ROWS entries, parallel to
# `bias_buffer`); the 3 zero bytes are dropped on load.
SCALE_TABLE_ENTRY_BYTES = 8
SCALE_TABLE_MULTIPLIER_BYTES = 4
SCALE_TABLE_SHIFT_BYTES = 1
SCALE_BUFFER_ENTRY_BITS = 8 * (SCALE_TABLE_MULTIPLIER_BYTES + SCALE_TABLE_SHIFT_BYTES)

assert SCALE_TABLE_MULTIPLIER_BYTES + SCALE_TABLE_SHIFT_BYTES <= SCALE_TABLE_ENTRY_BYTES
assert SCALE_TABLE_ENTRY_BYTES == 8, "one 64-bit AXI beat per channel is the ratified table format"

# ---------------------------------------------------------------------------
# Frame budget (flow_status.md S5): the target backbone, the clock/fps
# targets, and the section-6 cycle model from
# doc/cnn_accel_sizing_proposal.md, ported to Python so
# `test_cnn_accel_model.py` can assert the budget as a real (currently
# failing at PE_ROWS) test instead of a static markdown table that can
# silently go stale.
#
# `BACKBONE_TARGET` is the 9-layer target backbone from
# doc/cnn_accel_tiled_dataflow_proposal.md section 6, `pixels_320` being
# each layer's per-pixel output count at the reference 320x320 input
# (already reflecting that layer's cumulative downsampling). Per
# doc/cnn_accel_sizing_proposal.md section 3, an input of a different size
# scales every layer's pixel count by the input/reference area ratio.
# ---------------------------------------------------------------------------

CLOCK_HZ = 150_000_000
TARGET_FPS = 60
INPUT_W = 320
INPUT_H = 240

# Reference resolution the BACKBONE_TARGET pixel counts were measured at
# (doc/cnn_accel_tiled_dataflow_proposal.md section 6).
_REFERENCE_W = 320
_REFERENCE_H = 320


class BackboneLayer(NamedTuple):
    in_channels: int
    out_channels: int
    pixels_at_reference: int  # output pixel count at _REFERENCE_W x _REFERENCE_H


BACKBONE_TARGET = (
    BackboneLayer(3, 16, 25600),
    BackboneLayer(16, 32, 6400),
    BackboneLayer(32, 32, 6400),
    BackboneLayer(32, 64, 1600),
    BackboneLayer(64, 64, 1600),
    BackboneLayer(64, 128, 400),
    BackboneLayer(128, 128, 400),
    BackboneLayer(128, 256, 100),
    BackboneLayer(256, 256, 100),
)


def cycles_per_frame(
    pe_rows: int,
    pe_cols: int = PE_COLS,
    input_w: int = INPUT_W,
    input_h: int = INPUT_H,
) -> int:
    """Cycle count for one frame, `doc/cnn_accel_sizing_proposal.md` section 3:
    per-pixel cycles = ceil(9*in_channels/pe_cols), output-channel tile
    count OT = ceil(out_channels/pe_rows), layer cycles = pixels *
    cyc_per_pixel * OT, summed over `BACKBONE_TARGET`.
    """
    area_ratio = (input_w * input_h) / (_REFERENCE_W * _REFERENCE_H)
    total = 0.0
    for layer in BACKBONE_TARGET:
        pixels = layer.pixels_at_reference * area_ratio
        cycles_per_pixel = math.ceil(9 * layer.in_channels / pe_cols)
        out_tiles = math.ceil(layer.out_channels / pe_rows)
        total += pixels * cycles_per_pixel * out_tiles
    return round(total)


def frame_budget_cycles(clock_hz: int = CLOCK_HZ, fps: int = TARGET_FPS) -> int:
    """Cycle budget for one frame at `clock_hz` to sustain `fps`."""
    return clock_hz // fps

# ---------------------------------------------------------------------------
# Opcodes: instruction word W0 bits [7:0]. Values only -- VHDL type
# (`std_ulogic_vector(7 downto 0)`) and golden-model `OPCODE_*` names are
# derived from this table by their respective consumers.
# ---------------------------------------------------------------------------

OPCODES: dict[str, int] = {
    "HALT": 0x00,
    "CONV2D": 0x01,
    "DWCONV2D": 0x02,
    "POOL_MAX": 0x03,
    "POOL_AVG": 0x04,
    "FC": 0x05,
    # ISA v2.0 (doc/cnn_accel_top_v2_arch.md section 5): explicit data
    # movement between DDR and the local tensor scratchpad, plus the
    # local-to-local elementwise/resample family. The v1.2 compute opcodes
    # above keep their values (0x00-0x05), so a v1.2 program is a valid
    # v2.0 program; the v2.0 family starts at 0x10 to leave 0x06-0x0F free
    # for future compute opcodes rather than interleaving the two families.
    "LOAD": 0x10,
    "STORE": 0x11,
    "LOADW": 0x12,
    "ADD": 0x13,
    "UPSAMPLE": 0x14,
    "COPY": 0x15,
    "ACT": 0x16,
}

# ---------------------------------------------------------------------------
# Flags: bit indices into the W0 flags byte (bits [15:8] of the instruction
# word's first 32-bit word).
# ---------------------------------------------------------------------------

FLAGS: dict[str, int] = {
    "RELU_EN": 0,
    "BIAS_EN": 1,
    "REQUANT_EN": 2,
    "PAD_EN": 3,
    # ISA v1.1 (doc/tosa_compiler_plan.md section 5, extension 1 / HW
    # milestone H1): when set, the epilogue clamps to [clamp_min, clamp_max]
    # instead of the RELU_EN-derived [0 or -128, 127] range. RELU_EN is
    # ignored while CLAMP_EN is set.
    "CLAMP_EN": 4,
    # ISA v1.2 (doc/tosa_compiler_plan.md section 5, extension 2 / HW
    # milestone H2): when set, every output channel's (multiplier, shift)
    # comes from the DDR table at `scale_addr` (SCALE_TABLE_ENTRY_BYTES per
    # channel) instead of the descriptor's requant_scale/requant_shift,
    # which are then ignored. Only meaningful with REQUANT_EN.
    "PER_CHANNEL_EN": 5,
    # ISA v2.0 (doc/cnn_accel_top_v2_arch.md section 5): the epilogue's
    # activation comes from the 256-entry int8->int8 LUT in LOCAL_WEIGHT
    # instead of the RELU_EN/CLAMP_EN range logic. This is the H1
    # general-activation path (SiLU etc.), and it is what the standalone
    # `ACT` opcode uses unconditionally.
    "ACT_LUT_EN": 6,
    # ISA v2.0: this command's weights/bias/scale are already resident in
    # LOCAL_WEIGHT from a previous LOADW or conv, so the weight buffer must
    # not be refilled. This is the flag residency invariant R6 asserts on
    # (a repeated conv adds zero to WEIGHT_LOAD_BYTES).
    "WEIGHT_REUSE": 7,
}

# ---------------------------------------------------------------------------
# Storage spaces (ISA v2.0, doc/cnn_accel_top_v2_arch.md section 5.1).
#
# Every address-bearing operand carries a 2-bit tag naming the address
# space its address refers to. `DDR` is 0 so that a v1.2 instruction word
# -- whose W0 bytes 2-3 were `reserved, must be 0` -- decodes as
# "all operands in DDR", which is exactly v1.2 semantics. This is the
# mechanism that makes v2.0 backward compatible by construction rather
# than by convention.
# ---------------------------------------------------------------------------

SPACES: dict[str, int] = {
    "DDR": 0,
    "LOCAL_TENSOR": 1,
    "LOCAL_WEIGHT": 2,
    "RESERVED": 3,
}

SPACE_TAG_BITS = 2

# Bit index of each operand's tag inside the `spaces` byte (W0 bits
# [23:16]). Same idiom as `FLAGS` above: indices into a named byte of the
# instruction word, not absolute bit positions.
SPACE_FIELDS: dict[str, int] = {
    "SRC0": 0,
    "SRC1": 2,
    "DST": 4,
    "WGT": 6,
}

# ---------------------------------------------------------------------------
# Instruction word layout: an ordered, contiguous table of (field_name,
# width_bytes, signed) tuples. Byte offsets are DERIVED by walking this
# table in encounter order -- never hardcode an offset anywhere else.
#
# Reserved gaps that doc/cnn_accel_arch.md's ISA table documents as "must
# be 0" are explicit `RESERVED` entries so they participate in the walk
# (and therefore in `isa_total_bytes()`/the self-consistency test) without
# ever being emitted as a named, addressable field.
# ---------------------------------------------------------------------------

RESERVED = "reserved"

INSTR_WORD_BYTES = 64


class IsaField(NamedTuple):
    name: str
    width_bytes: int
    signed: bool = False


ISA_LAYOUT: tuple[IsaField, ...] = (
    IsaField("opcode", 1),
    IsaField("flags", 1),
    # W0 byte 2 (ISA v2.0): the four 2-bit operand storage-space tags, see
    # `SPACES`/`SPACE_FIELDS`. Reserved-must-be-0 in v1.2, and DDR == 0, so
    # v1.2 programs decode unchanged.
    IsaField("spaces", 1),
    IsaField(RESERVED, 1),  # W0 byte 3
    IsaField("in_addr", 4),
    IsaField("out_addr", 4),
    IsaField("weight_addr", 4),
    IsaField("bias_addr", 4),
    IsaField("in_width", 2),
    IsaField("in_height", 2),
    IsaField("in_channels", 2),
    IsaField("out_channels", 2),
    IsaField("kernel_h", 1),
    IsaField("kernel_w", 1),
    IsaField("stride_h", 1),
    IsaField("stride_w", 1),
    IsaField("pad_top", 1),
    IsaField("pad_bottom", 1),
    IsaField("pad_left", 1),
    IsaField("pad_right", 1),
    IsaField("requant_scale", 4, signed=True),
    IsaField("requant_shift", 1),
    IsaField(RESERVED, 3),  # W10 bytes 41-43
    IsaField("pool_kernel_h", 1),
    IsaField("pool_kernel_w", 1),
    IsaField("pool_stride_h", 1),
    IsaField("pool_stride_w", 1),
    IsaField("next_instr_addr", 4),
    # W13 (ISA v1.1, H1): output offset added after the rounded requant
    # shift, and the CLAMP_EN clamp bounds. All three are signed; all three
    # must be 0 in a v1.0 program (which is exactly what the reserved-must-
    # be-0 rule already guaranteed, so v1.0 programs run unchanged).
    IsaField("output_offset", 2, signed=True),
    IsaField("clamp_min", 1, signed=True),
    IsaField("clamp_max", 1, signed=True),
    # W14 (ISA v1.2, H2): byte address of the per-channel requantization
    # table (PER_CHANNEL_EN). Must be 0 in a v1.0/v1.1 program (reserved-
    # must-be-0), and is ignored while PER_CHANNEL_EN is clear.
    IsaField("scale_addr", 4),
    # W15 (ISA v2.0): transfer size in bytes for the data-movement and
    # elementwise family (LOAD/STORE/LOADW/ADD/UPSAMPLE/COPY/ACT). For ADD
    # these same bits instead carry the second source operand's address
    # (`src1_addr`), since ADD's size is implied by its tensor geometry.
    # Reserved-must-be-0 in v1.2, and every v1.2 opcode ignores it.
    IsaField("xfer_bytes", 4),
)


class IsaFieldOffset(NamedTuple):
    name: str
    offset_bytes: int
    width_bytes: int
    signed: bool


def isa_field_offsets() -> list[IsaFieldOffset]:
    """Derive `[(name, offset_bytes, width_bytes, signed), ...]` for every
    non-reserved field in `ISA_LAYOUT`, in table order, by walking the
    table and accumulating byte offsets. Reserved gaps advance the running
    offset but are not returned."""
    offset = 0
    fields = []
    for field in ISA_LAYOUT:
        if field.name != RESERVED:
            fields.append(IsaFieldOffset(field.name, offset, field.width_bytes, field.signed))
        offset += field.width_bytes
    return fields


def isa_reserved_ranges() -> list[tuple[int, int]]:
    """Derive `[(start_byte, end_byte_inclusive), ...]` for every reserved
    gap in `ISA_LAYOUT`, in table order."""
    offset = 0
    ranges = []
    for field in ISA_LAYOUT:
        if field.name == RESERVED:
            ranges.append((offset, offset + field.width_bytes - 1))
        offset += field.width_bytes
    return ranges


def isa_total_bytes() -> int:
    """Total byte width of `ISA_LAYOUT`, walking every entry including
    reserved gaps. Must equal `INSTR_WORD_BYTES`."""
    return sum(field.width_bytes for field in ISA_LAYOUT)
