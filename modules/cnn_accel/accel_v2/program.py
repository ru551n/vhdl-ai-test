"""Lower a `planner.PlannedProgram` into a chained ISA v2.0 `DescV2`
program plus a `MemoryImage` ready for CSV export/import
(`doc/cnn_accel_top_v2_arch.md` sections 5-7).

This is the only module that actually allocates DDR addresses for
weights/bias/scale tables/LUTs/the program itself (`planner.py` already
allocated addresses for tensors -- graph inputs, graph outputs and
spills -- via the same shared `DdrMap`, so the two modules' allocations
never collide). It reuses `cnn_accel_model.py`'s packers
(`pack_weights_for_hw`/`pack_bias_for_hw`/`pack_scale_table_for_hw`/
`pack_activation_planes`) for every byte layout -- the same "single
source of truth for the channel-tiled-plane/OHWI-tiled layout" rule
`reference.py` follows, so a CSV this module emits and the bytes
`reference.py` computes in memory describe the same tensors byte for
byte (exercised by `tests/test_program.py`'s round-trip test).

Two spec ambiguities resolved here (beyond `reference.py`'s ADD
rescale-pair ambiguity):

* **Ambiguity #2 -- `ADD`'s `xfer_bytes` field.** Section 5.1's W15 row
  says `xfer_bytes` is repurposed as `src1_addr` for `ADD`; section
  5.2's opcode table separately describes `ADD` as "elementwise,
  `xfer_bytes` long", which cannot be true simultaneously (the field
  cannot hold both a byte count and an address). The word-layout table
  (5.1) is the more precise, field-by-field source of truth, and
  explicitly names the field's `space_src1`-tagged address role for
  `ADD`; the opcode table's "`xfer_bytes` long" is read as loose,
  reused wording rather than a second, conflicting bit-level spec. This
  module therefore encodes `ADD.xfer_bytes = src1_addr` and derives the
  elementwise length from `in_width * in_height * in_channels`, exactly
  like every other shaped opcode (`CONV2D`/`POOL_*`/`UPSAMPLE`).
* **Ambiguity #3 -- the standalone `ACT` LUT's table address.** No
  field is named for it anywhere in section 5.1; `weight_addr` (W3,
  `space_wgt`-tagged) is the only slot whose role -- "the compile-time
  side-table for this instruction" -- already matches (it plays the
  same part for `CONV2D`'s per-channel scale table, `scale_addr`, one
  slot over). This module reuses `weight_addr`/`space_wgt` for `ACT`'s
  256-byte LUT address; `bias_addr`/`scale_addr` are left `0` (unused
  by `ACT`).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import cnn_accel_model as golden

from accel_v2 import isa
from accel_v2.ddrmap import DdrMap
from accel_v2.memimage import MemoryImage
from accel_v2.model import ActOp, AddOp, Conv2dOp, CopyOp, DepthToSpaceOp, PoolOp, UpsampleOp
from accel_v2.planner import (
    ComputeStep,
    ConstLoadStep,
    MoveStep,
    PlannedProgram,
    RowCopyStep,
    descriptor_count,
)

#: `MoveStep.kind -> DescV2 opcode` (section 5.2: `LOAD`/`STORE` are one
#: DMA family differing only in operand space tags).
_MOVE_OPCODE = {
    "cold_load": isa.OPCODE_LOAD,
    "reload": isa.OPCODE_LOAD,
    "spill": isa.OPCODE_STORE,
}


@dataclass
class WeightAllocation:
    """Where one `Conv2dOp`'s compile-time constants landed in DDR.
    `bias_addr`/`scale_addr` are `None` when the op has no bias / no
    per-channel scale table (`flags().bias_en`/`per_channel_en` clear),
    matching `DescV2`'s own "field is ignored unless the matching flag
    is set" convention."""

    weight_addr: int
    bias_addr: int | None
    scale_addr: int | None


@dataclass
class ProgramImage:
    """Result of `emit_program`. `descs` is the decoded chain (handy for
    assertions in tests without re-parsing `image`); `image` is the same
    `MemoryImage` a VHDL testbench would load via CSV (section 7),
    containing the program, weights/bias/scale/LUT tables and the seeded
    graph-input tensors -- everything `program_addr` needs to run
    end to end except the host's `START` pulse."""

    descs: list[isa.DescV2]
    image: MemoryImage
    program_addr: int


def _weight_allocations(planned: PlannedProgram, ddr_map: DdrMap) -> dict[int, WeightAllocation]:
    """`id(Conv2dOp) -> WeightAllocation`. A `weight_reuse` op's entry is
    an alias of `reused_weight_op`'s own entry -- it never allocates a
    second time, mirroring `planner.py`'s zero-byte traffic charge for
    the same case (`Conv2dOp.flags`'s `WEIGHT_REUSE` docstring)."""
    allocations: dict[int, WeightAllocation] = {}
    #: `id(weight list) -> the allocation its first owner got`. Two
    #: convolutions that share a `weight` object are the same convolution
    #: -- which is exactly what `tiler.py` produces when it clones one
    #: conv onto `S` strips -- and writing `S` byte-identical copies of a
    #: weight image into DDR wastes both the image and the `WEIGHTS`
    #: region, which a tiled program at real channel counts runs out of.
    #: No untiled case is affected: `Model.conv2d` draws a fresh list for
    #: every convolution, so nothing there ever shares one.
    by_weights: dict[int, WeightAllocation] = {}
    for op in planned.model.ops:
        if not isinstance(op, Conv2dOp):
            continue
        if op.weight_reuse:
            assert op.reused_weight_op is not None, "weight_reuse set without reused_weight_op"
            allocations[id(op)] = allocations[id(op.reused_weight_op)]
            continue
        shared = by_weights.get(id(op.weight))
        if shared is not None:
            allocations[id(op)] = shared
            continue

        desc = op.weight_layer_desc()
        weight_addr = ddr_map.alloc(
            DdrMap.WEIGHTS, golden.packed_weight_count(desc, golden.TILE_CHANNELS, golden.PE_ROWS)
        )
        bias_addr = None
        if op.bias is not None:
            bias_addr = ddr_map.alloc(DdrMap.BIAS, golden.packed_bias_count(desc, golden.PE_ROWS) * 4)
        scale_addr = None
        if op.per_channel_scale is not None:
            scale_addr = ddr_map.alloc(DdrMap.SCALE, golden.packed_scale_table_bytes(desc, golden.PE_ROWS))
        allocations[id(op)] = by_weights[id(op.weight)] = WeightAllocation(
            weight_addr, bias_addr, scale_addr
        )
    return allocations


def _write_weight_images(
    image: MemoryImage, planned: PlannedProgram, allocations: dict[int, WeightAllocation]
) -> None:
    """Pack and write every non-`weight_reuse` `Conv2dOp`'s weight/bias/
    scale table, using the exact same `cnn_accel_model.pack_*` calls
    `reference.py` charges bytes for (never a second, hand-rolled
    layout)."""
    written: set[int] = set()
    for op in planned.model.ops:
        if not isinstance(op, Conv2dOp) or op.weight_reuse:
            continue
        alloc = allocations[id(op)]
        if alloc.weight_addr in written:
            # A strip clone sharing its original's image (see
            # `_weight_allocations`): the bytes are already there, and
            # they are byte-identical by construction.
            continue
        written.add(alloc.weight_addr)
        desc = op.weight_layer_desc()

        packed_w = golden.pack_weights_for_hw(op.weight, desc, golden.TILE_CHANNELS, golden.PE_ROWS)
        image.write_bytes(alloc.weight_addr, bytes(v & 0xFF for v in packed_w))

        if op.bias is not None:
            packed_b = golden.pack_bias_for_hw(op.bias, desc, golden.PE_ROWS)
            image.write_bytes(alloc.bias_addr, b"".join(struct.pack("<i", v) for v in packed_b))

        if op.per_channel_scale is not None:
            packed_s = golden.pack_scale_table_for_hw(op.per_channel_scale, desc, golden.PE_ROWS)
            image.write_bytes(alloc.scale_addr, packed_s)


def _lut_addrs(planned: PlannedProgram, ddr_map: DdrMap, image: MemoryImage) -> dict[int, int]:
    """`id(ActOp) -> DDR address of its `isa.ACT_LUT_BYTES`-byte
    int8->int8 LUT."""
    addrs: dict[int, int] = {}
    for op in planned.model.ops:
        if not isinstance(op, ActOp):
            continue
        addr = ddr_map.alloc(DdrMap.LUT, isa.ACT_LUT_BYTES)
        image.write_bytes(addr, bytes(v & 0xFF for v in op.lut))
        addrs[id(op)] = addr
    return addrs


def _row_copy_opcode(src_space: int, dst_space: int) -> int:
    """`LOAD`, `STORE` or `COPY` for one plane of a `RowCopyOp`, chosen
    from the two resolved operand spaces and nothing else.

    Section 5.2: `LOAD`/`STORE` are one DMA family differing only in
    which end is `LOCAL_TENSOR`, and `COPY` is the elementwise engine's
    opaque byte move that is legal between any two spaces. A row copy
    therefore needs no opcode of its own -- which is the whole reason the
    tiling design fits inside ISA v2.1 unchanged. Where a row copy sits
    (group-input load, join window, strip store) is a fact about the
    graph, never about the descriptor."""
    if src_space == isa.SPACE_DDR and dst_space == isa.SPACE_LOCAL_TENSOR:
        return isa.OPCODE_LOAD
    if src_space == isa.SPACE_LOCAL_TENSOR and dst_space == isa.SPACE_DDR:
        return isa.OPCODE_STORE
    return isa.OPCODE_COPY


def _row_copy_descs(step: RowCopyStep, next_addrs: list[int]) -> list[isa.DescV2]:
    """One descriptor per activation plane. `next_addrs` is parallel to
    `step.transfers`: each plane chains to the following plane, and the
    last to whatever follows the whole step."""
    opcode = _row_copy_opcode(step.src_space, step.dst_space)
    return [
        isa.DescV2(
            opcode=opcode,
            space_src0=step.src_space,
            space_dst=step.dst_space,
            in_addr=src_addr,
            out_addr=dst_addr,
            xfer_bytes=nbytes,
            next_instr_addr=next_addr,
        )
        for (src_addr, dst_addr, nbytes), next_addr in zip(step.transfers, next_addrs)
    ]


def _const_load_descs(
    step: ConstLoadStep,
    next_addrs: list[int],
    allocations: dict[int, WeightAllocation],
) -> list[isa.DescV2]:
    """One `LOAD` per packed sub-image: DDR (where `_write_weight_images`
    put the bytes) into the scratchpad address the planner reserved.

    A plain `LOAD` with `space_dst = LOCAL_TENSOR`, not `LOADW`: `LOADW`
    writes the `LOCAL_WEIGHT` space -- the weight buffer inside the conv
    core -- which is a different thing entirely, is not addressable per
    output-channel tile, and is what `WEIGHT_LOAD_BYTES` was named after.
    What residency wants is the images sitting in the ordinary tensor
    scratchpad so that `st_wgt_req` can fetch tile `pass` out of them
    through port `r1`."""
    alloc = allocations[id(step.conv)]
    sources = {
        "weight": alloc.weight_addr,
        "bias": alloc.bias_addr,
        "scale": alloc.scale_addr,
    }
    descs = []
    for (kind, local_addr, nbytes), next_addr in zip(step.images, next_addrs):
        src = sources[kind]
        assert src is not None, f"program: resident {kind} image with no DDR allocation"
        descs.append(
            isa.DescV2(
                opcode=isa.OPCODE_LOAD,
                space_src0=isa.SPACE_DDR,
                space_dst=isa.SPACE_LOCAL_TENSOR,
                in_addr=src,
                out_addr=local_addr,
                xfer_bytes=nbytes,
                next_instr_addr=next_addr,
            )
        )
    return descs


def _move_descs(step: MoveStep, next_addrs: list[int]) -> list[isa.DescV2]:
    """One `LOAD`/`STORE` per `MoveStep.unit` chunk -- one descriptor for
    the whole move whenever the buffer is confined the strict way, which
    is every untiled tensor. See `MoveStep.unit` for why a plane-confined
    buffer cannot be moved in one request."""
    opcode = _MOVE_OPCODE[step.kind]
    return [
        isa.DescV2(
            opcode=opcode,
            space_src0=step.src_space,
            space_dst=step.dst_space,
            in_addr=step.src_addr + offset,
            out_addr=step.dst_addr + offset,
            xfer_bytes=nbytes,
            next_instr_addr=next_addr,
        )
        for (offset, nbytes), next_addr in zip(step.transfer_offsets, next_addrs)
    ]


def _compute_desc(
    step: ComputeStep,
    next_addr: int,
    weight_allocations: dict[int, WeightAllocation],
    lut_addrs: dict[int, int],
) -> isa.DescV2:
    op = step.op

    if isinstance(op, Conv2dOp):
        x = op.inputs[0]
        # R8. `WEIGHT_REUSE` does not mean "these weights are already in
        # the weight buffer, skip the fetch for pass 0"; `cmd_proc.vhd`
        # (:1177-1179) skips the weight refill for EVERY output-channel
        # pass, so with more than one pass every pass but the last-loaded
        # one would run on the wrong tile. It is correct only when the op
        # has a single pass, i.e. `out_channels <= PE_ROWS`. Nothing in
        # the catalogue has ever violated that, and the tiler never sets
        # the flag at all (a strip clone reuses the weight *object*, not
        # the descriptor bit) -- but the failure mode is silent numeric
        # corruption on the DUT only, so it is refused here, at the one
        # place that can see both the flag and the channel count.
        n_ot = -(-op.output.channels // golden.PE_ROWS)
        if op.weight_reuse and n_ot > 1:
            raise ValueError(
                f"program: conv '{op.name}' sets WEIGHT_REUSE with {op.output.channels} "
                f"output channels, which is {n_ot} output-channel passes. The hardware "
                "skips the weight refill for every pass, not just the first, so passes "
                "2.. would compute on whichever tile happened to be resident. "
                f"WEIGHT_REUSE is only correct for out_channels <= {golden.PE_ROWS}; drop "
                "`weight_reuse_from` and let this conv fetch its own weights."
            )
        alloc = weight_allocations[id(op)]
        clamp_min, clamp_max = op.clamp if op.clamp is not None else (0, 0)
        pad_top, pad_bottom, pad_left, pad_right = op.padding
        # Weight residency (design D6). `space_wgt` is ONE tag covering
        # all three side tables, so either the whole image group is in
        # the scratchpad or none of it is -- which is exactly how
        # `planner.py` places it. The addresses swap with the space; no
        # other field changes, and the descriptor is otherwise the one
        # this emitter has always produced.
        weight_addrs = step.weight_addrs or {}
        return isa.DescV2(
            opcode=isa.OPCODE_CONV2D,
            flags=op.flags(),
            space_src0=step.input_spaces[0],
            space_dst=step.output_space,
            space_wgt=step.weight_space,
            in_addr=step.input_addrs[0],
            out_addr=step.output_addr,
            weight_addr=weight_addrs.get("weight", alloc.weight_addr),
            bias_addr=weight_addrs.get("bias", alloc.bias_addr or 0),
            scale_addr=weight_addrs.get("scale", alloc.scale_addr or 0),
            in_width=x.width,
            in_height=x.height,
            in_channels=x.channels,
            out_channels=op.output.channels,
            kernel_h=op.kernel[0],
            kernel_w=op.kernel[1],
            stride_h=op.stride[0],
            stride_w=op.stride[1],
            pad_top=pad_top,
            pad_bottom=pad_bottom,
            pad_left=pad_left,
            pad_right=pad_right,
            pad_value=op.pad_value,
            requant_scale=op.requant_scale,
            requant_shift=op.requant_shift,
            output_offset=op.output_offset,
            clamp_min=clamp_min,
            clamp_max=clamp_max,
            next_instr_addr=next_addr,
        )

    if isinstance(op, PoolOp):
        x = op.inputs[0]
        clamp_min, clamp_max = op.clamp if op.clamp is not None else (0, 0)
        opcode = isa.OPCODE_POOL_MAX if op.mode == "max" else isa.OPCODE_POOL_AVG
        return isa.DescV2(
            opcode=opcode,
            flags=op.flags(),
            space_src0=step.input_spaces[0],
            space_dst=step.output_space,
            in_addr=step.input_addrs[0],
            out_addr=step.output_addr,
            in_width=x.width,
            in_height=x.height,
            in_channels=x.channels,
            pool_kernel_h=op.kernel[0],
            pool_kernel_w=op.kernel[1],
            pool_stride_h=op.stride[0],
            pool_stride_w=op.stride[1],
            pad_top=op.padding[0],
            pad_bottom=op.padding[1],
            pad_left=op.padding[2],
            pad_right=op.padding[3],
            pad_value=op.pad_value,
            requant_scale=op.requant_scale,
            requant_shift=op.requant_shift,
            output_offset=op.output_offset,
            clamp_min=clamp_min,
            clamp_max=clamp_max,
            next_instr_addr=next_addr,
        )

    if isinstance(op, AddOp):
        # Ambiguity #2 (module docstring): xfer_bytes carries src1_addr,
        # not a byte count -- the elementwise length comes from
        # in_width/in_height/in_channels, like every other shaped opcode.
        a = op.inputs[0]
        return isa.DescV2(
            opcode=isa.OPCODE_ADD,
            flags=1 << golden.FLAG_REQUANT_EN,
            space_src0=step.input_spaces[0],
            space_src1=step.input_spaces[1],
            space_dst=step.output_space,
            in_addr=step.input_addrs[0],
            out_addr=step.output_addr,
            xfer_bytes=step.input_addrs[1],
            in_width=a.width,
            in_height=a.height,
            in_channels=a.channels,
            requant_scale=op.requant_scale,
            requant_shift=op.requant_shift,
            next_instr_addr=next_addr,
        )

    if isinstance(op, DepthToSpaceOp):
        x = op.inputs[0]
        # The only elementwise descriptor that carries `out_channels`:
        # every other opcode in the family is channel-preserving, so the
        # field is left at 0 there and the engine derives everything from
        # `in_channels`. `dts_factor` (W10 byte 42) is ISA v2.2.
        return isa.DescV2(
            opcode=isa.OPCODE_DEPTH_TO_SPACE,
            space_src0=step.input_spaces[0],
            space_dst=step.output_space,
            in_addr=step.input_addrs[0],
            out_addr=step.output_addr,
            in_width=x.width,
            in_height=x.height,
            in_channels=x.channels,
            out_channels=op.output.channels,
            dts_factor=op.factor,
            next_instr_addr=next_addr,
        )

    if isinstance(op, UpsampleOp):
        x = op.inputs[0]
        return isa.DescV2(
            opcode=isa.OPCODE_UPSAMPLE,
            space_src0=step.input_spaces[0],
            space_dst=step.output_space,
            in_addr=step.input_addrs[0],
            out_addr=step.output_addr,
            in_width=x.width,
            in_height=x.height,
            in_channels=x.channels,
            next_instr_addr=next_addr,
        )

    if isinstance(op, CopyOp):
        return isa.DescV2(
            opcode=isa.OPCODE_COPY,
            space_src0=step.input_spaces[0],
            space_dst=step.output_space,
            in_addr=step.input_addrs[0],
            out_addr=step.output_addr,
            xfer_bytes=op.inputs[0].size_bytes,
            next_instr_addr=next_addr,
        )

    if isinstance(op, ActOp):
        # Ambiguity #3 (module docstring): the standalone LUT address
        # rides in weight_addr/space_wgt, the same "compile-time side
        # table" slot CONV2D uses for its own per-channel scale table.
        return isa.DescV2(
            opcode=isa.OPCODE_ACT,
            flags=1 << golden.FLAG_ACT_LUT_EN,
            space_src0=step.input_spaces[0],
            space_dst=step.output_space,
            space_wgt=isa.SPACE_DDR,
            in_addr=step.input_addrs[0],
            out_addr=step.output_addr,
            weight_addr=lut_addrs[id(op)],
            xfer_bytes=op.inputs[0].size_bytes,
            next_instr_addr=next_addr,
        )

    raise TypeError(f"program: unhandled op type {type(op)!r}")


def emit_program(planned: PlannedProgram) -> ProgramImage:
    """Lower `planned` into a `ProgramImage`: a chained `DescV2` list and
    the `MemoryImage` a testbench loads to run it. Allocates DDR space
    for weights/bias/scale/LUT tables and for the program itself out of
    `planned.ddr_map` -- the same map `Planner.plan` already used for
    tensor inputs/outputs/spills, so addresses never collide.

    Seeds every graph input's packed bytes into `image` first, exactly
    like `reference.run_reference` does (`reference.py`'s own docstring
    notes the two must agree byte for byte -- `tests/test_program.py`'s
    round-trip test checks this)."""
    ddr_map = planned.ddr_map
    image = MemoryImage()

    for t in planned.model.inputs:
        addr = planned.tensor_ddr_addr[t.name]
        packed = golden.pack_activation_planes(t.data, t.width, t.height, t.channels)
        image.write_bytes(addr, bytes(v & 0xFF for v in packed))

    weight_allocations = _weight_allocations(planned, ddr_map)
    _write_weight_images(image, planned, weight_allocations)
    lut_addrs = _lut_addrs(planned, ddr_map, image)

    # Allocate every descriptor's DDR address up front (one per planned
    # step, plus the closing HALT) so each step's next_instr_addr can
    # point at the following one already known, per section 6's chained-
    # descriptor program region.
    # One descriptor per step, except a `RowCopyStep`, which is one per
    # activation plane -- `planner.descriptor_count` is the single place
    # that rule is stated, and the planner charges the program fetch from
    # the same function.
    counts = [descriptor_count(step) for step in planned.steps]
    n_descs = sum(counts) + 1
    addrs = [ddr_map.alloc(DdrMap.PROGRAM, isa.INSTR_WORD_BYTES) for _ in range(n_descs)]
    program_addr = addrs[0]

    descs: list[isa.DescV2] = []
    cursor = 0
    for step, count in zip(planned.steps, counts):
        # Each descriptor chains to the one after it, which for a
        # multi-plane row copy means to the next plane of the same step.
        next_addrs = addrs[cursor + 1 : cursor + count + 1]
        if isinstance(step, RowCopyStep):
            descs.extend(_row_copy_descs(step, next_addrs))
        elif isinstance(step, ConstLoadStep):
            descs.extend(_const_load_descs(step, next_addrs, weight_allocations))
        elif isinstance(step, MoveStep):
            descs.extend(_move_descs(step, next_addrs))
        else:
            assert isinstance(step, ComputeStep)
            descs.append(_compute_desc(step, next_addrs[0], weight_allocations, lut_addrs))
        cursor += count
    descs.append(isa.DescV2(opcode=isa.OPCODE_HALT))

    for addr, desc in zip(addrs, descs):
        image.write_bytes(addr, isa.encode_desc(desc))

    return ProgramImage(descs=descs, image=image, program_addr=program_addr)


__all__ = ["WeightAllocation", "ProgramImage", "emit_program"]
