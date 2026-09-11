"""Bit-exact reference execution of a `planner.PlannedProgram`
(`doc/cnn_accel_top_v2_arch.md` sections 3-5, 10).

This is the "second implementation" the residency tests actually trust:
it walks the exact `ComputeStep`/`MoveStep` sequence `planner.Planner`
produced, reading/writing two address spaces (a `memimage.MemoryImage`
standing in for DDR, and a flat `bytearray` standing in for
`cnn_accel_tensor_mem`), and returns both the resulting tensor values and
the DDR/local traffic actually incurred -- so a test can assert
`planned.traffic == result.traffic` (the prediction matched what running
the program would really move) as well as check outputs.

Anti-fork guard: every opcode that existed in ISA v1.2 (`CONV2D`,
`POOL_MAX`, `POOL_AVG`) is executed by calling `cnn_accel_model.conv2d`/
`pool_max`/`pool_avg` directly on the unpacked LOGICAL activation --
*not* a second, hand-rolled convolution/pooling loop. `ADD`'s per-operand
rescale reuses `cnn_accel_model.round_shift_right_signed`/
`saturate_signed` (the same two primitives `bias_requantize_relu` is
built from), so its bit-exactness rests on the same, already-tested
rounding rule.

That guard now covers the ISA v2.0 family too: `ADD`, `UPSAMPLE` and
`ACT` are `cnn_accel_model.elementwise_add`/`upsample_nearest`/`act_lut`,
called on the unpacked LOGICAL activation exactly like the v1.2 opcodes
above. This module hand-rolls no op arithmetic of its own at all; what it
still owns alone is the *residency* model -- which address space each
operand is read from, and what traffic that costs -- which is the thing
the golden model deliberately does not have (it has one flat memory, see
`cnn_accel_model.decode_instruction`).

Weights/bias/per-channel scale tables are consumed directly from the
`Conv2dOp` in their LOGICAL form (never packed/unpacked through a byte
image here) -- they are compile-time constants that never travel through
`LOCAL_TENSOR`, so simulating their packed DDR bytes would add packing/
unpacking cost without exercising anything this module is responsible
for verifying (that byte image's fidelity is covered directly by
`cnn_accel_model.py`'s own tests and by `program.py`'s CSV round-trip
test). Their *byte counts* are still charged against `DdrTraffic`
(`packed_weight_count`/`packed_bias_count`/`packed_scale_table_bytes`,
the same calls `planner.py` makes), so the read/weight-byte totals stay
meaningful.

Channel `concat`/`split` execute no step at all -- they are pure buffer
aliasing (`planner.py`'s module docstring), so the bytes they "produce"
are already in memory, written there by the producers of their parts or
by whoever produced their parent. What this module still owes such a
tensor is its LOGICAL HWC value, for `tensor_data`; that is recovered
after the step loop by `_materialize_alias_values`, which composes a
concat result from its parts and slices a split view out of its parent
rather than re-reading memory (the buffer may legitimately have been
freed and reused by then).

Bank interleaving/arbitration inside `cnn_accel_tensor_mem` (section 4)
has no Python-visible effect on *which bytes end up where* -- every
legal `(base, length)` job stays inside one bank by construction (the
planner never allocates across a page it does not know exists) -- so
this module models the scratchpad as one flat byte array, matching
`planner.py`'s own scoping note.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cnn_accel_model as golden

from accel_v2 import isa
from accel_v2.memimage import MemoryImage
from accel_v2.model import (
    ActOp,
    AddOp,
    Conv2dOp,
    CopyOp,
    Model,
    Op,
    PLANE_CHANNELS,
    alias_byte_offset,
    alias_root,
    PoolOp,
    RowCopyOp,
    Tensor,
    DepthToSpaceOp,
    UpsampleOp,
)
from accel_v2.planner import (
    ComputeStep,
    ConstLoadStep,
    DdrTraffic,
    MoveStep,
    PlannedProgram,
    RowCopyStep,
    charge_weight_fill,
    descriptor_count,
    ifmap_passes,
)


@dataclass
class ExecutionResult:
    #: Actually-incurred traffic, directly comparable to
    #: `PlannedProgram.traffic` (same dataclass, same field meanings).
    traffic: DdrTraffic
    #: `Tensor.name -> LOGICAL HWC int8 values`, for every tensor that
    #: was ever produced or was a graph input.
    tensor_data: dict[str, list[int]] = field(default_factory=dict)


def _to_signed_bytes(raw: bytes) -> list[int]:
    return [b - 256 if b >= 128 else b for b in raw]


def _exec_conv2d(op: Conv2dOp, input_values: list[int]) -> list[int]:
    x = op.inputs[0]
    flags = op.flags()
    clamp_min, clamp_max = op.clamp if op.clamp is not None else (0, 0)
    pad_top, pad_bottom, pad_left, pad_right = op.padding

    desc = golden.LayerDesc(
        opcode=golden.OPCODE_CONV2D,
        flags=flags,
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
    )
    bias_values = op.bias if op.bias is not None else [0] * op.output.channels
    return golden.conv2d(input_values, op.weight, bias_values, desc, op.per_channel_scale)


def _exec_pool(op: PoolOp, input_values: list[int]) -> list[int]:
    x = op.inputs[0]
    flags = op.flags()
    clamp_min, clamp_max = op.clamp if op.clamp is not None else (0, 0)

    desc = golden.LayerDesc(
        opcode=golden.OPCODE_POOL_MAX if op.mode == "max" else golden.OPCODE_POOL_AVG,
        flags=flags,
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
    )
    return golden.pool_max(input_values, desc) if op.mode == "max" else golden.pool_avg(input_values, desc)


def _exec_add(op: AddOp, a_values: list[int], b_values: list[int]) -> list[int]:
    """`dst = sat_i8(requant(src0) + requant(src1))` (section 5.2), via
    `cnn_accel_model.elementwise_add` -- the same anti-fork rule as
    `_exec_conv2d`/`_exec_pool` above: the arithmetic lives in the golden
    model, this function only builds the descriptor that names it.

    One `(requant_scale, requant_shift)` pair covers both operands
    because the descriptor has only one (ambiguity #1, see `AddOp`'s
    docstring: this assumes both operands are already expressed on a
    common scale, as is typical for a residual/skip-connection add)."""
    desc = golden.LayerDesc(
        opcode=golden.OPCODE_ADD,
        requant_scale=op.requant_scale,
        requant_shift=op.requant_shift,
    )
    return golden.elementwise_add(a_values, b_values, desc)


def _exec_upsample(op: UpsampleOp, input_values: list[int]) -> list[int]:
    """Nearest-neighbour replication, via `cnn_accel_model.upsample_nearest`."""
    x = op.inputs[0]
    desc = golden.LayerDesc(
        opcode=golden.OPCODE_UPSAMPLE,
        in_width=x.width,
        in_height=x.height,
        in_channels=x.channels,
    )
    return golden.upsample_nearest(input_values, desc, op.factor)


def _exec_depth_to_space(op: DepthToSpaceOp, input_values: list[int]) -> list[int]:
    """Pixel-shuffle, via `cnn_accel_model.depth_to_space`.

    `out_channels` is the OUTPUT tensor's channel count, which is what the
    descriptor's own `out_channels` field carries -- the golden model reads
    it off the `LayerDesc`, so it must be filled in here even though every
    other elementwise op leaves it at zero."""
    x = op.inputs[0]
    desc = golden.LayerDesc(
        opcode=golden.OPCODE_DEPTH_TO_SPACE,
        in_width=x.width,
        in_height=x.height,
        in_channels=x.channels,
        out_channels=op.output.channels,
        dts_factor=op.factor,
    )
    return golden.depth_to_space(input_values, desc, op.factor)


def _exec_act(op: ActOp, input_values: list[int]) -> list[int]:
    """The standalone 256-entry LUT, via `cnn_accel_model.act_lut`."""
    return golden.act_lut(input_values, op.lut)


def _materialize_alias_values(model: Model, tensor_data: dict[str, list[int]]) -> None:
    """Fill in `tensor_data` for every `concat`/`split` tensor.

    Both are address arithmetic, not commands, so no step ever produced
    them -- but a test still wants their values (a concat result is
    frequently the graph output). Rules, mirroring `Tensor.alias_role`:

    * a tensor with `alias_parts` (a CONCAT result that owns a buffer) is
      the channel-wise concatenation of those parts, in plane order;
    * a tensor with an `alias_parent` and no parts (a SPLIT view, or a
      CONCAT that collapsed into a re-view) is the channel-wise slice of
      its parent starting at `alias_plane_offset * T`.

    Parts are resolved before parents, so a nested concat
    (`concat([concat([a, b]), c])`) composes bottom-up and never
    recurses into itself.
    """

    def value_of(t: Tensor) -> list[int]:
        cached = tensor_data.get(t.name)
        if cached is not None:
            return cached
        pixels = t.height * t.width

        if t.alias_parts:
            values = [0] * (pixels * t.channels)
            for part in t.alias_parts:
                part_values = value_of(part)
                first_channel = part.alias_plane_offset * PLANE_CHANNELS
                for pixel in range(pixels):
                    dst = pixel * t.channels + first_channel
                    src = pixel * part.channels
                    values[dst : dst + part.channels] = part_values[src : src + part.channels]
            tensor_data[t.name] = values
            return values

        if t.alias_parent is not None:
            parent = t.alias_parent
            parent_values = value_of(parent)
            first_channel = t.alias_plane_offset * PLANE_CHANNELS
            values = []
            for pixel in range(pixels):
                src = pixel * parent.channels + first_channel
                values.extend(parent_values[src : src + t.channels])
            tensor_data[t.name] = values
            return values

        raise KeyError(t.name)

    for tensor in model.tensors:
        if tensor.name in tensor_data:
            continue
        if tensor.alias_parts or tensor.alias_parent is not None:
            try:
                value_of(tensor)
            except KeyError:
                # A part that no step ever produced: the graph is dead
                # here, and there is no value to report. Not an error --
                # `planner.py` would already have refused anything that
                # actually needed those bytes.
                pass


def _read_back_row_copy_targets(
    planned: PlannedProgram,
    image: MemoryImage,
    local: bytearray,
    targets: dict[str, Tensor],
    tensor_data: dict[str, list[int]],
) -> None:
    """Recover the logical value of every tensor that was assembled by
    `RowCopyOp`s, by reading its buffer back once the program has run.

    A tiled group output is written `S` windows at a time and is the
    `output` of `S` different steps, none of which knows the whole
    tensor; the only place its value exists is the buffer itself. That is
    also the honest way to report it -- the oracle compares a tiled
    program's outputs against an untiled program's, and reading the bytes
    the strips actually wrote is precisely the claim under test.

    Charges no traffic: this is the test harness looking at memory, not
    an instruction the hardware would execute (`program.py` emits nothing
    for it), exactly as `_materialize_alias_values` charges nothing for
    composing a concat result."""
    for name, tensor in targets.items():
        root = alias_root(tensor)
        addr = planned.tensor_ddr_addr.get(root.name)
        if addr is None:
            # Not in DDR: an extract window into the scratchpad. Its
            # value is only interesting if some later op read it, and
            # such an op recorded it itself; skip rather than guess at a
            # local address the plan may since have reused.
            continue
        offset = alias_byte_offset(tensor)
        raw = image.read_bytes(addr + offset, tensor.size_bytes)
        tensor_data[name] = golden.unpack_activation_planes(
            _to_signed_bytes(raw), tensor.width, tensor.height, tensor.channels
        )


def run_reference(planned: PlannedProgram, image: MemoryImage) -> ExecutionResult:
    """Execute `planned` against `image` (mutated in place, exactly like
    the real DDR memory model would be) plus a fresh local scratchpad
    sized `planned.tensor_mem_bytes`. Seeds `image` with every graph
    input's packed bytes first (so this function alone is enough to
    populate an empty `MemoryImage`; `program.emit_program` does the
    same seeding for the DUT-facing image, and both must agree byte for
    byte -- see `tests/test_program.py`)."""
    local = bytearray(planned.tensor_mem_bytes)
    traffic = DdrTraffic()
    tensor_data: dict[str, list[int]] = {t.name: list(t.data) for t in planned.model.inputs}

    for t in planned.model.inputs:
        addr = planned.tensor_ddr_addr[t.name]
        packed = golden.pack_activation_planes(t.data, t.width, t.height, t.channels)
        image.write_bytes(addr, bytes(v & 0xFF for v in packed))

    def read_from(space: int, addr: int, nbytes: int) -> bytes:
        if space == isa.SPACE_DDR:
            return image.read_bytes(addr, nbytes)
        return bytes(local[addr : addr + nbytes])

    def write_to(space: int, addr: int, data: bytes) -> None:
        if space == isa.SPACE_DDR:
            image.write_bytes(addr, data)
        else:
            local[addr : addr + len(data)] = data

    # Byte ranges of every buffer the planner could not place in the
    # scratchpad and left in DDR (`PlannedProgram.ddr_placements`). Used
    # only to split the DDR totals into "traffic this program would have
    # had anyway" and "traffic the overflow fallback cost", so that
    # `planned.traffic == result.traffic` still compares every field.
    resident_ranges = [(addr, addr + size) for _, addr, size in planned.ddr_placements]
    # ...and, separately, the tensors that live in DDR *by design* (a
    # fusion group's boundaries). Same mechanism, different question --
    # see `DdrTraffic.pinned_read_bytes`.
    pinned_ranges = [(addr, addr + size) for _, addr, size in planned.pinned_placements]

    def in_ranges(ranges, addr: int) -> bool:
        return any(lo <= addr < hi for lo, hi in ranges)

    def is_resident(addr: int) -> bool:
        return in_ranges(resident_ranges, addr)

    def count(space: int, addr: int, nbytes: int, *, is_read: bool) -> None:
        if space == isa.SPACE_DDR:
            if is_read:
                traffic.read_bytes += nbytes
                if is_resident(addr):
                    traffic.ddr_resident_read_bytes += nbytes
                if in_ranges(pinned_ranges, addr):
                    traffic.pinned_read_bytes += nbytes
            else:
                traffic.write_bytes += nbytes
                if is_resident(addr):
                    traffic.ddr_resident_write_bytes += nbytes
                if in_ranges(pinned_ranges, addr):
                    traffic.pinned_write_bytes += nbytes
        else:
            if is_read:
                traffic.local_read_bytes += nbytes
            else:
                traffic.local_write_bytes += nbytes

    def unpack(t: Tensor, raw: bytes) -> list[int]:
        return golden.unpack_activation_planes(_to_signed_bytes(raw), t.width, t.height, t.channels)

    def pack(t: Tensor, values: list[int]) -> bytes:
        return bytes(v & 0xFF for v in golden.pack_activation_planes(values, t.width, t.height, t.channels))

    #: Every tensor a `RowCopyOp` wrote into, so its logical value can
    #: be recovered from memory after the run. A strip *store*'s
    #: destination is written a window at a time by several steps and is
    #: never the output of any single one, so nothing in the loop below
    #: can name its value -- but a group output (and every graph output
    #: of a tiled program) is exactly such a tensor.
    row_copy_targets: dict[str, Tensor] = {}

    for step in planned.steps:
        if isinstance(step, ConstLoadStep):
            # No bytes move here. This module consumes a convolution's
            # weights from the `Conv2dOp` in their LOGICAL form and never
            # builds their packed byte image (see the module docstring),
            # so there is nothing in `image` to copy and nothing in
            # `local` that would ever be read back -- only the traffic is
            # real, and it is charged in full.
            #
            # That is not a gap in the check, it is what makes the DUT
            # comparison sharp: the hardware really does fetch its tiles
            # from these scratchpad addresses, while the reference's
            # answer does not depend on them at all. A wrong local weight
            # address is therefore wrong on the DUT and right in the
            # reference -- the one direction of disagreement the standing
            # blind spot normally denies us.
            moved = step.nbytes
            traffic.read_bytes += moved
            traffic.local_write_bytes += moved
            traffic.tensor_load_count += len(step.images)
            continue

        if isinstance(step, RowCopyStep):
            moved = []
            for src_addr, dst_addr, nbytes in step.transfers:
                raw = read_from(step.src_space, src_addr, nbytes)
                write_to(step.dst_space, dst_addr, raw)
                count(step.src_space, src_addr, nbytes, is_read=True)
                count(step.dst_space, dst_addr, nbytes, is_read=False)
                moved.append(raw)
            planes = len(step.transfers)
            destination = step.op.output
            if step.op.dst_rows.r0 == 0 and step.op.dst_rows.r1 == destination.height:
                # The copy filled the destination completely, so the
                # bytes just moved ARE its packed image, plane by plane
                # in order -- no need to read anything back. This is the
                # *extract* form: a group input's strip, or a join
                # window. (The store form writes a row window of a taller
                # buffer and is recovered from memory after the run.)
                tensor_data[destination.name] = unpack(destination, b"".join(moved))
            if step.src_space == isa.SPACE_DDR and step.dst_space == isa.SPACE_LOCAL_TENSOR:
                traffic.tensor_load_count += planes
            elif step.src_space == isa.SPACE_LOCAL_TENSOR and step.dst_space == isa.SPACE_DDR:
                traffic.tensor_store_count += planes
            row_copy_targets[step.op.output.name] = step.op.output
            continue

        if isinstance(step, MoveStep):
            raw = read_from(step.src_space, step.src_addr, step.nbytes)
            write_to(step.dst_space, step.dst_addr, raw)
            count(step.src_space, step.src_addr, step.nbytes, is_read=True)
            count(step.dst_space, step.dst_addr, step.nbytes, is_read=False)
            # One `LOAD`/`STORE` retires per descriptor, and a
            # plane-confined buffer is moved one plane per descriptor
            # (`MoveStep.unit`). `spill_count`/`reload_count` stay per
            # *buffer*: they answer "how many values left the scratchpad",
            # not "how many instructions did it take".
            descs = descriptor_count(step)
            if step.kind == "spill":
                traffic.tensor_store_count += descs
                traffic.spill_count += 1
            else:
                traffic.tensor_load_count += descs
                if step.kind == "reload":
                    traffic.reload_count += 1
            continue

        assert isinstance(step, ComputeStep)
        op = step.op

        if isinstance(op, CopyOp):
            nbytes = op.inputs[0].size_bytes
            raw = read_from(step.input_spaces[0], step.input_addrs[0], nbytes)
            count(step.input_spaces[0], step.input_addrs[0], nbytes, is_read=True)
            write_to(step.output_space, step.output_addr, raw)
            count(step.output_space, step.output_addr, nbytes, is_read=False)
            # Unpacked from the bytes actually copied, not from the
            # source tensor's `tensor_data` entry: a COPY's source may be
            # a `split` view or a `concat` slice, whose logical values are
            # only materialized at the end of the run
            # (`_materialize_alias_values`). The byte image is authoritative
            # and available right here.
            tensor_data[op.output.name] = unpack(op.output, raw)
            continue

        input_values: list[list[int]] = []
        # A convolution's ifmap is streamed once per output-channel tile
        # (`planner.ifmap_passes`), so it is *charged* `n_ot` times -- from
        # DDR or from the scratchpad, whichever it sits in. It is still
        # *read* once here: the bytes do not change between passes, and
        # re-reading them would only make this loop slower. The multiplier
        # is imported rather than re-derived, so this module and the
        # planner cannot disagree about it.
        passes = ifmap_passes(op)
        for index, (t, space, addr) in enumerate(
            zip(op.inputs, step.input_spaces, step.input_addrs)
        ):
            nbytes = t.size_bytes
            raw = read_from(space, addr, nbytes)
            count(space, addr, nbytes * (passes if index == 0 else 1), is_read=True)
            input_values.append(unpack(t, raw))

        if isinstance(op, Conv2dOp):
            values = _exec_conv2d(op, input_values[0])
            charge_weight_fill(
                op, traffic, resident=step.weight_space == isa.SPACE_LOCAL_TENSOR
            )
        elif isinstance(op, PoolOp):
            values = _exec_pool(op, input_values[0])
        elif isinstance(op, AddOp):
            values = _exec_add(op, input_values[0], input_values[1])
        elif isinstance(op, UpsampleOp):
            values = _exec_upsample(op, input_values[0])
        elif isinstance(op, DepthToSpaceOp):
            values = _exec_depth_to_space(op, input_values[0])
        elif isinstance(op, ActOp):
            values = _exec_act(op, input_values[0])
            # The 256-entry LUT is refetched from DDR for every `ACT`
            # command (`planner.py` charges the same read). Its *values*
            # come straight off the op, like a conv's weights -- only the
            # byte count is modelled here.
            traffic.read_bytes += isa.ACT_LUT_BYTES
        else:
            raise TypeError(f"reference: unhandled op type {type(op)!r}")

        out_bytes = pack(op.output, values)
        write_to(step.output_space, step.output_addr, out_bytes)
        count(step.output_space, step.output_addr, len(out_bytes), is_read=False)
        tensor_data[op.output.name] = values

    traffic.read_bytes += (
        sum(descriptor_count(step) for step in planned.steps) + 1
    ) * isa.INSTR_WORD_BYTES

    _read_back_row_copy_targets(planned, image, local, row_copy_targets, tensor_data)
    _materialize_alias_values(planned.model, tensor_data)

    return ExecutionResult(traffic=traffic, tensor_data=tensor_data)


__all__ = ["ExecutionResult", "run_reference"]
