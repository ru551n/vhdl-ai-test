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

The only genuinely new arithmetic here is `UPSAMPLE` (nearest-neighbour
index replication) and the standalone `ACT` LUT (a 256-entry table
lookup) -- both ISA v2.0-only opcodes with **no** v1.2 counterpart in
`cnn_accel_model.py` to reuse or fork from, and both simple enough (a
pure index permutation; a table lookup) that there is no rounding/
saturation subtlety for a second implementation to silently diverge on.

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
from accel_v2.model import AddOp, Conv2dOp, CopyOp, Op, PoolOp, Tensor, UpsampleOp, ActOp
from accel_v2.planner import ComputeStep, DdrTraffic, MoveStep, PlannedProgram


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
        requant_scale=op.requant_scale,
        requant_shift=op.requant_shift,
        output_offset=op.output_offset,
        clamp_min=clamp_min,
        clamp_max=clamp_max,
    )
    return golden.pool_max(input_values, desc) if op.mode == "max" else golden.pool_avg(input_values, desc)


def _exec_add(op: AddOp, a_values: list[int], b_values: list[int]) -> list[int]:
    """`dst = sat_i8(requant(src0) + requant(src1))` (section 5.2),
    `requant(v) = round_shift_right_signed(v * requant_scale, 15 +
    requant_shift)` -- the same core rescale step `bias_requantize_relu`
    performs, applied identically to both operands since the descriptor
    carries only one `(requant_scale, requant_shift)` pair for `ADD`
    (ambiguity #1, see `AddOp`'s docstring: this assumes both operands
    are already expressed on a common scale, as is typical for a
    residual/skip-connection add)."""
    divisor_shift = 15 + op.requant_shift
    out = []
    for va, vb in zip(a_values, b_values):
        ra = golden.round_shift_right_signed(va * op.requant_scale, divisor_shift)
        rb = golden.round_shift_right_signed(vb * op.requant_scale, divisor_shift)
        out.append(golden.saturate_signed(ra + rb, 8))
    return out


def _exec_upsample(op: UpsampleOp, input_values: list[int]) -> list[int]:
    x = op.inputs[0]
    f = op.factor
    out_h, out_w, c = x.height * f, x.width * f, x.channels
    out = [0] * (out_h * out_w * c)
    for oy in range(out_h):
        iy = oy // f
        for ox in range(out_w):
            ix = ox // f
            src_base = (iy * x.width + ix) * c
            dst_base = (oy * out_w + ox) * c
            out[dst_base : dst_base + c] = input_values[src_base : src_base + c]
    return out


def _exec_act(op: ActOp, input_values: list[int]) -> list[int]:
    lut = op.lut
    return [lut[v & 0xFF] for v in input_values]


def _charge_weight_traffic(op: Conv2dOp, traffic: DdrTraffic) -> None:
    """Identical formulas to `planner.Planner.plan`'s own weight-traffic
    accounting -- kept as a private helper here (rather than imported
    from `planner.py`) only because `PoolOp`/`AddOp`/etc. never call it;
    both call sites build the same `LayerDesc` via
    `Conv2dOp.weight_layer_desc`, so they cannot drift."""
    if op.weight_reuse:
        return
    desc = op.weight_layer_desc()
    weight_bytes = golden.packed_weight_count(desc, golden.TILE_CHANNELS, golden.PE_ROWS)
    traffic.read_bytes += weight_bytes
    traffic.weight_bytes += weight_bytes
    if op.bias is not None:
        bias_bytes = golden.packed_bias_count(desc, golden.PE_ROWS) * 4
        traffic.read_bytes += bias_bytes
        traffic.weight_bytes += bias_bytes
    if op.per_channel_scale is not None:
        scale_bytes = golden.packed_scale_table_bytes(desc, golden.PE_ROWS)
        traffic.read_bytes += scale_bytes
        traffic.weight_bytes += scale_bytes


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

    def count(space: int, nbytes: int, *, is_read: bool) -> None:
        if space == isa.SPACE_DDR:
            if is_read:
                traffic.read_bytes += nbytes
            else:
                traffic.write_bytes += nbytes
        else:
            if is_read:
                traffic.local_read_bytes += nbytes
            else:
                traffic.local_write_bytes += nbytes

    def unpack(t: Tensor, raw: bytes) -> list[int]:
        return golden.unpack_activation_planes(_to_signed_bytes(raw), t.width, t.height, t.channels)

    def pack(t: Tensor, values: list[int]) -> bytes:
        return bytes(v & 0xFF for v in golden.pack_activation_planes(values, t.width, t.height, t.channels))

    for step in planned.steps:
        if isinstance(step, MoveStep):
            raw = read_from(step.src_space, step.src_addr, step.nbytes)
            write_to(step.dst_space, step.dst_addr, raw)
            count(step.src_space, step.nbytes, is_read=True)
            count(step.dst_space, step.nbytes, is_read=False)
            if step.kind == "spill":
                traffic.tensor_store_count += 1
                traffic.spill_count += 1
            else:
                traffic.tensor_load_count += 1
                if step.kind == "reload":
                    traffic.reload_count += 1
            continue

        assert isinstance(step, ComputeStep)
        op = step.op

        if isinstance(op, CopyOp):
            nbytes = op.inputs[0].size_bytes
            raw = read_from(step.input_spaces[0], step.input_addrs[0], nbytes)
            count(step.input_spaces[0], nbytes, is_read=True)
            write_to(step.output_space, step.output_addr, raw)
            count(step.output_space, nbytes, is_read=False)
            tensor_data[op.output.name] = list(tensor_data[op.inputs[0].name])
            continue

        input_values: list[list[int]] = []
        for t, space, addr in zip(op.inputs, step.input_spaces, step.input_addrs):
            nbytes = t.size_bytes
            raw = read_from(space, addr, nbytes)
            count(space, nbytes, is_read=True)
            input_values.append(unpack(t, raw))

        if isinstance(op, Conv2dOp):
            values = _exec_conv2d(op, input_values[0])
            _charge_weight_traffic(op, traffic)
        elif isinstance(op, PoolOp):
            values = _exec_pool(op, input_values[0])
        elif isinstance(op, AddOp):
            values = _exec_add(op, input_values[0], input_values[1])
        elif isinstance(op, UpsampleOp):
            values = _exec_upsample(op, input_values[0])
        elif isinstance(op, ActOp):
            values = _exec_act(op, input_values[0])
        else:
            raise TypeError(f"reference: unhandled op type {type(op)!r}")

        out_bytes = pack(op.output, values)
        write_to(step.output_space, step.output_addr, out_bytes)
        count(step.output_space, len(out_bytes), is_read=False)
        tensor_data[op.output.name] = values

    traffic.read_bytes += (len(planned.steps) + 1) * isa.INSTR_WORD_BYTES

    return ExecutionResult(traffic=traffic, tensor_data=tensor_data)


__all__ = ["ExecutionResult", "run_reference"]
