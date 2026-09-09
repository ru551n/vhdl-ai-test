"""Local tensor memory planner (`doc/cnn_accel_top_v2_arch.md` sections 3,
3.1, 4, 10).

Hardware "has no buffer table and no liveness tracking" (section 3.1) --
that analysis is entirely this module's job. `Planner.plan` walks a
`model.Model`'s ops in creation order and, for every tensor, decides:

* `LOCAL_TENSOR` whenever the on-chip scratchpad has room and the tensor
  is still live (has a future consumer) -- kept resident across any
  number of intervening commands, exactly as section 3.1 promises;
* an explicit `STORE` (spill, to the DDR `SPILL` arena) followed by an
  explicit `LOAD` (reload) when it does not fit, never a silent
  DDR round trip;
* `DDR` directly (no local buffer at all) for a graph input's or a
  spilled tensor's *last* remaining use, since paying for a reload only
  to immediately free the buffer again wastes both a local slot and an
  instruction for no bandwidth benefit -- see `_resolve_input`'s
  docstring for the exact rule and why it differs from the always-reload
  rule for values that were *previously* spilled;
* `DDR` for its whole lifetime -- no local buffer at any point --
  when it *cannot* be placed on chip at all: bigger than one
  `cnn_accel_tensor_mem` bank, or unable to fit even with every
  evictable buffer evicted. This is the fallback that stops the
  planner falling over on a real network. It is always available and
  always correct because section 3 defines ONE activation layout,
  used byte-identically in DDR and `LOCAL_TENSOR`, and operand space
  is a per-operand tag on each instruction: a buffer's slice offsets,
  plane order and alignment do not change with the space it sits in.
  It matters most for the `C2f` concat family, which is written by
  several scattered producers, stays live until the block's closing
  convolution, and is therefore deliberately never an eviction
  victim -- so it has no spill lowering, and without this it had no
  lowering at all. What it costs is bandwidth, reported separately in
  `DdrTraffic.ddr_resident_read_bytes`/`ddr_resident_write_bytes` and
  listed buffer by buffer in `PlannedProgram.ddr_placements`.

Channel `concat`/`split` (`model.Model.concat`/`split`) add a second job:
**buffer aliasing**. Those two operations produce no instruction at all --
they only say that one tensor's bytes live inside another's buffer (see
`Tensor.alias_parent`). This module is what makes that true:

* every tensor is resolved to its `alias_root` -- the tensor that owns an
  actual allocation -- and its address is that root's address plus
  `alias_byte_offset`, which is always a whole number of `H*W*T`-byte
  activation planes and therefore always 8-byte aligned;
* allocation, spilling and freeing are done **per alias family**, never
  per tensor. The family of a root is the root plus every tensor aliased
  into it (transitively). A family's buffer is allocated when the first
  of its members is produced or read, and freed only once *every* member
  has been produced *and* every consumer of every member has run -- the
  liveness rule that stops a concat buffer being recycled while one of
  its slices is still live.

Spatial tiling (`tiler.py`, arch doc section 3.2) adds three things to
that, all of them opt-in and none of them reachable from a graph that
does not ask:

* **`Tensor.pin_ddr`** -- a fusion-group boundary, written by `S` strip
  stores and read by the next group's `S'` strip loads. It takes the
  `place_in_ddr` path immediately and unconditionally, and its bytes are
  reported in `pinned_read_bytes`/`pinned_write_bytes` rather than in
  `ddr_resident_*`, so "did the overflow fallback cost anything?" stays
  a separate question from "what did the group boundaries cost?".
* **`Tensor.confine_unit_bytes`** -- per-*request* bank confinement
  instead of per-buffer, which is what lets a multi-plane activation
  strip be larger than one `cnn_accel_tensor_mem` bank. Only sound for a
  buffer every request against which is one plane or less; see
  `tiler._UNPLANNED_REQUEST_OPS` for the ops that are not, and
  `MoveStep.unit` for the spill lowering it forces.
* **`Conv2dOp.weights_resident`** -- the packed weight/bias/scale images
  `LOAD`ed into the scratchpad once (`ConstLoadStep`) and fetched per
  output-channel pass from `space_wgt = LOCAL_TENSOR`. All `S` strip
  clones of one convolution share one image, so a group's weights cross
  the bus once instead of once per strip. A request the allocator cannot
  place falls back to DDR weights silently, and the traffic charge
  follows the decision actually made.

The traffic prediction is exact, not a bound. Two things it once missed
are charged here: a convolution's ifmap is streamed once per
output-channel tile (`ifmap_passes`, arch doc section 5.4) and every
`ACT` refetches its whole LUT (`isa.ACT_LUT_BYTES`).

The output is a `PlannedProgram`: an ordered list of `ComputeStep`/
`MoveStep`s with every operand's space and address already resolved,
plus a `DdrTraffic` prediction that `program.py`'s actual DDR-image
emission and `reference.py`'s actual execution are both checked against
(the whole point of computing it up front, per the spec's residency
invariants R1-R6).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cnn_accel_constants as _const
import cnn_accel_model as golden

from accel_v2 import isa
from accel_v2.ddrmap import DdrMap
from accel_v2.model import (
    ActOp,
    Conv2dOp,
    Model,
    Op,
    RowCopyOp,
    Tensor,
    alias_byte_offset,
    alias_root,
)

#: Section 4 default: `g_num_banks=2 * g_bank_words=1024 * 8 bytes`.
DEFAULT_TENSOR_MEM_BYTES = 2 * 1024 * 8

#: Section 4 default bank size: `g_bank_words=1024 * 8 bytes`.
#:
#: The scratchpad is byte-addressed as one flat range, but it is NOT one
#: flat memory: it is `tensor_mem_bytes / bank_bytes` independent banks,
#: and `cnn_accel_tensor_mem` serves each `(addr, length)` request from
#: the single bank `addr` decodes to. A request with
#: `offset + beats > bank_words` is a caller bug that the hardware
#: reports and then *clamps* -- the part of the transfer past the bank
#: boundary is simply never done. So a local buffer must never straddle
#: a bank boundary, which is this module's job to guarantee (see
#: `_LocalAllocator`); nothing else in the toolchain checks it, and
#: neither `reference.py` (a flat `bytearray`) nor a value-only DUT
#: comparison can see the difference, because the program and the
#: reference both take their addresses from the same planner.
DEFAULT_BANK_BYTES = 1024 * 8

#: Bytes per packed bias entry (one `ACCUM_WIDTH` accumulator) and per
#: packed per-channel scale-table entry, matching `cmd_proc`'s own
#: `c_bias_entry_bytes` / `c_scale_entry_bytes`. They set the per-pass
#: request size of the bias and scale images, and therefore the
#: granularity at which a resident image must stay inside one bank.
_BIAS_ENTRY_BYTES = _const.ACCUM_WIDTH // 8
_SCALE_ENTRY_BYTES = _const.SCALE_TABLE_ENTRY_BYTES


@dataclass
class DdrTraffic:
    """Predicted (by `Planner`) or measured (by `reference.Reference`)
    traffic counters, deliberately named after the CSR map (section 8)
    so a test reads as a direct comparison against hardware counters:

    * `read_bytes`  ~ `DDR_RD_BYTES`  (all AXI reads, including the
      64-byte-per-instruction program fetch and weight/bias/scale fills)
    * `write_bytes` ~ `DDR_WR_BYTES`
    * `weight_bytes` ~ `WEIGHT_LOAD_BYTES` (a subtotal already included
      in `read_bytes`, not additional to it)
    * `tensor_load_count`/`tensor_store_count` ~ `TENSOR_LOAD_COUNT`/
      `TENSOR_STORE_COUNT` (explicit `LOAD`/`STORE` instructions retired)
    * `spill_count`/`reload_count`: finer-grained breakdown of the above
      two, this module's own bookkeeping (not literal CSR fields) --
      how many of those `LOAD`/`STORE`s were compiler-inserted
      spill/reload traffic versus a "plain" first-time `LOAD` of a graph
      input
    * `local_read_bytes`/`local_write_bytes` ~ `LOCAL_RD_BYTES`/
      `LOCAL_WR_BYTES`
    """

    read_bytes: int = 0
    write_bytes: int = 0
    weight_bytes: int = 0
    tensor_load_count: int = 0
    tensor_store_count: int = 0
    spill_count: int = 0
    reload_count: int = 0
    local_read_bytes: int = 0
    local_write_bytes: int = 0
    #: Subtotals of `read_bytes`/`write_bytes` (not additional to them)
    #: attributable to buffers the planner had to place in DDR because
    #: they could not be placed in the scratchpad at all -- see
    #: `Planner.plan`'s `place_in_ddr`. Zero for every program that fits
    #: locally, so a test can assert "this program paid nothing for
    #: overflow" as directly as it can assert "this one paid exactly
    #: N bytes".
    ddr_resident_read_bytes: int = 0
    ddr_resident_write_bytes: int = 0
    #: Subtotals of `read_bytes`/`write_bytes` (not additional to them)
    #: attributable to tensors that live in DDR *by design* --
    #: `Tensor.pin_ddr`, the full-resolution buffers on a fusion group's
    #: boundary (`tiler.py`). Split out from `ddr_resident_*_bytes` so the
    #: two questions stay separate: "did the overflow fallback cost
    #: anything?" (`ddr_resident_*`, still expected to be zero for a
    #: healthy plan) and "what did the group boundaries cost?"
    #: (`pinned_*`, expected to be exactly the tiling's own prediction).
    pinned_read_bytes: int = 0
    pinned_write_bytes: int = 0


@dataclass
class ComputeStep:
    """One compute/elementwise instruction with every operand's space
    and address already resolved. `input_spaces`/`input_addrs` are
    parallel to `op.inputs`."""

    op: Op
    input_spaces: list[int]
    input_addrs: list[int]
    output_space: int
    output_addr: int
    #: `space_wgt` for this instruction's compile-time side tables (a
    #: convolution's weight/bias/scale images). `SPACE_DDR` is the only
    #: value any untiled program has ever used; `SPACE_LOCAL_TENSOR`
    #: means the images were LOADed into the scratchpad by an earlier
    #: `ConstLoadStep` and every output-channel pass fetches its tile
    #: from there. `ACT`'s LUT is not covered -- it always rides in DDR.
    weight_space: int = isa.SPACE_DDR
    #: `kind -> scratchpad byte address` for a `weight_space ==
    #: SPACE_LOCAL_TENSOR` convolution, `None` otherwise.
    weight_addrs: dict[str, int] | None = None


@dataclass
class MoveStep:
    """One explicit `LOAD` (`kind="reload"` or `kind="cold_load"`) or
    `STORE` (`kind="spill"`) data-movement instruction the planner
    inserted (as opposed to a compute op just naming a `DDR` operand
    directly, which needs no separate instruction at all)."""

    tensor: Tensor
    src_space: int
    src_addr: int
    dst_space: int
    dst_addr: int
    nbytes: int
    kind: str  # "reload" | "spill"
    #: Largest number of bytes one descriptor of this move may transfer,
    #: or `None` for "all of them in one".
    #:
    #: A `LOAD`/`STORE` is a single request of `xfer_bytes`
    #: (`cmd_proc.vhd`'s `st_xfer_*`), and `cnn_accel_tensor_mem` serves
    #: it from the one bank its address decodes to -- so moving a
    #: plane-confined buffer (one that is deliberately allowed to span
    #: banks, `Tensor.confine_unit_bytes`) in one go would be clamped and
    #: silently truncated. Spilling one is therefore `ceil(nbytes/unit)`
    #: descriptors at consecutive offsets, exactly as a `RowCopyStep` is
    #: one per plane. Found by the DUT: `cnn_accel_tensor_mem` asserts on
    #: the straddle, and the first tiled case that evicted a multi-plane
    #: strip stopped the simulation on it.
    unit: int | None = None

    @property
    def transfer_offsets(self) -> list[tuple[int, int]]:
        """`(offset, nbytes)` per emitted descriptor, in address order."""
        if self.unit is None or self.unit >= self.nbytes:
            return [(0, self.nbytes)]
        return [
            (offset, min(self.unit, self.nbytes - offset))
            for offset in range(0, self.nbytes, self.unit)
        ]


@dataclass
class RowCopyStep:
    """One `model.RowCopyOp`, resolved into its `plane_count` per-plane
    sub-transfers.

    One planner step, several descriptors: the activation layout is
    channel-plane-major, so a row window of a multi-plane tensor is
    `plane_count` separate byte ranges (`plane_bytes` apart at the source,
    a *different* `plane_bytes` apart at the destination whenever the two
    tensors differ in height). Keeping them one step is what lets the
    planner's liveness, eviction and family bookkeeping treat the whole
    move as the single operation it logically is -- in particular a
    half-copied destination is never an eviction victim, because the
    planner never sees a moment between two of its planes.

    `program.py` emits one `LOAD`/`STORE`/`COPY` descriptor per entry of
    `transfers`, choosing the opcode from `src_space`/`dst_space` exactly
    as it does for a `MoveStep`."""

    op: RowCopyOp
    src_space: int
    dst_space: int
    #: `(src_addr, dst_addr, nbytes)` per activation plane, in plane order.
    transfers: list[tuple[int, int, int]] = field(default_factory=list)

    @property
    def nbytes(self) -> int:
        """Total bytes moved -- `rows * width * T` per plane."""
        return sum(n for _, _, n in self.transfers)


@dataclass
class ConstLoadStep:
    """The one-time `LOAD` of a convolution's packed weight/bias/scale
    images into the scratchpad, so that its output-channel passes can
    fetch their tiles with `space_wgt = LOCAL_TENSOR` instead of
    re-reading DDR (design D6 / hardware fact F7).

    Three sub-images, three descriptors, because `program.py` allocates
    them out of three different `DdrMap` regions and they are therefore
    not contiguous in DDR. Each lands in its **own** local buffer with
    its own confinement unit -- the request the hardware makes against
    the weight image is one `wgt_tile_bytes` tile, against the bias image
    `PE_ROWS * 4` bytes and against the scale image
    `PE_ROWS * SCALE_TABLE_ENTRY_BYTES`, and only a per-image unit keeps
    all three inside one bank without padding the weight region out to a
    whole number of tiles twice over.

    `images` is `(kind, local_addr, nbytes)` with `kind` one of
    `"weight"`, `"bias"`, `"scale"`; `program.py` looks the DDR source up
    by the same kind. The step carries no source address of its own
    because DDR addresses for compile-time constants are `program.py`'s
    to allocate, exactly as they always have been."""

    conv: Conv2dOp
    key: str
    images: list[tuple[str, int, int]] = field(default_factory=list)

    @property
    def nbytes(self) -> int:
        return sum(n for _, _, n in self.images)


Step = ComputeStep | MoveStep | RowCopyStep | ConstLoadStep


def ifmap_passes(op: Op) -> int:
    """How many times the hardware streams `op`'s `inputs[0]`.

    `cmd_proc` runs a `CONV2D` as `n_ot = ceil(out_channels / PE_ROWS)`
    output-channel passes (`st_geom_out2`: `n_planes_out_q <= n_ot_q`),
    and every pass re-kicks the ifmap feeder over the *whole* input
    tensor -- one `(row, tile)` request per plane row when there is more
    than one channel tile, one whole-tensor request otherwise, but either
    way `in_total_bytes` per pass. So a conv whose input sits in DDR
    reads it `n_ot` times over AXI, and a conv whose input is resident
    reads it `n_ot` times out of the scratchpad.

    That is hardware fact F4 of the tiling design, and it is where 184 of
    the 202 MB per frame of the untiled YOLOv8n baseline come from. Until
    this function existed both `planner.py` and `reference.py` charged a
    conv's ifmap exactly once, which is why `TrafficPolicy.read_bytes`
    could only ever be a lower bound.

    Every other opcode streams its inputs once: `POOL_*` also runs one
    pass per plane, but each pass reads one *plane* (`:1494-1496`), so
    the tensor is read once in total; the elementwise family issues one
    request per operand for the whole tensor.

    Stated here, once, and imported by `reference.py` -- the two must
    charge the same multiplier or every exact read assertion fails for a
    reason that has nothing to do with residency."""
    if isinstance(op, Conv2dOp):
        return -(-op.output.channels // golden.PE_ROWS)
    return 1


#: The three packed side tables a `CONV2D` fetches per output-channel
#: pass, as `(kind, total-bytes function, per-pass request function)`.
#: `cmd_proc`'s `st_wgt_req` walks exactly these three, in this order,
#: at `<base> + pass * <per-pass bytes>`.
def const_images(op: Conv2dOp) -> list[tuple[str, int, int]]:
    """`(kind, total bytes, per-pass request bytes)` for every packed
    side table of `op`, in `st_wgt_req`'s own order.

    The third element is what must stay inside one `cnn_accel_tensor_mem`
    bank when the image is resident: the hardware fetches one weight
    *tile*, one `PE_ROWS`-entry bias row and one `PE_ROWS`-entry scale
    row per pass, never the whole image. Totals come from the same
    `cnn_accel_model.packed_*` calls `program.py` uses to write the
    bytes, so an image is exactly as big as what lands in it."""
    desc = op.weight_layer_desc()
    n_ot = -(-op.output.channels // golden.PE_ROWS)
    images = []
    weight_bytes = golden.packed_weight_count(desc, golden.TILE_CHANNELS, golden.PE_ROWS)
    images.append(("weight", weight_bytes, weight_bytes // n_ot))
    if op.bias is not None:
        bias_bytes = golden.packed_bias_count(desc, golden.PE_ROWS) * _BIAS_ENTRY_BYTES
        images.append(("bias", bias_bytes, golden.PE_ROWS * _BIAS_ENTRY_BYTES))
    if op.per_channel_scale is not None:
        scale_bytes = golden.packed_scale_table_bytes(desc, golden.PE_ROWS)
        images.append(("scale", scale_bytes, golden.PE_ROWS * _SCALE_ENTRY_BYTES))
    return images


def weight_fill_bytes(op: Conv2dOp) -> int:
    """Bytes the weight/bias/scale refill moves for `op` over all its
    output-channel passes -- what the hardware's `WEIGHT_LOAD_BYTES`
    counter accumulates in `st_wgt_req`, whichever space `space_wgt`
    names. Zero for a `WEIGHT_REUSE` op, which skips the refill
    entirely.

    The single definition: `Planner.plan` predicts it and
    `reference.py` charges it again independently, and the two must
    agree or `weight_bytes_exact` fails for a reason unrelated to
    residency."""
    if op.weight_reuse:
        return 0
    return sum(total for _, total, _ in const_images(op))


def charge_weight_fill(op: Conv2dOp, traffic: "DdrTraffic", *, resident: bool) -> None:
    """Charge `op`'s weight/bias/scale refill against `traffic`.

    Called by `Planner.plan` (predicting) and by `reference.py`
    (executing), so the formula lives once. `WEIGHT_LOAD_BYTES` counts
    the refill whichever space it reads -- `cmd_proc`'s `cnt_wgt_bytes_q`
    increments in `st_wgt_req` before the space mux, so it is a measure
    of how much the weight buffer was filled, not of how much DDR was
    read. What residency moves is which of `read_bytes` (AXI) and
    `local_read_bytes` (scratchpad port `r1`) the same bytes land in."""
    fill = weight_fill_bytes(op)
    if not fill:
        return
    traffic.weight_bytes += fill
    if resident:
        traffic.local_read_bytes += fill
    else:
        traffic.read_bytes += fill


def descriptor_count(step: "Step") -> int:
    """How many 64-byte descriptors `program.py` will emit for `step`.

    One for every step except a `RowCopyStep` (one per activation plane),
    a `ConstLoadStep` (one per packed sub-image) and a `MoveStep` of a
    plane-confined buffer (one per plane, `MoveStep.unit`).
    The single place this is stated: `planner.py` charges the program
    fetch, `reference.py` charges it again independently, and
    `program.py` actually emits them, and all three must agree or a
    traffic assertion fails for a reason that has nothing to do with
    the tensors."""
    if isinstance(step, RowCopyStep):
        return len(step.transfers)
    if isinstance(step, ConstLoadStep):
        return len(step.images)
    if isinstance(step, MoveStep):
        return len(step.transfer_offsets)
    return 1


@dataclass
class PlannedProgram:
    model: Model
    steps: list[Step]
    traffic: DdrTraffic
    tensor_mem_bytes: int
    ddr_map: DdrMap
    #: DDR byte address of every tensor that ever touches DDR: graph
    #: inputs, graph outputs, and any spilled intermediate (keyed by
    #: `Tensor.name`). `program.py` uses this to know where to seed
    #: input data and where outputs land; `reference.py` uses it to seed
    #: and to read back results.
    tensor_ddr_addr: dict[str, int] = field(default_factory=dict)
    #: Bank size of the scratchpad this program was planned for, in
    #: bytes. `tensor_mem_bytes // bank_bytes` is the bank count.
    bank_bytes: int = DEFAULT_BANK_BYTES
    #: Every local allocation this plan made, in order, as
    #: `(alias-root tensor name, byte address, byte size)`. Recorded so a
    #: test can assert the *geometry* of a placement directly (e.g. that
    #: a buffer lies within one bank) rather than only its data: the
    #: emitted program and `reference.py` take their addresses from the
    #: same plan, so a placement bug is self-consistently wrong on both
    #: sides and invisible to a value-only comparison.
    local_placements: list[tuple[str, int, int]] = field(default_factory=list)
    #: Every buffer this plan could NOT place in the scratchpad and
    #: therefore left resident in DDR, in order, as `(alias-root tensor
    #: name, DDR byte address, byte size)`. Empty for a program that
    #: fits entirely locally -- which is every program that planned
    #: before this fallback existed, so a non-empty list is exactly the
    #: set of buffers whose placement changed. Recorded (like
    #: `local_placements`) so a test can assert the *placement* directly
    #: rather than inferring it from values, which the DUT and
    #: `reference.py` would agree on even if it were wrong.
    ddr_placements: list[tuple[str, int, int]] = field(default_factory=list)
    #: Every `Tensor.pin_ddr` buffer this plan gave a DDR home, as
    #: `(alias-root tensor name, DDR byte address, byte size)`. Kept out
    #: of `ddr_placements` on purpose: a pinned buffer is not a failure to
    #: place anything, it is the fusion-group boundary the tiler asked
    #: for, and a test that asserts "nothing overflowed" must still be
    #: able to say that about a tiled program.
    pinned_placements: list[tuple[str, int, int]] = field(default_factory=list)
    #: Placement name -> the largest single hardware request that will be
    #: made against it, for `tiling_checks.check_units_confined` to
    #: re-derive the bank-confinement property from the plan alone. Only
    #: buffers whose unit is *not* their whole size appear: a tiled
    #: activation strip (one plane) and a resident weight image (one
    #: tile). Deliberately a side table rather than a fourth element of
    #: `local_placements`, so the placement tuples -- and the digest that
    #: hashes them -- keep their shape.
    local_confine_units: dict[str, int] = field(default_factory=dict)


class _LocalAllocator:
    """First-fit free-list allocator over `[0, capacity)` that never
    places a buffer across a bank boundary. Buffers are freed the instant
    their last consumer has run (section 3.1: "Buffer reuse... [is] the
    producer's [compiler's] responsibility"), so the free list is usually
    fragmented in the same way a real heap is; the planner's job is only
    to decide *which* tensor to evict when first-fit fails, not to
    defragment.

    **Bank awareness.** `cnn_accel_tensor_mem` is `capacity / bank_bytes`
    *independent* banks and serves any one request entirely from the
    single bank its address decodes to; a request that would run past the
    end of that bank is clamped by the hardware (see `DEFAULT_BANK_BYTES`),
    silently truncating the transfer. Every allocation here is therefore
    required to satisfy `addr // bank_bytes == (addr + size - 1) //
    bank_bytes` -- **per `unit`**, where `unit` is the largest single
    request that will ever be made against the buffer (`try_alloc`'s
    argument, from `Tensor.confine_unit_bytes`). For every untiled
    tensor the unit is the whole buffer, which is the strict rule this
    planner has always applied and which every existing placement still
    gets; a tiled activation strip passes its plane size instead,
    because no request against an activation is larger than one channel
    plane, and that is what lets a multi-plane strip be bigger than a
    bank at all.

    The strategy is **"skip to the next bank boundary on straddle"**,
    not "align every buffer to a bank":

    * scan the free list first-fit exactly as before, but if the naturally
      aligned candidate inside a free block would straddle the boundary,
      retry at that boundary (still inside the same free block) and move
      on to the next block if it no longer fits there;
    * the skipped bytes are not lost -- they stay on the free list as an
      ordinary leading hole and are handed to the next buffer small enough
      to fit in them.

    The rejected alternative, rounding every buffer's *base* (or size) up
    to a whole bank, is what makes bank-awareness expensive: with the
    section-4 default geometry (8 KiB banks) and this project's tensors
    (0.5-2 KiB), it would put one tensor in each bank and cut the usable
    scratchpad by up to 16x, turning ordinary programs into spilling ones.
    The rule above costs *nothing at all* until a buffer would actually
    straddle, and even then wastes at most `size - 1` bytes, reusable by
    anything smaller. Its one real cost is a mild bias towards leaving
    small holes just below bank boundaries; with no defragmenter that
    bias is permanent, which is the tradeoff accepted here (measured on
    the whole `tb_cnn_accel_top` catalogue: zero placement changes,
    because no case ever straddled to begin with).

    A buffer larger than one bank cannot be placed legally at all --
    `Planner.plan` rejects it with an actionable message rather than
    splitting it across banks, since the hardware has no notion of a
    multi-bank transfer."""

    def __init__(self, capacity: int, bank_bytes: int) -> None:
        if bank_bytes <= 0:
            raise ValueError(f"bank_bytes must be positive, got {bank_bytes}")
        self.capacity = capacity
        self.bank_bytes = bank_bytes
        self._free: list[tuple[int, int]] = [(0, capacity)] if capacity > 0 else []

    def try_alloc(self, size: int, align: int = 8, unit: int | None = None) -> int | None:
        """Place `size` bytes so that **every `unit`-byte sub-range lies
        in one bank**, and return the address (or `None`).

        `unit` is the largest transfer the hardware will ever issue
        against this buffer -- `None` means "the whole buffer", which is
        the strict rule this planner has always enforced and what every
        untiled tensor uses. A tiled activation strip passes its
        `plane_bytes` instead (`Tensor.confine_unit_bytes`), because the
        hardware requirement is per *request*, and no request against an
        activation is bigger than one plane.

        With `unit == size` the loop below reduces, provably, to the
        original single "bump to the next bank boundary on straddle"
        step: `k` can only be 0, the boundary it straddles is the first
        one after `addr`, and the bumped address `beta - 0*unit == beta`
        cannot straddle again because `size <= bank_bytes`. That
        equivalence is what keeps every existing placement byte-identical
        (`tests/test_tiling_guard.py`).
        """
        if size <= 0:
            raise ValueError(f"allocation size must be positive, got {size}")
        if unit is None:
            unit = size
        if unit <= 0:
            raise ValueError(f"confinement unit must be positive, got {unit}")
        if unit > self.bank_bytes:
            # Caller (`Planner.alloc_local`) is expected to have rejected
            # this already with a much more actionable message; belt and
            # braces, since silently returning an illegal address is the
            # exact failure mode this class exists to prevent.
            return None
        for i, (start, length) in enumerate(self._free):
            addr = start + (-start) % align
            # Bank boundaries are whole multiples of `align` (a bank is a
            # whole number of 8-byte words) and so is `unit`, so every
            # bump below keeps the alignment guarantee.
            # Slide until nothing straddles. Which unit straddles first,
            # and how far the buffer then moves, depend ONLY on
            # `addr % bank_bytes` -- so a repeated residue means the
            # slide has entered a cycle and NO confined address exists
            # for this many units, which is what happens whenever `unit`
            # does not divide `bank_bytes` and there are enough units to
            # wrap around one. Detecting that is what keeps the search
            # from walking a large free block eight bytes at a time; a
            # buffer with no confined address falls back to DDR exactly
            # as one that is simply too big does.
            seen_residues: set[int] = set()
            while addr + size <= start + length:
                straddle = self._first_straddle(addr, size, unit)
                if straddle is None:
                    break
                residue = addr % self.bank_bytes
                if residue in seen_residues:
                    break
                seen_residues.add(residue)
                # Slide the buffer up so unit `k` *starts* on the boundary
                # it was straddling. `boundary` is strictly above unit
                # `k`'s own start, so this strictly increases `addr`.
                k, boundary = straddle
                addr = boundary - k * unit
            pad = addr - start
            if (
                pad < 0
                or length - pad < size
                or self._first_straddle(addr, size, unit) is not None
            ):
                continue
            end = addr + size
            replacement = []
            if pad > 0:
                replacement.append((start, pad))
            if length - pad - size > 0:
                replacement.append((end, length - pad - size))
            self._free[i : i + 1] = replacement
            return addr
        return None

    def _first_straddle(self, addr: int, size: int, unit: int) -> tuple[int, int] | None:
        """`(k, boundary)` for the first `unit`-sized sub-range of
        `[addr, addr+size)` that crosses a bank boundary, or `None` when
        none does. The final sub-range may be short (`size` need not be a
        whole number of units); a short one straddles by the same rule."""
        k = 0
        while k * unit < size:
            lo = addr + k * unit
            hi = min(lo + unit, addr + size)
            if lo // self.bank_bytes != (hi - 1) // self.bank_bytes:
                return k, (lo // self.bank_bytes + 1) * self.bank_bytes
            k += 1
        return None

    def clone(self) -> "_LocalAllocator":
        """A throwaway copy, for asking "could this size EVER be placed
        here?" without touching the real free list -- see
        `Planner.plan`'s `can_ever_fit`."""
        other = _LocalAllocator(self.capacity, self.bank_bytes)
        other._free = list(self._free)
        return other

    def free(self, addr: int, size: int) -> None:
        entries = sorted(self._free + [(addr, size)])
        merged: list[tuple[int, int]] = []
        for start, length in entries:
            if merged and merged[-1][0] + merged[-1][1] == start:
                prev_start, prev_len = merged[-1]
                merged[-1] = (prev_start, prev_len + length)
            else:
                merged.append((start, length))
        self._free = merged


def _plan_row_copy(
    op: RowCopyOp,
    exclude: set[str],
    resolve_input,
    resolve_output,
    traffic: DdrTraffic,
) -> RowCopyStep:
    """Resolve one `RowCopyOp` into a `RowCopyStep` of `plane_count`
    per-plane sub-transfers.

    Three things make this different from an ordinary op, and all three
    are consequences of the channel-plane-major layout (arch doc section
    3) plus the fact that a strip and the full tensor it came from have
    *different heights*:

    * the bytes moved are `planes * rows * width * T`, not either
      tensor's `size_bytes` -- so both operand resolutions are told the
      real figure rather than being allowed to charge a whole tensor;
    * the source is read **in place**. A row copy exists precisely so
      that a full-resolution group boundary never enters the scratchpad;
      letting `resolve_input`'s ordinary "more than one consumer left, so
      hoist it local" rule fire here would reintroduce the traffic
      tiling was built to remove (and, for a real group boundary, would
      simply not fit);
    * the plane stride differs at the two ends (`src.plane_bytes` vs
      `dst.plane_bytes`), which is why this cannot be one transfer even
      when the row ranges happen to line up.

    `LOAD`/`STORE` counting follows the descriptors: a DDR->local copy
    retires `plane_count` `LOAD`s and a local->DDR copy `plane_count`
    `STORE`s, exactly what `TENSOR_LOAD_COUNT`/`TENSOR_STORE_COUNT` will
    show. Neither is a spill or a reload -- no value is being evicted --
    so `spill_count`/`reload_count` stay untouched."""
    src, dst = op.inputs[0], op.output
    planes = src.plane_count
    rows = op.src_rows.rows
    per_plane = rows * src.row_bytes
    moved = planes * per_plane

    src_space, src_base = resolve_input(src, exclude, read_bytes=moved, in_place=True)
    dst_space, dst_base = resolve_output(dst, exclude, write_bytes=moved)

    transfers = [
        (
            src_base + plane * src.plane_bytes + op.src_rows.r0 * src.row_bytes,
            dst_base + plane * dst.plane_bytes + op.dst_rows.r0 * dst.row_bytes,
            per_plane,
        )
        for plane in range(planes)
    ]

    if src_space == isa.SPACE_DDR and dst_space == isa.SPACE_LOCAL_TENSOR:
        traffic.tensor_load_count += planes
    elif src_space == isa.SPACE_LOCAL_TENSOR and dst_space == isa.SPACE_DDR:
        traffic.tensor_store_count += planes

    return RowCopyStep(op=op, src_space=src_space, dst_space=dst_space, transfers=transfers)


class Planner:
    def __init__(
        self,
        tensor_mem_bytes: int = DEFAULT_TENSOR_MEM_BYTES,
        ddr_map: DdrMap | None = None,
        bank_bytes: int | None = None,
    ) -> None:
        """`bank_bytes` is `g_bank_words * 8`, i.e. the size of ONE
        `cnn_accel_tensor_mem` bank -- the largest transfer the hardware
        can serve, and therefore the largest buffer this planner may
        place (see `_LocalAllocator`). It defaults to the section-4
        default bank, narrowed to `tensor_mem_bytes` for a scratchpad
        smaller than one default bank (which is then a single bank, the
        only geometry that makes sense for it -- and what the small-mem
        cases in `cases.py` actually instantiate)."""
        if bank_bytes is None:
            bank_bytes = min(DEFAULT_BANK_BYTES, tensor_mem_bytes)
        if bank_bytes <= 0 or tensor_mem_bytes % bank_bytes != 0:
            raise ValueError(
                f"planner: tensor_mem_bytes={tensor_mem_bytes} must be a positive "
                f"whole multiple of bank_bytes={bank_bytes} (the scratchpad is a "
                "whole number of equally sized cnn_accel_tensor_mem banks)"
            )
        self.tensor_mem_bytes = tensor_mem_bytes
        self.bank_bytes = bank_bytes
        self.ddr_map = ddr_map if ddr_map is not None else DdrMap()

    def plan(self, model: Model) -> PlannedProgram:
        alloc = _LocalAllocator(self.tensor_mem_bytes, self.bank_bytes)
        local_placements: list[tuple[str, int, int]] = []
        ddr_placements: list[tuple[str, int, int]] = []
        pinned_placements: list[tuple[str, int, int]] = []
        local_confine_units: dict[str, int] = {}
        traffic = DdrTraffic()
        steps: list[Step] = []
        tensor_ddr_addr: dict[str, int] = {}

        # -- alias families (see the module docstring) ---------------------
        #
        # `root_of[name]` is the tensor that owns the buffer `name` lives
        # in (itself, when it is not aliased). Everything below is keyed
        # by ROOT name: residency, spill state, liveness and eviction all
        # operate on whole buffers, because a concat buffer's slices have
        # no independent existence.
        root_of: dict[str, Tensor] = {t.name: alias_root(t) for t in model.tensors}
        family: dict[str, list[Tensor]] = {}
        for t in model.tensors:
            family.setdefault(root_of[t.name].name, []).append(t)
        root_by_name: dict[str, Tensor] = {name: root_of[name] for name in family}

        #: root name -> its current LOCAL_TENSOR byte offset, only while
        #: resident.
        local_addr: dict[str, int] = {}
        #: root name -> its DDR spill address, only while spilled
        #: (evicted, not yet reloaded). Absent for a root that has never
        #: been spilled.
        spilled: dict[str, int] = {}
        #: root name -> its permanent DDR address, for a buffer that
        #: could not be placed in the scratchpad AT ALL (see
        #: `place_in_ddr`). Sticky: once a buffer lives in DDR every
        #: producer of every family member writes straight into it and
        #: every consumer reads straight out of it, so it is never
        #: reloaded, never spilled and never half-resident.
        ddr_resident: dict[str, int] = {}
        #: root name -> not-yet-executed consumer edges over the whole
        #: family, and not-yet-run producers of family members. A buffer
        #: is freed only when BOTH reach zero: a concat buffer is written
        #: by several producers and read through several slices, and it
        #: must outlive all of them.
        pending_consumers: dict[str, int] = {}
        pending_producers: dict[str, int] = {}
        total_consumers: dict[str, int] = {}
        for name, members in family.items():
            pending_consumers[name] = sum(len(t.consumers) for t in members)
            total_consumers[name] = pending_consumers[name]
            pending_producers[name] = 0
        # Counted by walking the ops rather than by counting members with
        # a `producer`, because those two stopped being the same number
        # when `RowCopyOp` arrived: `S` strip stores all write into one
        # pinned tensor, which therefore has several writers and no single
        # `producer` at all. For every graph without row copies the two
        # counts are identical (each op sets its output's `producer`), so
        # no existing plan moves.
        for op in model.ops:
            pending_producers[root_of[op.output.name].name] += 1

        def nbytes(t: Tensor) -> int:
            return t.size_bytes

        def confine_unit(t: Tensor) -> int:
            """The largest single hardware request that will ever be made
            against `t`'s buffer, which is what must stay inside one bank
            (`_LocalAllocator.try_alloc`). `None` means the whole buffer:
            the strict rule, and the only one an untiled tensor uses."""
            return t.confine_unit_bytes if t.confine_unit_bytes is not None else nbytes(t)

        def next_use(root_name: str, from_idx: int) -> int | None:
            """First op index >= `from_idx` that reads any member of
            `root_name`'s family."""
            best: int | None = None
            for member in family[root_name]:
                for c in member.consumers:
                    if c >= from_idx and (best is None or c < best):
                        best = c
            return best

        def remaining_uses(root_name: str, from_idx: int) -> int:
            return sum(
                1 for member in family[root_name] for c in member.consumers if c >= from_idx
            )

        def evict_one(exclude: set[str]) -> bool:
            """Evict the resident, non-excluded buffer whose next use is
            furthest in the future (Belady's rule: minimizes the chance
            the very next allocation has to evict again). `exclude` is
            the current op's own input roots -- a value about to be read
            can never be its own eviction victim.

            A whole alias family is spilled and reloaded as one buffer:
            its slices are byte ranges of it and have no separate
            existence, so there is nothing finer to evict.

            A buffer that is only *half written* -- a concat buffer whose
            remaining parts are produced by commands still ahead of us --
            is never a victim. Its later producers write straight to the
            buffer's address and have no reload of their own to bring the
            spilled bytes back, so evicting it would silently lose the
            slices already in it. Spilling and reloading a buffer that is
            about to be written again is also pure waste, so nothing is
            given up by the restriction; if it makes an allocation
            impossible, `alloc_local` says so rather than corrupting the
            program."""
            best_name: str | None = None
            best_next = -1
            for name in local_addr:
                if name in exclude:
                    continue
                if pending_producers[name]:
                    continue
                nxt = next_use(name, op_index)
                score = nxt if nxt is not None else 1 << 62
                if score > best_next:
                    best_next, best_name = score, name
            if best_name is None:
                return False
            victim = root_by_name[best_name]
            addr = local_addr.pop(best_name)
            size = nbytes(victim)
            ddr_addr = self.ddr_map.alloc(DdrMap.SPILL, size)
            steps.append(
                MoveStep(
                    tensor=victim,
                    src_space=isa.SPACE_LOCAL_TENSOR,
                    src_addr=addr,
                    dst_space=isa.SPACE_DDR,
                    dst_addr=ddr_addr,
                    nbytes=size,
                    kind="spill",
                    unit=victim.confine_unit_bytes,
                )
            )
            traffic.local_read_bytes += size
            traffic.write_bytes += size
            # One retired `STORE` per descriptor (see `MoveStep.unit`);
            # `spill_count` stays one per buffer.
            traffic.tensor_store_count += descriptor_count(steps[-1])
            traffic.spill_count += 1
            alloc.free(addr, size)
            spilled[best_name] = ddr_addr
            tensor_ddr_addr[best_name] = ddr_addr
            return True

        def can_ever_fit(size: int, exclude: set[str], unit: int) -> bool:
            """Would `size` bytes fit locally if every buffer `evict_one`
            is *allowed* to evict were evicted?

            Asked BEFORE the first eviction, and answered on a throwaway
            copy of the free list, because the eviction loop is not
            transactional: `evict_one` appends a real `STORE` step and
            charges real traffic, so discovering only afterwards that the
            buffer still does not fit would leave a trail of pointless
            spills in the program (before this fallback existed that did
            not matter -- the `ValueError` threw the whole plan away).

            The eviction loop below evicts one victim at a time, so if
            the all-victims-evicted state admits the allocation the loop
            is guaranteed to terminate in success; if it does not, no
            sequence of evictions can help and the caller places the
            buffer in DDR instead, having moved nothing."""
            probe = alloc.clone()
            for name, addr in local_addr.items():
                if name in exclude or pending_producers[name]:
                    continue
                probe.free(addr, nbytes(root_by_name[name]))
            return probe.try_alloc(size, unit=unit) is not None

        def alloc_local(root_name: str, size: int, exclude: set[str], unit: int) -> int | None:
            """Place `root_name`'s `size`-byte buffer in the scratchpad,
            spilling other buffers if that is what it takes, and return
            its byte address -- or `None` when it cannot be placed
            locally at all, which is the caller's cue to leave the buffer
            in DDR (`place_in_ddr`).

            The two ways a buffer is unplaceable are both permanent
            properties of this moment, not transient pressure:

            * its confinement `unit` -- the largest single transfer the
              hardware will make against it -- is larger than one
              `cnn_accel_tensor_mem` bank, so no legal address exists for
              it at any occupancy (the hardware serves every transfer from
              the single bank the address decodes to and clamps anything
              past that bank's end). For an untiled tensor the unit is the
              whole buffer, which is the rule this planner has always
              applied;
            * even with every evictable buffer evicted there is no
              bank-confined hole big enough -- the free list is
              fragmented and this planner has no defragmenter, or the
              buffers that *cannot* be evicted (this op's own operands, a
              half-written concat family) already own too much of the
              scratchpad.

            Neither is an error. `ValueError` is reserved for programs
            that cannot be lowered *at all*; "does not fit on chip" has a
            correct, if slower, lowering, and taking it is what makes the
            planner degrade instead of falling over."""
            if unit > self.bank_bytes:
                return None
            addr = alloc.try_alloc(size, unit=unit)
            if addr is None:
                if not can_ever_fit(size, exclude, unit):
                    return None
                while addr is None:
                    if not evict_one(exclude):  # pragma: no cover - can_ever_fit said yes
                        raise ValueError(
                            f"planner: {size} bytes for '{root_name}' do not fit in the "
                            f"{self.tensor_mem_bytes}-byte tensor memory "
                            f"({self.tensor_mem_bytes // self.bank_bytes} x "
                            f"{self.bank_bytes}-byte banks, and a buffer may not cross a "
                            "bank boundary) even after evicting every evictable buffer"
                        )
                    addr = alloc.try_alloc(size, unit=unit)
            local_placements.append((root_name, addr, size))
            if unit != size:
                local_confine_units[root_name] = unit
            return addr

        def place_in_ddr(root_name: str, size: int, *, pinned: bool = False) -> int:
            """Give `root_name`'s buffer a permanent DDR home and return
            its address, allocating one on first call.

            This is the whole fallback. It is correct because the
            section-3 activation layout is byte-identical in DDR and in
            `LOCAL_TENSOR`: a buffer's slice offsets, plane order and
            alignment do not depend on which space it sits in, so every
            address the alias arithmetic computes stays valid, and both
            operand spaces are per-operand tags on the instruction
            (`space_src0`/`space_src1`/`space_dst`) that the hardware
            binds to a DMA instead of a scratchpad port. A concat family
            in particular needs nothing else: its parts' producers write
            their slices straight to DDR at `base + plane_offset *
            H*W*T`, and its consumer reads the assembled tensor from
            there. What it costs is bandwidth -- every write and every
            read of this buffer is now real AXI traffic, counted in
            `ddr_resident_write_bytes`/`ddr_resident_read_bytes`."""
            addr = ddr_resident.get(root_name)
            if addr is not None:
                return addr
            # A graph input (or a previously spilled buffer) already has
            # a DDR home; reuse it rather than allocating a second copy.
            addr = tensor_ddr_addr.get(root_name)
            if addr is None:
                region = DdrMap.OUTPUTS if root_by_name[root_name].is_output else DdrMap.SPILL
                addr = self.ddr_map.alloc(region, size)
                tensor_ddr_addr[root_name] = addr
            ddr_resident[root_name] = addr
            (pinned_placements if pinned else ddr_placements).append((root_name, addr, size))
            return addr

        def pin_if_needed(root: Tensor) -> bool:
            """Give a `Tensor.pin_ddr` buffer its permanent DDR home the
            first time anything touches it, and say whether it has one.

            Unconditional and immediate, unlike `place_in_ddr`'s other
            caller: a pinned buffer is not a scratchpad candidate that
            failed to fit, it is a fusion-group boundary that is *defined*
            to be backing storage -- written by one strip's store, read by
            another strip's load, many commands apart, and larger than the
            scratchpad in every case that matters. Trying to place it
            locally first would be a guaranteed-wasted eviction sweep."""
            if not root.pin_ddr:
                return False
            if root.name not in ddr_resident:
                place_in_ddr(root.name, nbytes(root), pinned=True)
            return True

        def resolve_input(
            t: Tensor,
            exclude: set[str],
            *,
            read_bytes: int | None = None,
            in_place: bool = False,
            passes: int = 1,
        ) -> tuple[int, int]:
            """Return `(space, addr)` for consuming `t` as an operand of
            the op currently being planned (`op_index`), doing whatever
            spill-bookkeeping / reload / local allocation that requires.

            `t` may be an alias (a `split` view, or a `concat` operand):
            everything below is decided for its `alias_root`'s BUFFER,
            and only the final address adds `alias_byte_offset(t)`.

            Rule: a buffer that is *currently* in DDR (never yet loaded,
            or spilled) is reloaded into a fresh local buffer if its
            family has more than one remaining consumer (this one
            included) -- so later consumers reuse the local copy for free
            -- with one exception: a *previously spilled* buffer is
            **always** reloaded explicitly, even with exactly one
            remaining consumer, because "spill, then read directly from
            DDR at the single remaining use" would silently skip the
            explicit `LOAD` the residency policy requires for a value
            that once left the scratchpad (section 10 R5's "no hidden
            traffic" is about accounting, but the spec's own wording --
            "an explicit spill... and a later reload" -- names the LOAD
            as mandatory, not merely one of several equally-valid
            lowerings). A never-yet-loaded graph input has no such
            history, so its single-remaining-consumer case is left as an
            ordinary direct `DDR` operand read (identical AXI bytes, one
            fewer instruction, and exactly the "DDR round trip for
            initial inputs" the policy already allows).

            `read_bytes` overrides how many bytes this operand fetch
            costs: a `RowCopyOp` reads a row *window*, not the whole
            tensor, and charging it the tensor's size would make the
            tiling look as expensive as the thing it replaces.
            `passes` is how many times the hardware will stream this
            operand -- `ifmap_passes(op)`, i.e. the output-channel-tile
            count for a convolution's ifmap and 1 for everything else.
            It multiplies the operand fetch only: a reload is one
            transfer however many times the value is then read, so the
            `LOAD` below is charged once and the `n_ot` re-reads land on
            `local_read_bytes`, which is exactly what the hardware does.

            `in_place` says never to hoist the buffer into the scratchpad
            for this read -- also a `RowCopyOp` property, and the whole
            point of tiling: loading a full-resolution group boundary
            into the scratchpad just to copy 20 rows out of it is exactly
            the traffic the strips exist to avoid."""
            root = root_of[t.name]
            offset = alias_byte_offset(t)
            operand_bytes = (nbytes(t) if read_bytes is None else read_bytes) * passes

            if root.name in local_addr:
                traffic.local_read_bytes += operand_bytes
                return isa.SPACE_LOCAL_TENSOR, local_addr[root.name] + offset

            pinned = pin_if_needed(root)

            if root.name in ddr_resident:
                # Never reloaded: a buffer only lives in DDR because it
                # provably has no local home (or, pinned, because it is
                # backing storage by design), so there is nowhere to
                # reload it to, and a later producer of another slice of
                # it would have to find it in DDR anyway.
                traffic.read_bytes += operand_bytes
                if pinned:
                    traffic.pinned_read_bytes += operand_bytes
                else:
                    traffic.ddr_resident_read_bytes += operand_bytes
                return isa.SPACE_DDR, ddr_resident[root.name] + offset

            if root.is_input and root.name not in tensor_ddr_addr:
                tensor_ddr_addr[root.name] = self.ddr_map.alloc(DdrMap.INPUTS, nbytes(root))
            if root.name not in tensor_ddr_addr:
                raise ValueError(
                    f"planner: '{t.name}' is read by op {op_index} but its buffer "
                    f"('{root.name}') is neither resident nor backed by DDR -- it was "
                    "freed before this use, which is a liveness bug in the planner"
                )
            ddr_addr = tensor_ddr_addr[root.name]

            was_spilled = root.name in spilled
            if not in_place and (was_spilled or remaining_uses(root.name, op_index) > 1):
                size = nbytes(root)
                addr = alloc_local(root.name, size, exclude, confine_unit(root))
                # `addr is None`: the buffer has no local home. Reading it
                # straight out of DDR moves the same bytes as the reload
                # would have, one instruction fewer, and is the only
                # lowering left -- so take it instead of failing. The
                # buffer stays in `spilled` (if it was), so a later use
                # under less pressure gets its explicit reload back.
                if addr is not None:
                    steps.append(
                        MoveStep(
                            tensor=root,
                            src_space=isa.SPACE_DDR,
                            src_addr=ddr_addr,
                            dst_space=isa.SPACE_LOCAL_TENSOR,
                            dst_addr=addr,
                            nbytes=size,
                            kind="reload" if was_spilled else "cold_load",
                            unit=root.confine_unit_bytes,
                        )
                    )
                    traffic.read_bytes += size
                    traffic.local_write_bytes += size
                    traffic.tensor_load_count += descriptor_count(steps[-1])
                    if was_spilled:
                        traffic.reload_count += 1
                        del spilled[root.name]
                    local_addr[root.name] = addr
                    # The reload above is its own DDR<->local transaction;
                    # the compute op then reads the now-resident copy as a
                    # second, separate local-memory transaction (mirrors
                    # reference.py, which always counts a ComputeStep's
                    # operand fetch regardless of how the operand got local).
                    traffic.local_read_bytes += operand_bytes
                    return isa.SPACE_LOCAL_TENSOR, addr + offset

            traffic.read_bytes += operand_bytes
            return isa.SPACE_DDR, ddr_addr + offset

        def release(root_name: str) -> None:
            """Free `root_name`'s local buffer once its whole alias family
            is dead: every member produced, and every consumer of every
            member already run."""
            if pending_consumers[root_name] or pending_producers[root_name]:
                return
            if total_consumers[root_name] == 0:
                # Never read by anything: keep the (dead) buffer, exactly
                # as this planner always has -- freeing it here would
                # shuffle every later address for no benefit.
                return
            if root_name in local_addr:
                freed_addr = local_addr.pop(root_name)
                alloc.free(freed_addr, nbytes(root_by_name[root_name]))

        def resolve_output(
            out_t: Tensor, exclude: set[str], *, write_bytes: int | None = None
        ) -> tuple[int, int]:
            """Return `(space, addr)` for the tensor an op writes, doing
            whatever allocation / DDR fallback that requires.

            `write_bytes` overrides the byte charge for the same reason
            `resolve_input`'s `read_bytes` does: a `RowCopyOp` writes a row
            window, not a whole tensor."""
            out_root = root_of[out_t.name]
            out_offset = alias_byte_offset(out_t)
            written = nbytes(out_t) if write_bytes is None else write_bytes

            if pin_if_needed(out_root):
                traffic.write_bytes += written
                traffic.pinned_write_bytes += written
                return isa.SPACE_DDR, ddr_resident[out_root.name] + out_offset

            if out_root.is_output:
                if out_root.name not in tensor_ddr_addr:
                    tensor_ddr_addr[out_root.name] = self.ddr_map.alloc(
                        DdrMap.OUTPUTS, nbytes(out_root)
                    )
                traffic.write_bytes += written
                return isa.SPACE_DDR, tensor_ddr_addr[out_root.name] + out_offset

            if out_root.name in spilled:  # pragma: no cover - see evict_one
                raise ValueError(
                    f"planner: op {op_index} writes '{out_t.name}' into the buffer of "
                    f"'{out_root.name}', which is currently spilled to DDR -- a "
                    "half-written concat buffer must never be evicted (see evict_one)"
                )
            if out_root.name not in local_addr and out_root.name not in ddr_resident:
                addr = alloc_local(
                    out_root.name, nbytes(out_root), exclude, confine_unit(out_root)
                )
                if addr is None:
                    place_in_ddr(out_root.name, nbytes(out_root))
                else:
                    local_addr[out_root.name] = addr
            if out_root.name in ddr_resident:
                traffic.write_bytes += written
                traffic.ddr_resident_write_bytes += written
                return isa.SPACE_DDR, ddr_resident[out_root.name] + out_offset
            traffic.local_write_bytes += written
            return isa.SPACE_LOCAL_TENSOR, local_addr[out_root.name] + out_offset

        # -- resident weight images (design D6, hardware fact F7) ---------
        #
        # Every convolution that shares one `weight` list shares one
        # image: that is precisely what the tiler produces when it clones
        # a conv onto `S` strips, and sharing the image is the whole
        # saving. Keyed by the identity of that list -- the ops keep it
        # alive for the duration of this call -- and named after the
        # first conv to ask, so a placement reads as
        # `'<conv>#const#weight'`.
        image_of_op: dict[int, str] = {}
        image_users: dict[str, list[int]] = {}
        for index, candidate in enumerate(model.ops):
            if not isinstance(candidate, Conv2dOp) or not candidate.weights_resident:
                continue
            if candidate.weight_reuse:
                # No refill to redirect: `WEIGHT_REUSE` skips `st_wgt_req`
                # outright, so residency has nothing to say about it.
                continue
            key = f"{candidate.name}#const"
            for seen_key, users in image_users.items():
                if model.ops[users[0]].weight is candidate.weight:
                    key = seen_key
                    break
            image_of_op[index] = key
            image_users.setdefault(key, []).append(index)

        #: key -> `{kind: local addr}` once the image is resident, or
        #: `None` once it has been decided that it cannot be. Sticky
        #: either way: the decision is made at the image's first use and
        #: every later user must agree with it, or half the passes would
        #: read a scratchpad that never got loaded.
        image_addrs: dict[str, dict[str, int] | None] = {}
        #: key -> the `(addr, size)` buffers to hand back to the
        #: allocator after the image's last user.
        image_buffers: dict[str, list[tuple[int, int]]] = {}

        def load_const_image(key: str, conv: Conv2dOp, exclude: set[str]) -> dict[str, int] | None:
            """Place `conv`'s packed side tables in the scratchpad and
            emit the one-time `LOAD`, or return `None` when they do not
            fit -- in which case the caller leaves `space_wgt` at DDR and
            the program is merely the slower one it would have been
            anyway. Each sub-image gets its own buffer with its own
            confinement unit (see `ConstLoadStep`)."""
            placed: list[tuple[str, int, int]] = []
            buffers: list[tuple[int, int]] = []
            for kind, size, unit in const_images(conv):
                addr = alloc_local(f"{key}#{kind}", size, exclude, unit)
                if addr is None:
                    for freed_addr, freed_size in buffers:
                        alloc.free(freed_addr, freed_size)
                        local_placements.pop()
                        local_confine_units.pop(f"{key}#{placed.pop()[0]}", None)
                    return None
                placed.append((kind, addr, size))
                buffers.append((addr, size))
            steps.append(ConstLoadStep(conv=conv, key=key, images=placed))
            moved = sum(size for _, _, size in placed)
            traffic.read_bytes += moved
            traffic.local_write_bytes += moved
            traffic.tensor_load_count += len(placed)
            image_buffers[key] = buffers
            return {kind: addr for kind, addr, _ in placed}

        for op_index, op in enumerate(model.ops):
            exclude_this_op = {root_of[t.name].name for t in op.inputs}

            if isinstance(op, RowCopyOp):
                steps.append(_plan_row_copy(op, exclude_this_op, resolve_input, resolve_output, traffic))
            else:
                input_spaces: list[int] = []
                input_addrs: list[int] = []
                # Only `inputs[0]` is re-streamed by the output-channel
                # loop; `ADD`'s second operand is read once like every
                # other elementwise input.
                passes = ifmap_passes(op)
                for index, t in enumerate(op.inputs):
                    space, addr = resolve_input(
                        t, exclude_this_op, passes=passes if index == 0 else 1
                    )
                    input_spaces.append(space)
                    input_addrs.append(addr)

                # The weight LOAD is emitted before the operand
                # resolution's own steps would be, but after the operand
                # roots are known, so the images can never evict a buffer
                # this very command is about to read.
                key = image_of_op.get(op_index)
                weight_addrs = None
                if key is not None:
                    if key not in image_addrs:
                        image_addrs[key] = load_const_image(key, op, exclude_this_op)
                    weight_addrs = image_addrs[key]

                output_space, out_addr = resolve_output(op.output, exclude_this_op)
                steps.append(
                    ComputeStep(
                        op=op,
                        input_spaces=input_spaces,
                        input_addrs=input_addrs,
                        output_space=output_space,
                        output_addr=out_addr,
                        weight_space=(
                            isa.SPACE_LOCAL_TENSOR if weight_addrs else isa.SPACE_DDR
                        ),
                        weight_addrs=weight_addrs,
                    )
                )

            if isinstance(op, ActOp):
                # The standalone LUT is a compile-time side table in DDR
                # (`program.py` puts it at `weight_addr`/`space_wgt`), and
                # `cnn_accel_elementwise` refills all 256 entries for every
                # `ACT` command -- there is no cache across descriptors. It
                # is a plain DDR read, not weight-buffer traffic: the
                # hardware's `WEIGHT_LOAD_BYTES` counter is only touched by
                # the conv weight refill and by `LOADW`, never by this.
                traffic.read_bytes += isa.ACT_LUT_BYTES

            if isinstance(op, Conv2dOp):
                charge_weight_fill(
                    op,
                    traffic,
                    resident=isinstance(steps[-1], ComputeStep)
                    and steps[-1].weight_space == isa.SPACE_LOCAL_TENSOR,
                )

            # An image is freed the moment its last user has run, exactly
            # like any other buffer -- a group's weights must not squat in
            # the scratchpad for the whole program.
            key = image_of_op.get(op_index)
            if key is not None and op_index == image_users[key][-1]:
                for freed_addr, freed_size in image_buffers.pop(key, []):
                    alloc.free(freed_addr, freed_size)

            out_root_name = root_of[op.output.name].name
            pending_producers[out_root_name] -= 1
            for t in op.inputs:
                pending_consumers[root_of[t.name].name] -= 1
            release(out_root_name)
            for t in op.inputs:
                release(root_of[t.name].name)

        # +1 for the closing HALT. Every step is one fetched 64-byte
        # descriptor (section 6/CSR "incl. descriptors") except a
        # `RowCopyStep`, which is one per plane -- see `descriptor_count`.
        traffic.read_bytes += (
            sum(descriptor_count(step) for step in steps) + 1
        ) * isa.INSTR_WORD_BYTES

        return PlannedProgram(
            model=model,
            steps=steps,
            traffic=traffic,
            tensor_mem_bytes=self.tensor_mem_bytes,
            ddr_map=self.ddr_map,
            tensor_ddr_addr=tensor_ddr_addr,
            bank_bytes=self.bank_bytes,
            local_placements=local_placements,
            local_confine_units=local_confine_units,
            ddr_placements=ddr_placements,
            pinned_placements=pinned_placements,
        )


__all__ = [
    "DEFAULT_BANK_BYTES",
    "DEFAULT_TENSOR_MEM_BYTES",
    "DdrTraffic",
    "ComputeStep",
    "ConstLoadStep",
    "charge_weight_fill",
    "const_images",
    "weight_fill_bytes",
    "MoveStep",
    "RowCopyStep",
    "Step",
    "descriptor_count",
    "ifmap_passes",
    "PlannedProgram",
    "Planner",
]
