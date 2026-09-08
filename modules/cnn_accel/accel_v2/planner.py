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

The output is a `PlannedProgram`: an ordered list of `ComputeStep`/
`MoveStep`s with every operand's space and address already resolved,
plus a `DdrTraffic` prediction that `program.py`'s actual DDR-image
emission and `reference.py`'s actual execution are both checked against
(the whole point of computing it up front, per the spec's residency
invariants R1-R6).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cnn_accel_model as golden

from accel_v2 import isa
from accel_v2.ddrmap import DdrMap
from accel_v2.model import Conv2dOp, Model, Op, Tensor, alias_byte_offset, alias_root

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


Step = ComputeStep | MoveStep


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
    bank_bytes`.

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

    def try_alloc(self, size: int, align: int = 8) -> int | None:
        if size <= 0:
            raise ValueError(f"allocation size must be positive, got {size}")
        if size > self.bank_bytes:
            # Caller (`Planner.alloc_local`) is expected to have rejected
            # this already with a much more actionable message; belt and
            # braces, since silently returning an illegal address is the
            # exact failure mode this class exists to prevent.
            return None
        for i, (start, length) in enumerate(self._free):
            aligned_start = start + (-start) % align
            # Bank boundaries are whole multiples of `align` (a bank is a
            # whole number of 8-byte words), so bumping to one keeps the
            # alignment guarantee.
            bank_end = (aligned_start // self.bank_bytes + 1) * self.bank_bytes
            if aligned_start + size > bank_end:
                aligned_start = bank_end
            pad = aligned_start - start
            if length - pad < size:
                continue
            end = aligned_start + size
            replacement = []
            if pad > 0:
                replacement.append((start, pad))
            if length - pad - size > 0:
                replacement.append((end, length - pad - size))
            self._free[i : i + 1] = replacement
            return aligned_start
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
            pending_producers[name] = sum(1 for t in members if t.producer is not None)

        def nbytes(t: Tensor) -> int:
            return t.size_bytes

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
                )
            )
            traffic.local_read_bytes += size
            traffic.write_bytes += size
            traffic.tensor_store_count += 1
            traffic.spill_count += 1
            alloc.free(addr, size)
            spilled[best_name] = ddr_addr
            tensor_ddr_addr[best_name] = ddr_addr
            return True

        def can_ever_fit(size: int, exclude: set[str]) -> bool:
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
            return probe.try_alloc(size) is not None

        def alloc_local(root_name: str, size: int, exclude: set[str]) -> int | None:
            """Place `root_name`'s `size`-byte buffer in the scratchpad,
            spilling other buffers if that is what it takes, and return
            its byte address -- or `None` when it cannot be placed
            locally at all, which is the caller's cue to leave the buffer
            in DDR (`place_in_ddr`).

            The two ways a buffer is unplaceable are both permanent
            properties of this moment, not transient pressure:

            * it is larger than one `cnn_accel_tensor_mem` bank, so no
              legal address exists for it at any occupancy (the hardware
              serves every transfer from the single bank the address
              decodes to and clamps anything past that bank's end);
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
            if size > self.bank_bytes:
                return None
            addr = alloc.try_alloc(size)
            if addr is None:
                if not can_ever_fit(size, exclude):
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
                    addr = alloc.try_alloc(size)
            local_placements.append((root_name, addr, size))
            return addr

        def place_in_ddr(root_name: str, size: int) -> int:
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
                addr = self.ddr_map.alloc(DdrMap.SPILL, size)
                tensor_ddr_addr[root_name] = addr
            ddr_resident[root_name] = addr
            ddr_placements.append((root_name, addr, size))
            return addr

        def resolve_input(t: Tensor, exclude: set[str]) -> tuple[int, int]:
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
            initial inputs" the policy already allows)."""
            root = root_of[t.name]
            offset = alias_byte_offset(t)

            if root.name in local_addr:
                traffic.local_read_bytes += nbytes(t)
                return isa.SPACE_LOCAL_TENSOR, local_addr[root.name] + offset

            if root.name in ddr_resident:
                # Never reloaded: a buffer only lives in DDR because it
                # provably has no local home, so there is nowhere to
                # reload it to, and a later producer of another slice of
                # it would have to find it in DDR anyway.
                traffic.read_bytes += nbytes(t)
                traffic.ddr_resident_read_bytes += nbytes(t)
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
            if was_spilled or remaining_uses(root.name, op_index) > 1:
                size = nbytes(root)
                addr = alloc_local(root.name, size, exclude)
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
                        )
                    )
                    traffic.read_bytes += size
                    traffic.local_write_bytes += size
                    traffic.tensor_load_count += 1
                    if was_spilled:
                        traffic.reload_count += 1
                        del spilled[root.name]
                    local_addr[root.name] = addr
                    # The reload above is its own DDR<->local transaction;
                    # the compute op then reads the now-resident copy as a
                    # second, separate local-memory transaction (mirrors
                    # reference.py, which always counts a ComputeStep's
                    # operand fetch regardless of how the operand got local).
                    traffic.local_read_bytes += nbytes(t)
                    return isa.SPACE_LOCAL_TENSOR, addr + offset

            traffic.read_bytes += nbytes(t)
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

        for op_index, op in enumerate(model.ops):
            exclude_this_op = {root_of[t.name].name for t in op.inputs}
            input_spaces: list[int] = []
            input_addrs: list[int] = []
            for t in op.inputs:
                space, addr = resolve_input(t, exclude_this_op)
                input_spaces.append(space)
                input_addrs.append(addr)

            out_t = op.output
            out_root = root_of[out_t.name]
            out_offset = alias_byte_offset(out_t)
            if out_root.is_output:
                if out_root.name not in tensor_ddr_addr:
                    tensor_ddr_addr[out_root.name] = self.ddr_map.alloc(
                        DdrMap.OUTPUTS, nbytes(out_root)
                    )
                out_addr = tensor_ddr_addr[out_root.name] + out_offset
                output_space = isa.SPACE_DDR
                traffic.write_bytes += nbytes(out_t)
            else:
                if out_root.name in spilled:  # pragma: no cover - see evict_one
                    raise ValueError(
                        f"planner: op {op_index} writes '{out_t.name}' into the buffer of "
                        f"'{out_root.name}', which is currently spilled to DDR -- a "
                        "half-written concat buffer must never be evicted (see evict_one)"
                    )
                if out_root.name not in local_addr and out_root.name not in ddr_resident:
                    addr = alloc_local(out_root.name, nbytes(out_root), exclude_this_op)
                    if addr is None:
                        place_in_ddr(out_root.name, nbytes(out_root))
                    else:
                        local_addr[out_root.name] = addr
                if out_root.name in ddr_resident:
                    out_addr = ddr_resident[out_root.name] + out_offset
                    output_space = isa.SPACE_DDR
                    traffic.write_bytes += nbytes(out_t)
                    traffic.ddr_resident_write_bytes += nbytes(out_t)
                else:
                    out_addr = local_addr[out_root.name] + out_offset
                    output_space = isa.SPACE_LOCAL_TENSOR
                    traffic.local_write_bytes += nbytes(out_t)

            steps.append(
                ComputeStep(
                    op=op,
                    input_spaces=input_spaces,
                    input_addrs=input_addrs,
                    output_space=output_space,
                    output_addr=out_addr,
                )
            )

            if isinstance(op, Conv2dOp) and not op.weight_reuse:
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

            pending_producers[out_root.name] -= 1
            for t in op.inputs:
                pending_consumers[root_of[t.name].name] -= 1
            release(out_root.name)
            for t in op.inputs:
                release(root_of[t.name].name)

        # +1 for the closing HALT: every step corresponds to exactly one
        # fetched 64-byte descriptor (section 6/CSR "incl. descriptors").
        traffic.read_bytes += (len(steps) + 1) * isa.INSTR_WORD_BYTES

        return PlannedProgram(
            model=model,
            steps=steps,
            traffic=traffic,
            tensor_mem_bytes=self.tensor_mem_bytes,
            ddr_map=self.ddr_map,
            tensor_ddr_addr=tensor_ddr_addr,
            bank_bytes=self.bank_bytes,
            local_placements=local_placements,
            ddr_placements=ddr_placements,
        )


__all__ = [
    "DEFAULT_BANK_BYTES",
    "DEFAULT_TENSOR_MEM_BYTES",
    "DdrTraffic",
    "ComputeStep",
    "MoveStep",
    "Step",
    "PlannedProgram",
    "Planner",
]
