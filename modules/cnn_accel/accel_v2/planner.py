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
  rule for values that were *previously* spilled.

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
from accel_v2.model import Conv2dOp, Model, Op, Tensor

#: Section 4 default: `g_num_banks=2 * g_bank_words=1024 * 8 bytes`.
#: The planner treats the scratchpad as one flat byte-addressable range
#: (bank interleaving/arbitration is a `cnn_accel_tensor_mem` RTL concern
#: with no Python-visible effect on which bytes land where -- see the
#: module docstring of `reference.py` for the same scoping note).
DEFAULT_TENSOR_MEM_BYTES = 2 * 1024 * 8


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


class _LocalAllocator:
    """First-fit free-list allocator over `[0, capacity)`. Buffers are
    freed the instant their last consumer has run (section 3.1: "Buffer
    reuse... [is] the producer's [compiler's] responsibility"), so the
    free list is usually fragmented in the same way a real heap is; the
    planner's job is only to decide *which* tensor to evict when
    first-fit fails, not to defragment."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._free: list[tuple[int, int]] = [(0, capacity)] if capacity > 0 else []

    def try_alloc(self, size: int, align: int = 8) -> int | None:
        if size <= 0:
            raise ValueError(f"allocation size must be positive, got {size}")
        for i, (start, length) in enumerate(self._free):
            pad = (-start) % align
            if length - pad < size:
                continue
            aligned_start = start + pad
            end = aligned_start + size
            replacement = []
            if pad > 0:
                replacement.append((start, pad))
            if length - pad - size > 0:
                replacement.append((end, length - pad - size))
            self._free[i : i + 1] = replacement
            return aligned_start
        return None

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
    def __init__(self, tensor_mem_bytes: int = DEFAULT_TENSOR_MEM_BYTES, ddr_map: DdrMap | None = None) -> None:
        self.tensor_mem_bytes = tensor_mem_bytes
        self.ddr_map = ddr_map if ddr_map is not None else DdrMap()

    def plan(self, model: Model) -> PlannedProgram:
        alloc = _LocalAllocator(self.tensor_mem_bytes)
        traffic = DdrTraffic()
        steps: list[Step] = []
        tensor_ddr_addr: dict[str, int] = {}
        tensor_by_name = {t.name: t for t in model.tensors}

        #: tensor.name -> its current LOCAL_TENSOR byte offset, only
        #: while resident.
        local_addr: dict[str, int] = {}
        #: tensor.name -> its DDR spill address, only while spilled
        #: (evicted, not yet reloaded). Absent for a tensor that has
        #: never been spilled.
        spilled: dict[str, int] = {}
        #: tensor.name -> number of not-yet-executed consumers.
        remaining = {t.name: len(t.consumers) for t in model.tensors}

        def nbytes(t: Tensor) -> int:
            return t.size_bytes

        def next_use(t: Tensor, from_idx: int) -> int | None:
            for c in t.consumers:
                if c >= from_idx:
                    return c
            return None

        def evict_one(exclude: set[str]) -> bool:
            """Evict the resident, non-excluded tensor whose next use is
            furthest in the future (Belady's rule: minimizes the chance
            the very next allocation has to evict again). `exclude` is
            the current op's own input set -- a value about to be read
            can never be its own eviction victim."""
            best_name: str | None = None
            best_next = -1
            for name in local_addr:
                if name in exclude:
                    continue
                nxt = next_use(tensor_by_name[name], op_index)
                score = nxt if nxt is not None else 1 << 62
                if score > best_next:
                    best_next, best_name = score, name
            if best_name is None:
                return False
            victim = tensor_by_name[best_name]
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

        def alloc_local(size: int, exclude: set[str]) -> int:
            addr = alloc.try_alloc(size)
            while addr is None:
                if not evict_one(exclude):
                    raise ValueError(
                        f"planner: {size} bytes do not fit in the "
                        f"{self.tensor_mem_bytes}-byte tensor memory even "
                        "after evicting every evictable buffer"
                    )
                addr = alloc.try_alloc(size)
            return addr

        def resolve_input(t: Tensor, exclude: set[str]) -> tuple[int, int]:
            """Return `(space, addr)` for consuming `t` as an operand of
            the op currently being planned (`op_index`), doing whatever
            spill-bookkeeping / reload / local allocation that requires.

            Rule: a tensor that is *currently* in DDR (never yet loaded,
            or spilled) is reloaded into a fresh local buffer if it has
            more than one remaining consumer (itself included) -- so
            later consumers reuse the local copy for free -- with one
            exception: a *previously spilled* tensor is **always**
            reloaded explicitly, even with exactly one remaining
            consumer, because "spill, then read directly from DDR at the
            single remaining use" would silently skip the explicit
            `LOAD` the residency policy requires for a value that once
            left the scratchpad (section 10 R5's "no hidden traffic" is
            about accounting, but the spec's own wording -- "an explicit
            spill... and a later reload" -- names the LOAD as mandatory,
            not merely one of several equally-valid lowerings). A
            never-yet-loaded graph input has no such history, so its
            single-remaining-consumer case is left as an ordinary direct
            `DDR` operand read (identical AXI bytes, one fewer
            instruction, and exactly the "DDR round trip for initial
            inputs" the policy already allows)."""
            if t.name in local_addr:
                traffic.local_read_bytes += nbytes(t)
                return isa.SPACE_LOCAL_TENSOR, local_addr[t.name]

            if t.is_input and t.name not in tensor_ddr_addr:
                tensor_ddr_addr[t.name] = self.ddr_map.alloc(DdrMap.INPUTS, nbytes(t))
            ddr_addr = tensor_ddr_addr[t.name]

            was_spilled = t.name in spilled
            remaining_from_here = sum(1 for c in t.consumers if c >= op_index)
            if was_spilled or remaining_from_here > 1:
                size = nbytes(t)
                addr = alloc_local(size, exclude)
                steps.append(
                    MoveStep(
                        tensor=t,
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
                    del spilled[t.name]
                local_addr[t.name] = addr
                # The reload above is its own DDR<->local transaction;
                # the compute op then reads the now-resident copy as a
                # second, separate local-memory transaction (mirrors
                # reference.py, which always counts a ComputeStep's
                # operand fetch regardless of how the operand got local).
                traffic.local_read_bytes += size
                return isa.SPACE_LOCAL_TENSOR, addr

            traffic.read_bytes += nbytes(t)
            return isa.SPACE_DDR, ddr_addr

        for op_index, op in enumerate(model.ops):
            exclude_this_op = {t.name for t in op.inputs}
            input_spaces: list[int] = []
            input_addrs: list[int] = []
            for t in op.inputs:
                space, addr = resolve_input(t, exclude_this_op)
                input_spaces.append(space)
                input_addrs.append(addr)

            out_t = op.output
            out_size = nbytes(out_t)
            if out_t.is_output:
                out_addr = self.ddr_map.alloc(DdrMap.OUTPUTS, out_size)
                tensor_ddr_addr[out_t.name] = out_addr
                output_space = isa.SPACE_DDR
                traffic.write_bytes += out_size
            else:
                out_addr = alloc_local(out_size, exclude_this_op)
                local_addr[out_t.name] = out_addr
                output_space = isa.SPACE_LOCAL_TENSOR
                traffic.local_write_bytes += out_size

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

            for t in op.inputs:
                remaining[t.name] -= 1
                if remaining[t.name] == 0 and t.name in local_addr:
                    freed_addr = local_addr.pop(t.name)
                    alloc.free(freed_addr, nbytes(t))

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
        )


__all__ = [
    "DEFAULT_TENSOR_MEM_BYTES",
    "DdrTraffic",
    "ComputeStep",
    "MoveStep",
    "Step",
    "PlannedProgram",
    "Planner",
]
