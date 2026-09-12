# cnn_accel: INPUT_ADDR/OUTPUT_ADDR relocation + one-deep job queue (ISA v2.3)

Status: approved by user, implementation in progress.

## 1. Motivation

Today a compiled program's descriptor chain has every tensor's DDR
address baked in at compile time; `PROGRAM_BASE_ADDR` only points at the
chain. For streaming inference (the same compiled network run on many
frames back to back), the host would otherwise have to recompile or
patch the descriptor chain per frame. This feature lets the host instead
compile the network **once** and, per frame, only supply where the input
tensor is and where the output should land — plus lets the host queue
the *next* frame's addresses while the current frame is still running,
so the accelerator can go frame-to-frame without host-latency gaps.

## 2. Register map additions (nothing existing changes)

| Offset | Name | R/W | Bits |
|---|---|---|---|
| `0x44` | `INPUT_ADDR` | RW | graph input tensor's DDR base for the job about to start (or about to be queued) |
| `0x48` | `OUTPUT_ADDR` | RW | graph output tensor's DDR base, same deal |

`STATUS` (`0x08`) gains one bit, taken from the byte that was previously
all-reserved (`reserved1`, bits `[15:8]`):

- bit `8`: `QUEUED` — a job is latched waiting for the current one to
  finish. Auto-clears the instant that job is dispatched.
- `reserved1` shrinks to bits `[15:9]` (7 bits), still must-be-zero.

`ERR_CODE` (`STATUS` bits `[7:4]`, 4-bit field, values `0x0`-`0xF`) gains
one value in previously-unused space:

- `0xA` = `ERR_QUEUE_FULL` — a `CTRL.START` write arrived while
  `BUSY=1` and a job was already queued.

## 3. `CTRL.START` semantics (the only behavior change to an existing
register)

Today: `start_i <= regs_down.ctrl.start and not busy_q;` — a `START`
write while busy is silently dropped (`cnn_accel_csr.vhd:130`). New
behavior, all three cases:

- `BUSY=0`: starts immediately, exactly as today, latching
  `PROGRAM_BASE_ADDR`/`INPUT_ADDR`/`OUTPUT_ADDR` into internal
  "in-use" registers (`INPUT_ADDR`/`OUTPUT_ADDR` join the existing
  `program_base_addr_latch` pattern).
- `BUSY=1`, no job already queued: latches the just-written
  `INPUT_ADDR`/`OUTPUT_ADDR` into a **queued** pair, sets
  `STATUS.QUEUED`. `PROGRAM_BASE_ADDR` is not re-latched here — the
  queued job always reruns the same compiled program.
- `BUSY=1`, a job is already queued: rejected. `STATUS.ERROR` sets,
  `ERR_CODE=ERR_QUEUE_FULL`, same `IRQ_MASK.ERROR` cause as any other
  error. The **running** job is completely unaffected — only the
  rejected write's addresses are discarded.

On the running job's `STATUS.DONE` (which still asserts on *every*
completion, including one that immediately auto-restarts): if
`STATUS.QUEUED` was set, the queued `INPUT_ADDR`/`OUTPUT_ADDR` become
the new "in use" pair, `QUEUED` clears, and the same compiled program
auto-dispatches — no second `CTRL.START` write needed.

## 4. Address relocation mechanism (the ISA change)

Rejected alternative (do not implement): two new operand-space codes
(`SPACE_DDR_INPUT`/`SPACE_DDR_OUTPUT`). `SPACE_TAG_BITS` is 2 and all
4 values are already assigned (`DDR`/`LOCAL_TENSOR`/`LOCAL_WEIGHT`/
`RESERVED`), and both the `spaces` byte and the `flags` byte are fully
packed (8/8 bits each) — widening either means spilling into the one
byte that is *not* fully packed, `reserved_w0` (W0 byte 3, currently
required to be all-zero), and every one of the ~15 RTL sites that
already switch on `space_src0`/`space_dst` would need to learn a 3-bit
comparison instead of 2-bit. High blast radius for no functional gain
over the option below.

**Actual mechanism**: two new independent per-descriptor boolean flags,
taking 2 of the 8 bits of `reserved_w0` (which is otherwise still
all-zero-by-default, exactly like `pad_value`/`dts_factor` did to
`reserved_w10` bytes in ISA v2.1/v2.2 — this is an established pattern
in this codebase, not a new one):

- bit 0: `reloc_input` — when set **and** this operand's space is
  `SPACE_DDR`, add `INPUT_ADDR` (the latched register, not the raw bus
  value) to the descriptor's `in_addr` before use.
- bit 1: `reloc_output` — same, adding `OUTPUT_ADDR` to `out_addr`.
- bits `[7:2]`: still `reserved_w0`, still must-be-zero
  (`ERR_BAD_RESERVED` if not). `DescV2.reserved_w0`'s dataclass value
  is redefined as *only* this 6-bit value (right-justified) rather than
  the raw byte — `cases_error.py`'s existing
  `overrides={"reserved_w0": 1}` still triggers `ERR_BAD_RESERVED`
  unchanged, since bit value `1` still lands in the must-be-zero range
  under the new packing (bit 2 of the raw byte), not in the new
  `reloc_*` bits (bits 0-1).

This works uniformly for however many descriptors touch the graph's
input/output tensor — a tiled program's several row-copy planes each
carry their own `reloc_input`, exactly as approved.

Default (`reloc_input=reloc_output=False`, `INPUT_ADDR=OUTPUT_ADDR=0`)
reproduces every existing program's behavior exactly: relocation adds
zero. All 79 existing catalogue cases need no changes to pass, once
their config sets both registers to `0` (already the reset value, so in
practice: no change needed at all).

## 5. Compiler (`accel_v2/program.py`)

`emit_program` gets one new post-pass, after all descriptors are built
and before `ProgramImage` is returned: for each `DescV2`, if
`space_src0 == SPACE_DDR` and `in_addr` falls inside any graph input
tensor's `[tensor_ddr_addr, tensor_ddr_addr + size_bytes)` window (from
`planned.tensor_ddr_addr`/`planned.model.inputs`), set `reloc_input =
True`. Symmetrically for `space_dst`/`out_addr`/`planned.model.outputs`
→ `reloc_output`. Address-range matching (not "is this operand
syntactically the tensor") is what makes this correct for both tiled
(N row-copy planes) and aliased/CONCAT (N producer descriptors writing
different slices of one output tensor) cases without special-casing
either — the same technique `tbcase.py`'s own `_check_outputs_against`
already uses to find a tensor's producer(s).

Two known, disclosed v1 limitations: only `src0`/`dst` are checked
(never `src1`/`wgt`), so a graph input consumed as `ADD`'s *second*
operand would not be relocatable — no catalogue case does this today.

## 6. RTL

### `cnn_accel_cmd_proc.vhd`

- Decode `reloc_input`/`reloc_output` alongside the existing
  `space_src0`/`space_dst` decode (same "resolve once" registers,
  `cmd_proc.vhd:1234-1270` — new `reloc_input_q`/`reloc_output_q`
  computed there).
- Two new input ports, `input_addr`/`output_addr` (`std_ulogic_vector
  (31 downto 0)`), fed from `cnn_accel_csr`'s latched "in-use"
  registers.
- Compute `effective_in_addr <= desc_q.in_addr + unsigned(input_addr)
  when desc_q.reloc_input = '1' else desc_q.in_addr;` and the output
  equivalent as **combinational** signals (not registered, despite the
  file's usual `_q` convention) at the same resolve point. This was a
  deliberate deviation found during implementation: alignment/DDR-range
  validation of `in_addr`/`out_addr` runs *before* the "resolve once"
  block, in the same cycle `desc_q` is latched, so a registered
  `effective_*_addr` would still be one cycle stale when that
  validation reads it — the combinational version is what makes the
  validation see the real, relocated address. Every consume site that
  read `desc_q.in_addr`/`desc_q.out_addr` directly (alignment checks,
  range-end checks, request addresses, feed addresses, elementwise
  addresses — a dozen-odd sites across the file; tiling offsets like
  `out_pass_off_q`/`in_pass_off_q` still add on top of the *effective*,
  already-relocated base) switches to the new signal instead.
- `ERR_BAD_RESERVED` check (`cmd_proc.vhd:1069`) becomes
  `desc_q.reserved_w0(7 downto 2) /= "000000"` instead of comparing the
  whole byte to `x"00"`.
- **Performance counter snapshot (discovered during implementation, not
  in the original design pass).** Every run-scoped counter (`CMD_COUNT`,
  `CYCLE_COUNT`, `COMPUTE_CYCLES`, `STALL_CYCLES`, `DDR_RD_BYTES`,
  `DDR_WR_BYTES`, `TENSOR_LOAD_COUNT`, `TENSOR_STORE_COUNT`,
  `WEIGHT_LOAD_BYTES`, `LOCAL_RD_KIB`, `LOCAL_WR_KIB`) is cleared at
  `START` and free-runs until the next one — a pre-existing contract
  that assumed a host always reads a completed run's counters before
  issuing the next `START`. Auto-dispatch breaks that assumption: the
  queued job's `START` fires within a cycle or two of the completed
  job's own `DONE`, well before a host polling loop can read anything.
  Fixed by adding a second bank of `*_snap_q` registers per counter,
  latched once on `seq_done`/`seq_error` (before that `START` can touch
  the live ones) and held until the *next* completion — the CSR-facing
  `counters` record now reads the snapshot bank, not the live one.
  Four of the eleven counters (`CMD_COUNT`, `TENSOR_LOAD_COUNT`,
  `TENSOR_STORE_COUNT`, `WEIGHT_LOAD_BYTES`) were previously exempt from
  the `START` clear entirely (a lifetime-cumulative design with no
  currently-known consumer relying on that); this pass unifies all
  eleven onto the same per-run-then-snapshotted model, since nothing in
  the existing 85-case regression depends on cross-run accumulation and
  a uniform model is far less surprising.

### `cnn_accel_csr.vhd`

- New RW registers `INPUT_ADDR`/`OUTPUT_ADDR` (hdl-registers-generated,
  same as `PROGRAM_BASE_ADDR`).
- `program_base_addr_latch` process gains two more "used" registers
  (`input_addr_used_q`/`output_addr_used_q`), latched on `start_i`
  exactly like `program_base_addr_used_q` is today — except the value
  latched is the queued pair when this `start_i` pulse is the
  *auto-dispatch* one, or the live bus register otherwise.
- New `queue_tracking` process: `queued_q`, `queued_input_addr_q`,
  `queued_output_addr_q`. Sets on a `START` write while `busy_q='1'`
  and `queued_q='0'`; raises the new `ERR_QUEUE_FULL` cause (feeding
  the same error-latch path `seq_error` already drives) on a `START`
  write while `busy_q='1'` and `queued_q='1'`; clears (and triggers
  one-cycle `auto_dispatch_pulse`) on `seq_done` when `queued_q='1'`.
- `start_i <= (regs_down.ctrl.start and not busy_q) or
  auto_dispatch_pulse;`
- `irq`/`IRQ_MASK` logic unchanged — `ERR_QUEUE_FULL` rides the
  existing `ERROR` cause.

### `cnn_accel_top.vhd`

Wire the two new `cmd_proc` ports through from `csr_inst`'s new latched
outputs. No new top-level ports (still AXI-Lite + AXI4 + `irq`, as
today).

## 7. ISA source-of-truth changes

- `cnn_accel_constants.py`: no change to `SPACES`/`SPACE_TAG_BITS`.
- `isa.py`: `DescV2` gains `reloc_input: bool = False`, `reloc_output:
  bool = False`; `reserved_w0`'s semantic width shrinks from 8 to 6
  bits (still named `reserved_w0`, still defaults to `0`);
  `encode_desc`/`decode_desc` updated; new `ERR_QUEUE_FULL = 0xA`
  constant. `ISA_VERSION` bumps `0x0200` → `0x0203` (matching the
  `pad_value`/`dts_factor` precedent of bumping per added field-set,
  not per byte).
- `module_cnn_accel.py`'s `registers_hook()`: add `INPUT_ADDR`/
  `OUTPUT_ADDR` register entries, add `STATUS.queued` bit (shrinking
  `reserved1` to 7 bits), add `ERR_QUEUE_FULL` to whatever enum/comment
  documents `ERR_CODE` values there.

## 8. Testing

- Existing 79-case regression must stay 85/85 (this is the two-line
  proof of backward compatibility: every existing config's `INPUT_ADDR`/
  `OUTPUT_ADDR` default to `0`, every existing descriptor's
  `reloc_input`/`reloc_output` default to `False`).
- New Python-only unit coverage (fast, no simulator): `encode_desc`/
  `decode_desc` round-trip for `reloc_input`/`reloc_output` and the
  redefined `reserved_w0`; `emit_program`'s new post-pass, on a small
  synthetic tiled+aliased case, asserting exactly the expected
  descriptors get tagged.
- New VUnit-level coverage: a dedicated testbench,
  `test/tb_cnn_accel_streaming.vhd`, since `tb_cnn_accel_top`'s
  one-job-per-config model has no way to express a queued pair.
  `relocated_and_queued_pair` starts job A at the compiled (unrelocated)
  addresses, queues job B behind it with `INPUT_ADDR`/`OUTPUT_ADDR` set
  to a relocated pair while job A is still `BUSY`, then checks job A's
  `DONE`+result (with `STATUS.BUSY` still set — auto-dispatch's whole
  point is no gap) and job B's auto-dispatched `DONE`+result at the
  relocated addresses. `queue_full_rejected` additionally attempts a
  third `START` while the queue is already full, expecting
  `ERR_QUEUE_FULL` without disturbing either the running job A or the
  already-queued job B, then lets both run to completion to prove
  neither was corrupted by the rejected write.

## 9. Explicitly out of scope for this pass

- More than one queued frame.
- Queueing a *different* compiled program per frame (`PROGRAM_BASE_ADDR`
  is not queueable).
- Any change to `IRQ_MASK`'s two existing causes, or a third IRQ cause
  for queue-full (reuses `ERROR`, per prior decision).
- `src1`/`wgt` relocation (see §5's disclosed limitation).
