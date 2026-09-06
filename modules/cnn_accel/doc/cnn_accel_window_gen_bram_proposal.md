# cnn_accel_window_gen — block-RAM inference recovery proposal (`vhdesign`, M7)

Scope: `cnn_accel_window_gen`'s row-bank storage only (`src/cnn_accel_window_gen.vhd`
lines 127-157 declarations, `assemble_window` process lines 398-510). No RTL
changes are made by this document; it is the contract for the follow-up
`vhfill` round. Read-only investigation; `control`/write-side logic,
`row_ready`'s fencepost test, and the `window_m2s_t`/`pe_array` design already
ratified in `cnn_accel_tiled_dataflow_proposal.md` are treated as fixed unless
a section below says otherwise.

Numbers use the actually-elaborated generics from `module_cnn_accel.py`:
`g_max_kernel_size = 3`, `g_max_row_tile_words = 512`, `g_tile_channels = 8`
(`c_lane_width = 64` b), target `xc7a100t` (63,400 LUTs, 126,800 FFs, 135
BRAM36, 240 DSP48E1). Re-derive if these change (§8 risk 4 covers a larger
`g_max_kernel_size`).

## 1. Root cause

`assemble_window` (`cnn_accel_window_gen.vhd:398-510`) is a single
combinational process. For every `(kr, kc)` in `0 .. g_max_kernel_size-1`
it reads `row_banks(bank_idx)(input_col * n_tiles_i + rd_tile_i)`
(`:480`) with `bank_idx`/`input_col` computed **in the same process, same
cycle** as the read. That is a combinational (asynchronous) read of a
large array — the one thing that unconditionally blocks Xilinx 7-series
block-RAM inference (`shared/ModernVHDL.md` "Memory: infer the intended
RAM type, and prove it").

Measured (CI toolchain, Yosys v0.68, `xc7`, from
`tsfpga_mcp_out/projects/cnn_accel_window_gen/output.txt`):

| Resource | Count | Note |
|---|---|---|
| Total LUTs | 23,757 | ~37% of `xc7a100t`'s 63,400 |
| FDRE (FF) | 199 | |
| RAM64M (distributed RAM) | 4,608 | |
| Block RAM (RAMB18/36E1) | 0 | |
| DSP48E1 | 8 | unrelated to this proposal — see note below |

**The 4,608 figure is not noise, it decomposes exactly**: one single-port
512-deep x 1-bit distributed memory needs `ceil(512/64) = 8` RAM64M
cells (RAM64M is inherently 64-deep); x 64 bits (`c_lane_width`) = 512
cells per single-read-port 512x64 memory; x 3 (`kc` = 0..2, three
*independent* random-address reads issued against the same `row_banks(bank_idx)`
signal every cycle, which Yosys cannot share onto one port and instead
fully duplicates) = 1,536; x 3 (`g_max_kernel_size` banks, `kr` = 0..2) =
**4,608**. In other words: **the current design already pays the full
cost of "one physical single-read-port memory per tap position"
(§4.1's Option 1) — it just pays it in distributed RAM instead of block
RAM**, because the reads are combinational. Registering those same 9
reads is close to a wire-for-wire swap of primitive, not a new
architecture.

Note on the 8 DSP48E1: these come from the runtime multiplies
`cur_col_q * n_tiles_q` / `input_col * n_tiles_i` (address arithmetic,
`:330-331`, `:480`), not from the row-bank reads. None of the options
below change this; 8 DSPs is 3.3% of `xc7a100t`'s 240 and out of scope
for this proposal.

## 2. Xilinx 7-series block-RAM inference rules relied on below

(`shared/ModernVHDL.md`, `UG473`/`UG953` — RAMB36E1/RAMB18E1)

1. **Registered read port.** Read data is available one clock after the
   read address (and, in the templates used here, a plain `if
   rising_edge(clk)` assignment) is presented — never in the same cycle
   as the address. Any combinational/asynchronous read forces
   distributed RAM.
2. **Exactly two physical ports per block RAM**, in one of two modes:
   - **Simple Dual-Port (SDP)**: one write-only port + one read-only
     port, independent clocks/addresses, up to **72 bits wide x 512
     deep** (36 Kb) for a `RAMB36E1`. This is the mode
     `cnn_accel_weight_buffer.vhd` and `hdl-modules`' `fifo.vhd`
     (`memory_block`, `fifo.vhd:409-441`) already use.
   - **True Dual-Port (TDP)**: two fully independent read/write ports,
     but each port maxes out at **36 bits wide** (32 data + 4 parity) —
     narrower than SDP's 72. A `RAMB36E1` in TDP mode is really its two
     18 Kb halves used independently.
   - A third concurrent access (a third reader, or a reader plus two
     writers) always exceeds the primitive and forces either
     replication (separate physical block RAMs) or a fallback to
     distributed RAM.
3. **Read-during-write behavior** is fixed per the port's configured
   `WRITE_MODE_*` (`WRITE_FIRST`/`READ_FIRST`/`NO_CHANGE`) — do not rely
   on behavior the configured mode doesn't guarantee. `row_ready`
   (`:451-457`) already ensures a tap is never read at an address still
   being written the same cycle; every option below preserves that
   invariant unchanged.
4. **No reset on memory contents.** `row_banks` already has no reset
   branch and no initial-value aggregate on its `signal` declaration —
   keep it that way; resetting a large array forces per-bit reset
   muxing that itself defeats inference. Un-written cells are never
   read (`row_ready`/`in_frame` gate every access), so no initial value
   is semantically required either.

## 3. Reuse search (`shared/ReusableRTL.md`, `ModernVHDL.md` reuse order)

Searched `hdl-modules/modules/{fifo,hard_fifo,ring_buffer,common}` (no
`corvidex-mcp` server was available in this session — local search only).

| Candidate | What it is | Fits? |
|---|---|---|
| `fifo/src/fifo.vhd` | Synchronous FIFO, AXI-Stream-like handshake, optional `enable_peek_mode` (re-read a packet from its start without popping) | **No.** Peek mode replays one packet from its head; it does not give random access to an arbitrary earlier `(row, col)` cell — exactly the FIFO/shift-register insufficiency the module's own header comment already rules out (`cnn_accel_window_gen.vhd:17-33`). |
| `fifo/src/asynchronous_fifo.vhd` | CDC variant of the above | No — same semantics, wrong clock-domain problem entirely. |
| `hard_fifo/src/hard_fifo.vhd` | Wrapper around the `FIFO36E2` hard macro | No — hard FIFO primitive, pop-once, no random access. |
| `ring_buffer/src/ring_buffer_write_simple.vhd` | Segment-address generator for an FPGA-writes/CPU-reads DRAM ring buffer | No — it doesn't hold data at all, it hands out addresses for an external buffer; wrong abstraction level. |
| `common/*` | Handshake/CDC/width-conversion helpers | No memory primitive here. |

**Nothing in the vendored library provides a random-access, multi-read,
replay-tolerant line buffer** — this is a reasonable gap, since that
requirement is fairly specific to sliding-window generators with runtime
padding/stride. However, the **RAM-inference template itself is already
proven and reused twice in this project**: `fifo.vhd`'s `memory_block`
block (`fifo.vhd:409-441` — one array signal, one process for the
registered read, one process/branch for the synchronous write, no
reset) is the same idiom `cnn_accel_weight_buffer.vhd` explicitly credits
(`cnn_accel_weight_buffer.vhd:102-105`, confirmed clean in synthesis:
64 RAMB18E1 + 8 RAMB36E1, 0 distributed RAM, ~181 raw LUT primitives
total for all its control logic —
`tsfpga_mcp_out/projects/cnn_accel_weight_buffer/cnn_accel_weight_buffer_utilization.txt`).

**Conclusion**: no existing module or thin wrapper fits; a hand-written
inference template is justified, but it should be the *same* template
already proven twice in this codebase (array + one write process + one
read process, replicated per physical copy), not a new invention.

## 4. Options evaluated

All options replace `assemble_window`'s single combinational process with
an address-decode stage (unchanged combinational logic, `row_ready`,
`bank_idx`, `input_col` — none of this touches BRAM inference) feeding
one or more registered-read row-bank copies, plus an output stage that
packs registered read data into `m_window_m2s.data` exactly as today.

### 4.1 Option 1 — bank replication (recommended)

Replace the single `row_banks` array (`g_max_kernel_size` banks) with
`g_max_kernel_size * g_max_kernel_size` physical copies: bank `kr`'s
data is stored once per possible `kc` slot (`0 .. g_max_kernel_size-1`),
so tap `(kr, kc)` gets its own dedicated single-read-port memory. Each
copy uses the SDP idiom from §3 (one write-only port, one registered
read-only port).

- **Write side**: one incoming `(col, tile)` beat broadcasts its address
  and data unchanged to all `g_max_kernel_size` copies of its target
  bank (`cur_row_q mod g_max_kernel_size`). This is pure fan-out to
  independent physical write ports — **no write-bandwidth cost**, no
  arbitration, no contention (this directly answers the "write
  bandwidth" question in the brief: it's an area cost, not a throughput
  one).
- **Read side**: all `g_max_kernel_size**2` taps read their own copy in
  the same cycle — no port sharing, no serialization needed.
- **BRAM36**: `g_max_kernel_size**2` = **9**, each `512 x 64` b (32 Kb,
  88.9% of one 36 Kb block in 512x72 SDP mode) → **9 BRAM36 = 6.7% of
  `xc7a100t`'s 135**.
- **LUT/FF**: engineering estimate, not measured — `cnn_accel_weight_buffer`
  needed ~181 raw LUT primitives / 59 FDRE for materially simpler
  (flat-row) addressing; window_gen's per-tap `kr`/`kc` guard and
  channel-unpack logic is more involved, so budget **~1,500-3,000 LUTs /
  ~700-1,000 FF** (the extra FFs are mostly the 9 x 64 b = 576 b of
  registered read-data, plus a pipeline-valid register) — still two
  orders of magnitude below today's 23,757/199.
- **Added latency**: **+1 to +2 cycles** per window (one BRAM read
  register, optionally one output-packing register) versus today's
  0-cycle combinational read. `pe_array`'s own steady-state budget for
  the reference 3x3/`Ct=8`/8x8-PE config is `num_groups + 3` = 12
  cycles/window (`cnn_accel_pe_array_proposal.md` §5/§6); +1-2 cycles is
  a **~8-17% cycle overhead**, likely largely hidden if window N+1's
  address decode is allowed to start the cycle after window N is
  consumed (it already does — `row_ready` is a pure function of
  registered state that updates the cycle after `consume`, unchanged by
  this option).
- **`window_m2s_t` contract**: **unchanged.** Still one complete
  `K_h*K_w*Ct`-tap beat per accept; only internal latency changes.
- **Runtime kernel/stride/padding**: no interaction. Copy selection
  (`kr`, `kc`) is a structural/elaboration-time index, exactly like
  today's `tap_idx` destination selection (`:496-503`); the *address*
  into a given copy is still `input_col * n_tiles + rd_tile`, computed
  exactly as today. The existing `kr < kh and kc < kw` guard
  (`:465`) still disables unused copies for smaller runtime kernels.
- **Scaling risk**: cost is `g_max_kernel_size**2` — fine at 3 (9
  BRAM36), still fine at 5 (25, 18.5%), starting to matter at 7 (49,
  36%). Flagged in §8.

### 4.2 Option 2 — true dual-port halving

Idea per the brief: use TDP mode's 2 read/write-capable ports to serve 2
of the `K_w` reads per bank from one physical copy, halving the
replication factor to `ceil(K_w/2)` copies/bank.

- **Width problem (decisive)**: TDP tops out at 36 b/port; this design's
  word is 64 b (`c_lane_width = 8 * g_tile_channels`). A 64 b-wide TDP
  "copy" needs 2 physical `RAMB36E1`s in parallel (2 x 36 b lanes) to
  reach 64 b — so each TDP copy costs **2 BRAM36**, not 1.
- **BRAM36**: `g_max_kernel_size * ceil(g_max_kernel_size/2)` copies x 2
  BRAM36/copy = `3 * 2 * 2` = **12 BRAM36** for K=3 — **worse than
  Option 1's 9**, and the gap widens for larger K (K=5: `5*3*2=30` vs.
  Option 1's 25). The width penalty always outweighs the port-count
  saving at this word width.
- **Port problem**: a TDP port pair gives 2 read/write-capable ports,
  not "2 reads + 1 write". To still support writes, one of the 2 slots
  must be time-shared between "serve a read" and "accept a write",
  needing a write/read arbiter and a possible 1-cycle read stall
  whenever a write and a full 2-read cycle collide — real added control
  complexity for a device that (per Option 1) isn't even BRAM-constrained.
- **Runtime kernel/stride/padding**: which of the 2 shared slots serves
  a given `(kr, kc)` tap now depends on runtime `kw` (odd/even column
  count changes the pairing), not a fixed elaboration-time index — a
  genuinely harder, runtime-dependent allocation problem, unlike Option
  1's static mapping.
- **`window_m2s_t` contract**: unchanged (same "all taps, one beat"
  shape), but at strictly worse BRAM cost and materially higher control
  complexity than Option 1.
- **Verdict: reject.** No scenario at this word width where TDP beats
  SDP replication.

### 4.3 Option 3 — serialize tap delivery over multiple cycles

Two distinct variants, deliberately separated because they have very
different interface impact:

**3a — serialize internally, keep the beat contract.** Keep only
`g_max_kernel_size` row banks (matches the ~3 BRAM36 originally budgeted
in `cnn_accel_tiled_dataflow_proposal.md` §1), each with **one**
registered read port. Walk `kc = 0 .. kw_q - 1` over up to
`g_max_kernel_size` cycles (a small counter, runtime-bounded exactly like
`out_width_q`'s existing style — not a variable-bound `for` loop), reading
all `K_h` banks in parallel each cycle (one read/bank/cycle, which the
single port supports) into a `g_max_kernel_size**2 * c_lane_width`-bit
(576 b) tap-assembly register. Once the register is full, present it as
one `m_window_m2s` beat exactly as today.

- **BRAM36**: `g_max_kernel_size` = **3** — the smallest of all options,
  matching the original design intent (`cnn_accel_tiled_dataflow_proposal.md`
  §1: *"K_h rows read in parallel ...; K_w columns per row are
  sequential through that one port"* — this is what the accepted
  proposal already assumed the RTL would do; the current
  `assemble_window` process does not actually do it).
- **`window_m2s_t` contract**: **unchanged.**
- **Added latency — the real cost**: `window_gen` is currently
  single-buffered (the next window's address decode only starts the
  cycle after `consume`, §4.1). Serializing the read over up to
  `g_max_kernel_size` cycles adds **+(kw+1) to +(kw+2) cycles per
  window** (worst case K=3: **+4 to +5 cycles**) *serially* in front of
  `pe_array`'s own 12-cycle budget, i.e. ~16-17 cycles/window instead of
  ~12 — a **~25-30% steady-state throughput regression** (the reference
  network's 17.4 FPS, `cnn_accel_tiled_dataflow_proposal.md` §6, would
  drop to roughly 12.5-13 FPS). This is the quantified version of that
  proposal's own §8 risk 8 ("assumed never the bottleneck... re-check if
  `g_pe_cols` grows large enough that `groups_per_tile` shrinks below
  `K_w`") — it was never re-checked, and the assumption does not hold
  once the actual per-window budget (12 cycles) is this close to `K_w`
  (3-5 cycles of added latency).
- **Runtime kernel/stride/padding**: fine — the `kc` counter's bound is
  a registered `kw_q` compare, same discipline as existing counters; no
  correctness interaction.
- **Mitigation** (double-buffer the assembly one window ahead of
  consumption) would claw back most of the throughput loss but is a
  materially bigger diff than Option 1's registered-read swap — it
  needs speculative computation of the *next* `out_row_q`/`out_col_q`
  before the current window is actually consumed, which does not exist
  today.

**3b — serialize on the interface itself** (expose the column
sub-beats to `pe_array` instead of buffering them inside `window_gen`).
This is the variant the brief's phrasing warns about directly:
`window_m2s_t` would need new sideband state (e.g. a column/group
sub-beat counter or a narrower per-beat `data`), and `pe_array`'s `idle`
state (`cnn_accel_pe_array_proposal.md` §5) — which today latches the
**entire** window array on a single accept cycle and then sequences
`num_groups` internally, decoupled from `Kw`/`Ct` — would instead have
to accumulate across several accept cycles per tile, entangling
`window_gen`'s memory-access schedule with `pe_array`'s own MAC-group
schedule. That coupling does not exist today and is a materially larger,
cross-module redesign (both modules' handshake/backpressure contracts
per `shared/Axi4.md` would need re-verification). **Reject for this
milestone** — the BRAM problem does not require it (Option 1 is
cheaper and non-invasive); keep as a future direction only if a much
larger PE array ever makes `pe_array`'s per-window cycle budget shrink
below the memory's minimum service latency.

### 4.4 Option 4 — write-time pre-aligned wide word (considered, rejected)

An idea worth recording as explicitly evaluated and rejected: keep only
`g_max_kernel_size` banks, but widen each word to
`g_max_kernel_size * c_lane_width` bits and, at **write** time, also
duplicate each incoming column's data into the `g_max_kernel_size - 1`
neighboring word slots, so that a *single* aligned read at the window's
left column already returns all `K_w` taps of that row concatenated —
trading read-side replication for write-side replication, still with
only `g_max_kernel_size` physical read ports.

- **BRAM36 cost is not actually lower**: a `g_max_kernel_size *
  c_lane_width`-bit-wide word (192 b at K=3) needs
  `ceil(192/72) = 3` `RAMB36E1`s width-cascaded per bank (SDP tops out
  at 72 b/port) — `3 banks * 3 = 9 BRAM36`, **identical to Option 1**.
- **It is strictly worse on correctness risk**: the "pre-aligned wide
  word" trick only stays simple for `stride = 1`. With runtime
  `stride_w /= 1` the tap a given output column needs
  (`ocol*stride_w + kc - pad_left`) does not land on a fixed offset
  from the write-time alignment point, so *every* possible offset would
  still need to be duplicated at write time (the same fan-out as Option
  1, just organized as "wide reads from few banks" instead of "narrow
  reads from many banks") — no actual saving, and it reintroduces
  exactly the kind of clever, alignment-dependent indexing the module's
  own header comment already warns produces subtle replay bugs under
  runtime padding/stride (`cnn_accel_window_gen.vhd:17-33`).
- **Verdict: reject.** Isomorphic BRAM cost to Option 1, meaningfully
  higher bug risk. No scenario found where it wins.

## 5. Comparison summary

| | Today (broken) | **1. Bank replication** | 2. TDP halving | 3a. Serialize (internal) | 3b. Serialize (interface) | 4. Pre-aligned wide word |
|---|---|---|---|---|---|---|
| BRAM36 | 0 | **9** | 12 | 3 | 3 | 9 |
| LUT (est.) | 23,757 (measured) | ~1,500-3,000 | similar to (1) + arbiter | ~1,000-2,000 | fewer in `window_gen`, more in `pe_array` | ~2,000-3,500 |
| FF (est.) | 199 (measured) | ~700-1,000 | similar to (1) | ~700-900 (576 b assembly reg) | redistributed across both modules | ~700-900 |
| Added latency/window | 0 (combinational) | +1 to +2 cyc | +1 to +2 cyc (+ write stalls) | **+4 to +5 cyc** | data-dependent, no single number | +1 to +2 cyc |
| Steady-state throughput vs. today | baseline (but 0 BRAM) | ~92-98% | ~85-95% (+ stalls) | **~70-75%** | unresolved without a redesign | ~92-98% |
| `window_m2s_t` change? | — | No | No | No | **Yes** | No |
| `pe_array` change? | — | No | No | No | **Yes, materially** | No |
| Runtime kernel/stride/pad interaction | (broken but works) | none (structural mapping) | new runtime allocation problem | none | new coupling to `pe_array` groups | correctness risk under stride≠1 |

## 6. Recommendation

> ### RATIFIED 2026-09: **Option 3a**, not the recommended Option 1.
>
> The architect ratified **Option 3a — serialize internally, keep the
> beat contract** (§4.3), overriding the recommendation below, and
> **retained Option 1 (bank replication, §4.1) as the designated
> speed-optimization path** to be taken when throughput, rather than
> area, becomes the binding constraint.
>
> Rationale for the override: area is the binding constraint now. Option
> 3a costs **3 BRAM36** against Option 1's 9 (2.2% vs 6.7% of an
> `xc7a100t`) and has the lowest LUT estimate of any option, and it is
> what `cnn_accel_tiled_dataflow_proposal.md` §1 already assumed the RTL
> did. The accepted price is the **~25-30% steady-state throughput
> regression** quantified in §4.3 (reference network ~17.4 FPS ->
> ~12.5-13 FPS). That price is explicitly accepted, not overlooked.
>
> Because Option 1 is now a planned future step rather than a rejected
> alternative, §4.1 and the §5 comparison table must be kept accurate;
> do not delete them once 3a is built. The upgrade path is: swap the
> `g_max_kernel_size` single-port banks for `g_max_kernel_size**2`
> replicated copies and drop the `kc` walk. Option 3a's own
> double-buffering mitigation (§4.3, §8 risk 6) is the intermediate step
> if partial throughput recovery is wanted without paying the 9 BRAM36.
>
> **Decision points as ratified.** DP2, DP3 and DP4 were all accepted,
> but DP2 and DP3 were written in Option 1's terms; under Option 3a they
> read:
>
> - **DP1 (ratified, overridden):** Option 3a. Option 1 retained as the
>   speed path. Option 2, 3b and 4 remain rejected.
> - **DP2 (ratified, restated):** BRAM scaling is
>   `g_max_kernel_size` (**3**, linear), not `g_max_kernel_size**2`.
>   This is strictly better than the accepted bound and removes §8
>   risk 1 as a practical concern: even K=7 costs 7 BRAM36. The
>   `g_max_kernel_size**2` re-estimate becomes required again only if
>   the design later moves to Option 1.
> - **DP3 (ratified, restated):** the accepted added latency is
>   **+4 to +5 cycles/window** (§4.3), not +1 to +2, with the resulting
>   ~70-75% relative throughput. Per §8 risk 2 these are estimates: they
>   must be re-measured against real `vunit-mcp` / `tsfpga-mcp` results
>   once the RTL exists, and the measured numbers written back into §5.
> - **DP4 (ratified, unchanged):** reuse the `fifo.vhd` `memory_block`
>   idiom already adopted by `cnn_accel_weight_buffer.vhd` as the
>   per-bank template. Applies identically under Option 3a — only the
>   number of instances differs.
>
> The §7 implementation sketch is written for Option 1 and is therefore
> **not** directly applicable; see §7.1 for the Option 3a scope note.
> In particular, §7 item 5's "fence expected BRAM36 (~9)" becomes
> **exactly 3** under 3a.

**Recommend Option 1 (bank replication, §4.1).** It reuses a template
already proven twice in this codebase (§3), costs 9 BRAM36 (6.7% of
`xc7a100t`), adds only 1-2 cycles of latency, requires **zero** change
to `window_m2s_t` or to the not-yet-built `pe_array`, and has no
interaction with runtime kernel size/stride/padding beyond what already
exists. It is, in effect, "give the distributed RAM the project is
already building (§1) a registered port and let the BRAM inference rules
in §2 do their job" rather than a new architecture.

### Decision points for the architect to ratify

- **DP1 — accept Option 1 over Options 2-4.** Rationale: strictly better
  BRAM cost than Option 2 and Option 4 at this word width, strictly
  better latency/throughput than Option 3, no interface churn. Rejecting
  this needs a reason stronger than "it was the original ~3-BRAM36
  budget" (§4.3 already shows that budget cost real throughput that was
  never actually re-checked).
- **DP2 — accept the ~9 BRAM36 / `g_max_kernel_size**2` scaling.**
  Fine through K=5 (18.5% of `xc7a100t`); if a future network needs
  `g_max_kernel_size > 5`, re-run this estimate before raising the
  generic (§8 risk 1).
- **DP3 — accept +1 to +2 cycles/window added latency**, unmeasured
  until real RTL exists; if `pe_array`'s eventual measured per-window
  cycle count is much lower than the reference config's 12 (e.g. a
  future wider PE array), re-run the throughput comparison in §5.
- **DP4 — reuse `fifo.vhd`'s `memory_block` idiom** (already adopted by
  `cnn_accel_weight_buffer.vhd`) as the per-copy template rather than a
  novel one, per `shared/ReusableRTL.md`'s reuse-before-authoring policy
  (§3).

## 7. Implementation sketch (for the `vhfill` round, not built here)

Not RTL — a scope note for whoever fills this in:

1. Replace `row_bank_arr_t`/`row_banks` (`:153-157`) with
   `g_max_kernel_size**2` instances of the `weight_mem`/`memory_block`
   idiom (`cnn_accel_weight_buffer.vhd:108-116`, `:245-251`) — one array
   signal, one write process (broadcast to all `g_max_kernel_size`
   copies of the target bank), one read process per copy (`signal <=
   array(to_integer(addr));` on `rising_edge(clk)`, no reset).
2. Split `assemble_window` (`:398-510`) into: (a) the existing
   combinational address-decode logic (`row_ready`, `bank_idx`,
   `input_col`/`input_row`, `in_frame` — unchanged, this was never the
   problem), driving each copy's registered read address; (b) a new
   registered stage that reads the 9 copies and packs `data_i` from
   registered read outputs instead of `row_banks` directly.
3. `window_valid` must now trail the read-address computation by the
   pipeline's added latency (§4.1) — becomes a small shift
   register/valid-delay alongside the read pipeline, not a same-cycle
   function of `row_ready`.
4. Testbench impact: `tb_cnn_accel_window_gen.vhd`'s golden model already
   only checks data/handshake correctness, not cycle-exact latency
   (confirm before `vhfill`); if it does assume 0-cycle combinational
   readout, its latency assumption needs updating alongside the RTL.
5. Add the resource check this class of bug needs
   (`shared/ModernVHDL.md` "check the utilization report" /
   `tsfpga`'s `BuildResultCheckers`): fence expected BRAM36 (~9,
   non-zero) and a LUT ceiling well below today's 23,757 in
   `module_cnn_accel.py`'s build project for this module, so a future
   regression back to combinational reads fails CI instead of being
   found at place-and-route.

### 7.1 Implementation sketch — Option 3a (the ratified one)

Supersedes §7 for the `vhfill` round. Also a scope note, not RTL:

1. Keep **`g_max_kernel_size` row banks** (not `**2`). Rebuild each as
   the `weight_mem`/`memory_block` idiom
   (`cnn_accel_weight_buffer.vhd:108-116`, `:245-251`) with **one
   registered read port** — `signal <= array(to_integer(addr));` on
   `rising_edge(clk)`, no reset. The registered read is the entire point:
   it is what §2's inference rules need and what today's combinational
   read denies them.
2. Add a **`kc` column counter**, `0 .. kw_q - 1`, bounded by a
   *registered* `kw_q` compare — the same discipline as the existing
   `out_width_q` counters, and explicitly **not** a variable-bound `for`
   loop (that is the GHDL synth crash already hit on this project).
   Each cycle it issues one read address to **all `K_h` banks in
   parallel**; one read per bank per cycle is exactly what the single
   port supports.
3. Accumulate the results into a
   `g_max_kernel_size**2 * c_lane_width`-bit (576 b) **tap-assembly
   register**, then present the full window as one `m_window_m2s` beat.
   `window_m2s_t` is unchanged, and `pe_array` is untouched — preserving
   that is the whole reason 3a was chosen over 3b.
4. `window_valid` becomes a **valid-delay/shift register** trailing the
   read pipeline plus the `kc` walk, not a same-cycle function of
   `row_ready` (§7 item 3, §8 risk 3 — confirm nothing else depends on
   the same-cycle behavior).
5. Do **not** implement the double-buffering mitigation (§4.3) in this
   milestone. It is the designated intermediate step if throughput needs
   partial recovery later; building it now would obscure whether plain
   3a's measured numbers match the §5 estimates.
6. Fence the result in `module_cnn_accel.py`: BRAM36 **exactly 3**
   (non-zero is the point — 0 means inference silently failed again) and
   a LUT ceiling far below today's 23,950. Per this project's standing
   rule, the limits go in **from a CI run**, never from a local Yosys.
   `cnn_accel_conv_core`'s limits must be re-baselined in the same pass,
   since window_gen contributes 23,950 of its 35,056 LUTs.

## 8. Open questions / risks for the architect

1. **`g_max_kernel_size**2` scaling** (DP2): fine at 3 (9 BRAM36) and 5
   (25), starts to matter at 7 (49, 36% of `xc7a100t`). No action needed
   now; re-run this estimate if a future network needs a larger max
   kernel.
2. **Latency/throughput numbers in §5/§6 are engineering estimates**,
   not measured — no RTL exists yet for the registered-read version.
   Re-measure with real `vunit-mcp`/`tsfpga-mcp` results once `vhfill`
   produces it, per this project's "never claim ... success without a
   real tool result" policy.
3. **`window_valid`'s move from same-cycle to pipelined** (§7 item 3) is
   the one place this proposal does touch behavior other modules may
   assume — confirm no other module/testbench relies on `window_valid`
   responding to `row_ready` in the same cycle.
4. **No `corvidex-mcp` server was available this session** — the
   organization-wide search for existing RAM/line-buffer implementations
   was limited to the locally vendored `hdl-modules` tree (§3). Re-run
   the wider-org search when that server is reachable, in case a better
   fit exists outside this repository.
5. **This proposal does not touch the 8 DSP48E1** (§1 note) — separate,
   much smaller issue (3.3% of `xc7a100t`), out of scope here.
6. **Option 3a's double-buffering mitigation** (§4.3) was identified but
   not designed — if a future PE array shrinks the per-window cycle
   budget enough that even Option 1's +1-2 cycles matters, that
   mitigation (or Option 1 outright) is the fallback; re-open this
   document rather than reaching for Option 3b's interface change first.
