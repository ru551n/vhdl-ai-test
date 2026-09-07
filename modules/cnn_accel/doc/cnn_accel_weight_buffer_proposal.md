# cnn_accel_weight_buffer — vhdesign proposal

Input: `modules/cnn_accel/doc/cnn_accel_weight_buffer_req.md`.
Related: `doc/cnn_accel_arch.md` ("Type policy", "Interface record policy",
"Reset policy", "Portability target" — BRAM-inference intent).

## 1. Requirements summary

Double-buffered (ping-pong), two-region (weight + bias) on-chip cache for
one layer's weights/bias. One bank is filled over an internal AXI4-Stream
(`s_stream_m2s`/`s2m`, `axi_stream_pkg` record type per the arch doc's
interface record policy) from the weight `cnn_accel_axi_read_dma` instance
while the other bank is read out through two simple synchronous
random-access ports toward `cnn_accel_pe_array` (weight) and
`cnn_accel_bias_requant` (bias). `fill_bank_sel`/`read_bank_sel` pick which
bank each side uses; `fill_is_bias` routes fill traffic to the weight or
bias region of the currently-selected fill bank. Synchronous active-high
`reset` (`= reset_internal`) must return both banks' fill progress to empty
so an aborted layer's partially-filled bank can never look complete.

## 2. Interface (as given by the requirement, widths resolved below)

| Port | Dir | Type |
|---|---|---|
| `clk` | in | `std_ulogic` |
| `reset` | in | `std_ulogic := '0'` |
| `s_stream_m2s` | in | `axi_stream_pkg.axi_stream_m2s_t` |
| `s_stream_s2m` | out | `axi_stream_pkg.axi_stream_s2m_t` |
| `fill_bank_sel` | in | `std_ulogic` |
| `fill_is_bias` | in | `std_ulogic` |
| `read_bank_sel` | in | `std_ulogic` |
| `weight_rd_addr` | in | `std_ulogic_vector(weight_addr_width - 1 downto 0)` |
| `weight_rd_data` | out | `std_ulogic_vector(8*pe_rows*pe_cols - 1 downto 0)` |
| `bias_rd_addr` | in | `std_ulogic_vector(bias_addr_width - 1 downto 0)` |
| `bias_rd_data` | out | `std_ulogic_vector(accum_width*pe_rows - 1 downto 0)` |

Generics: `weight_buffer_depth : positive` (rows per bank, per region —
see §3.2 for why one depth serves both regions), `pe_rows`, `pe_cols :
positive`, and one **added** generic, `accum_width : positive := 32`
(see §3.1). Per this IP's own established convention (`cnn_accel_pkg.vhd`'s
`c_`-prefixed constants, every `*_req.md`'s `g_`-prefixed generics table),
generics keep the `g_` prefix and internal constants the `c_` prefix in the
RTL, superseding `shared/TsfpgaCodingConventions.md`'s unprefixed-upstream
default for this IP specifically (an already-established, documented
in-IP deviation, not a new one introduced here).

Bank encoding: `fill_bank_sel`/`read_bank_sel` = `'0'` selects bank A,
`'1'` selects bank B.

## 3. Design decisions (resolving requirement ambiguities)

### 3.1 `bias_rd_data` width — requirement typo fixed

The requirement's port table gives `bias_rd_data`'s width as
`std_ulogic_vector(8*g_accum_width-1 downto 0)`, described in the same row
as "sized for `g_pe_rows` int32 lanes". Taken literally that is
`8*32=256` bits per lane group with only one implied lane — inconsistent
with "one int32 bias per output channel lane" (which needs
`g_accum_width` bits per lane, `g_pe_rows` lanes, i.e.
`g_accum_width*g_pe_rows` bits total) and, worse, `g_accum_width` is not
even listed in this module's own Generics table. This is treated as a
documentation defect in the (non-hand-owned) port table, not part of the
preserved Functional Description. Resolution: add generic
`g_accum_width : positive := 32` (matching `doc/cnn_accel_arch.md`'s
top-level generic of the same name/default — "accumulator width"/int32),
and size `bias_rd_data` as `std_ulogic_vector(g_accum_width*g_pe_rows - 1
downto 0)`.

### 3.2 One depth generic serves both regions

The requirement gives only `g_weight_buffer_depth` (no separate bias
depth). Both the weight region and the bias region of a bank are sized to
`g_weight_buffer_depth` rows; a real layer only ever fills
`out_channels/g_pe_rows`-ish bias rows (far fewer than
`g_weight_buffer_depth`), so most bias rows in a bank go unused in
practice — a documented BRAM-cost simplification given the requirement
supplies no second depth generic. A future revision could add
`g_bias_buffer_depth` if resource pressure demands it.

### 3.3 "Tile" addressing, not raw byte addressing, for both read ports

The Generics table's own words — "shape the read port so
`cnn_accel_pe_array` can pull one tile per cycle" — plus the fact that
`axi_stream_pkg.axi_stream_m2s_t.data` is fixed at 128 bits
(`axi_stream_data_sz`, hdl-modules `axi_stream_pkg.vhd`) rule out "one
full row per stream beat": a weight row (`8*g_pe_rows*g_pe_cols` bits,
512 bits at the IP's default 8x8) or a bias row (`g_accum_width*g_pe_rows`
bits, 256 bits at the default) cannot fit in one beat. Resolution: each
bank/region is organized as `g_weight_buffer_depth` **rows** (one row =
one full read-out tile: `g_pe_rows*g_pe_cols` weight bytes, or `g_pe_rows`
bias lanes). The fill stream writes **one lane per accepted beat** (one
weight byte from `s_stream_m2s.data(7 downto 0)`, or one bias lane from
`s_stream_m2s.data(g_accum_width-1 downto 0)`, selected by
`fill_is_bias`), auto-advancing an internal lane index within the current
row; the row (tile) pointer only advances once a row's lanes are all
written. `weight_rd_addr`/`bias_rd_addr` address rows (widths
`num_bits_needed(g_weight_buffer_depth-1)`, reusing `math.math_pkg` per
`shared/ReusableRTL.md` rather than hand-rolling a log2 helper).

This directly resolves the requirement's literal wording too: "the
selected bank's write pointer reaches `g_weight_buffer_depth`" is exactly
the row/tile pointer described here, compared against `g_weight_buffer_depth`
rows (not against a byte count).

### 3.4 One write pointer *pair* per bank, not one pointer shared across regions

Re-reading the Functional Description ("a write pointer... resets to 0
whenever `cnn_accel_layer_ctrl` starts a new fill for that bank") as a
single literal pointer shared between weight and bias regions breaks
correct bias addressing (a bias fill starting mid-way through the weight
pointer's count would never start its own region at row 0). The
requirement gives no explicit "start new fill" pulse port, so "starts a
new fill for that bank" is implemented as an edge on `fill_bank_sel`
newly selecting that bank (a session-start event), at which point **both**
that bank's weight-row-pointer/lane-index and bias-row-pointer/lane-index
reset to 0 together. Within one such fill session, the weight sub-fill
(`fill_is_bias='0'`) and bias sub-fill (`fill_is_bias='1'`) each then
advance their own independent row/lane counters — exactly reproducing the
literal spec text (pointer(s) zero "whenever a new fill starts for that
bank") while giving each region correct addressing from row 0 regardless
of fill order (weight-then-bias, interleaved, or bias-only).

### 3.5 Fill-stream `last`/`user` fields

Not consumed: the requirement's Functional Description does not tie any
weight-buffer behavior to `TLAST`/`TUSER` framing (the weight `dma_done`
pulse living in `cnn_accel_axi_read_dma`/`cnn_accel_layer_ctrl` is the
completion signal per `doc/cnn_accel_arch.md`'s DMA table). Left
unconnected/ignored in this revision; noted for `vhdoc`.

## 4. Clock/reset

Single clock `clk`, synchronous active-high `reset` (`reset_internal`,
per `doc/cnn_accel_arch.md` "Reset policy"). Reset clears, for both banks:
weight row pointer, weight lane index, bias row pointer, bias lane index
(back to 0/empty) and the `fill_bank_sel` edge-tracking register. Memory
contents are **not** cleared by reset (BRAM-inference intent; clearing
would require a depth-deep clear loop, defeats single-cycle abort). This
is sufficient for the requirement's abort guarantee: after reset, a new
fill of the same bank always starts writing at row 0 again, so a
previously-aborted partial fill can never silently "complete" a stale
row range — the pointer state that would have made it look complete is
gone. Read-data output registers are not reset (no completeness contract
of their own; "value irrelevant until valid" per `vhfill`'s reset
minimization gate) and keep only declaration initial values.

## 5. Architecture and dataflow

```
                         +-------------------- bank A/B (x2) -----------------+
s_stream_m2s.data(7:0) ->|  weight_mem: array(0 to depth-1) of                |
  (fill_is_bias='0')     |    std_ulogic_vector(8*pe_rows*pe_cols-1 downto 0) |-> weight_rd_data
                         |  written 1 byte/beat into the current row's lane   |   (registered,
s_stream_m2s.data        |    [weight_lane_idx], row advances every           |    1 cyc latency)
  (accum_width-1:0)   -->|    pe_rows*pe_cols beats                          |
  (fill_is_bias='1')     |  bias_mem: array(0 to depth-1) of                  |
                         |    std_ulogic_vector(accum_width*pe_rows-1 downto 0)|-> bias_rd_data
                         |  written 1 lane/beat, row advances every pe_rows   |   (registered,
                         |    beats                                          |    1 cyc latency)
                         +-----------------------------------------------------+
```

Two identical bank memories (index 0/1), each holding both regions.
`fill_bank_sel`/`fill_is_bias` pick which memory + region a fill beat
targets; `read_bank_sel` picks which memory the two read ports serve
(independent from the fill side — the whole point of ping-pong).

## 6. BRAM-inference idiom

Followed the project's existing inferred simple-dual-port idiom
(`hdl-modules/modules/fifo/src/fifo.vhd`, `memory_block`/`memory` process:
`type mem_t is array (natural range <>) of word_t; signal mem : mem_t(0 to
depth-1); process begin wait until rising_edge(clk); read_data <=
mem(read_addr); if write then mem(write_addr) <= write_data; end if; end
process;`) rather than a ready-made hdl-modules primitive: no hdl-modules
module combines two banks, two independently-addressed sub-regions with
different element widths, and a byte/lane-serial-to-wide-row write path,
so a hand-written inferred dual-port RAM array (two arrays per bank, one
process for the byte/lane-serial write side, one process for the
registered wide read side, matching `fifo.vhd`'s split) is the expected
outcome here, per the task's own allowance.

## 7. Numeric types

Row/lane counters: `unsigned`. Row pointer width:
`num_bits_needed(g_weight_buffer_depth)` bits (must represent the value
`g_weight_buffer_depth` itself, the "reached depth" backpressure
boundary, without wraparound — one bit wider than the read-address port's
`num_bits_needed(g_weight_buffer_depth-1)`). Lane index width:
`num_bits_needed(lanes-1)`. All from `math.math_pkg.num_bits_needed`
(reused, not reimplemented, per `shared/ReusableRTL.md`).

## 8. Latency/throughput

Fill: 1 beat accepted per cycle when ready (no internal stall beyond the
end-of-bank backpressure). Read: fixed 1-cycle registered latency on both
`weight_rd_addr`/`data` and `bias_rd_addr`/`data`, matching
`cnn_accel_pe_array`'s expected weight-fetch latency (per the
requirement's Functional Description).

## 9. Corner cases covered by the verification plan (§10)

- Ping-pong: fill bank A while reading bank B and vice versa, concurrently.
- Fill pointer auto-increment across lanes/rows; reset-to-0 on a new fill
  session (`fill_bank_sel` edge onto a bank).
- Weight vs. bias region routing via `fill_is_bias` within one session.
- One-cycle read latency, both read ports.
- Backpressure: `ready` deasserts once the selected bank/region's row
  pointer reaches `g_weight_buffer_depth`; verified for both regions.
- `reset` pulsed mid-fill: pointer must not leave a partially-filled bank
  addressable as if a subsequent fill had completed cleanly (see §4).

## 10. Verification plan

VUnit-5 testbench `tb_cnn_accel_weight_buffer` (`architecture tb`),
small generics (`g_weight_buffer_depth=8`, `g_pe_rows=2`, `g_pe_cols=2`,
`g_accum_width=16`) to keep row/lane counts directed and readable. No
built-in VUnit AXI4-Stream VC is used: the record port type
(`axi_stream_m2s_t`/`s2m_t`) is driven directly by small hand-written
non-blocking helper procedures (push one fill beat, wait for
`ready`/`valid`) — using the VUnit VC would require unpacking to flat
`tvalid`/`tready`/`tdata` signals and back for no behavioral benefit here
(this module's fill port never bursts/reorders; it is a single
point-to-point stream with simple backpressure). Read ports are exercised
directly (no BFM needed, they are plain synchronous ports). Test cases:
`test_ping_pong_fill_a_read_b`, `test_ping_pong_fill_b_read_a`,
`test_fill_pointer_autoincrement_and_reset_on_new_fill`,
`test_bias_vs_weight_region_routing`, `test_read_latency_one_cycle`,
`test_backpressure_weight_region_full`, `test_backpressure_bias_region_full`,
`test_reset_mid_fill_does_not_leak_partial_bank`.

## 11. 2026-09 rework: single-buffering, decoupled bias depth, prefetch FIFO

Supersedes the double-buffering (ping-pong) decisions above (§3.2, §3.4,
§4, §5, §9, §10) for the *current* RTL/testbench — kept historical above
rather than rewritten, since they document why ping-pong was chosen in
the first place; this section documents why it was later dropped and
what replaced it. `doc/cnn_accel_weight_buffer.md` reflects only the
current (post-rework) behavior.

**Motivation.** The weight/bias fill path was never meant to be
continuously fed from an always-resident DDR4 image: `cnn_accel_layer_ctrl`
streams one output-channel pass's weight+bias set in per pass, from a
`cnn_accel_axi_read_dma` instance, ahead of that pass's compute. Once that
is the model, DDR4/AXI bandwidth is not the constraint — a modern DDR4
interface can stream a whole pass's weights in a small fraction of that
pass's compute time — so hiding the *whole* fetch behind compute (the
ping-pong bank's reason to exist) buys little, while paying for it in
on-chip block RAM twice over (two full banks) is a real, fixed cost on a
resource-constrained Artix-7. Single-buffering plus a *shallow* prefetch
FIFO (`g_fill_fifo_depth`, default 32) gets the same practical benefit —
absorbing DMA burst/request latency so the fill stream doesn't have to be
cycle-accurate with compute — without paying for a second copy of the
(much larger) weight region.

**Depth sizing (`g_weight_buffer_depth`, replaces the old arbitrary
512).** A layer's packed weight image is `OT*T*kernel_h*kernel_w` rows
(`pack_weights_for_hw()`'s own contract, §3.2/`doc/cnn_accel_weight_buffer.md`).
Across the target network's backbone, the worst case is layer 9:
`kernel_h=kernel_w=3`, `in_channels=256`, `g_tile_channels=8` ⇒
`T=ceil(256/8)=32` tiles, giving `3*3*32 = 288` rows — independent of
`OT` (the `g_weight_buffer_depth` contract only needs to hold *one*
output-channel-tile pass's rows at a time, since a new `fill_start`
begins the next pass's fill). `_WEIGHT_BUFFER_DEPTH` in
`module_cnn_accel.py` is set to exactly 288, not padded for headroom: with
weights now streamed per pass rather than resident for the whole run,
there is no "might need more later" case to pad against — a future layer
needing more rows would need a value bump here, an explicit, reviewable
change, not silent headroom.

**Depth sizing (`g_bias_buffer_depth`, new, independent of
`g_weight_buffer_depth`).** The bias region only ever needs
`ceil(out_channels/g_pe_rows)` rows — for `g_pe_rows=8` and the target
network's largest `out_channels` (64), that is 8 rows, orders of
magnitude shallower than the weight region. Under the old shared-depth
scheme (§3.2) the bias region was sized identically to the weight region
(512, later 288, rows) and sat at ~99.9% dead — Yosys still had to infer
a full block RAM for it. Decoupling the two generics lets
`g_bias_buffer_depth` default to 8, small enough that Yosys can map it to
LUTRAM/registers instead of a block RAM (`doc/cnn_accel_weight_buffer.md`'s
Generics table).

**Whole-row write (row-assembly register).** The pre-rework fill path
wrote one lane per accepted beat directly into the region memory with a
per-lane decoded write enable — semantically a wide memory word, but
structurally forcing Yosys's `memory_collect` to treat every lane as an
independently-writable sub-memory, since no evidence in the netlist ties
the per-lane writes back together into one wide write. That fragmented
the weight region into one RAMB18 per lane (64 lanes ⇒ 64 RAMB18, plus a
mostly-dead 8-RAMB36 bias region ⇒ measured 72 block RAMs pre-rework).
The rework instead accumulates incoming lanes into a `c_weight_row_width`/
`c_bias_row_width`-bit row-assembly register
(`weight_row_assemble_q`/`bias_row_assemble_q`,
`cnn_accel_weight_buffer.vhd:174-180`) and issues exactly ONE wide write
to the region memory per completed row (`:287-333`) — the same
`memory_block` idiom `cnn_accel_window_gen.vhd`'s M7 rework later reused
(`doc/cnn_accel_window_gen_bram_proposal.md`).

**Measured 2026-09 (weight_buffer-only netlist build, local dev Yosys,
`synth_xilinx -family xc7`, default generics incl. `g_fill_fifo_depth=32`):**
881 LUTs, 1093 FFs, 15 block RAMs (15 RAMB18, 0 RAMB36), 0 DSP. Block RAMs
dropped 72 → 15 as intended. LUTs/FFs went *up* from the pre-rework
baseline (175 LUTs, 59 FFs) rather than down: the two whole-row
assembly registers (512 + 256 = 768 bits between them) plus the
depth-32 prefetch FIFO's distributed-RAM storage (too shallow for Yosys
to prefer a block RAM over LUT-mapped `RAM32M` primitives) now cost
register/LUT area the old per-lane design didn't pay. This is an accepted
trade — BRAM was the scarce/fragmented resource this rework targeted;
LUTs/FFs are comparatively abundant on the target XC7A100T — not a
regression to chase down further in this round. See
`module_cnn_accel.py`'s `cnn_accel_weight_buffer` build project comment
for the checker thresholds this measurement was re-baselined against.

## 12. Proposed replacement for the hand-owned Functional Description

**Why the current text is wrong.** M7b (commit `2d78d59`, "perf(cnn_accel):
stream weights into a single-buffered weight_buffer") deleted the A/B
ping-pong design entirely: `fill_bank_sel`/`read_bank_sel` no longer exist
anywhere in `cnn_accel_weight_buffer.vhd`, replaced by a `fill_start` pulse,
a single pair of region memories, and a row-assembly-register + prefetch-
FIFO fill path (§11 above; `doc/cnn_accel_weight_buffer.md`,
`src/cnn_accel_weight_buffer.vhd`). `cnn_accel_weight_buffer_req.md`'s
hand-owned `## Functional Description` still narrates the old bank-select
behavior verbatim ("while `fill_bank_sel` selects bank X... the pointer
resets to 0 whenever `cnn_accel_layer_ctrl` starts a new fill for that
bank... while `read_bank_sel` selects bank Y") — every noun in it (`bank
X`, `bank Y`, `fill_bank_sel`, `read_bank_sel`) refers to ports and
concepts that were removed by M7b and do not exist in the RTL, generics
table, ports table, or `doc/cnn_accel_weight_buffer.md` it sits alongside.
This was flagged rather than fixed at the time (`flow_status.md`'s M7b
"Open" note, and this proposal's own §11 preamble) because the section is
user-owned. The replacement text below describes only what the current
single-buffer RTL actually does, grounded in
`src/cnn_accel_weight_buffer.vhd` and `doc/cnn_accel_weight_buffer.md`
(both already correct).

**Paste target:** `modules/cnn_accel/doc/cnn_accel_weight_buffer_req.md`,
replacing its entire `## Functional Description` section (the text after
the `<!-- functional-spec: hand-owned below this line -->` marker) with
the block below.

```markdown
## Functional Description

Single-buffered: one weight region and one, independently-sized, bias
region, each written by its own fill path and read through its own
synchronous read port. There is no bank select of any kind and no second
copy of either region.

Fill path: a `fill_start` pulse begins a new fill session, resetting both
regions' write row pointers, lane indices and row-assembly registers to 0
together (a beat presented the same cycle as `fill_start` is not
accepted). Incoming `s_stream` bytes then auto-increment a lane index into
a row-assembly register for the region selected by `fill_is_bias` (`'0'`
weight, `'1'` bias); once a row's lanes are all written, the assembled row
is committed to that region's memory with a single wide write, the lane
index wraps to 0, and the row pointer advances. Weight-region and
bias-region row pointers, lane indices and row-assembly registers are
fully independent of each other, so weight-then-bias, bias-then-weight, or
interleaved fill order all work identically, and one region reaching its
own depth (asserting backpressure on `s_stream_s2m.ready` for that region
only) never affects the other region's fill.

`cnn_accel_layer_ctrl` pulses `fill_start` once per output-channel-tile
pass (not once per layer): weights and bias for the pass currently
executing are streamed in fresh from DDR4 ahead of that pass, read
through `weight_rd_addr`/`bias_rd_addr` while the pass runs, and then
overwritten in place by the next pass's `fill_start`/fill stream. An
optional shallow prefetch FIFO on the fill stream (`g_fill_fifo_depth`,
default 32, `0` disables it) absorbs DDR4/DMA burst latency in place of
the ping-pong bank this module previously used for that purpose.

Read path: `weight_rd_addr`/`bias_rd_addr` are registered synchronous read
addresses into their respective region (no bank selection), one cycle of
read latency, matching `cnn_accel_pe_array`'s expected weight-fetch
latency. Reads of already-committed rows are correct while a fill of
later rows in the same region is still in progress — single-buffering
means there is no second copy to read from, but the row-assembly register
never touches memory until a row is complete, so a read can never observe
a partially-written row.
```

## Implementation Notes (vhfill)

(filled in during/after implementation)
