# cnn_accel_weight_buffer

## Purpose

Single-buffered on-chip cache for one CNN layer's weights, bias and (ISA
v1.2, HW milestone H2) per-channel requant table, sized/organized for BRAM
inference. `fill_start` pulses once to begin a new fill
session (resetting the write pointers), the whole weight set (weight
region, then/interleaved with the bias and scale regions, selected by
`fill_is_bias`/`fill_is_scale`)
is streamed in over `s_stream`, the pass runs while `cnn_accel_pe_array`/
`cnn_accel_bias_requant` read through `weight_rd_addr`/`bias_rd_addr`, then
the next pass's `fill_start` pulse begins the next refill. There is no
second bank to hide the refill behind compute any more — an optional
shallow prefetch FIFO on the fill stream (`g_fill_fifo_depth`) absorbs
DDR4/DMA burst latency instead (see below). See
`modules/cnn_accel/doc/cnn_accel_weight_buffer_req.md` and
`modules/cnn_accel/doc/cnn_accel_weight_buffer_proposal.md`.

## Entity and architecture

Entity `cnn_accel_weight_buffer`, architecture `a`
(`modules/cnn_accel/src/cnn_accel_weight_buffer.vhd`).

## Generics

| Name | Type | Meaning |
|---|---|---|
| `g_weight_buffer_depth` | positive | Rows in the weight region. **Depth contract (proposal doc §4, D10):** must hold a whole layer's weight rows, `ceil(kernel_h*kernel_w*in_channels/g_pe_cols)` — not just one input-channel tile's `groups_per_tile` — because `cnn_accel_pe_array.weight_rd_addr` rebases to 0 only on `first_tile` of a new output pixel and then runs continuously, one row per group, across every input-channel tile of that pixel (never wrapping mid-pixel). Minimum across the target network: 288 rows (layer 9); `module_cnn_accel.py`'s `_WEIGHT_BUFFER_DEPTH` build-time generic uses this recommended value (see proposal doc §11 for the sizing rationale). The address width (`weight_rd_addr`, and the internal weight row-pointer width) is always derived from this generic via `num_bits_needed`, never hard-coded, so raising the depth needs no other RTL change. |
| `g_bias_buffer_depth` | positive := 8 | Rows in the (separate, independently-sized, much shallower) bias region — decoupled from `g_weight_buffer_depth` so it can fall out of block RAM into LUTRAM/registers: a real layer only ever needs `ceil(out_channels/g_pe_rows)` bias rows (proposal doc §3.2/§11). `bias_rd_addr` and the internal bias row-pointer width are derived from this generic, independently of `g_weight_buffer_depth`. |
| `g_pe_rows` | positive | Output-channel parallelism: rows per read tile, and bias lanes per read tile. |
| `g_pe_cols` | positive | Input-channel/MAC parallelism: weight lanes per read tile, together with `g_pe_rows`. |
| `g_accum_width` | positive := 32 | Bit width of one bias lane (int32 accumulator width elsewhere in this IP); added by `vhdesign` to give `bias_rd_data` a well-typed width not resolvable from the requirement alone (proposal doc §3.1). |
| `g_fill_fifo_depth` | natural := 32 | Depth of the shallow prefetch FIFO placed on the fill stream, ahead of the row-assembly/write logic, to absorb DDR4/DMA burst latency now that there is no second bank to hide it behind. `0` means no FIFO is instantiated (fill stream connects straight through, exactly the old single-bank timing). Reuses hdl-modules' `fifo.fifo` unmodified; must be a power of two whenever nonzero (that entity's own constraint). |

## Ports

| Name | Dir | Type/width | Description |
|---|---|---|---|
| `clk` | in | `std_ulogic` | Single clock domain. |
| `reset` | in | `std_ulogic := '0'` | Synchronous active-high (`= reset_internal` at the IP top level). |
| `s_stream_m2s` | in | `axi_stream_pkg.axi_stream_m2s_t` | Fill-side AXI4-Stream, from the weight/bias `cnn_accel_axi_read_dma` instance. One accepted beat writes one lane: a weight byte on `data(7 downto 0)`, a bias lane on `data(g_accum_width-1 downto 0)`, or (H2) a per-channel scale entry on `data(c_scale_entry_width-1 downto 0)`, selected by `fill_is_bias`/`fill_is_scale`. `last`/`user` are not consumed (proposal doc §3.5). |
| `s_stream_s2m` | out | `axi_stream_pkg.axi_stream_s2m_t` | `ready` deasserts once the selected region's own write pointer reaches its depth (`g_weight_buffer_depth` or `g_bias_buffer_depth`). |
| `fill_start` | in | `std_ulogic := '0'` | Pulse: starts a new fill session — resets the weight and bias write row pointers/lane indices/row-assembly registers to 0. Must be pulsed once before streaming a new output-channel pass's weight/bias set (replaces the old `fill_bank_sel`-edge-triggered "new fill session" detector — proposal doc §3.4). A beat presented the same cycle as `fill_start` is not accepted. |
| `fill_is_bias` | in | `std_ulogic` | Routes fill beats to the weight region (`'0'`) or bias region (`'1'`), unless `fill_is_scale='1'`. |
| `fill_is_scale` | in | `std_ulogic := '0'` | ISA v1.2 (H2): `'1'` routes fill beats to the per-channel scale region, overriding `fill_is_bias`. One `c_scale_entry_width` (40)-bit lane on `data(39 downto 0)` per accepted beat = the first 40 bits of one 8-byte little-endian DDR table entry (int32 multiplier, uint8 shift), so a 64-bit DMA beat carrying one entry needs no reshuffling. Left at `'0'` the region is never written and every other port is bit-identical to the pre-H2 buffer. |
| `weight_rd_addr` | in | `std_ulogic_vector(num_bits_needed(g_weight_buffer_depth-1)-1 downto 0)` | Row (tile) address into the weight region, from `cnn_accel_pe_array`. |
| `weight_rd_data` | out | `std_ulogic_vector(8*g_pe_rows*g_pe_cols-1 downto 0)` | One int8 weight per active PE, registered, 1-cycle read latency. |
| `bias_rd_addr` | in | `std_ulogic_vector(num_bits_needed(g_bias_buffer_depth-1)-1 downto 0)` | Row (tile) address into the bias region, from `cnn_accel_bias_requant`. |
| `bias_rd_data` | out | `std_ulogic_vector(g_accum_width*g_pe_rows-1 downto 0)` | One int32 (`g_accum_width`-bit) bias per output-channel lane, registered, 1-cycle read latency. |
| `scale_rd_data` | out | `std_ulogic_vector(c_scale_entry_width*g_pe_rows-1 downto 0)` | ISA v1.2 (H2): one per-channel requant entry (multiplier `[31:0]`, shift `[39:32]` per lane, `cnn_accel_pkg.c_scale_entry_width`) per output-channel lane of the scale-region row addressed by `bias_rd_addr`, registered, 1-cycle read latency — same timing as `bias_rd_data`. |

## Clocking and reset

Single clock (`clk`), synchronous active-high `reset` (`reset_internal`).
`reset` clears the weight row pointer, weight lane index, weight
row-assembly register, bias row pointer, bias lane index and bias
row-assembly register (all back to 0/empty, `cnn_accel_weight_buffer.vhd:
270-276`) — the same reset the `fill_start` pulse itself performs
(`:277-286`), since a new fill session and a reset both need to guarantee
the next fill starts writing at row 0. This is sufficient for the
requirement's abort guarantee: after `reset`, a new fill always starts
writing at row 0 again, so a partially-filled region from an aborted
layer can never be left addressable as if it had completed cleanly — the
pointer state that would have made it look complete is gone.

Region memory contents (`weight_mem`/`bias_mem`) are **not** cleared by
`reset` — BRAM-inference intent; a depth-deep clear loop would defeat the
single-cycle-abort requirement and clearing is unnecessary since the
pointer reset alone already prevents an aborted region from being read as
complete. Read-data output registers (`weight_rd_data`/`bias_rd_data`) are
also not reset (no completeness contract of their own — "value irrelevant
until valid") and keep only their declaration initial values.

## Interfaces/protocols

**Fill port (`s_stream_m2s`/`s2m`):** AXI4-Stream per `shared/Axi4.md`,
but a single point-to-point stream with simple backpressure — no bursting,
no reordering, no same-ID ordering concerns (no `id` field used), optionally
passing through the internal prefetch FIFO first (`g_fill_fifo_depth > 0`).
One accepted beat (`s_stream_m2s.valid = '1' and s_stream_s2m.ready = '1'`,
sampled at `rising_edge(clk)`) writes exactly one lane into the row-
assembly register addressed by the selected region's own lane index, then
auto-advances that index; once a row's lanes are all written, the whole
assembled row is committed to memory with a single wide write and the row
pointer advances (`cnn_accel_weight_buffer.vhd:287-333`) — this, not the
former double buffering, is what lets Yosys infer one wide block RAM per
region instead of one RAMB18 per lane. Backpressure is honored strictly:
`ready` deasserts combinationally as soon as the selected region's own row
pointer reaches its depth (`g_weight_buffer_depth` or
`g_bias_buffer_depth`), and a beat presented while `ready = '0'` is never
written (verified: the row a stalled beat targets keeps its prior content,
`tb_cnn_accel_weight_buffer.vhd`'s `test_backpressure_*` cases). A
weight-region-full condition is a `layer_error`-worthy condition surfaced
upward by `cnn_accel_layer_ctrl`, not silently-dropped data — this module
only implements the backpressure signal, not the error escalation itself.

**Read ports (`weight_rd_addr`/`data`, `bias_rd_addr`/`data`):** plain
synchronous random-access ports, not AXI4-Stream — no valid/ready
handshake at all. `*_rd_addr` is sampled every `rising_edge(clk)`
unconditionally (no enable), and `*_rd_data` reflects the addressed row,
one cycle later (`cnn_accel_weight_buffer.vhd:343-349`). Reads are
concurrent with fills into the *other* region's or a not-yet-written
row's memory — single-buffering means there is no second copy to read
while a fill is in progress, but the weight and bias regions' fill and
read pointers are otherwise independent of each other.

**Region independence:** `s_stream_s2m.ready` is purely a function of
`fill_is_bias` and the currently-selected region's own write pointer
(`cnn_accel_weight_buffer.vhd:210-213`) — the weight region and bias
region each carry an independent row pointer, lane index and row-assembly
register (`weight_wr_row_q`/`bias_wr_row_q`,
`weight_lane_q`/`bias_lane_q`, `weight_row_assemble_q`/
`bias_row_assemble_q`, `:172-180`), so one region being full never
affects the other region's `ready`.

## Functional behavior

- One weight region (`g_weight_buffer_depth` rows of
  `g_pe_rows*g_pe_cols` int8 lanes), one, independently-sized, bias
  region (`g_bias_buffer_depth` rows of `g_pe_rows` `g_accum_width`-bit
  lanes) — proposal doc §3.2/§3.3/§11 — and, since ISA v1.2 (H2), a
  per-channel scale region of the **same** depth, lane count and read
  address as the bias region (`g_bias_buffer_depth` rows of `g_pe_rows`
  `c_scale_entry_width`-bit lanes), because the bias and the requant table
  are tiled identically (`cnn_accel_model.pack_bias_for_hw` /
  `pack_scale_table_for_hw`: entry `ot*g_pe_rows + r` -> row `ot`, lane
  `r`) and consumed together per beat by `cnn_accel_bias_requant`.
- **Fill path:** incoming `s_stream` lanes (optionally through the
  prefetch FIFO) auto-increment a lane index into a row-assembly register
  for the region selected by `fill_is_scale`/`fill_is_bias` (`fill_is_scale
  ='1'` = scale, else `fill_is_bias` `'0'` = weight, `'1'` = bias); once a
  row's lanes are all written, the assembled row is committed to that
  region's memory with one wide write, the lane index wraps to 0 and the
  row pointer advances. A scale beat carries one whole table entry on
  `data(c_scale_entry_width-1 downto 0)`.
- **New fill session:** a `fill_start` pulse resets **all** regions' row
  pointers, lane indices and row-assembly registers to 0 together — not a
  single pointer shared across regions (that reading of the requirement
  text would break bias addressing whenever a bias fill starts mid-way
  through the weight pointer's count; proposal doc §3.4). Within one fill
  session, the weight, bias and scale sub-fills each advance their own
  independent row/lane counters, so fill order (weight-then-bias-then-
  scale, interleaved, or bias-only) does not matter. A beat presented the
  same cycle as `fill_start` is not accepted.
- **Read path:** `weight_rd_addr`/`bias_rd_addr` are registered
  synchronous read addresses into their respective region, one cycle of
  read latency, matching `cnn_accel_pe_array`'s expected weight-fetch
  latency; `bias_rd_addr` reads the bias and scale regions in lock-step
  (`bias_rd_data`/`scale_rd_data`).
- **Prefetch FIFO:** when `g_fill_fifo_depth > 0`, `s_stream` first drains
  into hdl-modules' `fifo.fifo` (unmodified), carrying `fill_is_bias` and
  `fill_is_scale` alongside the payload bits (the payload is the widest
  of a weight byte, a bias lane and a 40-bit scale entry) through the FIFO
  so a beat already accepted into the FIFO is always routed to the region
  it was destined for at accept time, regardless of any later region-
  select change while it is still buffered. With `g_fill_fifo_depth = 0`
  the fill stream connects straight through with no FIFO instantiated at
  all.

## Weight row layout contract (D10, proposal doc §4)

This module is layout-agnostic — it stores/returns whatever
`c_weight_row_width`-bit rows it is fed, addressed by a flat
`weight_rd_addr`. It has no opinion on what a row *means*. But the
depth sizing above and the fill content are only correct if the fill
source and `cnn_accel_pe_array`'s read sequencing agree on that meaning,
so the contract is pinned here rather than left implicit:

- **Row order is TILE-MAJOR**, not the golden model's native layout:
  `cnn_accel_pe_array` walks `weight_rd_addr` as `t -> kr -> kc` (input-
  channel tile index hoisted *above* kernel position) within one output-
  pixel/output-channel-tile pass, resetting only on that pass's
  `first_tile`. The golden model (`cnn_accel_model.py`'s `conv2d`/OHWI
  weights) is `ic` fastest-varying *inside* `kr,kc`
  (`cnn_accel_model.py:401,416`) — the opposite nesting. For
  `in_channels=32`, `g_tile_channels=8` (`Ct=8`), tile 1's rows need
  `ic in 8..15` at *every* `(kr,kc)`: a strided gather across the whole
  `kr,kc` range of the OHWI array, not a contiguous slice.
- **The gather happens outside this module, in software, once** —
  `cnn_accel_model.pack_weights_for_hw()` performs exactly this repack at
  compile time into the byte image the weight DMA fill path streams in;
  it is not, and must never be, implemented in this RTL (no runtime-
  variable-bound loop or dynamic slice could do it without breaking the
  GHDL-synthesis constraints this module's fill path already works
  around — see `cnn_accel_weight_buffer.vhd:190-207`'s comment). If this
  module's row width, `g_pe_rows`/`g_pe_cols` shape, or row order
  contract ever changes, `pack_weights_for_hw()` must change with it —
  the two are not independently versioned.
- **Open risk, flagged not fixed here (out of this module's/this
  round's scope):** *within* one row, `pack_weights_for_hw()`'s own
  docstring fixes lane `c*g_pe_rows + r` (`c` outer, `r` inner —
  `cnn_accel_model.py:494-498`), but `cnn_accel_pe_array.vhd:258`
  extracts `weight_lane := r*g_pe_cols + c` (`r` outer, `c` inner) — the
  transpose of that convention, and not the same lane index whenever
  `r /= c`. Nothing currently catches this: there is no `cnn_accel_top`/
  DMA/`layer_ctrl` RTL yet and no test drives `pack_weights_for_hw()`'s
  output through real `cnn_accel_pe_array`/`cnn_accel_weight_buffer`
  hardware end-to-end. Needs a ratified decision (which side is
  authoritative) before `cnn_accel_top` integration; not addressed here
  since fixing it means either editing `cnn_accel_pe_array.vhd` (out of
  scope for this change) or `cnn_accel_model.py` (not this module's
  contract to redefine unilaterally).
- **Depth**: see the `g_weight_buffer_depth` generic entry above. A
  layer's packed image is always exactly
  `OT*T*kernel_h*kernel_w*g_tile_channels*g_pe_rows` int8 values
  (`pack_weights_for_hw()`'s own contract), i.e. `OT*T*kernel_h*kernel_w`
  rows; `g_weight_buffer_depth` must be at least the largest such row
  count across every layer/output-channel-tile pass in the target
  network (288 = `3*3*32`, layer 9's worst case) — configured at exactly
  288, no headroom, since weights are now streamed per output-channel
  pass rather than double-buffered on-chip (`module_cnn_accel.py`'s
  `_WEIGHT_BUFFER_DEPTH`; see proposal doc §11 for the full rationale).

## Timing/latency

- **Fill:** 1 beat accepted per cycle while `ready = '1'` (no internal
  stall beyond end-of-region backpressure and, when `g_fill_fifo_depth >
  0`, whatever elasticity the prefetch FIFO itself adds/removes).
- **Read:** fixed 1-cycle registered latency on both `weight_rd_addr`/
  `data` and `bias_rd_addr`/`data` (verified,
  `test_read_latency_one_cycle`: address applied one edge, `*_rd_data`
  checked valid exactly one edge later, both for weight and bias ports,
  at a non-zero row to also confirm the addressed row — not just row 0 —
  is the one returned).
- **Row commit:** a region's memory receives exactly one wide write per
  completed row (on the beat carrying that row's final lane), not one
  write per lane — the row-assembly register absorbs all intermediate
  lane writes.
- **Backpressure edge:** `ready` deasserts in the same cycle the last row
  of the selected region is accepted, i.e. once that region's own row
  pointer reaches its depth (verified for both regions,
  `test_backpressure_weight_region_full`/`test_backpressure_bias_region_full`
  below), and reasserts immediately (same cycle, combinationally) once
  `fill_is_bias` selects a region whose own pointer has not yet reached
  depth.

## Dependencies

- `axi_stream.axi_stream_pkg` (hdl-modules) for the `axi_stream_m2s_t`/
  `axi_stream_s2m_t` record types on `s_stream_m2s`/`s2m`.
- `math.math_pkg.num_bits_needed` (hdl-modules) for `weight_rd_addr`/
  `bias_rd_addr` width and the internal row-pointer widths — reused, not
  reimplemented, per `shared/ReusableRTL.md`.
- `fifo.fifo` (hdl-modules), instantiated unmodified when
  `g_fill_fifo_depth > 0`, for the prefetch FIFO on the fill stream (see
  proposal doc §6/§5 for why it fits here even though no ready-made
  hdl-modules primitive fits the region memories themselves).
- No hdl-modules RAM primitive is instantiated for the region memories:
  `weight_mem`/`bias_mem` are a hand-written inferred simple-dual-port
  RAM idiom (one array signal per region, one write process, one read
  process), following `hdl-modules/modules/fifo/src/fifo.vhd`'s
  `memory_block` pattern — see proposal doc §6 for why no ready-made
  hdl-modules primitive fits (two independently-addressed regions of
  different element widths and depths, and a byte/lane-serial-to-wide-row
  write path).

## Implementation notes

- Weight and bias row pointers are sized independently:
  `num_bits_needed(g_weight_buffer_depth)`/`num_bits_needed(g_bias_buffer_depth)`
  bits (one bit wider than the corresponding read-address ports'
  `num_bits_needed(depth-1)`), because each pointer must represent the
  value of its own depth itself — the "reached depth" backpressure
  boundary — without wrapping around
  (`cnn_accel_weight_buffer.vhd:113-123`).
- The weight and bias lane-serial-to-wide-row writes use a constant-
  bound `for` loop with the lane index as a per-lane enable, rather than
  a dynamically-bounded slice, because GHDL's synthesis backend rejects
  dynamic-bound slices (ghdl/ghdl#2658) — this only ever updates a plain
  register (the row-assembly register), never the memory itself, so it
  costs no BRAM fragmentation either way (`:287-333`).
- `bias_rd_data`'s width is `g_accum_width*g_pe_rows` bits. The
  requirement's port table states `std_ulogic_vector(8*g_accum_width-1
  downto 0)` "sized for `g_pe_rows` int32 lanes" — literally `8*32=256`
  bits with only one implied lane, inconsistent with its own "one int32
  bias per output channel lane" wording and with `g_accum_width` not even
  appearing in the requirement's Generics table. Treated as a
  documentation defect in the (non-hand-owned) port table, not part of
  the preserved Functional Description; resolved per proposal doc §3.1.
- Fill-stream `last`/`user` are left unconnected: no weight-buffer
  behavior in the requirement's Functional Description ties to
  `TLAST`/`TUSER` framing (the weight-DMA-done pulse living in
  `cnn_accel_axi_read_dma`/`cnn_accel_layer_ctrl` is the completion signal
  per `doc/cnn_accel_arch.md`'s DMA table) — proposal doc §3.5.
- No `--@` (unfinished-design-direction) markers remain in
  `cnn_accel_weight_buffer.vhd`.

## Verification notes

VUnit-5 testbench `tb_cnn_accel_weight_buffer` (`architecture tb`),
`modules/cnn_accel/test/tb_cnn_accel_weight_buffer.vhd`, small directed
generics (`g_weight_buffer_depth=4`, `g_bias_buffer_depth=2`,
`g_pe_rows=2`, `g_pe_cols=2`, `g_accum_width=16`, `g_fill_fifo_depth=0`)
to keep row/lane counts directed and readable; weight and bias depths are
deliberately different to exercise the decoupled address widths, and the
prefetch FIFO is disabled (straight-through) so every backpressure/
latency check sees the exact same-cycle timing as the region memories
themselves — the FIFO datapath itself is exercised by the default-generic
instance in `cnn_accel_conv_core` / the weight_buffer netlist build. No
built-in VUnit AXI4-Stream VC is used: the record port type
(`axi_stream_m2s_t`/`s2m_t`) is driven directly by small hand-written
non-blocking helper procedures, each honoring `ready` before advancing —
appropriate here since this module's fill port never bursts/reorders and
is a single point-to-point stream with simple backpressure, so a full
VUnit `axi_stream_master` VC would add unpacking overhead for no
behavioral benefit (proposal doc §10). Read ports are exercised directly
(no BFM needed — plain synchronous ports, no handshake).

Test cases (`lib.tb_cnn_accel_weight_buffer.<name>`), all 8 passing:

- `test_fill_then_read_weight_and_bias` — fill and read back both
  regions with deterministic patterns.
- `test_fill_start_restarts_pointer_and_drops_same_cycle_beat` —
  `fill_start` resets both regions' write pointers/lane
  indices/row-assembly registers to 0 together, and a beat presented the
  same cycle as `fill_start` is not accepted.
- `test_bias_vs_weight_region_routing` — `fill_is_bias` correctly routes
  beats to the weight region vs. the bias region within one fill session.
- `test_read_latency_one_cycle` — 1-cycle registered read latency, both
  read ports, at a non-zero row.
- `test_backpressure_weight_region_full` / `test_backpressure_bias_region_full` —
  `ready` deasserts exactly once the selected region's own row pointer
  reaches its own depth; a beat presented while full is not accepted (row
  content unchanged); the *other* region's own pointer and `ready` are
  unaffected by one region being full — each region's independence is
  checked explicitly, in both directions.
- `test_reset_mid_fill_does_not_leak_partial_fill` — `reset` pulsed
  mid-fill, followed by a full re-fill with a different data pattern:
  every row (including row 0) must show only the post-abort pattern, and
  the buffer correctly reports full (`ready` deasserted) once the
  re-fill completes.
- `test_read_already_committed_rows_during_active_fill` — reads of
  already-committed rows return correct data while a fill of later rows
  is still in progress (single-buffered concurrent read/write — no
  second bank needed to make this safe, since the row-assembly register
  only ever touches the memory once a row is complete).

The ISA v1.2 (H2) scale region has no dedicated case in this testbench:
its fill/read path is the bias region's, reused with a third select and a
wider lane, and it is exercised end to end (fill via `fill_is_scale` after
the bias sub-fill, read through `bias_rd_addr`, consumed per lane by
`cnn_accel_bias_requant`) by `tb_cnn_accel_conv_core`'s
`conv3x3_per_channel` vector case in every `g_pe_rows` config; the
`PER_CHANNEL_EN=0` cases of the same run prove the untouched region does
not disturb the others.

**`module_cnn_accel.py` / `setup_vunit` generics:** none recommended. This
testbench exposes only `runner_cfg` (no `stall_probability_percent` or
similar per-test generic to sweep) — unlike the AXI4-Stream-elastic
modules in `canny`/`axi_stream_join` that randomize a downstream
consumer's `tready` gaps, `cnn_accel_weight_buffer`'s fill port is a
single point-to-point stream whose only backpressure condition
(region-full) is already exhaustively exercised by the directed
`test_backpressure_*` cases, and its read ports have no `valid`/`ready`
handshake at all to randomize. A future revision could add a generic to
vary stall patterns on the fill side if this module ever grows a VUnit
AXI4-Stream VC-driven producer, but nothing in the current protocol or
requirement calls for it.

**Netlist build (`module_cnn_accel.py`'s `cnn_accel_weight_buffer` Yosys
project):** default generics (`g_weight_buffer_depth=288`,
`g_bias_buffer_depth=8`, `g_fill_fifo_depth=32` default). Measured 2026-09
(local dev Yosys, `synth_xilinx -family xc7`): 881 LUTs, 1093 FFs, 15
block RAMs (15 RAMB18, 0 RAMB36), 0 DSP — down from the pre-rework 72
block RAMs (the old 2-bank, per-lane-byte-write-enable weight region
fragmented into 64 separate 1024x8 memories), at the cost of higher
LUT/FF usage from the two whole-row assembly registers and the
distributed-RAM-mapped depth-32 prefetch FIFO (see
`module_cnn_accel.py`'s own comment on this build project for the full
breakdown).
