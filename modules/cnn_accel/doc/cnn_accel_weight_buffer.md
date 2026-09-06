# cnn_accel_weight_buffer

## Purpose

Double-buffered (ping-pong) on-chip cache for one CNN layer's weights and
bias, sized/organized for BRAM inference. While `cnn_accel_layer_ctrl`
directs the weight/bias `cnn_accel_axi_read_dma` instance to fill one bank
over an internal AXI4-Stream port, `cnn_accel_pe_array`/
`cnn_accel_bias_requant` read the *other* bank for the layer currently
executing, so the next layer's weight/bias fetch overlaps this layer's
compute. See `modules/cnn_accel/doc/cnn_accel_weight_buffer_req.md` and
`modules/cnn_accel/doc/cnn_accel_weight_buffer_proposal.md`.

## Entity and architecture

Entity `cnn_accel_weight_buffer`, architecture `a`
(`modules/cnn_accel/src/cnn_accel_weight_buffer.vhd`).

## Generics

| Name | Type | Meaning |
|---|---|---|
| `g_weight_buffer_depth` | positive | Rows per bank, per region (weight region and bias region both use this same depth — see proposal doc §3.2). **Depth contract (proposal doc §4, D10):** must hold a whole layer's weight rows, `ceil(kernel_h*kernel_w*in_channels/g_pe_cols)` — not just one input-channel tile's `groups_per_tile` — because `cnn_accel_pe_array.weight_rd_addr` rebases to 0 only on `first_tile` of a new output pixel and then runs continuously, one row per group, across every input-channel tile of that pixel (never wrapping mid-pixel). Minimum across the target network: 288 rows (layer 9); `module_cnn_accel.py`'s `_WEIGHT_BUFFER_DEPTH` build-time generic uses the recommended 512. The address width (`weight_rd_addr`/`bias_rd_addr`, and the internal row-pointer width) is always derived from this generic via `num_bits_needed`, never hard-coded, so raising the depth needs no other RTL change. |
| `g_pe_rows` | positive | Output-channel parallelism: rows per read tile, and bias lanes per read tile. |
| `g_pe_cols` | positive | Input-channel/MAC parallelism: weight lanes per read tile, together with `g_pe_rows`. |
| `g_accum_width` | positive := 32 | Bit width of one bias lane (int32 accumulator width elsewhere in this IP); added by `vhdesign` to give `bias_rd_data` a well-typed width not resolvable from the requirement alone (proposal doc §3.1). |

## Ports

| Name | Dir | Type/width | Description |
|---|---|---|---|
| `clk` | in | `std_ulogic` | Single clock domain. |
| `reset` | in | `std_ulogic := '0'` | Synchronous active-high (`= reset_internal` at the IP top level). |
| `s_stream_m2s` | in | `axi_stream_pkg.axi_stream_m2s_t` | Fill-side AXI4-Stream, from the weight/bias `cnn_accel_axi_read_dma` instance. One accepted beat writes one lane: a weight byte on `data(7 downto 0)`, or a bias lane on `data(g_accum_width-1 downto 0)`, selected by `fill_is_bias`. `last`/`user` are not consumed (proposal doc §3.5). |
| `s_stream_s2m` | out | `axi_stream_pkg.axi_stream_s2m_t` | `ready` deasserts once the selected bank+region's write pointer reaches `g_weight_buffer_depth`. |
| `fill_bank_sel` | in | `std_ulogic` | Bank filled by `s_stream` (`'0'` = bank A, `'1'` = bank B), from `cnn_accel_layer_ctrl`. |
| `fill_is_bias` | in | `std_ulogic` | Routes fill beats to the weight region (`'0'`) or bias region (`'1'`) of the bank selected by `fill_bank_sel`. |
| `read_bank_sel` | in | `std_ulogic` | Bank served by both read ports (`'0'` = bank A, `'1'` = bank B), from `cnn_accel_layer_ctrl`; independent of `fill_bank_sel` — the ping-pong property. |
| `weight_rd_addr` | in | `std_ulogic_vector(num_bits_needed(g_weight_buffer_depth-1)-1 downto 0)` | Row (tile) address into the selected bank's weight region, from `cnn_accel_pe_array`. |
| `weight_rd_data` | out | `std_ulogic_vector(8*g_pe_rows*g_pe_cols-1 downto 0)` | One int8 weight per active PE, registered, 1-cycle read latency. |
| `bias_rd_addr` | in | `std_ulogic_vector(num_bits_needed(g_weight_buffer_depth-1)-1 downto 0)` | Row (tile) address into the selected bank's bias region, from `cnn_accel_bias_requant`. |
| `bias_rd_data` | out | `std_ulogic_vector(g_accum_width*g_pe_rows-1 downto 0)` | One int32 (`g_accum_width`-bit) bias per output-channel lane, registered, 1-cycle read latency. |

## Clocking and reset

Single clock (`clk`), synchronous active-high `reset` (`reset_internal`).
`reset` clears, for both banks: the weight row pointer, weight lane index,
bias row pointer, bias lane index (all back to 0/empty) and the
`fill_bank_sel` edge-tracking register (`cnn_accel_weight_buffer.vhd:135`,
`:176-181`). This is sufficient for the requirement's abort guarantee: after
`reset`, a new fill of the same bank always starts writing at row 0 again,
so a partially-filled bank from an aborted layer can never be left
addressable as if it had completed cleanly — the pointer state that would
have made it look complete is gone.

Bank memory contents (`weight_mem`/`bias_mem`) are **not** cleared by
`reset` — BRAM-inference intent; a depth-deep clear loop would defeat the
single-cycle-abort requirement and clearing is unnecessary since the
pointer reset alone already prevents an aborted bank from being read as
complete. Read-data output registers (`weight_rd_data`/`bias_rd_data`) are
also not reset (no completeness contract of their own — "value irrelevant
until valid") and keep only their declaration initial values.

## Interfaces/protocols

**Fill port (`s_stream_m2s`/`s2m`):** AXI4-Stream per `shared/Axi4.md`,
but a single point-to-point stream with simple backpressure — no bursting,
no reordering, no same-ID ordering concerns (no `id` field used). One
accepted beat (`s_stream_m2s.valid = '1' and s_stream_s2m.ready = '1'`,
sampled at `rising_edge(clk)`) writes exactly one lane into the row/lane
addressed by the currently-selected bank+region's own write pointer, then
auto-advances that pointer (lane index, then row index once a row's lanes
are all written). Backpressure is honored strictly: `ready` deasserts
combinationally as soon as the selected bank+region's row pointer reaches
`g_weight_buffer_depth`, and a beat presented while `ready = '0'` is never
written (verified: the row a stalled beat targets keeps its prior content,
`tb_cnn_accel_weight_buffer.vhd:362-370`). A weight-region-full condition
is a `layer_error`-worthy condition surfaced upward by
`cnn_accel_layer_ctrl`, not silently-dropped data — this module only
implements the backpressure signal, not the error escalation itself.

**Read ports (`weight_rd_addr`/`data`, `bias_rd_addr`/`data`):** plain
synchronous random-access ports, not AXI4-Stream — no valid/ready
handshake at all. `*_rd_addr` is sampled every `rising_edge(clk)`
unconditionally (no enable), and `*_rd_data` reflects the addressed row of
the bank selected by `read_bank_sel`, one cycle later.

**Region/bank independence:** `s_stream_s2m.ready` is purely a function of
`fill_is_bias` and the *currently selected* bank+region's own write
pointer (`cnn_accel_weight_buffer.vhd:156-159`) — the weight region and
bias region of a bank each carry an independent row pointer and lane
index (`weight_wr_row_q`/`bias_wr_row_q`, `weight_lane_q`/`bias_lane_q`,
one pair of pointers per region per bank, `:124-132`), so one region being
full never affects the other region's `ready`, and the fill-side bank
never affects the read-side bank's data (`fill_bank_sel`/`read_bank_sel`
are fully independent selectors into the same two-bank memory array).

## Functional behavior

- Two banks (A = index 0, B = index 1), each holding a weight region
  (`g_weight_buffer_depth` rows of `g_pe_rows*g_pe_cols` int8 lanes) and a
  bias region (`g_weight_buffer_depth` rows of `g_pe_rows`
  `g_accum_width`-bit lanes) — proposal doc §3.2/§3.3.
- **Fill path:** while `fill_bank_sel` selects bank X, incoming
  `s_stream` bytes auto-increment a write pointer into bank X's weight
  region (`fill_is_bias = '0'`) or bias region (`fill_is_bias = '1'`): one
  lane per accepted beat, auto-advancing a lane index; once a row's lanes
  are all written, the lane index wraps to 0 and the row pointer advances
  (`cnn_accel_weight_buffer.vhd:190-214`).
- **New fill session:** an edge on `fill_bank_sel` newly selecting bank X
  resets **both** of bank X's regions' row pointers and lane indices to 0
  together (`:185-189`) — not a single pointer shared across regions (that
  reading of the requirement text would break bias addressing whenever a
  bias fill starts mid-way through the weight pointer's count; proposal
  doc §3.4). Within one fill session, the weight sub-fill and bias
  sub-fill each advance their own independent row/lane counters, so fill
  order (weight-then-bias, interleaved, or bias-only) does not matter.
- **Read path:** while `read_bank_sel` selects bank Y, `weight_rd_addr`/
  `bias_rd_addr` are registered synchronous read addresses into bank Y,
  one cycle of read latency (`:227-233`), matching `cnn_accel_pe_array`'s
  expected weight-fetch latency.

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
  network (288, layer 9) — configured at 512 for headroom
  (`module_cnn_accel.py`'s `_WEIGHT_BUFFER_DEPTH`).

## Timing/latency

- **Fill:** 1 beat accepted per cycle while `ready = '1'` (no internal
  stall beyond end-of-region backpressure).
- **Read:** fixed 1-cycle registered latency on both `weight_rd_addr`/
  `data` and `bias_rd_addr`/`data` (verified,
  `tb_cnn_accel_weight_buffer.vhd:318-344`: address applied one edge,
  `*_rd_data` checked valid exactly one edge later, both for weight and
  bias ports, at a non-zero row to also confirm the addressed row — not
  just row 0 — is the one returned).
- **Backpressure edge:** `ready` deasserts in the same cycle the last row
  of the selected bank+region is accepted, i.e. once that region's row
  pointer reaches `g_weight_buffer_depth` (verified for both regions,
  `test_backpressure_weight_region_full`/`test_backpressure_bias_region_full`
  below), and reasserts immediately (same cycle, combinationally) once
  `fill_is_bias`/`fill_bank_sel` selects a region whose own pointer has
  not yet reached depth.

## Dependencies

- `axi_stream.axi_stream_pkg` (hdl-modules) for the `axi_stream_m2s_t`/
  `axi_stream_s2m_t` record types on `s_stream_m2s`/`s2m`.
- `math.math_pkg.num_bits_needed` (hdl-modules) for `weight_rd_addr`/
  `bias_rd_addr` width and the internal row-pointer width — reused, not
  reimplemented, per `shared/ReusableRTL.md`.
- No hdl-modules RAM primitive is instantiated: bank memories are a
  hand-written inferred simple-dual-port RAM idiom (one array signal per
  bank/region, one write process, one read process), following
  `hdl-modules/modules/fifo/src/fifo.vhd`'s `memory_block` pattern — see
  proposal doc §6 for why no ready-made hdl-modules primitive fits (two
  banks, two independently-addressed regions of different element widths,
  and a byte/lane-serial-to-wide-row write path).

## Implementation notes

- Row-pointer width is `num_bits_needed(g_weight_buffer_depth)` bits (one
  bit wider than the read-address ports' `num_bits_needed(g_weight_buffer_depth-1)`),
  because the pointer must represent the value `g_weight_buffer_depth`
  itself — the "reached depth" backpressure boundary — without wrapping
  around (`cnn_accel_weight_buffer.vhd:90-95`).
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
generics (`g_weight_buffer_depth=4`, `g_pe_rows=2`, `g_pe_cols=2`,
`g_accum_width=16`) to keep row/lane counts directed and readable. No
built-in VUnit AXI4-Stream VC is used: the record port type
(`axi_stream_m2s_t`/`s2m_t`) is driven directly by small hand-written
non-blocking helper procedures (`push_fill_beat`/`push_weight_row`/
`push_bias_row`, each honoring `ready` before advancing) — appropriate
here since this module's fill port never bursts/reorders and is a single
point-to-point stream with simple backpressure, so a full VUnit
`axi_stream_master` VC would add unpacking overhead for no behavioral
benefit (proposal doc §10). Read ports are exercised directly (no BFM
needed — plain synchronous ports, no handshake).

Test cases (`lib.tb_cnn_accel_weight_buffer.<name>`), all 8 passing:

- `test_ping_pong_fill_a_read_b` / `test_ping_pong_fill_b_read_a` — fill
  one bank while reading the other, concurrently; both ping-pong
  directions.
- `test_fill_pointer_autoincrement_and_reset_on_new_fill` — lane/row
  pointer auto-increment, and reset-to-0 on a new fill session
  (`fill_bank_sel` edge onto a bank).
- `test_bias_vs_weight_region_routing` — `fill_is_bias` correctly routes
  beats to the weight region vs. the bias region within one fill session.
- `test_read_latency_one_cycle` — 1-cycle registered read latency, both
  read ports, at a non-zero row.
- `test_backpressure_weight_region_full` / `test_backpressure_bias_region_full` —
  `ready` deasserts exactly once the selected region's row pointer
  reaches `g_weight_buffer_depth`; a beat presented while full is not
  accepted (row content unchanged); the *other* region's own pointer and
  `ready` are unaffected by one region being full — each region's
  independence is checked explicitly, in both directions.
- `test_reset_mid_fill_does_not_leak_partial_bank` — `reset` pulsed
  mid-fill, followed by a full re-fill with a different data pattern:
  every row (including row 0) must show only the post-abort pattern, and
  the bank correctly reports full (`ready` deasserted) once the re-fill
  completes.

**Fixed during this verification pass:** the two backpressure-independence
checks (`test_backpressure_weight_region_full`/
`test_backpressure_bias_region_full`) originally read
`s_stream_s2m.ready` in the same delta cycle as the preceding
`fill_is_bias <= ...` signal assignment, with no intervening `wait` —
since a VHDL signal assignment only takes effect once its driving process
next suspends/resumes, the check observed the *pre*-assignment value of
the combinationally-derived `ready_i` (confirmed via waveform: the
recorded `fill_is_bias` trace showed no transition at all before the
failing check). Every other place in this testbench that changes an input
and then checks a derived output already inserts the established
`wait for c_settle;` (1 ns) idiom first; these two spots were the only
ones missing it. The RTL's `ready_i` expression
(`cnn_accel_weight_buffer.vhd:156-159`) already correctly depends only on
`fill_is_bias` and the region it selects, independent of the other
region — no RTL change was needed. Fix: add the missing
`wait for c_settle;` after each `fill_is_bias <= ...` and before the
following `check_true`, matching the testbench's own established
pattern; all 8 tests then pass against the unmodified RTL.

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
