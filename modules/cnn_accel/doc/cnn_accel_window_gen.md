# cnn_accel_window_gen

## Purpose

Configurable `K_h x K_w`, stride `S_h x S_w`, zero-padded, input-channel-
tiled sliding-window generator over row-major int8 input: buffers
`K_h - 1` full rows (BRAM-inference intent) plus the current row, and for
every valid output position (only stride-aligned positions are emitted —
no wasted beats) emits `T = ceil(cfg_in_channels / g_tile_channels)`
consecutive windows of `K_h * K_w * g_tile_channels` int8 taps, one per
input-channel tile (`first_tile`/`last_tile` sideband-flagged, both `1`
when `T = 1`; unused final-tile channel lanes zero-padded, D11 — see
`doc/cnn_accel_tiled_dataflow_proposal.md` sections 1/2/9). Generalizes
the fixed-3x3, fixed-border-dilation pattern in
`modules/canny/src/canny_window3x3.vhd` to arbitrary, per-instruction
`K_h`/`K_w`/stride/padding — not a reusable instance of that entity
(different generality and no Canny-specific border-dilation semantics).
One instance, time-multiplexed across instructions by
`cnn_accel_layer_ctrl`; used both for `CONV2D`/`DWCONV2D`/`FC` windows
(consumed by `cnn_accel_pe_array`) and for `POOL_MAX`/`POOL_AVG` windows
(consumed by `cnn_accel_pool`) — the windowing operation itself does not
differ between the two uses, only the consumer. See
`modules/cnn_accel/doc/cnn_accel_window_gen_req.md` and
`modules/cnn_accel/doc/cnn_accel_window_gen_proposal.md`.

## Entity and architecture

Entity `cnn_accel_window_gen`, architecture `a`.

## Generics

| Generic | Type | Meaning / constraints |
|---|---|---|
| `g_max_kernel_size` | `positive` | Upper bound on `cfg_kernel_h`/`cfg_kernel_w` individually; sizes the row-bank count and the window's tap grid. Contract: `g_max_kernel_size >= 2`, checked by an elaboration-time `assert`. |
| `g_max_row_tile_words` | `positive` | Upper bound on `cfg_in_width * ceil(cfg_in_channels / g_tile_channels)` ("row-tile-word count"); sizes each row bank's depth (BRAM-inference intent). Bounding the *product* directly — not `cfg_in_width` alone — avoids sizing width and channel-tile-count independently (see `doc/cnn_accel_tiled_dataflow_proposal.md` section 1). Replaces the retired `g_max_fmap_width`. Checked by a `severity failure` assert at `start` (runtime `cfg_*` values, not a true generic-only elaboration bound). |
| `g_tile_channels` | `positive` | Input channels processed in parallel per beat/tile (`Ct`). Contract: `g_tile_channels * 8 <= axi_stream_data_sz` (128), checked by an elaboration-time `assert` — one tile's channels must fit in `s_stream_m2s.data`'s low bytes. `cfg_in_channels` need not be a multiple of it: `T = ceil(cfg_in_channels / g_tile_channels)` tiles are emitted per output pixel, the last zero-padded if partial (D11). |

## Ports

| Port | Dir | Type | Description |
|---|---|---|---|
| `clk` | in | `std_ulogic` | Clock. |
| `reset` | in | `std_ulogic` | Synchronous active-high reset (`reset_internal` at the IP top level). Default `'0'`. |
| `cfg_kernel_h`, `cfg_kernel_w` | in | `std_ulogic_vector(7 downto 0)` | Kernel height/width, latched at `start`. Contract: each `<= g_max_kernel_size`. |
| `cfg_stride_h`, `cfg_stride_w` | in | `std_ulogic_vector(7 downto 0)` | Stride height/width, latched at `start`. |
| `cfg_pad_top`, `cfg_pad_bottom`, `cfg_pad_left`, `cfg_pad_right` | in | `std_ulogic_vector(7 downto 0)` | Zero-padding on each side, latched at `start`. |
| `cfg_in_width`, `cfg_in_height`, `cfg_in_channels` | in | `std_ulogic_vector(15 downto 0)` | Input frame size, latched at `start`. |
| `start` | in | `std_ulogic` | Pulse, from `cnn_accel_layer_ctrl`: latches all `cfg_*` ports above and resets row/column/tile counters and row-bank pointers for a new frame. |
| `done` | out | `std_ulogic` | Pulse: the final tile beat of the final window of the frame has been accepted (`m_window_s2m.ready = '1'` the same cycle). |
| `s_stream_m2s` / `s_stream_s2m` | in / out | `axi_stream_pkg.axi_stream_m2s_t` / `axi_stream_s2m_t` | Raster-order int8 input pixels, from the ifmap `cnn_accel_axi_read_dma`. One beat per `(column, channel-tile)` cell; `data` low `8 * g_tile_channels` bits hold that tile's channels. |
| `m_window_m2s` / `m_window_s2m` | out / in | `cnn_accel_pkg.window_m2s_t` / `window_s2m_t` | `T = ceil(cfg_in_channels / g_tile_channels)` consecutive beats per output pixel (one per input-channel tile), to the opcode-selected `cnn_accel_pe_array`/`cnn_accel_pool`. `data` (width `window_data_width(g_max_kernel_size, g_tile_channels)`) low `cfg_kernel_h * cfg_kernel_w * g_tile_channels * 8` bits hold the window: tap `i` (row-major, `i = row * cfg_kernel_w + col`), channel `c` within the tile, at bits `8*(i*g_tile_channels + c) + 7 downto 8*(i*g_tile_channels + c)`; remaining high bits are `0`; a partial final tile's unused lanes are `0` (D11). `first_tile`/`last_tile` flag the first/last of the `T` beats (both `1` when `T = 1`). `last` is `1` only for the final (`last_tile`) beat of the final output window of the frame. |

## Clocking and reset

Single clock domain (`clk`). Synchronous, active-high `reset` clears
`active_q` and the row/column/tile position counters (`cur_row_q`/
`cur_col_q`/`wr_tile_q`/`out_row_q`/`out_col_q`/`rd_tile_q`); the row-bank
contents are not reset (harmless — see "Implementation notes").

## Interfaces/protocols

`s_stream` is a plain AXI4-Stream link (`axi_stream_pkg`); `m_window` is
the `cnn_accel_pkg.window_m2s_t`/`s2m_t` handshake pair — same
valid/ready shape, plus the `first_tile`/`last_tile` sidebands (stable
sidebands per `shared/Axi4.md` rule 25, not `TLAST`-style framing — see
`doc/cnn_accel_tiled_dataflow_proposal.md` section 7). Full backpressure
on both links per `shared/Axi4.md`. `m_window_m2s.valid` (`window_valid`)
is a pure combinational function of registered state, never of
`s_stream_m2s.valid`/`m_window_s2m.ready`, so it is never combinationally
dependent on its own channel's `ready`. `s_stream_s2m.ready` is gated by
whether a currently-pending window (if any) has been consumed (`active_q
and (not window_valid or m_window_s2m.ready)`) — the standard, allowed
cross-channel "input readiness depends on output readiness" shape, not a
same-channel violation; this now gates on "tile `T-1` of the oldest still-
needed row consumed", not "the one pending window consumed", but the
shape is unchanged.

## Functional behavior

> The paragraph below is copied verbatim from
> `cnn_accel_window_gen_req.md`'s hand-owned "Functional Description"
> section (authoritative; not paraphrased here).

On `start`, reset row/column/tile counters and line-buffer pointers; as
`s_stream` beats arrive (one per `(column, channel-tile)` cell), write
into the line-buffer bank for the current row (ping-pong across `K_h` row
banks, oldest bank recycled once no longer needed by any tap); after
enough rows/columns have been buffered to form a full window at a
stride-aligned output position, assemble and emit `T =
ceil(cfg_in_channels / g_tile_channels)` consecutive `K_h x K_w x
g_tile_channels` windows, one per input-channel tile (substituting `0`
for any tap that falls in the padding region, per `cfg_pad_*`), as `T`
consecutive `m_window_m2s` beats; `first_tile`/`last_tile` mark the
first/last of those `T` beats (both `1` when `T = 1`); `last` marks only
the final (`last_tile`) beat of the final output window of the frame.
Pulses `done` once that final beat has been accepted
(`m_window_s2m.ready='1'`).

**D11 — partial-tile zero-padding.** When `cfg_in_channels` is not a
multiple of `g_tile_channels`, only the *last* tile of the frame is
partial; that tile's unused channel lanes are driven to `0` at write
time, deterministically — never left as whatever `s_stream_m2s.data`
happens to carry there.

### As-built detail beyond the requirement's own wording

- "Ping-pong across `K_h` row banks" is implemented as `g_max_kernel_size`
  full-row banks (not just `K_h` of them — `K_h` is a runtime `cfg_*`
  value, `g_max_kernel_size` the generic upper bound sizing the actual
  hardware array), addressed by `physical_row mod g_max_kernel_size`.
  Since `cfg_kernel_h <= g_max_kernel_size` is enforced at `start`, at
  most `g_max_kernel_size` distinct physical rows are ever simultaneously
  needed, so a bank is only ever overwritten once genuinely no longer
  needed by any pending or future output row.
- "Oldest bank recycled once no longer needed by any tap" is enforced
  indirectly, not by explicit per-bank bookkeeping: `s_stream_s2m.ready`
  is held low whenever a window is pending and not yet consumed, so a
  new input row can never overwrite a bank while any tap of a
  not-yet-consumed window still needs to read it. See
  `doc/cnn_accel_window_gen_proposal.md` "Architecture and dataflow" for
  the full row-bank design and why a FIFO/shift-register pipeline (the
  technique `canny_window3x3.vhd` uses for its own, non-configurable
  window) is insufficient here.
- Window readiness ("enough rows/columns have been buffered") is decided
  by the `row_ready` fencepost test described in the proposal doc's
  "Algorithms" section — ready once the largest *real* (non-padding) row/
  column a given output position needs has been fully written, with
  degenerate handling for windows that are entirely padding on one or
  more sides. `row_ready` is channel/tile-agnostic and unchanged by the
  channel-tiling retrofit — every tile of a given output pixel reads the
  same row/column range (`doc/cnn_accel_tiled_dataflow_proposal.md`
  section 2's orthogonality argument); tiling only changes which
  `(input_col * n_tiles_q + rd_tile_q)` row-bank cell supplies each tap.

## Timing/latency

Latency is variable and data-dependent, not a fixed per-instance
constant: the first window becomes ready once enough real rows/columns
have satisfied the `row_ready` test for output position `(0, 0)` — as
little as zero input beats for a window whose entire real range is
top/left padding, up to `kh - pad_top` full rows for a window with no
padding at all. See `doc/cnn_accel_window_gen_proposal.md`
"Latency/throughput" for the full derivation. Throughput: full 1:1 is
achievable whenever `m_window_s2m.ready` keeps up, since accepting an
input beat and consuming a pending window can happen in the same cycle
(`s_stream_s2m.ready`'s definition does not force a stall between them).

## Registers/configuration

None (no CSR-mapped registers in this module; `cfg_*` inputs are driven
combinationally by `cnn_accel_layer_ctrl` and latched internally at
`start`).

## Dependencies

- `ieee.std_logic_1164`, `ieee.numeric_std`.
- `axi_stream.axi_stream_pkg` (`hdl-modules/modules/axi_stream/src/axi_stream_pkg.vhd`) —
  `axi_stream_m2s_t`/`axi_stream_s2m_t` record types (`s_stream` only),
  `axi_stream_data_sz` constant.
- `cnn_accel.cnn_accel_pkg` — `window_m2s_t`/`window_s2m_t` record types
  and the `window_data_width` sizing function (`m_window`).
- No `hdl-modules` FIFO primitive is instantiated (see "Implementation
  notes" and the proposal doc's "Design rationale" for why the row-bank
  design was chosen over reusing `fifo.fifo_wrapper` as
  `canny_window3x3.vhd` does).

## Implementation notes

- Row banks (`row_bank_arr_t`, `g_max_kernel_size` entries of
  `row_bank_t`, each `g_max_row_tile_words` cells of `c_lane_width =
  8 * g_tile_channels` bits) are plain read/write arrays — BRAM-inference
  intent, not literal FIFOs — addressed by `physical_row mod
  g_max_kernel_size` on write and by direct `(input_row mod
  g_max_kernel_size, input_col * n_tiles_q + tile)` random access on
  read, with no pop/shift ordering constraint. Cell `col * n_tiles + tile`
  holds column `col`'s channel-tile `tile` (not one cell per column any
  more — see `doc/cnn_accel_tiled_dataflow_proposal.md` section 1). This
  is the key generalization over
  `canny_window3x3.vhd`'s FIFO + column-tap-shift-register technique;
  see `doc/cnn_accel_window_gen_proposal.md` "Design rationale: full-row
  banks, not FIFOs/shift registers" for the full comparison and the two
  concrete failure modes (padding-induced replay; stride not dividing
  the frame evenly) that technique cannot handle for this module's
  configurable stride/padding.
- The row-bank contents are not reset by `reset` — harmless because
  `active_q` is reset, `s_stream_s2m.ready`/`m_window_m2s.valid` both
  derive from `active_q`, and the very next `start` begins overwriting
  the banks from `(0, 0)` before any window can legitimately read stale
  content (no output is produced without first writing at least the
  row(s)/column(s) `row_ready` requires it to have written).
- `assemble_window` uses an explicit sensitivity list, not
  `process(all)` — mirrors `canny_window3x3.vhd`'s identical note: GHDL
  7.0.0-dev's `(all)` inference has not reliably tracked signals read
  only through nested loops/array indexing in this repo's experience, so
  an explicit list is used defensively even though the failure was not
  independently reproduced for this specific process.
- Output dimension (`out_width_q`/`out_height_q`) is computed once per
  `start` pulse via plain integer division, not a per-cycle datapath —
  deliberate (see proposal doc "Numeric types and widths"), not an
  accuracy concession.

## Verification notes

See `modules/cnn_accel/test/tb_cnn_accel_window_gen.vhd`
(`tb_cnn_accel_window_gen`/architecture `tb`) and
`doc/cnn_accel_window_gen_proposal.md`'s "Verification plan" for the
pre-tiling test list. Key corner cases covered: kernel/stride shapes from
`1x1` up to `c_kernel_max x c_kernel_max` (square and non-square,
`c_kernel_max = 3`), all-sides zero-padding combinations (none,
symmetric, two asymmetric — including one that clips a window's real-row
range to a single row), randomized independent backpressure on both
links (including mid-tile-sequence, `T = 2`), a dedicated zero-stall
full-throughput case, a mid-frame `reset` abort followed by a fresh,
ordinary frame to confirm no leftover state leaks in, and three
dedicated channel-tiling cases: `T = 1` partial (`in_channels = 3`,
`g_tile_channels = 8`, confirming the 5 unused lanes read as `0` even
when the testbench deliberately drives non-zero garbage there on
`s_stream`), `T > 1` exact (`in_channels = 16`), and `T > 1` partial
(`in_channels = 20`, `T = 3`, only the last tile partial). Golden values
are computed by an independently re-derived VHDL function
(`golden_window`/`golden_tap`, now with a channel-tile outer loop) in the
testbench, not by calling into the RTL under test; the scoreboard checks
`first_tile`/`last_tile`/`last`/`data` on every beat of every test, so
tile-sequencing correctness is exercised across every frame, not just the
dedicated tiling tests. A whole-simulation structural safety net
(`done_relation_check`) additionally confirms, on every `done` pulse
across every test, that it coincides exactly with acceptance of the
frame's final (`last`/`last_tile`) window beat.

Verified via `vunit-mcp`: `vunit_compile` then `vunit_run_tests` on
`cnn_accel.tb_cnn_accel_window_gen.*` under both GHDL and NVC — all 8
tests (`test_kernel_stride_shapes`, `test_padding_all_sides`,
`test_backpressure`, `test_full_throughput`,
`test_channel_tiling_partial_single`, `test_channel_tiling_exact_multi`,
`test_channel_tiling_partial_multi`, `test_reset_mid_frame_abort`) pass.
Full-project regression (`test_patterns=['*']`) is 100% green under GHDL.
See `doc/cnn_accel_window_gen_proposal.md` "Implementation Notes
(vhfill)" for the real architectural bug (FIFO/shift-register design
generalizing incorrectly to configurable padding/stride) found and fixed
during the original implementation, plus two testbench-only bugs (a
frame-streaming deadlock on non-dividing stride, and a `done`-pulse-count
delta-cycle race) fixed
alongside it.
