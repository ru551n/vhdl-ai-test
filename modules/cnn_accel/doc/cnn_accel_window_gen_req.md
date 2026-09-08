# cnn_accel_window_gen — requirement

## Responsibility

Configurable `K_h x K_w`, stride `S_h x S_w`, padded (with a configurable
pad value, ISA v2.1 -- see `cfg_pad_value`), input-channel-
tiled sliding-window generator over row-major int8 input: buffers
`K_h - 1` full rows (BRAM-inference intent) plus the current row, and for
every valid output position (only stride-aligned positions are emitted —
no wasted beats) emits `T = ceil(cfg_in_channels / g_tile_channels)`
consecutive windows of `K_h * K_w * g_tile_channels` int8 taps — one per
input-channel tile, `first_tile`/`last_tile` sideband-flagged (both `1`
when `T = 1`) — see doc/cnn_accel_tiled_dataflow_proposal.md sections 1/2
for the tiling rationale and D11's final-tile zero-padding rule (below).
Generalizes the fixed-3x3, fixed-border-dilation pattern in
`modules/canny/src/canny_window3x3.vhd` to arbitrary, per-instruction
`K_h`/`K_w`/stride/padding — not a reusable instance of that entity
(different generality and no Canny-specific border-dilation semantics).
One instance, time-multiplexed across instructions by
`cnn_accel_layer_ctrl`; used both for `CONV2D`/`DWCONV2D`/`FC` windows
(consumed by `cnn_accel_pe_array`) and for `POOL_MAX`/`POOL_AVG` windows
(consumed by `cnn_accel_pool`) — the windowing operation itself does not
differ between the two uses, only the consumer.

## Generics

| Generic | Type | Purpose |
|---|---|---|
| `g_max_kernel_size` | positive | upper bound on `K_h`/`K_w`, sizes internal tap registers |
| `g_max_row_tile_words` | positive | upper bound on `cfg_in_width * ceil(cfg_in_channels / g_tile_channels)`, sizes line-buffer BRAM depth (replaces the retired `g_max_fmap_width`, which bounded row width alone — see doc/cnn_accel_tiled_dataflow_proposal.md section 1) |
| `g_tile_channels` | positive | input channels processed in parallel per beat/tile (`Ct`); `cfg_in_channels` need not be a multiple of it — see D11 below |

## Ports

| Port | Dir | Type | Purpose |
|---|---|---|
| `clk`, `reset` | in | `std_ulogic` | `reset` = `reset_internal` |
| `cfg_kernel_h`/`w`, `cfg_stride_h`/`w` | in | `std_ulogic_vector(7 downto 0)` | latched at `start` |
| `cfg_pad_top`/`bottom`/`left`/`right` | in | `std_ulogic_vector(7 downto 0)` | latched at `start`; taps outside `[0, in_width) x [0, in_height)` are filled with `cfg_pad_value` |
| `cfg_pad_value` | in | `std_ulogic_vector(7 downto 0)` | **ISA v2.1**; latched at `start`. The signed int8 value a padded tap takes -- the input tensor's quantization zero-point, not necessarily 0. Defaults to 0 (the pre-v2.1 zero-fill). `cnn_accel_top` drives it from the descriptor on the POOL instance and ties it to 0 on the CONV instance; see `doc/cnn_accel_top_v2_arch.md` §5 for why zero-padding a zero-point-shifted tensor is wrong for max pooling |
| `cfg_in_width`/`height`/`channels` | in | `std_ulogic_vector(15 downto 0)` | latched at `start` |
| `start` | in | `std_ulogic` | pulse, from `cnn_accel_layer_ctrl` |
| `done` | out | `std_ulogic` | pulse, last tile beat of the last window of the frame emitted |
| `s_stream_m2s`/`s2m` | in/out | `axi_stream_pkg.axi_stream_m2s_t`/`s2m_t` | raster-order int8 input pixels from the ifmap `cnn_accel_axi_read_dma`; one beat per `(column, channel-tile)` cell, `data` low `8 * g_tile_channels` bits |
| `m_window_m2s`/`s2m` | out/in | `cnn_accel_pkg.window_m2s_t`/`s2m_t`, `data` width = `window_data_width(g_max_kernel_size, g_tile_channels)` (only the low `K_h*K_w*g_tile_channels*8` bits meaningful) | `T` beats per output pixel (one per input-channel tile), `first_tile`/`last_tile` sideband-flagged, to the opcode-selected `handshake_splitter` (`cnn_accel_pe_array` / `cnn_accel_pool`) |

## Protocols

AXI4-Stream in and out, full backpressure (`TREADY`) on both links per
`shared/Axi4.md`'s mandatory-default rule — every inter-stage link in
this pipeline has a working ready/valid handshake, no global stall.

## Clock/reset

Synchronous active-high `reset_internal`: line-buffer write/read pointers
and row/column counters must return to a clean idle state on host abort.

<!-- functional-spec: hand-owned below this line -->

## Functional Description

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
partial (a direct consequence of `T` being a ceiling division); that
tile's unused channel lanes (channel index `>= cfg_in_channels mod
g_tile_channels`, or the whole tile full when the remainder is 0) are
driven to `0` at write time, deterministically — never left as whatever
`s_stream_m2s.data` happens to carry there.
