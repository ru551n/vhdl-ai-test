# cnn_accel_weight_buffer — requirement

## Responsibility

Single-buffered on-chip cache for one layer's weights + bias
(BRAM-inference intent). `fill_start` pulses to begin a new fill session;
the weight/bias `cnn_accel_axi_read_dma` instance's AXI4-Stream fills it
while `cnn_accel_pe_array`/`cnn_accel_bias_requant` read it for the layer
currently executing. An optional shallow prefetch FIFO on the fill
stream (`g_fill_fifo_depth`) absorbs DDR4/DMA burst latency in place of
the ping-pong bank this module previously had (see
`doc/cnn_accel_weight_buffer_proposal.md` §11 for the 2026-09 rework
that dropped double-buffering — flagged against the hand-owned
Functional Description below, which still describes the pre-rework
behavior).

## Generics

| Generic | Type | Purpose |
|---|---|---|
| `g_weight_buffer_depth` | positive | rows in the weight region |
| `g_bias_buffer_depth` | positive | rows in the (separate, independently-sized) bias region |
| `g_pe_rows`, `g_pe_cols` | positive | shape the read port so `cnn_accel_pe_array` can pull one tile per cycle |
| `g_fill_fifo_depth` | natural | depth of the optional prefetch FIFO on the fill stream; `0` disables it |

## Ports

| Port | Dir | Type | Purpose |
|---|---|---|
| `clk`, `reset` | in | `std_ulogic` | `reset` = `reset_internal` |
| `s_stream_m2s`/`s2m` | in/out | `axi_stream_pkg.axi_stream_m2s_t`/`s2m_t` | weight/bias bytes from the weight `cnn_accel_axi_read_dma` |
| `fill_start` | in | `std_ulogic` | pulse: starts a new fill session, from `cnn_accel_layer_ctrl` (replaces `fill_bank_sel`/`read_bank_sel` from the pre-rework double-buffered design) |
| `fill_is_bias` | in | `std_ulogic` | routes incoming bytes to the weight region or the bias region |
| `weight_rd_addr` | in | `std_ulogic_vector` | from `cnn_accel_pe_array` |
| `weight_rd_data` | out | `std_ulogic_vector(8*g_pe_rows*g_pe_cols-1 downto 0)` | one int8 weight per active PE, registered read |
| `bias_rd_addr` | in | `std_ulogic_vector` | from `cnn_accel_bias_requant` |
| `bias_rd_data` | out | `std_ulogic_vector(8*g_accum_width-1 downto 0)` sized for `g_pe_rows` int32 lanes | one int32 bias per output channel lane |

## Protocols

AXI4-Stream fill port (backpressure honored: `s_stream_s2m.ready`
deasserts once the selected region's own write pointer reaches its depth
(`g_weight_buffer_depth` or `g_bias_buffer_depth`), which is a
`layer_error`-worthy condition surfaced upward by `cnn_accel_layer_ctrl`
rather than silently dropped data). Simple synchronous read ports (not
AXI4-Stream: random-access, one cycle latency) toward
`cnn_accel_pe_array`/`cnn_accel_bias_requant`.

## Clock/reset

Synchronous active-high `reset_internal`: fill pointers must return to
empty/idle on host abort so a partially-filled region from an aborted
layer is never read as if complete.

<!-- functional-spec: hand-owned below this line -->

## Functional Description

Fill path: while `fill_bank_sel` selects bank X, incoming `s_stream`
bytes auto-increment a write pointer into bank X's weight region (or
bias region when `fill_is_bias='1'`); the pointer resets to 0 whenever
`cnn_accel_layer_ctrl` starts a new fill for that bank. Read path: while
`read_bank_sel` selects bank Y, `weight_rd_addr`/`bias_rd_addr` are
registered synchronous read addresses into bank Y, one cycle of read
latency, matching `cnn_accel_pe_array`'s expected weight-fetch latency
(a `vhdesign`-time pipeline-alignment detail).
