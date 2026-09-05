# cnn_accel_weight_buffer — requirement

## Responsibility

Double-buffered (ping-pong) on-chip cache for one layer's weights + bias
(BRAM-inference intent). Bank A/B: one is being filled by the weight/bias
`cnn_accel_axi_read_dma` instance's AXI4-Stream while the other is being
read by `cnn_accel_pe_array`/`cnn_accel_bias_requant` for the layer
currently executing, so the next layer's weight fetch overlaps this
layer's compute.

## Generics

| Generic | Type | Purpose |
|---|---|---|
| `g_weight_buffer_depth` | positive | elements per bank |
| `g_pe_rows`, `g_pe_cols` | positive | shape the read port so `cnn_accel_pe_array` can pull one tile per cycle |

## Ports

| Port | Dir | Type | Purpose |
|---|---|---|
| `clk`, `reset` | in | `std_ulogic` | `reset` = `reset_internal` |
| `s_stream_m2s`/`s2m` | in/out | `axi_stream_pkg.axi_stream_m2s_t`/`s2m_t` | weight/bias bytes from the weight `cnn_accel_axi_read_dma` |
| `fill_bank_sel` | in | `std_ulogic` | which bank `s_stream` fills, from `cnn_accel_layer_ctrl` |
| `fill_is_bias` | in | `std_ulogic` | routes incoming bytes to the weight region or the bias region of the selected bank |
| `read_bank_sel` | in | `std_ulogic` | which bank the read ports below serve, from `cnn_accel_layer_ctrl` |
| `weight_rd_addr` | in | `std_ulogic_vector` | from `cnn_accel_pe_array` |
| `weight_rd_data` | out | `std_ulogic_vector(8*g_pe_rows*g_pe_cols-1 downto 0)` | one int8 weight per active PE, registered read |
| `bias_rd_addr` | in | `std_ulogic_vector` | from `cnn_accel_bias_requant` |
| `bias_rd_data` | out | `std_ulogic_vector(8*g_accum_width-1 downto 0)` sized for `g_pe_rows` int32 lanes | one int32 bias per output channel lane |

## Protocols

AXI4-Stream fill port (backpressure honored: `s_stream_s2m.ready`
deasserts once the selected bank's write pointer reaches
`g_weight_buffer_depth`, which is a `layer_error`-worthy condition
surfaced upward by `cnn_accel_layer_ctrl` rather than silently dropped
data). Simple synchronous read ports (not AXI4-Stream: random-access
within the active bank, one cycle latency) toward `cnn_accel_pe_array`/
`cnn_accel_bias_requant`.

## Clock/reset

Synchronous active-high `reset_internal`: bank fill/read pointers must
return to empty/idle on host abort so a partially-filled bank from an
aborted layer is never read as if complete.

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
