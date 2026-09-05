# cnn_accel_window_gen — requirement

## Responsibility

Configurable `K_h x K_w`, stride `S_h x S_w`, zero-padded sliding-window
generator over row-major int8 input: buffers `K_h - 1` full rows
(BRAM-inference intent) plus the current row, and for every valid output
position (only stride-aligned positions are emitted — no wasted beats)
emits one window of `K_h * K_w * g_line_buffer_channels` int8 taps.
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
| `g_max_fmap_width` | positive | upper bound on row width, sizes line-buffer BRAM depth |
| `g_line_buffer_channels` | positive | channels processed in parallel per beat |

## Ports

| Port | Dir | Type | Purpose |
|---|---|---|
| `clk`, `reset` | in | `std_ulogic` | `reset` = `reset_internal` |
| `cfg_kernel_h`/`w`, `cfg_stride_h`/`w` | in | `std_ulogic_vector(7 downto 0)` | latched at `start` |
| `cfg_pad_top`/`bottom`/`left`/`right` | in | `std_ulogic_vector(7 downto 0)` | latched at `start`; zero-fill taps outside `[0, in_width) x [0, in_height)` after padding |
| `cfg_in_width`/`height`/`channels` | in | `std_ulogic_vector(15 downto 0)` | latched at `start` |
| `start` | in | `std_ulogic` | pulse, from `cnn_accel_layer_ctrl` |
| `done` | out | `std_ulogic` | pulse, last window of the frame emitted |
| `s_stream_m2s`/`s2m` | in/out | `axi_stream_pkg.axi_stream_m2s_t`/`s2m_t` | raster-order int8 input pixels from the ifmap `cnn_accel_axi_read_dma` |
| `m_window_m2s`/`s2m` | out/in | `axi_stream_pkg.axi_stream_m2s_t`/`s2m_t`, `data` width = `g_max_kernel_size^2 * g_line_buffer_channels * 8` (only the low `K_h*K_w*channels*8` bits meaningful) | to the opcode-selected `handshake_splitter` (`cnn_accel_pe_array` / `cnn_accel_pool`) |

## Protocols

AXI4-Stream in and out, full backpressure (`TREADY`) on both links per
`shared/Axi4.md`'s mandatory-default rule — every inter-stage link in
this pipeline has a working ready/valid handshake, no global stall.

## Clock/reset

Synchronous active-high `reset_internal`: line-buffer write/read pointers
and row/column counters must return to a clean idle state on host abort.

<!-- functional-spec: hand-owned below this line -->

## Functional Description

On `start`, reset row/column counters and line-buffer pointers; as
`s_stream` beats arrive, write into the line-buffer bank for the current
row (ping-pong across `K_h` row banks, oldest bank recycled once no
longer needed by any tap); after enough rows/columns have been buffered
to form a full window at a stride-aligned output position, assemble the
`K_h x K_w x channels` taps (substituting `0` for any tap that falls in
the padding region, per `cfg_pad_*`) and present them as one
`m_window_m2s` beat; `last` marks the final window of the final output
row. Pulses `done` once the final window has been accepted
(`m_window_s2m.ready='1'`).
