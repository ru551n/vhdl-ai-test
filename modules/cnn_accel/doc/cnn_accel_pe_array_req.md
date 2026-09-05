# cnn_accel_pe_array — requirement

## Responsibility

int8 x int8 multiply-accumulate array, `g_pe_rows` x `g_pe_cols`
parallelism, int32 (`g_accum_width`) accumulation. Executes `CONV2D`,
`DWCONV2D` and `FC` (the latter two as documented in
`doc/cnn_accel_arch.md`'s ISA section: `FC` is a degenerate 1x1-spatial
`CONV2D`; `DWCONV2D` uses the same array with cross-channel accumulation
disabled per output channel). Consumes one window beat from
`cnn_accel_window_gen` and the corresponding weight tile from
`cnn_accel_weight_buffer` per (set of) cycle(s), and emits one int32
accumulator result per output pixel/channel to `cnn_accel_bias_requant`.

## Generics

| Generic | Type | Purpose |
|---|---|---|
| `g_pe_rows` | positive | output-channel parallelism |
| `g_pe_cols` | positive | input-channel/MAC parallelism per cycle |
| `g_accum_width` | positive | accumulator width (int32 default) |
| `g_max_kernel_size` | positive | upper bound on `K_h*K_w`, sizes the per-window MAC sequencing counter |

## Ports

| Port | Dir | Type | Purpose |
|---|---|---|
| `clk`, `reset` | in | `std_ulogic` | `reset` = `reset_internal` |
| `cfg_opcode` | in | `std_ulogic_vector(7 downto 0)` | selects `CONV2D`/`DWCONV2D`/`FC` accumulation-grouping mode |
| `cfg_in_channels`, `cfg_out_channels` | in | `std_ulogic_vector(15 downto 0)` | latched at layer start |
| `s_window_m2s`/`s2m` | in/out | `axi_stream_pkg.axi_stream_m2s_t`/`s2m_t` | from `cnn_accel_window_gen` (via the opcode `handshake_splitter`) |
| `weight_rd_addr` | out | `std_ulogic_vector` | to `cnn_accel_weight_buffer` |
| `weight_rd_data` | in | `std_ulogic_vector(8*g_pe_rows*g_pe_cols-1 downto 0)` | from `cnn_accel_weight_buffer` |
| `m_accum_m2s`/`s2m` | out/in | `axi_stream_pkg.axi_stream_m2s_t`/`s2m_t`, `data` width = `g_accum_width*g_pe_rows` | one int32 accumulator per output channel lane, to `cnn_accel_bias_requant` (via the opcode `handshake_mux`) |

## Protocols

AXI4-Stream in and out, full backpressure per `shared/Axi4.md`'s
mandatory-default rule.

## Clock/reset

Synchronous active-high `reset_internal`: partial-sum accumulator
registers must not leak a stale sum into the next layer/tile after a host
abort.

<!-- functional-spec: hand-owned below this line -->

## Functional Description

For each accepted window beat, sequences `weight_rd_addr` over the
window's `K_h*K_w*in_channels` taps (grouped `g_pe_cols` at a time),
accumulating `tap * weight` into `g_pe_rows` parallel int32 accumulators
(one per output channel currently in flight); `DWCONV2D` mode keeps each
accumulator scoped to a single input channel (no cross-channel sum);
`CONV2D`/`FC` mode sums across all `in_channels`. Once a window's full
MAC sequence completes, the `g_pe_rows` accumulators are presented as one
`m_accum_m2s` beat (`last` mirrors the incoming window's `last`).
Exact DSP-packing/pipelining microarchitecture is deferred to
`vhdesign`/`vhfill` (see `doc/cnn_accel_arch.md` "Open items").
