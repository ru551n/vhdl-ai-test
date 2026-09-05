# cnn_accel_axi_read_dma — requirement

## Responsibility

Generic AXI4 read master: accepts a `dma_req` (`addr`, byte `length`),
issues one or more `AR` bursts (respecting the 4 KiB burst-boundary rule
and the 256-beat `ARLEN` limit per `shared/Axi4.md`), accepts `R` beats
with backpressure, and republishes them as an internal AXI4-Stream
(`axi_stream_m2s_t`/`s2m_t`, `data` width = `g_axi_data_width`, `last`
asserted on the final beat of the whole request). Reports `dma_done` (one
pulse) when the requested byte range has been fully streamed out, or
`resp_error` if any `RRESP` was not `OKAY`. Instantiated three times (by
`cnn_accel_sequencer` for instructions, by `cnn_accel_layer_ctrl` for
weight/bias and ifmap) — identical entity, different callers/consumers.

Internally reuses `hdl-modules` `axi.axi_read_pipeline` (registers the AR/
R channels for timing) and `axi.axi_read_throttle` (limits outstanding
beats so the consumer's `axi_stream_s2m.ready` backpressure is honored
without overrunning an internal beat-count FIFO) rather than
re-implementing AXI4 read bookkeeping from scratch.

## Generics

| Generic | Type | Purpose |
|---|---|---|
| `g_axi_addr_width` | positive | |
| `g_axi_data_width` | positive | also the internal AXI4-Stream `data` width |
| `g_axi_id_width` | natural | this instance's fixed `ARID`/`RID` value is a `vhdesign`-time constant per instantiation site (instruction / weight / ifmap), not a generic |

## Ports

| Port | Dir | Type | Purpose |
|---|---|---|
| `clk`, `reset` | in | `std_ulogic` | `reset` = `reset_internal` |
| `req_m2s` | in | `cnn_accel_pkg.dma_req_m2s_t` | `valid`, `addr`, `length` |
| `req_s2m` | out | `cnn_accel_pkg.dma_req_s2m_t` | `ready` |
| `dma_done` | out | `std_ulogic` | pulse, full `length` streamed out |
| `resp_error` | out | `std_ulogic` | pulse, non-`OKAY` `RRESP` seen |
| `m_axi_ar_m2s`/`s2m` | out/in | `axi_pkg` AR-channel slice | to `axi.axi_simple_read_crossbar` |
| `m_axi_r_m2s`/`s2m` | in/out | `axi_pkg` R-channel slice | to `axi.axi_simple_read_crossbar` |
| `m_stream_m2s` | out | `axi_stream_pkg.axi_stream_m2s_t` | to the consumer (sequencer decode / `cnn_accel_weight_buffer` / `cnn_accel_window_gen`), via an `axi_stream_fifo` elasticity stage |
| `m_stream_s2m` | in | `axi_stream_pkg.axi_stream_s2m_t` | |

## Protocols

AXI4 read channels only (`AR`/`R`) at its master port; AXI4-Stream at its
consumer port. `shared/Axi4.md` rules 6-19 (excluding write-channel rules)
apply; a single `ARID` per instance is used so this instance's own
traffic is always self-ordered (rule 13's "single read ID" simplification
— no reordering to manage).

## Clock/reset

Synchronous active-high `reset_internal`: an in-flight DMA must be
abortable on host `ABORT` without leaving a stale outstanding-beat count
or a stuck `req_s2m.ready='0'` (per `doc/cnn_accel_arch.md` "Reset
policy").

<!-- functional-spec: hand-owned below this line -->

## Functional Description

On `req_m2s.valid` and `req_s2m.ready`, latch `addr`/`length`; split into
`AR` bursts of at most 256 beats x `g_axi_data_width/8` bytes each,
never crossing a 4 KiB boundary (per `shared/Axi4.md` rule 7); issue bursts
back-to-back via `axi.axi_read_pipeline`/`axi.axi_read_throttle`; forward
each accepted `R` beat's `RDATA` onto `m_stream_m2s.data` with `valid=1`,
`last=1` only on the very last beat of the very last burst of this
request; count bytes/beats to detect "fully streamed" and pulse
`dma_done`; if any `RRESP /= OKAY`, still drain the burst (per rule 15,
the slave always returns a defined response) but latch and pulse
`resp_error` once the request completes.
