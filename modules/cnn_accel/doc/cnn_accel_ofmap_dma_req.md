# cnn_accel_ofmap_dma — requirement

## Responsibility

Output-activation write-back: accepts a `dma_req` (`addr`, byte `length`)
and an AXI4-Stream of int8 output pixels, and writes them to DDR as one
or more AXI4 write bursts. Thin, generic wrapper around `hdl-modules`
`dma_axi_write_simple.dma_axi_write_simple` (reused unmodified for the
`AW`/`W`/`B` bookkeeping) adding only the `dma_req`/`dma_done`/`resp_error`
control-plane adaptation used by this IP (`cnn_accel_layer_ctrl` drives
address/length per layer rather than the wrapped module's own native
control interface).

## Generics

| Generic | Type | Purpose |
|---|---|---|
| `g_axi_addr_width` | positive | |
| `g_axi_data_width` | positive | |

## Ports

| Port | Dir | Type | Purpose |
|---|---|---|
| `clk`, `reset` | in | `std_ulogic` | `reset` = `reset_internal` |
| `req_m2s`/`s2m` | in/out | `cnn_accel_pkg.dma_req_*` | `addr`, `length` from `cnn_accel_layer_ctrl` |
| `dma_done` | out | `std_ulogic` | pulse, full `length` written |
| `resp_error` | out | `std_ulogic` | pulse, non-`OKAY` `BRESP` seen |
| `s_stream_m2s`/`s2m` | in/out | `axi_stream_pkg.axi_stream_m2s_t`/`s2m_t` | from the opcode-converged output mux (`cnn_accel_bias_requant` / `cnn_accel_pool` max-path) |
| `m_axi_aw_m2s`/`s2m`, `m_axi_w_m2s`/`s2m`, `m_axi_b_m2s`/`s2m` | out/in | `axi_pkg` channel slices | direct to top-level `m_axi` (no arbitration, single writer) |

## Protocols

AXI4 write channels (`AW`/`W`/`B`) at its master port; AXI4-Stream at its
producer-facing port. `shared/Axi4.md` rules 6-19 (write-channel subset)
apply.

## Clock/reset

Synchronous active-high `reset_internal`, same abort rationale as
`cnn_accel_axi_read_dma`.

<!-- functional-spec: hand-owned below this line -->

## Functional Description

On `req_m2s.valid` and `req_s2m.ready`, latch `addr`/`length` and forward
them to the wrapped `dma_axi_write_simple` instance's control interface;
as stream beats arrive on `s_stream_m2s`, hand them to the wrapped
module's data-in port; propagate its completion/`BRESP`-error signals to
`dma_done`/`resp_error`. `s_stream_s2m.ready` is exactly the wrapped
module's own input-ready (no extra buffering added here beyond what the
wrapped module already provides).
