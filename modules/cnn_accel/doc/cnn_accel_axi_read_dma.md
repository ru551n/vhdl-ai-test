# cnn_accel_axi_read_dma

## Purpose

Generic AXI4 read-DMA master. Accepts one `cnn_accel_pkg.dma_req_t`
(byte `addr`/`length`) at a time via `req_m2s`/`req_s2m`, splits it into
one or more `AR` bursts that respect the 4 KiB burst-boundary rule and
the 256-beat `ARLEN` limit, issues them back-to-back through
`axi.axi_read_throttle`/`axi.axi_read_pipeline`, and republishes every
accepted `R` beat as an AXI4-Stream (`data` width = `g_axi_data_width`)
to the consumer via an `axi_stream_fifo` elasticity stage. `dma_done`
pulses once the whole request has been streamed out to the true
external boundary (`m_stream_s2m`); `resp_error` pulses on the same
cycle if any `RRESP` in the request was not `OKAY` (the burst still
fully drains regardless). Intended to be instantiated once per AXI read
master port needed by the IP (instruction fetch, weight/bias fetch,
ifmap fetch), same entity/architecture each time.

## Entity and architecture

Entity `cnn_accel_axi_read_dma`, architecture `a`.

## Generics

| Generic | Type | Meaning / constraints |
|---|---|---|
| `g_axi_addr_width` | `positive` | Meaningful low bits of the fixed-max-width `axi_pkg` address fields. |
| `g_axi_data_width` | `positive` | AXI `RDATA` width and the internal/`m_stream` AXI4-Stream `data` width. `req_m2s.req.addr`/`.length` must both be multiples of `g_axi_data_width / 8` (word-aligned assumption, honored by every caller — see "Interfaces/protocols"). |
| `g_axi_id_width` | `natural` | Width of the (fixed-value) `ARID`/`RID` field. The value driven is always the hardwired constant `0` — this generic only sizes the field, it does not select the value. |

## Ports

| Port | Dir | Type | Description |
|---|---|---|---|
| `clk` | in | `std_ulogic` | Clock. |
| `reset` | in | `std_ulogic` | Synchronous active-high reset (`reset_internal` at the IP top level). Default `'0'`. |
| `req_m2s` | in | `cnn_accel_pkg.dma_req_m2s_t` | `valid`, `req.addr`/`req.length` (both `unsigned(31 downto 0)`, bytes). |
| `req_s2m` | out | `cnn_accel_pkg.dma_req_s2m_t` | `ready`. High only while idle (see FSM below). |
| `dma_done` | out | `std_ulogic` | One-cycle pulse, once per accepted request, the cycle after the request's last stream beat is accepted at `m_stream_s2m`. Default `'0'`. |
| `resp_error` | out | `std_ulogic` | One-cycle pulse, same cycle as `dma_done`, if any `RRESP` in the request was not `OKAY`. Default `'0'`. |
| `m_axi_ar_m2s` | out | `axi_pkg.axi_m2s_a_t` | AR channel (`ARVALID`/`ARADDR`/`ARLEN`/`ARSIZE`/`ARBURST`/`ARID`). |
| `m_axi_ar_s2m` | in | `axi_pkg.axi_s2m_a_t` | AR channel (`ARREADY`). |
| `m_axi_r_m2s` | out | `axi_pkg.axi_m2s_r_t` | R channel, `RREADY` only (this master's output). |
| `m_axi_r_s2m` | in | `axi_pkg.axi_s2m_r_t` | R channel, `RVALID`/`RDATA`/`RRESP`/`RLAST`/`RID` (this master's inputs). |
| `m_stream_m2s` | out | `axi_stream_pkg.axi_stream_m2s_t` | To the consumer, through an internal `axi_stream_fifo` elasticity stage. `data(g_axi_data_width - 1 downto 0)` holds one `R` beat's data; `last` marks the request's final beat. |
| `m_stream_s2m` | in | `axi_stream_pkg.axi_stream_s2m_t` | |

## Clocking and reset

Single clock domain (`clk`). Synchronous, active-high `reset`. On
`reset = '1'`: the top request FSM returns to `IDLE` (`req_s2m.ready`
goes high again the next cycle) and `err_pending_q` clears. The reused
hdl-modules submodules (`axi_read_pipeline`, `axi_read_throttle`,
`axi_r_fifo`, `axi_stream_fifo`) have no `reset` port of their own
(hdl-modules resetless-by-default convention) — any beat already
latched inside them at the moment of `reset` is *not* discarded there;
it still surfaces at `axi_r_fifo`'s consumer side at some later,
real-bus-timing-dependent point. This entity accounts for that
explicitly: on `reset`, `stale_beats_q` (a register deliberately **not**
cleared by `reset`, only ever incremented by it or decremented by the
drain logic below) is incremented by however many beats the
just-abandoned request had already gotten accepted into the pipe
(`beats_issued_q`) but not yet counted as consumed (`r_beat_count_q`).
A subsequently accepted request is held in an internal `DRAIN_STALE`
state — `req_s2m.ready` already low, no `AR` issued yet, nothing
forwarded to `m_stream_m2s` — for as long as `stale_beats_q` is nonzero,
unconditionally popping and discarding exactly that many leftover beats
off `axi_r_fifo` before ever issuing its own first `AR`. This works
regardless of when the abort happens relative to the abandoned
request's `AR`/`R` progress, and correctly accumulates across
back-to-back aborts. Exercised by `test_reset_mid_transfer`.

## Interfaces/protocols

- `req_m2s`/`req_s2m`: single-beat request handshake, `ready`-qualified,
  accepted only while idle (one request in flight at a time; no request
  pipelining across `dma_done` boundaries).
- `m_axi_ar_*`/`m_axi_r_*`: standard AXI4 AR/R master, full backpressure.
  `ARID`/expected `RID` is a hardwired `0` on every instance (not a
  generic-selected value) — safe because the documented downstream
  consumer (`axi.axi_simple_read_crossbar`) port-locks for the duration
  of a whole burst, so no two instances' traffic is ever interleaved
  there regardless of the `ARID` each one drives.
- `m_stream_m2s`/`s2m`: AXI4-Stream, full backpressure, one `R` beat per
  stream beat, `last` on the request's true final beat (independent of
  how many `AR` bursts that beat's data happened to travel in, since a
  single `ARID` per instance and the crossbar's port-locking guarantee
  in-order `R` beat arrival for the whole request).
- **Word-aligned assumption**: `req_m2s.req.addr`/`.length` must both be
  multiples of `g_axi_data_width / 8`. Honored by every caller
  (`cnn_accel_sequencer`, `cnn_accel_layer_ctrl`); no narrow/unaligned
  first/last-beat handling is implemented. `length = 0` is accepted as a
  degenerate case: no `AR` is issued at all, and `dma_done` pulses the
  cycle after accept (`resp_error` stays `'0'`).

## Functional behavior

Top FSM: `IDLE -> [DRAIN_STALE] -> ACTIVE -> WAIT_DRAIN -> DONE_PULSE ->
IDLE`. On accepting a request in `IDLE`, `addr_q`/`bytes_remaining_q`
are latched from `req.addr`/`.length`, `total_beats_q = length /
(g_axi_data_width / 8)`, and `r_beat_count_q`/`err_pending_q`/
`beats_issued_q` reset to `0`. If `stale_beats_q /= 0` (see "Clocking
and reset"), the FSM first passes through `DRAIN_STALE`; otherwise (or
once drained) it enters `ACTIVE`, where two sub-logics run concurrently:

- **AR-issue**: while `bytes_remaining_q /= 0`, computes this burst's
  byte count as `min(bytes_remaining_q, bytes to next 4 KiB boundary,
  one max-length burst)`, drives `AR` (`ARADDR = addr_q`, `ARLEN`/
  `ARSIZE`/`ARBURST = INCR` derived from the computed burst), and on
  acceptance advances `addr_q`/`bytes_remaining_q` by that burst's byte
  count.
- **R-consume**: pops `axi_r_fifo`'s consumer side whenever the output
  `axi_stream_fifo` has room, tags `last` on the beat where
  `r_beat_count_q + 1 = total_beats_q`, and latches `err_pending_q` on
  any non-`OKAY` `RRESP` without stopping the pop (the burst still fully
  drains).

Once R-consume reaches the last beat, the FSM moves to `WAIT_DRAIN` and
waits until that beat is actually accepted at the true external boundary
(`m_stream_s2m`), then pulses `dma_done`/`resp_error` for one cycle in
`DONE_PULSE` before returning to `IDLE`.

## Timing/latency

One outstanding AR burst issued at a time (no cross-burst AR
pipelining within one instance); within a burst, `axi_read_pipeline`/
`axi_read_throttle` support a transaction every cycle
(`full_ar_throughput => true`). `axi_r_fifo` and `axi_stream_fifo` each
add up to one cycle of latency on their pop-side combinational
passthrough. `dma_done` is asserted the cycle after the last stream beat
is accepted at `m_stream_s2m`, not at internal beat-count completion.

## Registers/configuration

None (no CSR-mapped registers in this module).

## Dependencies

- `ieee.std_logic_1164`, `ieee.numeric_std`.
- `cnn_accel.cnn_accel_pkg` (`modules/cnn_accel/src/cnn_accel_pkg.vhd`) —
  `dma_req_t`/`dma_req_m2s_t`/`dma_req_s2m_t`.
- `axi.axi_pkg` (`hdl-modules/modules/axi/src/axi_pkg.vhd`) —
  `axi_read_m2s_t`/`axi_read_s2m_t`, `axi_m2s_a_t`/`axi_s2m_a_t`,
  `axi_m2s_r_t`/`axi_s2m_r_t`, `axi_max_burst_length_beats`, `to_len`,
  `to_size`, `axi_a_burst_incr`, `axi_resp_okay`.
- `axi_stream.axi_stream_pkg` (`hdl-modules/modules/axi_stream/src/axi_stream_pkg.vhd`) —
  `axi_stream_m2s_t`/`axi_stream_s2m_t`.
- Reused entities (direct instantiation, no new datapath RTL for
  AR/R throttling or elasticity): `axi.axi_read_throttle`,
  `axi.axi_read_pipeline`, `axi.axi_r_fifo`, `axi_stream.axi_stream_fifo`.

## Implementation notes

- Internally, one bundled `axi_read_m2s_t`/`axi_read_s2m_t` pair (AR+R
  together) runs FSM -> `axi_read_throttle` -> `axi_read_pipeline`, only
  split into the four `m_axi_ar_*`/`m_axi_r_*` ports at the entity
  boundary — no behavior is added by the split.
- **`axi_r_fifo`/`axi_read_throttle.data_fifo_depth` sizing: `2 *
  axi_max_burst_length_beats` (512), not `1x`.** `axi_read_throttle`'s
  own `block_address_transactions` gating blocks issuing a new `AR`
  whenever the *current* burst's length is `>=` the FIFO's
  empty-and-not-yet-negotiated space; a FIFO sized to exactly one
  max-length burst (256) deadlocks the moment a full 256-beat burst is
  negotiated but not yet drained (its own empty space drops to exactly
  0, blocking that same burst's own onward `AR` forever). Matches the
  ratio used by `axi_read_throttle`'s own reference testbench
  (`hdl-modules/modules/axi/test/tb_axi_read_throttle.vhd`,
  `data_fifo_depth := 2 * max_burst_length_beats`). Found via the
  `test_multi_burst_over_256_beats` corner case, which is the only test
  case that drives a full 256-beat burst.
- `axi_stream_fifo` depth is a local constant (`16`), not a generic —
  the AXI-side elasticity is already provided by `axi_r_fifo`; this
  stage only decouples R-side pop timing from stream-side push timing
  and absorbs short consumer stalls.
- `ARID`/`RID` hardwired to `(others => '0')`, a local constant, not
  driven from `g_axi_id_width` (which only sizes the field).
- `stale_beats_q`/`DRAIN_STALE` (see "Clocking and reset") is the only
  non-obvious piece of state carried across a `reset` — every other
  register is fully re-latched from the next accepted request.

## Verification notes

See `modules/cnn_accel/test/tb_cnn_accel_axi_read_dma.vhd`
(`tb_cnn_accel_axi_read_dma`/architecture `tb`) and
`doc/cnn_accel_axi_read_dma_proposal.md` section 9/10 for the full test
list. `bfm.axi_read_slave` terminates `m_axi_ar`/`m_axi_r` against the
VUnit memory model (randomized address/data stall probabilities and
`RRESP` injection); `bfm.axi_stream_slave`/a hand-rolled `queue_t`-driven
checker validates `m_stream_m2s`/`s2m` against reference data.

Test cases (all pass, GHDL, VUnit 5): `test_single_short_burst`,
`test_multi_burst_over_256_beats` (> 256 beats, forces the
`axi_r_fifo` depth issue above), `test_4k_boundary_split`,
`test_backpressure` (randomized `m_stream_s2m.ready` stalls),
`test_resp_error` (non-`OKAY` `RRESP` injection, exactly one
`resp_error` pulse aligned with `dma_done`), `test_reset_mid_transfer`
(reset mid-burst, then a fresh request served correctly — exercises
`stale_beats_q`/`DRAIN_STALE`), `test_zero_length`.

Run with:
```
python3 run.py -o vunit_out_axi_read_dma 'cnn_accel.tb_cnn_accel_axi_read_dma*' -p 8
```
