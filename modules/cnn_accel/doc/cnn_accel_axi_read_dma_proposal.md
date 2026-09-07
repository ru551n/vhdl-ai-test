# cnn_accel_axi_read_dma — vhdesign proposal

Input: `modules/cnn_accel/doc/cnn_accel_axi_read_dma_req.md`.
Related: `doc/cnn_accel_arch.md` ("Type policy", "Interface record policy",
"Reset policy"), `shared/Axi4.md` rules 6-19 (AR/R only) and 20-26
(AXI4-Stream), `shared/ReusableRTL.md` ("Reuse before authoring new RTL").

## 1. Requirements summary

Generic AXI4 read master. Accepts a `cnn_accel_pkg.dma_req_t`
(`addr`/`length`, both bytes), splits it into one or more `AR` bursts that
respect the 4 KiB burst-boundary rule (`shared/Axi4.md` rule 7) and the
256-beat `ARLEN` limit (rule 6), issues them back-to-back through
`axi.axi_read_pipeline`/`axi.axi_read_throttle`, and republishes every
accepted `R` beat as an internal AXI4-Stream (`axi_stream_pkg` records,
`data` width = `g_axi_data_width`) via an `axi_stream_fifo` elasticity
stage. `dma_done` pulses once the whole request has been streamed out to
the consumer; `resp_error` pulses once (at the same time as `dma_done`) if
any `RRESP` in the request was not `OKAY`. Instantiated three times
(instruction fetch, weight/bias fetch, ifmap fetch) with the same
entity/architecture — differentiation between the three is by physical
AXI master port at `cnn_accel_top`/the crossbar, not by `ARID` value.

## 2. Interface (as given by the requirement, one direction defect fixed below)

| Generic | Type | Purpose |
|---|---|---|
| `g_axi_addr_width` | positive | meaningful low bits of the fixed-max-width `axi_pkg` address fields |
| `g_axi_data_width` | positive | AXI `RDATA` width and the internal/`m_stream` AXI4-Stream `data` width |
| `g_axi_id_width` | natural | width of the (fixed-value) `ARID`/`RID` field; the value itself is a hardwired `0`, see §3.1 |

| Port | Dir | Type | Purpose |
|---|---|---|---|
| `clk` | in | `std_ulogic` | |
| `reset` | in | `std_ulogic` | `= reset_internal`, synchronous active-high |
| `req_m2s` | in | `cnn_accel_pkg.dma_req_m2s_t` | `valid`, `req.addr`, `req.length` |
| `req_s2m` | out | `cnn_accel_pkg.dma_req_s2m_t` | `ready` |
| `dma_done` | out | `std_ulogic` | one-cycle pulse |
| `resp_error` | out | `std_ulogic` | one-cycle pulse |
| `m_axi_ar_m2s` | out | `axi_pkg.axi_m2s_a_t` | AR channel, to `axi.axi_simple_read_crossbar` |
| `m_axi_ar_s2m` | in | `axi_pkg.axi_s2m_a_t` | AR channel |
| `m_axi_r_m2s` | out | `axi_pkg.axi_m2s_r_t` | R channel (`RREADY`) — direction fixed, see §3.2 |
| `m_axi_r_s2m` | in | `axi_pkg.axi_s2m_r_t` | R channel (`RVALID`/`RDATA`/`RRESP`/`RLAST`) — direction fixed, see §3.2 |
| `m_stream_m2s` | out | `axi_stream_pkg.axi_stream_m2s_t` | to the consumer, through an `axi_stream_fifo` elasticity stage |
| `m_stream_s2m` | in | `axi_stream_pkg.axi_stream_s2m_t` | |

## 3. Design decisions (resolving requirement ambiguities)

### 3.1 `ARID`/`RID` is a hardwired constant, not a generic

Per the requirement's own generics-table note, the fixed `ARID` value used
by an instantiation site is *not* a top-level generic. `axi.
axi_simple_read_crossbar` (the documented downstream consumer of
`m_axi_ar`/`m_axi_r`) differentiates its input ports physically — it locks
onto one input port for the duration of that port's whole burst (AR
through the matching R beats) before arbitrating again — so no two
instances' traffic is ever interleaved at the crossbar regardless of the
`ARID` value each one drives. Resolution: this entity drives
`m_axi_ar_m2s.id <= (others => '0')` unconditionally (a local constant,
`g_axi_id_width` only sizes the field), identical at all three
instantiation sites. `shared/Axi4.md` rule 13's "single read ID"
simplification is satisfied trivially per instance; `RID` on the way back
is never inspected.

### 3.2 `m_axi_r_m2s`/`s2m` port-direction defect

The requirement's port table lists `in/out` for `m_axi_r_m2s`/`s2m`, but
that is inverted relative both to this module's own master role and to
the `m_axi_ar_m2s`/`s2m` row's own `out/in` pattern: for an AXI4 read
master, `RREADY` (`axi_m2s_r_t.ready`) is an output and
`RVALID`/`RDATA`/`RRESP`/`RLAST` (`axi_s2m_r_t`) are inputs, exactly
mirroring the AR row. Treated as a documentation defect in the
(non-hand-owned) port table, not part of the preserved Functional
Description. Resolution: `m_axi_r_m2s : out axi_pkg.axi_m2s_r_t`,
`m_axi_r_s2m : in axi_pkg.axi_s2m_r_t`.

### 3.3 Split AR/R ports vs. `axi_read_pipeline`/`axi_read_throttle`'s bundled ports

`axi.axi_read_pipeline` and `axi.axi_read_throttle` both use one bundled
`axi_read_m2s_t`/`axi_read_s2m_t` port pair (AR and R together), matching
`axi.axi_simple_read_crossbar`'s own bundled `axi_read_m2s_vec_t`/
`axi_read_s2m_vec_t` port. Internally this entity keeps one bundled
`axi_read_m2s_t`/`axi_read_s2m_t` pair end to end (FSM output ->
`axi_read_throttle` -> `axi_read_pipeline`) and only splits it into the
four `m_axi_ar_*`/`m_axi_r_*` ports (per §2) at the entity boundary, by
assigning/reading the `.ar`/`.r` sub-fields — no behavior is added by the
split, it is a boundary-only relabeling to match the requirement's port
table.

### 3.4 Word-aligned `addr`/`length` assumption

The requirement does not state an alignment requirement for
`req_m2s.req.addr`/`.length`. Every caller (`cnn_accel_sequencer`,
`cnn_accel_layer_ctrl`) is internal to this IP and known to only ever
request whole-beat ranges. Design assumption, to be honored by every
caller: `addr` and `length` are both multiples of
`c_bytes_per_beat = g_axi_data_width / 8`. This lets the beat-count
computed once at request-accept time (`total_beats = length /
c_bytes_per_beat`) be exact, and removes any need for narrow/unaligned
first/last-beat handling (`shared/Axi4.md` rule 11) inside this module.
`length = 0` is accepted as a degenerate, immediate-`dma_done` case (no
`AR` issued at all) — see §3.8.

### 3.5 Burst split uses total-beat counting, not per-burst `last` tagging

Because a single `ARID` is used per instance and `axi.
axi_simple_read_crossbar` locks a port for the duration of its whole
burst, `R` beats for this request are guaranteed to arrive in the same
order the `AR`s were issued (`shared/Axi4.md` rule 13) — no per-burst
bookkeeping is needed to know which `R` beat is the final one of the
whole request. The R-side beat counter (§6, `r_beat_count_q`) simply
counts beats 0 .. `total_beats - 1` computed once at request-accept time;
the stream `last` is asserted exactly on beat `total_beats - 1`,
independently of how many `AR` bursts that beat's data happened to travel
in.

### 3.6 Internal R-side elasticity FIFO and `axi_read_throttle` wiring

Per the requirement text ("limits outstanding beats so the consumer's
`axi_stream_s2m.ready` backpressure is honored without overrunning an
internal beat-count FIFO"), `axi.axi_read_throttle`'s `data_fifo_level`
input must be driven by a real FIFO that actually holds the outstanding
`R` beats. `axi.axi_r_fifo` (same `axi` library, already packs
`axi_s2m_r_t`/tracks a level — see `hdl-modules/modules/axi/test/
tb_axi_read_throttle.vhd` for the same wiring pattern) is reused for
this, sized to `c_r_fifo_depth = axi_pkg.axi_max_burst_length_beats`
(256) — one full max-length burst always fits, which is sufficient for
`axi_read_throttle` to never let outstanding beats exceed the FIFO's
capacity. This does not double-buffer across bursts (no burst N+1 AR is
issued while burst N is still fully in the FIFO); given `axi.
axi_simple_read_crossbar`'s own port-locking already serializes bursts
from different instances, full cross-burst pipelining inside one instance
is a possible future optimization, not required by the requirement's
"issue bursts back-to-back" (which only requires no *unnecessary* gap,
not overlap). `axi_read_throttle`'s `full_ar_throughput` generic is fixed
`true` (favor throughput over logic at this IP's scale).

`axi.axi_r_fifo`'s *input* side (the consumer-facing side, `input_m2s.
ready` in / `input_s2m` data out) is popped by the stream-conversion logic
in §6 whenever the downstream `axi_stream_fifo`'s `input_s2m.ready` is
`'1'`, which is exactly how `m_stream_s2m` backpressure propagates back
through to `axi_read_throttle` and ultimately gates new `AR`s.

### 3.7 Output elasticity FIFO sizing

`axi_stream_fifo` (per the requirement's own ports-table note) sits
between the stream-conversion logic and `m_stream_m2s`/`s2m`. Sized to
`c_stream_fifo_depth = 16` (a local constant, not exposed as a generic):
the AXI-side elasticity is already provided by `c_r_fifo_depth`, so this
stage only needs to decouple the two clocked domains of logic (R-side
pop vs. stream-side push) and absorb short consumer stalls without
forcing `axi.axi_r_fifo` to back up; 16 is a small, arbitrary-but-ample
choice given `g_axi_data_width`-wide entries.

### 3.8 `dma_done`/`resp_error` timing and the `length = 0` case

"Fully streamed out" (requirement's own wording) is read literally as
the boundary named in the ports table — `m_stream_m2s`/`s2m`, i.e. the
actual hand-off to the consumer, not an internal buffering stage.
`dma_done` pulses the cycle after `m_stream_m2s.valid = '1'` and
`.last = '1'` is accepted (`m_stream_s2m.ready = '1'`). `resp_error` is
latched (`err_pending_q`) the cycle any popped `R` beat's `resp /=
axi_resp_okay` (the burst is still drained per rule 15 — `err_pending_q`
does not stop the R-side counter/pop logic) and pulses on the same cycle
as `dma_done`, then clears. `req_s2m.ready` is `'0'` for the whole
request, including while the final beats drain out of the elasticity
FIFO, and returns to `'1'` only after `dma_done` (or after `reset`, see
§4) — this is also what "instantiated once per request, one request at a
time" implies structurally (no request pipelining across `dma_done`
boundaries is required or provided).

`length = 0`: accepted with no `AR` issued at all (`total_beats = 0`);
`dma_done` pulses one cycle after accept, `resp_error` stays `'0'`.

## 4. Clock/reset

Synchronous active-high `reset` (`= reset_internal` per `doc/
cnn_accel_arch.md` "Reset policy"). On `reset = '1'`: the top request
FSM (§6) returns to `IDLE` (so `req_s2m.ready` becomes `'1'` again next
cycle, satisfying the requirement's "must be abortable... without leaving
a stale outstanding-beat count or a stuck `req_s2m.ready = '0'`"),
`err_pending_q` clears, and the R-side beat counter/AR-side byte-remaining
registers are irrelevant once `IDLE` is re-entered (their next use always
re-latches from a fresh `req_m2s`). `axi.axi_read_pipeline`, `axi.
axi_read_throttle`, `axi.axi_r_fifo` and `axi_stream_fifo` are all plain
`clk`-only submodules (no `reset` port) — any beats already in flight
inside them at the moment of `reset` are simply discarded: `m_axi_ar_m2s.
valid`/`m_axi_r_m2s.ready` are forced low the cycle after `reset` by the
FSM re-entering `IDLE` and not driving a new `AR`/accepting further `R`
data, and no `m_stream_m2s.valid` is generated for a discarded, abandoned
request. This matches `shared/Axi4.md` rule 18 ("after reset, no in-flight
transaction may be assumed; pending handshakes are dropped").

## 5. Architecture and dataflow

```
req_m2s/s2m --> [top request FSM] --ar_gen_m2s/s2m (axi_read_m2s_t/s2m_t, bundled AR+R)-->
  axi_read_throttle --throttled_m2s/s2m--> axi_read_pipeline --right_m2s/s2m-->
  (split .ar/.r) --> m_axi_ar_m2s/s2m, m_axi_r_m2s/s2m

R data return path (ar_gen_s2m.r, after throttle+pipeline):
  --> axi_r_fifo (output_s2m <= ar_gen_s2m.r; output_m2s.ready --> ar_gen_m2s.r.ready)
      --level--> axi_read_throttle.data_fifo_level
  axi_r_fifo.input_s2m (popped data) --> [stream-conversion: last = (r_beat_count_q =
    total_beats_q - 1); latch err_pending_q on resp /= OKAY] --> axi_stream_fifo.input_m2s/s2m
      --> axi_stream_fifo.output_m2s/s2m == m_stream_m2s/s2m
  axi_r_fifo.input_m2s.ready <= axi_stream_fifo.input_s2m.ready (backpressure passthrough)
```

Two concurrently-running pieces of state, both gated by the single top
FSM being outside `IDLE` (i.e. a request is active):

- **AR-issue**: splits the latched `req.addr`/`.length` into successive
  bursts (§6.2) and drives `ar_gen_m2s.ar` until every burst for this
  request has been accepted.
- **R-consume**: independently pops `axi_r_fifo` (§6.3), counts beats,
  tags `last`, latches `err_pending_q`, and feeds `axi_stream_fifo`.

`req_s2m.ready` is asserted only in the top FSM's `IDLE` state, i.e. once
R-consume for the *previous* request has fully drained through
`m_stream`.

## 6. State machines

### 6.1 Top request FSM

| State | `req_s2m.ready` | Transition |
|---|---|---|
| `IDLE` | `'1'` | on `req_m2s.valid`: latch `addr_q`/`bytes_remaining_q := req.length`, `total_beats_q := length / c_bytes_per_beat`, `r_beat_count_q := 0`, `err_pending_q := '0'`; if `length = 0` go straight to `DONE_PULSE`, else -> `ACTIVE` |
| `ACTIVE` | `'0'` | AR-issue and R-consume run concurrently (§6.2/§6.3); when R-consume's `r_beat_count_q = total_beats_q` (all beats popped and pushed into `axi_stream_fifo`) -> `WAIT_DRAIN` |
| `WAIT_DRAIN` | `'0'` | wait until the beat carrying `last = '1'` is accepted at `m_stream_s2m` (i.e. leaves `axi_stream_fifo`) -> `DONE_PULSE` |
| `DONE_PULSE` | `'0'` | one cycle: `dma_done <= '1'`, `resp_error <= err_pending_q` -> `IDLE` |

(`length = 0` skips `ACTIVE`/`WAIT_DRAIN` entirely — no beat is ever
generated, so there is nothing to wait on.)

### 6.2 AR-issue sub-logic (runs while `ACTIVE`, until its own bursts are all issued)

Per burst, while `bytes_remaining_q /= 0`:

1. `bytes_to_4k_c = 4096 - addr_q(11 downto 0)` (or `4096` if
   `addr_q(11 downto 0) = 0`).
2. `burst_bytes_c = min(bytes_remaining_q, bytes_to_4k_c,
   axi_max_burst_length_beats * c_bytes_per_beat)`.
3. `burst_beats_c = burst_bytes_c / c_bytes_per_beat`.
4. Drive `ar_gen_m2s.ar = (valid => '1', id => 0, addr => addr_q, len =>
   to_len(burst_beats_c), size => to_size(g_axi_data_width), burst =>
   axi_a_burst_incr)`; hold until `ar_gen_s2m.ar.ready = '1'`.
5. On acceptance: `addr_q += burst_bytes_c`; `bytes_remaining_q -=
   burst_bytes_c`.

This sub-logic idles (no `AR` driven) once `bytes_remaining_q = 0`, even
though R-consume may still be draining earlier bursts' data.

### 6.3 R-consume sub-logic (runs while `ACTIVE`, until `r_beat_count_q = total_beats_q`)

Combinational: `axi_r_fifo.input_m2s.ready <= axi_stream_fifo_input_s2m.
ready`. On `axi_r_fifo.input_s2m.valid = '1' and axi_r_fifo.input_m2s.
ready = '1'`:

- `axi_stream_fifo_input_m2s <= (valid => '1', data => axi_r_fifo.
  input_s2m.data(g_axi_data_width - 1 downto 0), last => (r_beat_count_q
  = total_beats_q - 1))`.
- if `axi_r_fifo.input_s2m.resp /= axi_resp_okay`: `err_pending_q <=
  '1'`.
- `r_beat_count_q <= r_beat_count_q + 1`.

## 7. Numeric types and widths

- `addr_q`, `bytes_remaining_q`: `unsigned(31 downto 0)` (matches
  `cnn_accel_pkg.dma_req_t.addr`/`.length`).
- `total_beats_q`, `r_beat_count_q`: `unsigned(31 downto 0)` (a beat
  count can exceed 255 across multiple bursts even though each burst's
  own `ARLEN` is `<= 255`).
- `bytes_to_4k_c`, `burst_bytes_c`, `burst_beats_c`: local `unsigned`
  variables/signals sized to comfortably hold `4096`
  (`unsigned(12 downto 0)` minimum; widened to `unsigned(31 downto 0)`
  for the `min()` comparison against `bytes_remaining_q`).
- `axi_pkg.to_len`/`to_size`/`axi_a_burst_incr` used for the `AR` fields;
  no hand-rolled `AxLEN`/`AxSIZE` encoding.
- `ieee.numeric_std` throughout; no `std_logic_arith`/`_unsigned`/
  `_signed`.

## 8. Latency/throughput

One outstanding burst at a time (§3.6); within a burst, `axi_read_pipeline`
/`axi_read_throttle` allow a transaction every cycle
(`full_ar_throughput`/`full_data_throughput = true`). `axi_r_fifo` and
`axi_stream_fifo` each add up to one cycle of latency on their pop-side
combinational passthrough plus whatever depth-driven latency their own
internal `fifo.fifo` instance has; not a concern for this IP's throughput
budget (DMA read is not the pipeline's critical path per `doc/
cnn_accel_arch.md`).

## 9. Corner cases covered by the verification plan (§10)

- Single short burst (< 256 beats, no 4 KiB crossing).
- Request > 256 beats: multiple `AR`s, single logical stream, `last` only
  on the true final beat.
- Request straddling a 4 KiB boundary at a beat count that is not a
  multiple of 256: split driven by `bytes_to_4k_c`, not just `ARLEN`.
- Consumer backpressure (`m_stream_s2m.ready` random stalls): no beat
  loss/duplication, `last` on the correct beat regardless of stall
  pattern (rules 20-22).
- Non-`OKAY` `RRESP` mid-burst: burst still fully drains (rule 15);
  exactly one `resp_error` pulse, aligned with `dma_done`.
- `dma_done` fires exactly once per request, never for a request that
  never completes (e.g. abandoned by `reset`).
- Reset mid-transfer: `req_s2m.ready` returns to `'1'` promptly after
  `reset` deasserts and a fresh request is accepted correctly (no stale
  `bytes_remaining_q`/`r_beat_count_q` leaking into the new request).
- `length = 0`.

## 10. Verification plan

VUnit-5 self-checking testbench, `tb_cnn_accel_axi_read_dma.vhd`,
`bfm.axi_read_slave` terminating `m_axi_ar`/`m_axi_r` against the VUnit
memory model (random `address_stall_probability`/`data_stall_probability`
and randomized `RRESP` injection for the error case), a `queue_t`-driven
reference-data checker process on `m_stream_m2s`/`s2m` (manual
push/check rather than `check_axi_stream` — see the narrow-`TDATA`
gotcha when `g_axi_data_width` is not tested at exactly a multiple of 8
per beat is a non-issue here since `g_axi_data_width` is always a byte
multiple, but the checker is still hand-rolled to keep `last`-position
and per-beat data checks in one place), and a `run.py` test per §9's
corner case plus the mandatory list from the task brief (single short
burst; >256-beat request; 4 KiB-straddling request; randomized consumer
backpressure; one non-`OKAY` `RRESP` per request producing exactly one
`resp_error` pulse; `dma_done` exactly once; reset mid-transfer followed
by a fresh, correctly-served request).

## Implementation Notes (vhfill)

(empty — to be filled during `vhfill`)
