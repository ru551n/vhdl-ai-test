# cnn_accel_ofmap_dma

## Purpose

Output-activation write-back DMA: a thin wrapper around hdl-modules'
`dma_axi_write_simple.dma_axi_write_simple` (the plain, non-AXI-Lite
entity, reused unmodified) that adapts its native continuously-running
ring-buffer register interface to this IP's one-shot
`cnn_accel_pkg.dma_req_t` (byte `addr`/`length`) control plane. Accepts
one AXI4-Stream of already-AXI-width int8 output-activation beats
(`s_stream_m2s`/`s2m`, from the opcode-converged output mux) and writes
exactly `length` bytes starting at `addr` to DDR through
`m_axi_aw`/`w`/`b` (single writer, no arbitration, direct to the
top-level `m_axi`). `dma_done` pulses once the full length has been
written (last `BRESP` observed); `resp_error` pulses on the same cycle
if any `BRESP` in the request was not `OKAY` (the request is still
"fully attempted" regardless). Because the wrapped core is forced onto
its single-AXI-beat-packet optimized path (`packet_length_beats => 1`),
every accepted stream beat becomes its own one-beat `AW`/`W` burst — the
smallest legal packet size, not "AXI bursts of the maximum length
possible"; see "Implementation notes" for the trade-off this buys.
Intended to be instantiated once, as the IP's ofmap write-back master
(mirrors `cnn_accel_axi_read_dma`'s role on the read side, same
`dma_req`/`dma_done`/`resp_error` contract and abort rationale).

## Entity and architecture

Entity `cnn_accel_ofmap_dma`, architecture `a`.

## Generics

| Generic | Type | Meaning / constraints |
|---|---|---|
| `g_axi_addr_width` | `positive` | Passed straight through as the wrapped core's `address_width` generic; sizes the internal segment/pointer address arithmetic inside `dma_axi_write_simple`/`ring_buffer_write_simple`. |
| `g_axi_data_width` | `positive` | AXI `WDATA` width and the `s_stream_m2s`/`s2m` `data` width consumed (only the low `g_axi_data_width` bits of the fixed-width `axi_stream_m2s_t.data` field are used). Passed to the wrapped core as *both* `stream_data_width` and `axi_data_width` (forced equal — see "Implementation notes"). Constrained by the wrapped core's own generic to `axi_pkg.axi_data_width_t` (an AXI-legal power-of-two width), which is what makes `c_bytes_to_beats_shift = log2(g_axi_data_width/8)` exact. `req_m2s.req.addr`/`.length` must both be multiples of `g_axi_data_width / 8` (see "Interfaces/protocols"). |

Unlike `cnn_accel_axi_read_dma`, there is no `g_axi_id_width` generic
here: `m_axi_aw_m2s`'s `id` field is whatever the wrapped
`dma_axi_write_simple` core drives internally — this wrapper never
reads or overrides it.

## Ports

| Port | Dir | Type | Description |
|---|---|---|---|
| `clk` | in | `std_ulogic` | Clock. |
| `reset` | in | `std_ulogic` | Synchronous active-high reset (`reset_internal` at the IP top level). Default `'0'`. |
| `req_m2s` | in | `cnn_accel_pkg.dma_req_m2s_t` | `valid`, `req.addr`/`req.length` (both `unsigned(31 downto 0)`, bytes). |
| `req_s2m` | out | `cnn_accel_pkg.dma_req_s2m_t` | `ready`. High only while idle (see FSM below). |
| `dma_done` | out | `std_ulogic` | One-cycle pulse, once per accepted request, on the cycle of the `B` handshake that completes it (or, for `length = 0`, one cycle after accept). Default `'0'`. |
| `resp_error` | out | `std_ulogic` | One-cycle pulse, same cycle as `dma_done`, if any `BRESP` in the request was not `OKAY`. Default `'0'`. |
| `s_stream_m2s` | in | `axi_stream_pkg.axi_stream_m2s_t` | From the producer (opcode-converged output mux). `data(g_axi_data_width - 1 downto 0)` holds one payload beat; no `last` handling (byte count comes from `length`, not stream framing). |
| `s_stream_s2m` | out | `axi_stream_pkg.axi_stream_s2m_t` | `ready` wired straight through from the wrapped core's own `stream_ready` — no elasticity buffering added by this wrapper. |
| `m_axi_aw_m2s` | out | `axi_pkg.axi_m2s_a_t` | AW channel (`AWVALID`/`AWADDR`/`AWLEN`/`AWSIZE`/`AWBURST`/`AWID`), pure field fan-out from the wrapped core's `axi_write_m2s.aw`. |
| `m_axi_aw_s2m` | in | `axi_pkg.axi_s2m_a_t` | AW channel (`AWREADY`). |
| `m_axi_w_m2s` | out | `axi_pkg.axi_m2s_w_t` | W channel (`WVALID`/`WDATA`/`WSTRB`/`WLAST`), pure field fan-out. |
| `m_axi_w_s2m` | in | `axi_pkg.axi_s2m_w_t` | W channel (`WREADY`). |
| `m_axi_b_m2s` | out | `axi_pkg.axi_m2s_b_t` | B channel (`BREADY`), tied high by the wrapped core; also snooped by this wrapper for `aw_issued_q`/`bresp_acked_q`/`outstanding_q`/`error_latched_q`. |
| `m_axi_b_s2m` | in | `axi_pkg.axi_s2m_b_t` | B channel (`BVALID`/`BRESP`/`BID`). |

## Clocking and reset

Single clock domain (`clk`). Synchronous, active-high `reset`. The
wrapped `dma_axi_write_simple`/`ring_buffer_write_simple` core has **no
reset port at all** (hdl-modules resetless-by-default convention,
verified by grep across both sources — zero matches for "reset"). This
wrapper cannot reset the wrapped core's internals directly and never
asks it to abandon an already-asserted `AW`/`W` transaction (a basic
AXI4 rule, not a wrapped-module limitation — a master may not withdraw
`AWVALID`/`WVALID` before the handshake). Instead:

- Every register that `req_s2m.ready`/`dma_done`/`resp_error` depend on
  (`state_q`, `addr_q`, `length_q`, `expected_beats_q`, `aw_issued_q`,
  `bresp_acked_q`, `error_latched_q`) is owned by this wrapper and is
  synchronously cleared on `reset = '1'`.
- `outstanding_q` (`unsigned(7 downto 0)`) — the count of `AW`-accepted-
  but-`B`-not-yet-seen transactions — is **deliberately not cleared by
  `reset`**, updated unconditionally every cycle (including during
  `reset`) from `aw_handshake_i`/`b_handshake_i`. Clearing it on `reset`
  would make the wrapper falsely believe the bus was already quiet.
- On `reset = '1'`: if `outstanding_q`'s next value (computed the same
  cycle, accounting for any `AW`/`B` handshake on the reset cycle
  itself) is `0`, `state_q` goes straight to `s_idle` (`req_s2m.ready`
  high again the next cycle). Otherwise `state_q` goes to `s_drain`:
  `req_s2m.ready` held `'0'`, `enable_i` held `'0'` (no new segment can
  be requested from the ring buffer), and the FSM waits for
  `outstanding_next_i = 0` — i.e. for every already-issued transaction's
  `BRESP` to be observed (guaranteed in bounded time since the wrapped
  core ties its own `axi_write_m2s.b.ready` high) — before returning to
  `s_idle`.
- A normal (non-aborted) completion always reaches `outstanding_q = 0`
  at the same cycle `bresp_acked_q` reaches `expected_beats_q`, so
  `s_drain` is only ever entered on an aborted (mid-request) `reset`.

Exercised by `test_abort_mid_request_drains_before_ready_returns` and
`test_abort_with_zero_outstanding_returns_ready_next_cycle`.

## Interfaces/protocols

- `req_m2s`/`req_s2m`: single-beat request handshake, `ready`-qualified,
  accepted only while idle (one request in flight at a time; no request
  pipelining across `dma_done` boundaries).
- `m_axi_aw_*`/`m_axi_w_*`/`m_axi_b_*`: standard AXI4 AW/W/B master,
  full backpressure. Pure record field fan-out/fan-in from/to the
  wrapped core's bundled `axi_write_m2s`/`s2m` — no logic added on this
  path at all.
- `s_stream_m2s`/`s2m`: AXI4-Stream consumer port, full backpressure via
  the wrapped core's own `stream_ready` (itself forced low by the ring
  buffer whenever `enable = '0'`, i.e. during `s_idle`/`s_drain` and
  once a request's segment budget is exhausted — see "Functional
  behavior"). No `last`/packet framing is consumed; exactly one stream
  beat is expected per `g_axi_data_width`-sized chunk of `length`.
- **Word-aligned assumption**: `req_m2s.req.addr`/`.length` must both be
  multiples of `g_axi_data_width / 8`. This is inherent to the wrapped
  `dma_axi_write_simple` core, not an extra constraint invented by this
  wrapper: it always drives a full-width `WSTRB` and has no
  partial-packet flush at any `packet_length_beats` value, for *any*
  packet size — choosing the minimum packet size (one AXI beat, see
  "Implementation notes") makes this the smallest achievable version of
  that constraint. **An unaligned tail is not detected or reported**: an
  under-length final chunk would simply never complete, and `dma_done`
  would never fire.

  **Guaranteed statically since decision D1 (2026-09-07).** This is no
  longer an open question: a concurrent elaboration assert in
  `cnn_accel_ofmap_dma.vhd` bounds `g_axi_data_width` at
  `cnn_accel_constant_max_axi_data_width` (64 bits = 8 bytes/beat,
  generated from `cnn_accel_constants.MAX_AXI_DATA_WIDTH`). Under decision
  S6 every activation request is one whole channel-tiled plane, so `addr`
  and `length` are always multiples of `T = 8` bytes; a bus of at most 8
  bytes per beat is therefore aligned **unconditionally, for every layer
  geometry**, with no runtime check and no host-side contract. Above that
  width alignment would depend on runtime descriptor fields
  (`out_width * out_height` even, base address aligned), which cannot be
  checked at elaboration at all — hence a hard bound rather than a weaker
  guarantee. The assert is `severity failure`: exceeding it is a hang, not
  a suboptimal configuration.

  `length = 0` is accepted as a degenerate case
  (`s_zero_len` state): no `AW`/`W` activity at all, and `dma_done`
  pulses one cycle after accept (`resp_error` stays `'0'`).

## Functional behavior

Top FSM: `s_idle -> [s_drain] -> s_active -> s_idle`, with a dedicated
one-cycle `s_zero_len` branch out of `s_idle` for `length = 0`. On
accepting a request in `s_idle`, `addr_q`/`length_q` are latched from
`req.addr`/`.length`, `expected_beats_q = length >> log2(g_axi_data_width
/ 8)` is computed, and `aw_issued_q`/`bresp_acked_q`/`error_latched_q`
reset to `0`. If `expected_beats_q = 0` the FSM goes to `s_zero_len` (one
cycle, pulses `dma_done`, returns to `s_idle`); otherwise it goes to
`s_active`.

**Ring-buffer register-plane adaptation** (drives the wrapped core's
`regs_down`, degenerate single-shot mapping, one request at a time):

- `buffer_start_address <= addr_q`.
- `buffer_end_address <= addr_q + length_q + c_axi_data_width_bytes`
  (one extra segment/beat of padding beyond `addr + length`).
- `buffer_read_address <= buffer_start_address` (kept equal to the
  start address at all times, never advanced).
- `config.enable <= '1' when state_q = s_active and aw_issued_q <
  expected_beats_q else '0'` — a combinational, self-limiting pulse for
  exactly this request's duration; it also stops the ring buffer from
  ever requesting a segment beyond this request's length (the ring
  buffer itself has no other notion of "this region is full", since
  `buffer_read_address` never moves).
- `interrupt_mask`/`interrupt_status` tied to their `_init` (all-zero)
  values; `regs_up`/`interrupt` are connected but never read anywhere in
  this architecture. Completion/error tracking taps `m_axi_b_*`
  directly instead (see below), since `packet_length_beats => 1` makes
  "one packet accepted" and "one AXI beat's `BRESP` received" the same
  event.

**Completion/error accounting** (all in this wrapper's own resettable
state): `aw_issued_q` counts `AW` handshakes since the request was
accepted; `bresp_acked_q` counts `B` handshakes since the same point,
separately from `aw_issued_q` because `dma_done` must mean fully
*written* (response received), not merely fully issued. On a `B`
handshake, `error_latched_q` is set (and stays set) if `BRESP /=
axi_resp_okay`; when `bresp_acked_q + 1 = expected_beats_q`, the FSM
returns to `s_idle` and, on that same cycle, `dma_done` pulses and
`resp_error` pulses if `error_latched_q` was already set *or* this final
beat's own `BRESP` was non-`OKAY`.

`outstanding_q` (see "Clocking and reset") is updated unconditionally
every cycle from `aw_handshake_i`/`b_handshake_i`, independent of
`state_q`.

## Timing/latency

No AR/AW pipelining stage or elasticity FIFO is added on the write
datapath by this wrapper itself — `m_axi_aw`/`w`/`b` are pure
combinational field fan-out/fan-in to/from the wrapped
`dma_axi_write_simple` core's own `axi_write_m2s`/`s2m`, and
`s_stream_s2m.ready` is the wrapped core's own `stream_ready` with no
extra buffering. Because `packet_length_beats => 1`, every accepted
stream beat becomes its own single-beat `AW`+`W` burst inside the
wrapped core (no multi-beat accumulation FSM elaborated). `dma_done`/
`resp_error` are combinational on the completing `B` handshake itself
(same cycle, not a cycle later) — this differs from
`cnn_accel_axi_read_dma`'s registered `DONE_PULSE` state, which pulses
one cycle after its analogous completion event.

## Registers/configuration

None (no CSR-mapped registers in this module). The wrapped core's
`regs_down`/`regs_up` are plain, un-latched record ports driven directly
by this wrapper's own FSM every cycle — not a host-visible register
file (that would be `dma_axi_write_simple_axi_lite`, not instantiated
here).

## Dependencies

- `ieee.std_logic_1164`, `ieee.numeric_std`.
- `cnn_accel.cnn_accel_pkg` (`modules/cnn_accel/src/cnn_accel_pkg.vhd`) —
  `dma_req_t`/`dma_req_m2s_t`/`dma_req_s2m_t`.
- `axi.axi_pkg` (`hdl-modules/modules/axi/src/axi_pkg.vhd`) —
  `axi_write_m2s_t`/`axi_write_s2m_t`, `axi_m2s_a_t`/`axi_s2m_a_t`,
  `axi_m2s_w_t`/`axi_s2m_w_t`, `axi_m2s_b_t`/`axi_s2m_b_t`,
  `axi_resp_okay`, `_init` constants.
- `axi_stream.axi_stream_pkg`
  (`hdl-modules/modules/axi_stream/src/axi_stream_pkg.vhd`) —
  `axi_stream_m2s_t`/`axi_stream_s2m_t`.
- `math.math_pkg` — `log2` (used to derive `c_bytes_to_beats_shift`).
- `dma_axi_write_simple.dma_axi_write_simple_register_record_pkg` —
  `dma_axi_write_simple_regs_down_t`/`_up_t` and their `_init` constants.
- Reused entity (direct instantiation, no new datapath RTL for the
  AW/W/B bookkeeping or the ring-buffer pointer logic):
  `dma_axi_write_simple.dma_axi_write_simple` (the plain, non-AXI-Lite
  entity — `dma_axi_write_simple_axi_lite` is *not* used).

## Implementation notes

- **`packet_length_beats => 1`, `stream_data_width => axi_data_width =>
  g_axi_data_width`**: lands the wrapped core on its "optimized
  implementation for single-beat packets" generate branch (no
  persistent multi-beat accumulation FSM at all), and avoids elaborating
  its internal `common.width_conversion` block entirely (that block's
  beat-accumulation counter has no reset port — if a request were
  aborted mid-accumulation, the next request's first accumulated AXI
  beat could silently be corrupted by leftover partial bytes, with no
  way for this wrapper to observe or flush that counter). The trade-off,
  recorded as an open question in the proposal doc rather than resolved
  here: every beat becomes its own single-beat burst, far below "AXI
  bursts of the maximum length possible" — acceptable robustness-first
  choice for a first thin implementation, revisit with a real
  `packet_length_beats > 1` only if ofmap write bandwidth is measured to
  be a bottleneck.
- **`buffer_end_address`/`buffer_read_address` as actually wired**
  (`buffer_end_address <= addr + length + one_beat`,
  `buffer_read_address <= buffer_start_address`, both from the source
  comments and confirmed at `cnn_accel_ofmap_dma.vhd:148-168`) is *not*
  what the proposal doc's §3.2 originally specified
  (`buffer_end_address <= addr + length`, `buffer_read_address <= addr +
  length`). The as-built version is required by
  `ring_buffer_write_simple`'s own simulation-only assertion ("initial
  read address should be start address", checked on every `enable`
  rising edge) — `buffer_read_address` must equal
  `buffer_start_address`, not `addr + length`, or every request would
  fail that assertion. The one extra segment of padding on
  `buffer_end_address` is what still lets exactly `expected_beats_q`
  segments through before the ring buffer's own "never more than
  size-1 segments outstanding" self-block would otherwise kick in one
  segment early. **This is a real, deliberate deviation from the
  proposal's §3.2 as originally written, not a bug** — the entity's own
  header comment and inline comments explain the reasoning in full —
  but the proposal doc's own "Implementation Notes" section
  (`cnn_accel_ofmap_dma_proposal.md`, end of file) was left empty rather
  than updated to record the change; a reader of the proposal alone
  would be misled about what `buffer_read_address` is actually wired to.
- `outstanding_q` is a fixed `unsigned(7 downto 0)` (255 outstanding
  transactions), not generic-driven — a generous but unverified bound
  on realistic AXI interconnect/DDR-controller outstanding-transaction
  depth (flagged as an open question in the proposal doc, not a hard
  architectural limit).
- `expected_beats_q`/`aw_issued_q`/`bresp_acked_q` are all
  `unsigned(31 downto 0)`, sized to match `dma_req_t.length`'s width
  rather than a tightly-computed minimum (a `g_axi_data_width`-
  independent, always-safe choice for a first thin implementation, not
  yet revisited against a real resource measurement).
- No AXI ID generic/handling: `m_axi_aw_m2s.id` is whatever the wrapped
  core drives; this wrapper does not read, override, or expose it via a
  generic (unlike `cnn_accel_axi_read_dma`'s hardwired-`0`
  `g_axi_id_width`-sized `ARID`/`RID`).
- **No resource-footprint measurement is registered for this entity
  yet.** `module_cnn_accel.py` explicitly excludes both
  `cnn_accel_axi_read_dma` and `cnn_accel_ofmap_dma` from its
  per-entity Yosys netlist builds and from the Vivado-only
  `VivadoNetlistProject` registrations, noting both are "under
  concurrent development elsewhere, not yet registered here" — so,
  unlike several sibling `cnn_accel` entities, there is currently no
  "Measured" LUT/FF/BRAM/DSP comment to cite for this module.

## Verification notes

See `modules/cnn_accel/test/tb_cnn_accel_ofmap_dma.vhd`
(`tb_cnn_accel_ofmap_dma`/architecture `tb`) and
`doc/cnn_accel_ofmap_dma_proposal.md` section 8 for the full test list.
`bfm.axi_write_slave` terminates `m_axi_aw`/`w`/`b` against the VUnit
memory model; written data is checked byte-exactly with
`set_expected_word` + `check_expected_was_written`. Directed AXI
backpressure and held-back `BRESP`s come from switching the BFM's
per-channel stall probabilities between `0.0`/`1.0` at runtime. VUnit's
slave always answers `BRESP = OKAY`, so the `resp_error` test uses a
passive wire-level override of the `resp` field between the BFM and the
DUT for one chosen `B` beat — the BFM still owns every handshake and
data byte. The producer stream is driven directly by a small
`ready`-honoring procedure, since several tests interleave cycle-exact
checks between individual beats.

Test cases (all pass, GHDL, VUnit 5): `test_single_beat_request`,
`test_multi_beat_request` (verifies per-beat `dma_done` timing — not
before the last `BRESP`), `test_back_to_back_requests` (second request's
ring-buffer reinit actually takes effect, independent beat count),
`test_resp_error_latched_and_reported_at_completion` (`BRESP` error on a
non-final beat of a multi-beat request; `resp_error` only pulses with
the completing `dma_done`, never earlier), `test_stream_backpressure`
(idle cycles on the producer stream between beats),
`test_axi_backpressure` (`AWVALID` held stable while the slave is
stalled, no beat accepted early),
`test_abort_mid_request_drains_before_ready_returns` (reset with
outstanding `BRESP`s pending — `req_s2m.ready` drops immediately and
returns only after every pending `BRESP` is observed; a fresh request
afterwards behaves like any other first request),
`test_abort_with_zero_outstanding_returns_ready_next_cycle`
(reset while idle — `ready` returns the very next cycle, no spurious
`s_drain` detour), `test_zero_length_request` (no `AW`/`W` activity at
all, `dma_done` pulses, `ready` returns one cycle later).

Run with:
```
python3 run.py -o vunit_out_ofmap_dma 'cnn_accel.tb_cnn_accel_ofmap_dma*' -p 8
```
