# axi_stream_join — proposal

## Requirements summary

Generic 2-input AXI4-Stream rendezvous, per
`modules/axi_stream_join/doc/axi_stream_join_req.md`: emit a joined
transfer (data concatenated, `a` in upper bits, `b` in lower bits) only
when both inputs are simultaneously valid, applying combined backpressure
to both inputs together. Not canny-specific.

## Interface (copied from the requirement's structural section)

Generics: `g_data_width_a : positive`, `g_data_width_b : positive`.

Ports: `clk`, `rst_n`, `s_axis_a_{tvalid,tready,tdata,tuser,tlast}`,
`s_axis_b_{tvalid,tready,tdata,tuser,tlast}`,
`m_axis_{tvalid,tready,tdata,tuser,tlast}` — `s_axis_a_tuser`/
`s_axis_b_tuser`/`m_axis_tuser` are `std_logic_vector(1 downto 0)`
(bit0=SOF, bit1=border) per the project's internal-link `tuser` convention
(`doc/canny_arch.md`); `m_axis_tdata` width is
`g_data_width_a + g_data_width_b`.

## Clock/reset behavior

Single clock, synchronous active-low reset. No stateful elements are
required for the join function itself (`handshake_merger` is purely
combinational); `rst_n` is only used by the simulation-only consistency
assertion's reporting path, not by any functional logic.

## Architecture and dataflow

Thin wrapper around hdl-modules `common.handshake_merger`
(`num_interfaces => 2`), reused unmodified for the 2-way valid/ready
arbitration, per `shared/ReusableRTL.md`:

```
s_axis_a_tvalid ---> input_valid(0) --\
s_axis_b_tvalid ---> input_valid(1) --+--> handshake_merger --> result_valid/ready/last
m_axis_tready   ---> result_ready   --/
                  input_ready(0/1) ---> s_axis_a_tready / s_axis_b_tready
```

`handshake_merger`'s own contract: `input_ready <= (others => result_ready
and result_valid)`; `result_valid <= and(input_valid)`; `result_last <=
or(input_last)` (plus an elaboration-time-configurable assertion that all
`input_last` values agree when `result_valid`). This module does not feed
`input_last` from a per-input `tlast` at all (see "Numeric types and
widths" below on why `tlast`/`tuser(0)` are taken structurally from
`s_axis_a`, not from `handshake_merger`'s own last-OR behavior) — instead
`input_last` is left at its default (`'1'`) on both lanes so
`handshake_merger`'s internal consistency assertion becomes a no-op
(`or('1','1') = '1'`, always satisfied), and this module's own explicit
assertion (below) does the real `tlast`/`tuser(0)` agreement check instead
with a message that correctly identifies it as a synchronization bug
between the two forks, not a "packet length" mismatch as
`handshake_merger`'s own generic message would describe it.

On top of `handshake_merger`, this module adds (no internal register,
combinational, gated by the same fire condition `handshake_merger` uses
internally: `result_valid and result_ready`, i.e. `input_valid(0) and
input_valid(1) and m_axis_tready`):

- `m_axis_tdata <= s_axis_a_tdata & s_axis_b_tdata` (concatenation, `a` in
  the upper `g_data_width_a` bits).
- `m_axis_tuser(1) <= s_axis_a_tuser(1) or s_axis_b_tuser(1)` (border
  OR-reduction).
- `m_axis_tuser(0) <= s_axis_a_tuser(0)`; `m_axis_tlast <= s_axis_a_tlast`
  (structural passthrough from `a`, per the requirement's rationale: both
  forks originate from the same upstream beat, so they must already agree
  whenever `handshake_merger` fires).
- A simulation-only (`-- pragma synthesis_off`/`on`, or an assertion the
  synthesis tool is expected to strip — see Implementation Notes once
  `vhfill` picks a concrete mechanism) check: on every cycle where
  `handshake_merger`'s fire condition holds, assert
  `s_axis_a_tuser(0) = s_axis_b_tuser(0)` and `s_axis_a_tlast =
  s_axis_b_tlast`, reporting a clear "fork desynchronization" failure
  message (severity error) if not.

## State machines

None — purely combinational.

## Algorithms

None beyond the concatenation/OR/passthrough described above.

## Numeric types and widths

- `s_axis_a_tdata`/`s_axis_b_tdata`/`m_axis_tdata`: `std_logic_vector`
  (opaque payload, no arithmetic performed on it by this module — treated
  purely as bits to concatenate, per the numeric-type gate's rule that
  `std_logic_vector` is fine as an interface/opaque-payload type).
- `handshake_merger`'s ports are `std_ulogic`/`std_ulogic_vector`; this
  module's own ports are `std_logic`/`std_logic_vector` per project
  convention (`AGENTS.md`) — direct port-map connection between
  `std_logic` and `std_ulogic` is legal in VHDL (both resolve from the same
  base type family: `std_ulogic` is the unresolved version of the
  `std_logic` subtype), no conversion function needed at the boundary.

## Latency/throughput

Zero-latency (combinational), matching `handshake_merger`'s own contract.
Throughput: one joined beat per cycle when both inputs are continuously
valid and `m_axis_tready` is held high; otherwise gated 1:1 by
`handshake_merger`'s combinational fire condition — no internal buffering,
so a stall on either input or on `m_axis_tready` stalls all three sides
simultaneously (this is `handshake_merger`'s documented behavior, not a new
elastic stage).

## Corner cases

- Only one input valid: no output beat, that input's `tready` stays low
  (per `handshake_merger`: `input_ready <= (others => result_ready and
  result_valid)`, and `result_valid` requires **both** inputs valid) — the
  valid-but-unmatched input is correctly held/backpressured, not dropped.
- Both inputs valid but `m_axis_tready='0'`: `result_valid='0'` is not
  forced by `handshake_merger` itself (`result_valid <= and(input_valid)`
  regardless of `result_ready`) — `m_axis_tvalid` is asserted combinationally
  whenever both inputs are valid, independent of downstream readiness, per
  AXI4-Stream's `TVALID`-must-not-wait-for-`TREADY` rule; `input_ready` is
  what's gated by `result_ready and result_valid`, correctly implementing
  "advance only when the downstream also accepts."
- Fork desynchronization (`s_axis_a_tuser(0) /= s_axis_b_tuser(0)` or
  `s_axis_a_tlast /= s_axis_b_tlast` while both valid and both accepted):
  flagged by the simulation-only assertion described above; not otherwise
  handled at runtime (documented as an upstream bug this module cannot
  recover from, per the requirement).

## Selected patterns (`shared/DesignPatterns.md`)

"Ready/valid elastic stage" pattern's zero-latency combinational variant —
this module has no registered stage of its own (unlike every other
canny-pipeline module, which wraps `handshake_pipeline`); its elasticity
comes entirely from whatever registration already exists inside the two
upstream producers it joins.

## AXI4/AXI4-Stream protocol decisions (`shared/Axi4.md`)

Full AXI4-Stream backpressure on all three sides (`s_axis_a`, `s_axis_b`,
`m_axis`) — no shortcut. `TVALID` on `m_axis` is not gated by
`m_axis_tready` (correct per protocol); `TREADY` on each input is gated by
both the other input's `TVALID` and `m_axis_tready` (correct "elastic
join" semantics, delegated entirely to `handshake_merger`).

## Verification plan

- Dedicated VUnit unit test under `modules/axi_stream_join/test/` (per
  `doc/canny_arch.md` "Verification hooks").
- Use VUnit's raw `axi_stream_master`/`axi_stream_slave` VCs (three
  instances: two masters feeding `s_axis_a`/`s_axis_b`, one slave checking
  `m_axis`) with independent randomized `stall_config` on all three per
  `shared/Vunit.md` §12, plus a directed zero-stall sanity case.
- Directed cases: only-`a`-valid held for N cycles then `b` arrives (no
  spurious output beat, no dropped `a` beat); simultaneous arrival every
  cycle at full throughput (1 output beat/cycle, no bubbles); a directed
  negative case that intentionally desynchronizes `tuser(0)`/`tlast`
  between `a` and `b` and expects the consistency assertion to fire
  (VUnit mock-logger pattern, since this is an expected-failure check).
- Prefer non-blocking `push_axi_stream`/`check_axi_stream(...,
  blocking => false)` for stimulus/checks per `shared/Vunit.md` §12.
- A small Python reference is not needed for this module (the expected
  behavior is a pure structural concat/OR, cheap enough to compute inline
  in the testbench) — reserve the Python-golden-model machinery
  (`shared/Vunit.md` "Python reference models") for the canny-specific
  numeric stages and the full-pipeline integration test.

## Implementation Notes (vhfill)

(empty — to be filled in by `vhfill`)
