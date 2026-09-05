# canny_sobel3x3 — proposal

## Requirements summary

3x3 Sobel gradient magnitude (11-bit, L1 norm) and direction (2-bit,
4-sector) computed from one accepted smoothed-pixel window beat, forked
into two independent AXI4-Stream masters (`m_axis_mag`, `m_axis_dir`) per
`modules/canny/doc/canny_sobel3x3_req.md`. The two forks must
never desynchronize: `s_axis_tready` only asserts once both downstream
consumers can (eventually, per the joint-acceptance semantics below)
accept the current beat, and both fork valids trace back to the *same*
one-cycle-registered value.

## Interface (copied from the requirement's structural section)

No generics (fixed 8-bit input pixel width, fixed 11-bit magnitude width).

Ports: `clk`, `reset`, `s_axis_{tvalid,tready,tdata(71:0),tuser(1:0),tlast}`
(same 72-bit window packing as `canny_gaussian3x3`'s input: `w_tl`=71:64,
`w_tm`=63:56, `w_tr`=55:48, `w_ml`=47:40, `w_mm`=39:32 (unused by Sobel),
`w_mr`=31:24, `w_bl`=23:16, `w_bm`=15:8, `w_br`=7:0), and two independent
output masters `m_axis_mag_{tvalid,tready,tdata(10:0),tuser(1:0),tlast}` /
`m_axis_dir_{tvalid,tready,tdata(1:0),tuser(1:0),tlast}`.

## Clock/reset behavior

Single clock, synchronous active-high reset. `reset` clears the one shared
data/valid register that both forks are driven from (`reg_valid`/
`reg_data` in the architecture below), which combinationally forces both
`m_axis_mag_tvalid` and `m_axis_dir_tvalid` low the same cycle — see
"Deviation" note under Implementation Notes for the one documented gap
this leaves in `common.handshake_splitter`'s own un-resettable internal
bookkeeping.

## Architecture and dataflow

```
s_axis_t* --> [combinational Gx/Gy/mag/dir + border forcing] --> reg_data_in (16 bits: mag(11) & dir(2) & tuser(2) & tlast(1))
                                                                        |
                                                     reg_valid/reg_data register (reset-clearable)
                                                                        |
                                              reg_valid --> input_valid --\
                                                                          +--> common.handshake_splitter (num_interfaces=2) --> splitter_input_ready --\
                                              splitter_input_ready -------/                                                                          |
                                                                                                                                    (feeds back into the register's accept condition)
                                              reg_data (sliced) --> combinationally --> m_axis_mag_t* / m_axis_dir_t* data+user+last
                                              output_ready(0/1) <-- m_axis_mag_tready / m_axis_dir_tready
                                              output_valid(0/1) --> m_axis_mag_tvalid / m_axis_dir_tvalid
```

Two reuse decisions, per `shared/ReusableRTL.md`'s submodule table
convention:

- **`common.handshake_splitter`(`num_interfaces => 2`) — reuse, unmodified.**
  This is the piece the requirement explicitly calls out: a hand-written
  `s_axis_tready <= m_axis_mag_tready and m_axis_dir_tready` AND-gate looks
  equivalent in the steady state but is *wrong* the instant the two forks
  stall for different durations — if `mag` accepts a beat while `dir`
  stalls, a naive AND-gated single-valid design keeps presenting the
  *same* (already-consumed) beat to `mag` with `tvalid='1'` until `dir`
  also catches up, which is a duplicate-beat protocol violation on `mag`.
  `handshake_splitter`'s per-output `transaction_done_sticky` bit
  (confirmed via `hdl-modules:modules/common/src/handshake_splitter.vhd`)
  exists precisely to prevent this: once an individual output has
  transacted, its `output_valid` drops (independently of the other
  output) until the *whole* joint transaction (both outputs) is done, at
  which point both sticky bits clear together and the next beat is
  admitted. It has no `data` ports at all (pure control/handshake), so
  this module still owns all the data multiplexing/registration itself.
- **`common.handshake_pipeline` — evaluated, not reused for the register.**
  `handshake_pipeline` would be the natural reuse for "one combinational
  function, registered one cycle" (as `canny_gaussian3x3` does), but it
  exposes no reset port at all (confirmed against
  `hdl-modules:modules/common/src/handshake_pipeline.vhd` — only relies on
  its signals' power-up default values). The requirement's Clock/reset
  section is explicit that `reset` must clear the output register(s), so
  a `handshake_pipeline` instance could not satisfy that requirement on a
  mid-stream reset. The register is therefore hand-written instead (a
  single flip-flop pipeline stage implementing the same
  `input_ready <= output_ready or not output_valid` elastic-register
  logic `handshake_pipeline`'s own
  `full_throughput=true, pipeline_control_signals=false` mode uses
  internally — see that file — plus a synchronous `reset` clear that
  primitive cannot provide). This is a deliberately minimal amount of new
  RTL (one register, one mux condition), not a reimplementation of
  `handshake_splitter`'s fork logic.

On top of these two pieces, this module supplies (new, genuinely
module-specific logic):

- `unpack_window`: slices the 8 taps Sobel actually uses (`tl, tm, tr, ml,
  mr, bl, bm, br`; `mm` is unused) out of `s_axis_tdata`.
- Combinational `Gx`, `Gy` (signed, 12-bit headroom), `ax=|Gx|`, `ay=|Gy|`
  (unsigned, 11-bit), `mag = ax + ay` (11-bit unsigned), and the 4-sector
  direction classification — the exact formulas from
  `doc/canny_sobel3x3_req.md`'s Functional Description.
  Border forcing (`s_axis_tuser(1)='1'` ⇒ `mag=0`, `dir="00"`) is applied
  after the raw calculation, before packing into the register.
- Packing/unpacking of the shared 16-bit register word
  (`mag(11) & dir(2) & tuser(2) & tlast(1)`).

## State machines

None — the hand-written register is a single flip-flop pipeline stage
(no FSM), and `handshake_splitter` is itself combinational/single-register
(no explicit state machine either, per its source above).

## Algorithms

`Gx = (tr + 2*mr + br) - (tl + 2*ml + bl)`, `Gy = (bl + 2*bm + br) - (tl +
2*tm + tr)` (signed, range -1020..1020 each, computed with 12-bit signed
headroom to avoid any intermediate-sum overflow).
`mag = |Gx| + |Gy|` (unsigned, max 2040, fits 11 bits exactly as required).
Direction: `ax=|Gx|`, `ay=|Gy|`; `"00"` if `ay <= (ax >> 1)`; else `"10"`
if `ax <= (ay >> 1)`; else `"01"` if `Gx`/`Gy` have the same sign
(`sign(Gx) xor sign(Gy) = '0'`, using each value's own MSB/sign bit),
else `"11"`. The `ax=ay=0` tie resolves to `"00"` because the `"00"`
branch is checked first (matches the requirement's explicit tie-break
note). Border (`s_axis_tuser(1)='1'`) forces `mag=0`/`dir="00"` after the
above.

## Numeric types and widths

- Window taps: `unsigned(7 downto 0)` (raw 0..255 pixel values).
- `Gx`/`Gy`: `signed(11 downto 0)` — headroom above the provable
  -1020..1020 range so no intermediate sum/doubling can overflow.
- `ax`/`ay`: `unsigned(10 downto 0)` (11-bit, matches `mag`'s width;
  actual max value 1020 fits comfortably).
- `mag`: `unsigned(11 downto 0)` internally (extra carry-safety bit during
  the `ax + ay` addition), truncated to `std_logic_vector(10 downto 0)`
  for the register/port — safe because the true maximum (2040) fits in 11
  bits, so the discarded top bit is always `'0'` for any legal input.
- `dir`: `std_logic_vector(1 downto 0)`, one of the 4 literal sector
  codes.
- All AXI4-Stream data ports remain `std_logic`/`std_logic_vector` per
  project convention; `common.handshake_splitter`'s ports are
  `std_ulogic`/`std_ulogic_vector` — same direct-connection legality note
  as `axi_stream_join`.

## Latency/throughput

One cycle of latency from an accepted `s_axis` beat to the corresponding
`m_axis_mag`/`m_axis_dir` beat becoming visible (the hand-written
register). Throughput: 1:1 with `s_axis` once running, sustained even
under backpressure, as long as both `mag`/`dir` consumers keep up with
their own combined demand — `s_axis_tready` only reasserts once the
current registered beat has been consumed by *both* forks (directly or
because it is being consumed by both in the same cycle the next beat
would load), per `handshake_splitter`'s joint-acceptance contract.

## Corner cases

- One fork stalls, the other doesn't: `handshake_splitter`'s sticky
  bookkeeping holds the already-transacted fork's `tvalid` low (per-output)
  until the slower fork also transacts, then both are released together
  for the *next* beat — never re-presenting (duplicating) the already-sent
  beat to the fast fork, and never advancing `s_axis_tready` until the
  slow fork has actually caught up.
- Border beat (`s_axis_tuser(1)='1'`): `mag`/`dir` forced to `0`/`"00"`;
  `tuser`/`tlast` still pass through unchanged on both forks (identical
  value on both, since both are sliced from the one shared register).
- `ax=ay=0` (flat/zero-gradient window, not necessarily a border beat):
  resolves to direction `"00"` per the tie-break rule above.
- Reset while one fork has transacted the current beat and the other has
  not (mid-join-window reset): see "Implementation Notes" deviation below.

## Selected patterns (`shared/DesignPatterns.md`)

"Ready/valid elastic stage" pattern, one-cycle-registered variant (like
`canny_gaussian3x3`'s `handshake_pipeline` use) — but with the output side
forked 1-to-2 via `handshake_splitter`'s joint-acceptance variant of the
pattern instead of a single elastic consumer.

## AXI4/AXI4-Stream protocol decisions (`shared/Axi4.md`)

Full AXI4-Stream backpressure on all three links (`s_axis`, `m_axis_mag`,
`m_axis_dir`). `s_axis_tready` is a registered decision (the hand-written
register's accept condition), never combinationally looped from
`m_axis_mag_tvalid`/`m_axis_dir_tvalid`. `m_axis_mag_tvalid`/
`m_axis_dir_tvalid` are combinational functions of `reg_valid` and
`handshake_splitter`'s internal sticky state (per its source), never
waiting on `m_axis_mag_tready`/`m_axis_dir_tready` to *assert* validity
(correct per protocol — only whether a transaction is registered as
"already done" for that output is affected by its own `tready`).

## Verification plan

- Dedicated VUnit unit test under `modules/canny/test/`, using
  VUnit's raw `axi_stream_master`/`axi_stream_slave` VCs directly for all
  three streams (input window, `mag` output, `dir` output) — `tuser` is 2
  bits on all three, which is fine for the raw VCs but would fail
  `bfm.axi_stream_master/slave`'s byte-alignment assertion, same rationale
  as `canny_window3x3`.
- Independent, independently-randomized `stall_probability_percent_mag`/
  `stall_probability_percent_dir` generics (plus one for the input side)
  swept via `module_canny.py`'s `add_vunit_config` — at least one
  all-zero full-throughput config, plus configs with asymmetric nonzero
  values (e.g. 10 / 30) specifically to exercise the differently-stalled-
  forks case `handshake_splitter` exists for.
- Separate concurrent checking processes for `mag` and `dir` (each with
  its own non-blocking `check_axi_stream`), so a stalled `dir` consumer
  cannot block progress checking `mag`, and vice versa.
- Random-window correctness test with per-beat expected `Gx`/`Gy`/`mag`/
  `dir` computed inline in the testbench per the exact requirement
  formulas (including the tie-break and border-forcing rules).
- Directed case exercising each of the 4 direction sectors with
  hand-picked tap values.
- Border-forces-zero directed case (`s_axis_tuser(1)='1'`).
- Every-beat check (not just spot checks) that `mag` and `dir` carry the
  same `tuser`/`tlast` value whenever compared beat-for-beat, even when
  their respective downstream consumers stall independently.
- Full-throughput (all stalls zero) timing check via `check_relation`.

## Implementation Notes (vhfill)

- **Deviation from a literal reading of the requirement's
  `s_axis_tready <= m_axis_mag_tready and m_axis_dir_tready`.** That
  expression is accurate as a *steady-state* description (both consumers
  must eventually accept before a new beat is admitted) but is not
  implemented as a literal one-line combinational AND — `s_axis_tready`
  is instead the hand-written register's accept condition
  (`splitter_input_ready or not reg_valid`), where `splitter_input_ready`
  is `handshake_splitter`'s own `and()`-of-sticky-bits signal. This is
  necessary (not just a style choice) to get the asymmetric-stall corner
  case right — see "Architecture and dataflow" above for the concrete bug
  a literal AND-gate would have.
- **Known limitation (reset gap), same class as `canny_window3x3`'s FIFO
  caveat:** `common.handshake_splitter` exposes no reset port. The
  hand-written `reg_valid`/`reg_data` register *is* cleared by `reset`,
  which immediately forces both `m_axis_mag_tvalid` and `m_axis_dir_tvalid`
  low (satisfying the requirement's "clears both output registers" at the
  externally-visible level). However, if `reset` is pulsed at the exact
  moment one fork has already transacted the current beat and the other
  has not, that one fork's `transaction_done_sticky` bit is left set
  internally; the next beat after reset will then have its valid
  suppressed on that one fork for exactly one beat before the sticky bits
  self-resynchronize on the next full join — a single dropped beat on one
  fork, immediately after a reset that lands mid-join. This is out of
  scope for this module's verification plan (this project's established
  convention, per `tb_axi_stream_join`/`tb_canny_window3x3`, is to pulse
  `reset` once before any traffic, never mid-stream), and is not
  reachable under that convention.
