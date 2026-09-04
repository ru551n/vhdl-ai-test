# canny_threshold — proposal

## Requirements summary

Double-threshold classification of the suppressed magnitude into
none/weak/strong, per `modules/canny_threshold/doc/canny_threshold_req.md`:
`m_axis_tdata = "10"` (strong) if `s_axis_tdata >= g_thresh_high`; `"01"`
(weak) if `g_thresh_low <= s_axis_tdata < g_thresh_high`; else `"00"`
(none). `g_thresh_low <= g_thresh_high` is assumed and checked by an
elaboration-time assertion. Border (`s_axis_tuser(1)`) forces `"00"`
regardless of magnitude. `s_axis_tuser`/`s_axis_tlast` pass through
unchanged.

## Interface (copied from the requirement's structural section)

Generics: `g_thresh_low : natural`, `g_thresh_high : natural`.

Ports: `clk`, `rst_n`, `s_axis_{tvalid,tready,tdata,tuser,tlast}`,
`m_axis_{tvalid,tready,tdata,tuser,tlast}` per the requirement's port
table. `s_axis_tdata` is `std_logic_vector(10 downto 0)` (suppressed
magnitude, 0..2047 representable range); `s_axis_tuser`/`m_axis_tuser` are
2 bits (bit0=SOF, bit1=border); `m_axis_tdata` is 2 bits
(`"00"`=none, `"01"`=weak, `"10"`=strong).

## Clock/reset behavior

Single clock, synchronous active-low reset. `rst_n` is wired to the
`canny_threshold` entity per project convention, but the reused
`common.handshake_pipeline` primitive (see "Architecture and dataflow"
below) exposes **no reset port at all** (confirmed against
`hdl-modules:modules/common/src/handshake_pipeline.vhd`) — same situation
as `canny_window3x3`'s reused `fifo.fifo_wrapper`
(`doc/canny_window3x3.md` "Known limitation"). This module has no other
internal registers of its own (purely a combinational compare function
feeding the reused elastic stage), so `rst_n` has no functional effect
beyond what `handshake_pipeline`'s own VHDL default initial values
already provide (`output_valid <= '0'` etc. at time zero). This is
documented as a known limitation, following the established
`canny_window3x3` precedent, not silently dropped. Every existing
testbench in this repo (`tb_axi_stream_join`, `tb_canny_window3x3`)
already applies `rst_n` only once, before any beat is accepted, which is
unaffected by this limitation.

## Architecture and dataflow

Thin wrapper around hdl-modules `common.handshake_pipeline`
(default generics: `full_throughput => true`,
`pipeline_control_signals => true`, `pipeline_data_signals => true` —
the full skid-buffer mode), reused unmodified for the single elastic
registered stage, per `shared/ReusableRTL.md`. This module supplies the
combinational compare function and the lane data width;
`handshake_pipeline` supplies the register and the
`s_axis_tready`/`m_axis_tvalid` handshake logic — same pattern as
`canny_gaussian3x3`'s (planned) use of the same primitive.

```
s_axis_tdata, s_axis_tuser(1) --> classify() --> compare_result (2 bits) --\
s_axis_tuser (2 bits) -------------------------------------------------------+--> pack (5 bits) --> handshake_pipeline --> unpack --> m_axis_tdata/tuser/tlast
s_axis_tlast -----------------------------------------------------------------/
```

- `classify(magnitude, border, g_thresh_low, g_thresh_high)` is a pure
  combinational function: returns `"00"` when `border = '1'`
  (unconditionally, checked first); else `"10"` when
  `to_integer(unsigned(magnitude)) >= g_thresh_high`; else `"01"` when
  `to_integer(unsigned(magnitude)) >= g_thresh_low`; else `"00"`.
- The compare result, `s_axis_tuser`, and `s_axis_tlast` are packed into
  a single `c_lane_width = 5`-bit `input_data` lane (`compare_result &
  s_axis_tuser & s_axis_tlast`, MSB to LSB) fed into one
  `common.handshake_pipeline` instance (`data_width => c_lane_width`).
  `s_axis_tuser`/`s_axis_tlast` ride through the same instance as extra
  data-bit lanes (per requirement), so they can never desync from
  `m_axis_tdata`.
- `output_data` is unpacked: `m_axis_tdata <= output_data(4 downto 3)`,
  `m_axis_tuser <= output_data(2 downto 1)`, `m_axis_tlast <=
  output_data(0)`.
- `input_ready`/`input_valid`/`output_ready`/`output_valid` connect
  directly to `s_axis_tready`/`s_axis_tvalid`/`m_axis_tready`/
  `m_axis_tvalid`.
- An elaboration-time assertion (`assert g_thresh_low <= g_thresh_high
  report ... severity failure;`, concurrent statement at architecture
  level, per the `fifo.vhd`/`axi_master.vhd` hdl-modules precedent found
  via `vhdl-rag-mcp`) enforces the requirement's stated precondition.

## State machines

None in this module's own code (`handshake_pipeline`'s internal skid-
buffer FSM is entirely internal to the reused primitive).

## Algorithms

Two-threshold compare (`classify` function above); no other algorithm.

## Numeric types and widths

- `s_axis_tdata`: `std_logic_vector(10 downto 0)`, converted to `natural`
  via `to_integer(unsigned(...))` only inside `classify` for the
  magnitude comparisons — the port itself stays `std_logic_vector` (an
  opaque payload at the interface), per project numeric-type convention.
- `g_thresh_low`/`g_thresh_high`: `natural`, compared directly against
  the converted magnitude as plain integers (no synthesis-unfriendly
  real/floating math involved).
- `m_axis_tdata`: `std_logic_vector(1 downto 0)`, an enumerated-style
  code (`"00"`/`"01"`/`"10"`), not an arithmetic quantity.
- `handshake_pipeline`'s ports are `std_ulogic`/`std_ulogic_vector`; this
  module's own ports are `std_logic`/`std_logic_vector` per project
  convention — direct port-map connection is legal (same base type
  family), no conversion function needed, matching `axi_stream_join`'s
  own precedent.

## Latency/throughput

One cycle of latency (`handshake_pipeline`'s full skid-buffer register),
full throughput (one beat/cycle) sustained when neither side stalls,
matching `handshake_pipeline`'s documented `full_throughput => true`
contract.

## Corner cases

- `s_axis_tdata = g_thresh_low - 1` / `g_thresh_low` / `g_thresh_high - 1`
  / `g_thresh_high`: the four boundary values directly exercise the
  `>=`/`<` comparisons at their exact edges (see Verification plan).
- `s_axis_tuser(1) = '1'` (border): forces `"00"` regardless of
  `s_axis_tdata`'s magnitude, even when the magnitude alone would
  classify as weak/strong — checked first in `classify`, unconditionally.
- `g_thresh_low = g_thresh_high`: legal per the assertion (`<=`, not
  `<`); no magnitude can land in `"01"` in that configuration (`mag >=
  g_thresh_high` and `mag >= g_thresh_low` become the same test), so
  every non-border beat is either `"00"` or `"10"` — not specifically
  exercised by this module's own directed tests (the chosen
  `g_thresh_low=100`/`g_thresh_high=200` keep them distinct) but provably
  correct from the `classify` function's structure.
- `g_thresh_low > g_thresh_high`: rejected at elaboration by the
  assertion; not a runtime corner case.

## Selected patterns (`shared/DesignPatterns.md`)

"Ready/valid elastic stage" pattern via `handshake_pipeline`'s full
skid-buffer mode — same pattern as every other single-register
canny-pipeline stage (`canny_gaussian3x3`, planned).

## AXI4/AXI4-Stream protocol decisions (`shared/Axi4.md`)

Full AXI4-Stream backpressure on both sides, entirely delegated to
`handshake_pipeline`'s own contract (`input_ready`/`output_valid` driven
by its internal skid-buffer FSM, never a function of the other side's
signal in the same cycle in a way that creates a combinational loop).

## Verification plan

- Dedicated VUnit unit test under `modules/canny_threshold/test/`, using
  VUnit's raw `axi_stream_master`/`axi_stream_slave` VCs directly (not
  the hdl-modules `bfm.*` wrappers — `s_axis_tuser`/`m_axis_tuser` are 2
  bits, which fails `bfm.axi_stream_master`'s `user_width mod 8 = 0`
  assertion, per `shared/Vunit.md` §12's byte-alignment caveat).
- Concrete generics for the test DUT: `g_thresh_low => 100`,
  `g_thresh_high => 200` (within the 0..2040 magnitude range quoted in
  the task, comfortably inside the full 0..2047 11-bit port range).
- `stall_probability_percent` generic, swept via
  `module_canny_threshold.py` (0 for the dedicated full-throughput test,
  nonzero e.g. 20 otherwise), applied independently to the master and
  slave VC `stall_config`.
- `test_boundary_values`: directed, exactly `g_thresh_low - 1`,
  `g_thresh_low`, `g_thresh_high - 1`, `g_thresh_high`, non-border,
  checking the expected `"00"`/`"01"`/`"10"` code at each.
- `test_random_data`: random magnitude across the full `0..2047` 11-bit
  range, random border bit, expected value computed inline in the
  testbench (re-deriving the `classify` logic independently from the
  requirement, not copy-pasted from the RTL) — cheap enough that a
  separate Python golden model is not warranted, following
  `axi_stream_join`'s own precedent for a structurally simple compare/
  passthrough module.
- `test_border_forces_none`: directed, magnitude values that would
  otherwise classify as weak/strong, with `s_axis_tuser(1) = '1'`,
  expecting `"00"` regardless.
- `test_full_throughput`: zero stall on both sides, `check_relation` on
  total elapsed time vs. beat count + 1-cycle pipeline latency + small
  margin.
- **Deviation from the plan above, discovered during the RED run**:
  `check_axi_stream`/`vunit_lib.axi_stream_slave`'s TDATA mismatch
  detection is gated by a `for idx in tkeep'range loop` (see
  `verification_components/src/axi_stream_slave.vhd`); `tkeep`'s width
  is `data_length/8`, so for `m_axis_tdata` (`data_length => 2`, not a
  multiple of 8) `tkeep` is a null-range vector, the loop body never
  runs, and TDATA is *silently never checked* — the RED run passed all
  4 tests against a deliberately-wrong `classify` stub. Worked around by
  dropping `check_axi_stream` entirely: the main process pushes the
  expected `(tdata, tlast, tuser)` into a `queue_t`
  (`vunit_lib.queue_pkg`, pulled in transitively by `vunit_context`),
  and a separate concurrent `checker_proc` drains actual output beats
  via `pop_axi_stream` and compares them with `check_equal` directly —
  this both bypasses the broken tkeep-gated comparison and (running in
  its own process rather than interleaved push-then-blocking-pop in one
  process) preserves the zero-stall full-throughput timing property.
  See "New gotchas" in the final task report.

## Implementation Notes (vhfill)

The `classify` function's `--@` placeholder (always returning `"00"`)
was filled in exactly as designed: `mag >= thresh_high` -> `"10"`,
`elsif mag >= thresh_low` -> `"01"`, `else` -> `"00"`, with the border
check (`border = '1'` -> `"00"`) unconditionally ahead of both — no
deviation from this document's stated logic. The RED run (with the
stub) failed 3 of 4 tests with `TDATA mismatch ... Got 00. Expected 10`
(as expected); `test_border_forces_none` legitimately passed even
against the stub since all its inputs have `border = '1'`, which the
stub already handled correctly. No other deviations from this document.
