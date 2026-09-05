# canny_threshold

## Purpose

Double-threshold classification of the suppressed magnitude into
none/weak/strong. See `modules/canny/doc/canny_threshold_req.md`
and `modules/canny/doc/canny_threshold_proposal.md`.

## Entity and architecture

Entity `canny_threshold`, architecture `a`.

## Generics

| Name | Type | Meaning | Constraints |
|---|---|---|---|
| `thresh_low` | natural | below this: "none" | `thresh_low <= thresh_high` (elaboration-time assertion) |
| `thresh_high` | natural | at/above this: "strong"; between: "weak" | `thresh_low <= thresh_high` (elaboration-time assertion) |

## Ports

| Name | Mode | Type/width | Description |
|---|---|---|---|
| `clk` | in | std_logic | single clock domain |
| `s_axis_tvalid` | in | std_logic | input valid |
| `s_axis_tready` | out | std_logic | backpressure to producer |
| `s_axis_tdata` | in | std_logic_vector(10 downto 0) | suppressed magnitude |
| `s_axis_tuser` | in | std_logic_vector(1 downto 0) | bit0=SOF, bit1=border |
| `s_axis_tlast` | in | std_logic | end-of-line |
| `m_axis_tvalid` | out | std_logic | output valid |
| `m_axis_tready` | in | std_logic | backpressure from consumer |
| `m_axis_tdata` | out | std_logic_vector(1 downto 0) | `"00"`=none, `"01"`=weak, `"10"`=strong |
| `m_axis_tuser` | out | std_logic_vector(1 downto 0) | passthrough unchanged |
| `m_axis_tlast` | out | std_logic | passthrough unchanged |

## Clocking and reset

Single clock (`clk`), resetless. The reused `common.handshake_pipeline`
primitive exposes no reset port at all (confirmed against
`hdl-modules:modules/common/src/handshake_pipeline.vhd`), and this module
has no other internal registers of its own (purely a combinational
compare function feeding the reused elastic stage), so its VHDL default
initial values (`output_valid <= '0'` at time zero) already give the
correct power-up state — no reset port is needed, per
`shared/TsfpgaCodingConventions.md`'s resetless-by-default policy.

## Interfaces/protocols

Full AXI4-Stream elastic handshake on both sides per `shared/Axi4.md`,
entirely delegated to `common.handshake_pipeline`'s own contract:
`input_ready`/`output_valid` are driven by its internal skid-buffer FSM,
never combinationally dependent on the other side's signal within the
same cycle (no handshake loop).

## Functional behavior

- `classify(magnitude, border, thresh_low, thresh_high)`: combinational
  function returning `"00"` when `border = '1'` (checked first,
  unconditional); else `"10"` when `magnitude >= thresh_high`; else
  `"01"` when `magnitude >= thresh_low`; else `"00"`.
- The compare result (2 bits), `s_axis_tuser` (2 bits), and `s_axis_tlast`
  (1 bit) are packed into one 5-bit lane fed through a single
  `common.handshake_pipeline` instance; `m_axis_tdata`/`m_axis_tuser`/
  `m_axis_tlast` are unpacked from the pipeline's `output_data`.
- `m_axis_tuser`/`m_axis_tlast` are structural passthroughs of
  `s_axis_tuser`/`s_axis_tlast`, delay-matched to `m_axis_tdata` through
  the same elastic lane (cannot desync from the data path).

## Timing/latency

One cycle of latency (`handshake_pipeline`'s full skid-buffer register,
default generics), full throughput (one beat/cycle) sustained when
neither side stalls.

## Registers/configuration

None (no run-time configuration registers; `thresh_low`/`thresh_high`
are generics, fixed at elaboration).

## Dependencies

- `common.handshake_pipeline` (hdl-modules,
  `modules/common/src/handshake_pipeline.vhd`), instantiated once with
  default generics (`full_throughput => true`,
  `pipeline_control_signals => true`, `pipeline_data_signals => true`,
  i.e. the full skid-buffer mode), `data_width => 5` (2-bit compare
  result + 2-bit `tuser` + 1-bit `tlast`).

## Implementation notes

- `compare_result` is a pure combinational function of `s_axis_tdata`/
  `s_axis_tuser(1)` (border): `border = '1'` forces `"00"`; else
  `magnitude >= thresh_high` -> `"10"` (strong), `elsif magnitude >=
  thresh_low` -> `"01"` (weak), `else` -> `"00"` (none) — exactly as
  designed in the proposal, no deviation.
- Single `common.handshake_pipeline` instance (`data_width => 5`)
  carries the packed lane `compare_result(1:0) & s_axis_tuser(1:0) &
  s_axis_tlast`, unpacked back into `m_axis_tdata`/`m_axis_tuser`/
  `m_axis_tlast` on the output side.
- No other deviations from `doc/canny_threshold_proposal.md`.

## Verification notes

- Dedicated VUnit unit test under `modules/canny/test/`, using
  VUnit's raw `axi_stream_master`/`axi_stream_slave` VCs directly (not
  the hdl-modules `bfm.*` wrappers — `tuser` is 2 bits, which fails
  `bfm.axi_stream_master`'s `user_width mod 8 = 0` assertion).
- Concrete generics for the test DUT: `thresh_low => 100`,
  `thresh_high => 200`.
- `stall_probability_percent` generic swept via
  `module_canny.py` (0 for the dedicated full-throughput test,
  nonzero otherwise).
- Directed boundary test at exactly `thresh_low-1`/`thresh_low`/
  `thresh_high-1`/`thresh_high`.
- Random-data correctness test across the full `0..2047` 11-bit input
  range, with expected value computed inline in the testbench.
- Directed border-forces-`"00"` test.
- Full-throughput (zero stall) timing check via `check_relation`.
- **Deviation, discovered during the RED run**: does *not* use
  `check_axi_stream` to check the output side. `check_axi_stream`/
  `vunit_lib.axi_stream_slave`'s TDATA mismatch detection is gated by a
  `for idx in tkeep'range loop`, and `tkeep`'s width is `data_length/8`
  — for `m_axis_tdata` (`data_length => 2`, not a multiple of 8) `tkeep`
  is a null-range vector, so the loop never runs and TDATA is silently
  never checked. Worked around with a manual expected-value `queue_t`
  (pushed by the main process) drained by a separate `checker_proc`
  process via `pop_axi_stream` + `check_equal`, which also keeps the
  full-throughput timing property (no interleaved blocking pop in the
  push process). See the project's "New gotchas" for this session.
