# canny_hysteresis

## Purpose

Last stage of the canny pipeline: local (3x3) hysteresis over a
pre-classified window — a pixel is a final edge if it is `strong`, or
`weak` with at least one `strong` neighbor. Drives the IP's top-level
`m_axis` directly, so `m_axis_tuser` is 1 bit (SOF only); the incoming
`border` bit is consumed (forces `edge='0'`) and not re-exposed. See
`modules/canny/doc/canny_hysteresis_req.md` and
`modules/canny/doc/canny_hysteresis_proposal.md`.

## Entity and architecture

Entity `canny_hysteresis`, architecture `a`.

## Generics

None.

## Ports

| Name | Mode | Type/width | Description |
|---|---|---|---|
| `clk` | in | std_logic | single clock domain |
| `reset` | in | std_logic | synchronous active-high reset |
| `s_axis_tvalid` | in | std_logic | input valid |
| `s_axis_tready` | out | std_logic | backpressure to producer |
| `s_axis_tdata` | in | std_logic_vector(17 downto 0) | 3x3 classification window, row-major MSB-to-LSB, 2 bits/tap: `w_tl`=17:16 ... `w_br`=1:0 |
| `s_axis_tuser` | in | std_logic_vector(1 downto 0) | bit0=SOF, bit1=border |
| `s_axis_tlast` | in | std_logic | end-of-line |
| `m_axis_tvalid` | out | std_logic | output valid |
| `m_axis_tready` | in | std_logic | backpressure from consumer |
| `m_axis_tdata` | out | std_logic_vector(7 downto 0) | final edge byte, bit0=edge, bits 7:1='0' |
| `m_axis_tuser` | out | std_logic_vector(0 downto 0) | bit0=SOF passthrough only — top-level boundary, no border bit |
| `m_axis_tlast` | out | std_logic | passthrough of `s_axis_tlast` |

## Clocking and reset

Single clock (`clk`), synchronous active-high reset (`reset`). `reset`
gates the boundary signals of the wrapped `common.handshake_pipeline`
instance (which itself has no reset port): `input_valid`/`s_axis_tready`/
`m_axis_tvalid` are held to their reset-safe values, and `output_ready`
is forced so any pre-reset pipeline state drains within at most 2 cycles.
See `doc/canny_hysteresis_proposal.md` "Clock/reset behavior" for the full
rationale.

## Interfaces/protocols

Full AXI4-Stream elastic handshake on both sides per `shared/Axi4.md`,
delegated to `common.handshake_pipeline`'s own verified ready/valid
contract (full skid-buffer mode: `full_throughput`,
`pipeline_control_signals`, `pipeline_data_signals` all `true`).

## Functional behavior

- The 9 2-bit taps are unpacked from `s_axis_tdata`. Classification codes:
  `"10"`=strong, `"01"`=weak, else=none.
- `edge = '1'` if `w_mm = "10"`, or (`w_mm = "01"` and any of the other 8
  taps `= "10"`); forced to `'0'` if `s_axis_tuser(1)` (border) is `'1'`.
- `m_axis_tdata = "0000000" & edge`.
- `m_axis_tuser(0)` passes `s_axis_tuser(0)` (SOF) through, one cycle
  delayed by the pipeline register (same delay as the data path — both
  ride through the same `handshake_pipeline` instance).
- `m_axis_tlast` passes `s_axis_tlast` through, same delay.

## Timing/latency

One clock cycle of registered latency; one beat/cycle at full throughput
once primed (immediately — no fill/priming delay, unlike
`canny_window3x3`).

## Registers/configuration

None (no run-time configuration registers).

## Dependencies

- `common.handshake_pipeline` (hdl-modules,
  `modules/common/src/handshake_pipeline.vhd`), instantiated with its
  default generics (full skid-buffer mode), wrapping a 9-bit packed
  `(sof & edge_byte)` payload plus `tlast` as `input_last`/`output_last`.

## Implementation notes

- `compute_edge` implements the hysteresis rule exactly as designed:
  `edge = (center = strong) or (center = weak and any-of-the-other-8-taps =
  strong)`, then forced to `'0'` when `border = '1'`.
- Reset-gating around `common.handshake_pipeline` (which has no reset port
  of its own): `pipeline_input_valid <= s_axis_tvalid and not reset`;
  `pipeline_output_ready <= m_axis_tready or reset` (force-drains the
  pipeline within its own latency while `reset = '1'`); `s_axis_tready <=
  pipeline_input_ready and not reset`; `m_axis_tvalid <= pipeline_output_valid
  and not reset`. No deviation from the proposal's design was needed.
- No RTL bugs were caught by RED — the first `run.py` invocation reported a
  **false pass** while the RTL was still fully stubbed, due to a
  testbench bug (missing `wait_until_idle` before `test_runner_cleanup`;
  see `doc/canny_hysteresis_proposal.md` "Implementation Notes (vhfill)"
  for the full root cause and fix). After that fix, RED correctly failed
  all 4 tests (watchdog timeout), and GREEN was reached on the first
  `vhfill` pass.

## Verification notes

- Dedicated VUnit unit test under `modules/canny/test/`, using
  VUnit's raw `axi_stream_master`/`axi_stream_slave` VCs directly on both
  `s_axis` (2-bit tuser) and `m_axis` (1-bit tuser) — both widths violate
  the hdl-modules `bfm.axi_stream_*` wrapper's `user_width mod 8 = 0`
  assertion.
- `stall_probability_percent` generic swept via
  `module_canny.py` (0 for the dedicated full-throughput test,
  nonzero otherwise), independently randomized `stall_config` on both
  sides.
- Directed cases for all 4 combinations of (center=strong/weak/none) x (a
  neighbor strong or not), a directed border-forces-edge-0 case (also
  checking SOF passthrough on the same beat), a random-data correctness
  test, and a full-throughput zero-stall timing check via
  `check_relation`.
