# canny_nms

## Purpose

Non-maximum suppression: keeps the center magnitude of a 3x3 window only
if it is a local maximum along the gradient direction sector; otherwise
suppresses it to 0. Consumes the single joined AXI4-Stream produced by
`axi_stream_join` (windowed magnitude + direction resynchronized).

## Entity and architecture

Entity `canny_nms`, architecture `a`.

## Generics

None (fixed 11-bit magnitude width).

## Ports

| Name | Mode | Type/width | Description |
|---|---|---|---|
| `clk` | in | std_logic | single clock domain |
| `reset` | in | std_logic | synchronous active-high reset |
| `s_axis_tvalid` | in | std_logic | input valid |
| `s_axis_tready` | out | std_logic | backpressure to producer |
| `s_axis_tdata` | in | std_logic_vector(100 downto 0) | bits 100:2 = 3x3 magnitude window (9x11, row-major, same packing as `canny_window3x3`'s output: `w_tl` at 100:90 ... `w_br` at 12:2); bits 1:0 = direction sector |
| `s_axis_tuser` | in | std_logic_vector(1 downto 0) | bit0=SOF, bit1=border |
| `s_axis_tlast` | in | std_logic | end-of-line |
| `m_axis_tvalid` | out | std_logic | output valid |
| `m_axis_tready` | in | std_logic | backpressure from consumer |
| `m_axis_tdata` | out | std_logic_vector(10 downto 0) | suppressed magnitude |
| `m_axis_tuser` | out | std_logic_vector(1 downto 0) | passthrough unchanged |
| `m_axis_tlast` | out | std_logic | passthrough unchanged |

## Clocking and reset

Single clock (`clk`), synchronous active-high reset (`reset`). Reused
`common.handshake_pipeline` has no reset port; this module gates the
externally visible `m_axis_tvalid` with `reset`
(`m_axis_tvalid <= pipeline_output_valid and not reset`) so the output reads
as cleared for the duration of reset, per the requirement's "reset clears
the `handshake_pipeline` output register" — see
`doc/canny_nms_proposal.md` "Clock/reset behavior" for the full rationale.

## Interfaces/protocols

Two full AXI4-Stream elastic links (`s_axis`, `m_axis`), backpressure
handled entirely by `common.handshake_pipeline`'s own skid-buffer FSM
(default generics: `full_throughput => true`, `pipeline_control_signals
=> true`, `pipeline_data_signals => true`).

## Functional behavior

`direction_in = s_axis_tdata(1 downto 0)`; the 9 magnitude taps
(`w_tl..w_br`) are unpacked from `s_axis_tdata(100 downto 2)`. Neighbor
pair selected by `direction_in`:
- `"00"` (0 deg): `w_ml`, `w_mr`.
- `"01"` (45 deg): `w_tr`, `w_bl`.
- `"10"` (90 deg): `w_tm`, `w_bm`.
- `"11"` (135 deg): `w_tl`, `w_br`.

`nms_result = w_mm` if `w_mm >= neighbor_a and w_mm >= neighbor_b`
(non-strict, ties kept), else `0`. Forced to `0` when `s_axis_tuser(1)`
(border) is `'1'`. `m_axis_tuser` passes `s_axis_tuser` through unchanged
regardless of the compare result (this module does no windowing of its
own). `nms_result & s_axis_tuser` is registered through
`common.handshake_pipeline`'s `input_data`/`output_data` (13 bits:
`nms_result` in bits 12:2, `s_axis_tuser` in bits 1:0); `s_axis_tlast`
rides on `handshake_pipeline`'s dedicated `input_last`/`output_last` port.

## Timing/latency

Fixed 1-cycle latency (input accepted -> matching output beat), full
throughput once started — `handshake_pipeline`'s documented skid-buffer
behavior.

## Registers/configuration

One 13-bit + 1-bit (`last`) register stage inside `handshake_pipeline`
(skid-buffer mode also uses an internal skid register for backpressure
absorption); no other state in this module.

## Dependencies

- `common.handshake_pipeline` (hdl-modules,
  `modules/common/src/handshake_pipeline.vhd`), instantiated with default
  generics, reused unmodified as the elastic register + handshake stage.

## Implementation notes

- The direction-sector-to-neighbor-pair mapping and the local-maximum
  compare (`compare` process in `src/canny_nms.vhd`) are purely
  combinational and evaluated on the raw `s_axis_tdata`/`s_axis_tuser`
  inputs, before the `common.handshake_pipeline` register stage — only the
  13-bit result (11-bit magnitude + 2-bit `tuser`) is registered, not the
  full 101-bit window.
- `s_axis_tlast` rides `handshake_pipeline`'s own dedicated
  `input_last`/`output_last` port; `s_axis_tuser` is packed alongside the
  NMS result into the pipeline's generic `data_width => 13` payload
  instead of using a second parallel pipeline, so `tuser`/`tdata` can never
  desync.
- `common.handshake_pipeline` has no reset port at all (confirmed against
  its source). `m_axis_tvalid` is gated with `reset`
  (`m_axis_tvalid <= pipeline_output_valid and not reset`) so the
  externally-visible output reads as cleared during reset, without
  modifying the reused primitive. This pattern is intended to be reused
  verbatim by `canny_gaussian3x3`/`canny_sobel3x3`, which also reuse
  `handshake_pipeline`.
- RED caught two deliberately planted bugs during `vhtestgen`/first
  `vhfill` pass: (1) the `"00"` sector was wrongly mapped to the same
  neighbor pair as `"10"` (`w_tm`/`w_bm`) instead of `w_ml`/`w_mr`; (2) the
  local-maximum test used strict `>` instead of the required non-strict
  `>=` (which affects the tie-keeps-the-value case). Both are fixed in the
  current `compare` process.
- A first RED run of the testbench unexpectedly reported all tests PASS
  despite the two planted bugs above. Root cause: the tb's main process
  called `test_runner_cleanup` immediately after its push/check loops,
  without draining the non-blocking `push_axi_stream`/
  `check_axi_stream(blocking => false)` command queues first — those calls
  only enqueue a message on the VC actor and return immediately, well
  before any real clocked bus activity happens. Fixed by adding
  `use vunit_lib.sync_pkg.all;` and calling
  `wait_until_idle(net, as_sync(axi_master))` /
  `wait_until_idle(net, as_sync(axi_slave))` before any `now`-based timing
  check and before `test_runner_cleanup`. See project memory note
  `vunit_nonblocking_check_drain_gotcha` — the same pattern (and likely the
  same false-GREEN risk) exists in other modules' testbenches that use
  this push/check idiom without a drain.

## Verification notes

- All 4 direction sectors covered with directed local-maximum-kept,
  local-non-maximum-suppressed, and tie (kept, non-strict `>=`) cases.
- Random-data correctness computed against an independently re-derived
  inline model of the requirement's formula.
- Border forces `m_axis_tdata` to 0 while passing `m_axis_tuser(1)`
  through unchanged.
- Full-throughput (zero stall) timing check via `check_relation`.
- Randomized `stall_config` on both AXI4-Stream sides is mandatory per
  `shared/Vunit.md` §12, plus one directed zero-stall case.
