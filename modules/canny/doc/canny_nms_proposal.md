# canny_nms — proposal

## Requirements summary

Non-maximum suppression along the gradient direction sector, per
`modules/canny/doc/canny_nms_req.md`: keep the center magnitude of a
3x3 window only if it is a local maximum (non-strict `>=`) along the
gradient direction; else suppress to 0. Consumes the single joined
AXI4-Stream produced by `axi_stream_join` (windowed magnitude + direction
resynchronized), not two separate streams itself. Fixed 11-bit magnitude
width, no generics.

## Interface (copied from the requirement's structural section)

No generics. Ports: `clk`, `reset`, `s_axis_{tvalid,tready,tdata,tuser,tlast}`,
`m_axis_{tvalid,tready,tdata,tuser,tlast}` per the requirement's port
table. `s_axis_tdata` is 101 bits: bits 100:2 are the 3x3 magnitude window
(same 9x11 row-major packing as `canny_window3x3`'s output, `w_tl` at bits
100:90 down to `w_br` at bits 12:2), bits 1:0 are the direction sector.
`s_axis_tuser`/`m_axis_tuser` are 2 bits (bit0=SOF, bit1=border).
`m_axis_tdata` is 11 bits (suppressed magnitude).

## Clock/reset behavior

Single clock, synchronous active-high reset. `common.handshake_pipeline`
(the hdl-modules primitive this module wraps) has **no reset port of its
own** — verified via `vhdl-rag-mcp` (`modules/common/src/handshake_pipeline.vhd`
entity ports: `clk`, `input_*`, `output_*`, no `rst`/`reset`). To satisfy
the requirement's "synchronous active-high reset clears the
`handshake_pipeline` output register" without modifying the reused
primitive, this module gates the externally visible `m_axis_tvalid` with
`reset`: `m_axis_tvalid <= pipeline_output_valid and not reset`. This makes the
output register read as cleared (no spurious valid beat) for the entire
duration of reset, satisfying the requirement's observable behavior, while
leaving `handshake_pipeline`'s internal state untouched (it naturally
re-synchronizes once `s_axis_tvalid` resumes after reset deasserts,
since nothing valid was ever presented to it during reset in the intended
integration — `s_axis_tvalid` is expected low during reset by the upstream
producer, same convention as every other module in this pipeline). This
decision has no precedent in an already-GREEN module in this repository
(`canny_gaussian3x3`, cited by the requirement as "same pattern," is not
yet implemented) — recorded here as the design decision for `vhfill` to
implement, and to be reused verbatim by `canny_gaussian3x3`/`canny_sobel3x3`
when their turn comes.

## Architecture and dataflow

Thin wrapper around hdl-modules `common.handshake_pipeline`, per
`shared/ReusableRTL.md`: this module supplies the combinational NMS-compare
function and the packed data width; `handshake_pipeline` supplies the
elastic register and `s_axis_tready`/`m_axis_tvalid` handshake logic (the
same reuse shape the requirement documents for the not-yet-built
`canny_gaussian3x3`).

```
s_axis_tdata(100:2) --> unpack 9x11-bit taps --\
s_axis_tdata(1:0)   --> direction_in           +--> NMS compare (comb.) --> nms_result (11b)
s_axis_tuser(1)     --> border forcing --------/

nms_result & s_axis_tuser --> handshake_pipeline.input_data (13b)
s_axis_tlast              --> handshake_pipeline.input_last
s_axis_tvalid/tready      --> handshake_pipeline.input_valid/input_ready (= s_axis_tready)

handshake_pipeline.output_data(12:2) --> m_axis_tdata
handshake_pipeline.output_data(1:0)  --> m_axis_tuser
handshake_pipeline.output_last       --> m_axis_tlast
handshake_pipeline.output_valid and not reset --> m_axis_tvalid
```

- The NMS compare is computed **before** the pipeline register (on the raw
  `s_axis_tdata`/`s_axis_tuser`), not after — this mirrors
  `canny_gaussian3x3`'s documented shape ("this module supplies the
  combinational function ... `handshake_pipeline` supplies the elastic
  register") and is cheaper than registering the full 101-bit input and
  computing after: only the 13-bit result (11-bit suppressed magnitude +
  2-bit `tuser` passenger) needs to be registered, not the whole window.
- `s_axis_tuser`/`s_axis_tlast` ride through the same `handshake_pipeline`
  instance as extra data-bit lanes (`s_axis_tuser` packed alongside
  `nms_result` in `input_data`; `s_axis_tlast` on `handshake_pipeline`'s
  own dedicated `input_last`/`output_last` port) — not a second parallel
  pipeline — so they can never desync from `m_axis_tdata`, per the
  requirement.
- Packing order for `input_data` follows `canny_window3x3`'s `pack_lane`
  convention (data bits, then passenger bits): `nms_result & s_axis_tuser`
  (bits 12:2 = data, bit 1 = border, bit 0 = SOF).
- `handshake_pipeline` is instantiated with its default generics
  (`full_throughput => true`, `pipeline_control_signals => true`,
  `pipeline_data_signals => true`) — the full skid-buffer mode, giving
  the best timing characteristics and full throughput once primed, at a
  fixed 1-cycle latency from an accepted input beat to the corresponding
  output beat.

## State machines

None in this module's own RTL — `handshake_pipeline`'s internal skid-buffer
FSM (reused, not reimplemented) is the only sequencing logic in the
design.

## Algorithms

Direction-sector-to-neighbor-pair mapping (per requirement, exact):
- `"00"` (0 deg): compare `w_mm` against `w_ml`, `w_mr`.
- `"01"` (45 deg): compare `w_mm` against `w_tr`, `w_bl`.
- `"10"` (90 deg): compare `w_mm` against `w_tm`, `w_bm`.
- `"11"` (135 deg): compare `w_mm` against `w_tl`, `w_br`.

`nms_result = w_mm` if `w_mm >= neighbor_a and w_mm >= neighbor_b`, else
`0` (non-strict `>=`, ties kept — documented simplification per
requirement). Forced to `0` unconditionally when `s_axis_tuser(1)`
(border) is `'1'`.

## Numeric types and widths

- Window taps / `nms_result` / `m_axis_tdata`: `std_logic_vector(10 downto
  0)`, compared via `unsigned(...)` per project convention (`ieee.numeric_std`,
  no `std_logic_unsigned`).
- `s_axis_tdata`/`m_axis_tdata`: opaque `std_logic_vector` payload at the
  interface boundary; only the NMS compare function converts to
  `unsigned` for the `>=` comparisons, per project numeric-type
  conventions.
- `handshake_pipeline`'s own ports are `std_ulogic`/`std_ulogic_vector`;
  direct port-map connection to this module's `std_logic`/
  `std_logic_vector` signals is legal (same base type family), no
  conversion function needed — same precedent as `axi_stream_join`'s
  `handshake_merger` instantiation.

## Latency/throughput

Fixed 1-cycle latency (an accepted input beat's suppressed magnitude
appears on `m_axis` exactly one cycle later, per `handshake_pipeline`'s
full-skid-buffer-mode FSM), full throughput once started (one output beat
per accepted input beat, sustained indefinitely when neither side stalls).

## Corner cases

- Border (`s_axis_tuser(1) = '1'`): `m_axis_tdata` forced to 0
  unconditionally, regardless of the compare result; `m_axis_tuser(1)`
  still passes the border flag through unchanged (a later stage can still
  see "this was a border sample" even though the value was zeroed), per
  the requirement and per `doc/canny_arch.md`'s "growing border" rule
  ("each module ... still forces its numeric output to 0 and passes
  `TUSER(1)` through unchanged").
- Tie (`w_mm = neighbor_a` and/or `w_mm = neighbor_b`): kept (non-strict
  `>=`), a documented simplification per the requirement, not a bug.
- Reset asserted mid-stream: `m_axis_tvalid` forced low via the `reset`
  gate described above, regardless of `handshake_pipeline`'s own internal
  state; the requirement's port table has no explicit "in-flight
  transaction dropped" language beyond the generic AXI4 reset rules
  (`shared/Axi4.md` rule 18: "no in-flight transaction may be assumed"
  after reset) so this is consistent.

## Selected patterns (`shared/DesignPatterns.md`)

"Ready/valid elastic stage" pattern via `handshake_pipeline`'s full
skid-buffer mode — same pattern documented for the not-yet-built
`canny_gaussian3x3` (a combinational function feeding a reused elastic
register), not a hand-rolled register/FSM.

## AXI4/AXI4-Stream protocol decisions (`shared/Axi4.md`)

Full AXI4-Stream backpressure on both `s_axis`/`m_axis`, delegated
entirely to `handshake_pipeline`'s own proven `input_ready`/`output_valid`
handshake logic (skid-buffer mode: `TVALID`/`TREADY` decoupled correctly,
no combinational loop, payload held stable in the skid register while
`output_ready='0'`).

## Verification plan

- Dedicated VUnit unit test under `modules/canny/test/`, using VUnit's
  raw `axi_stream_master`/`axi_stream_slave` VCs directly (`s_axis_tuser`/
  `m_axis_tuser` are 2 bits, not a multiple of 8, which fails
  `bfm.axi_stream_master/slave`'s `user_width mod 8 = 0` assertion — same
  rationale as `canny_window3x3`'s testbench, `shared/Vunit.md` §12).
- `stall_probability_percent` generic, swept via `module_canny.py` (0
  for the dedicated full-throughput test, nonzero otherwise), applied to
  both the master and slave VC `stall_config`.
- `test_direction_sectors`: directed cases covering all 4 direction
  sectors, each with a local-maximum-kept case, a local-non-maximum-
  suppressed case, and a tie case (must be kept, non-strict `>=`).
- `test_random_data`: random 9-tap windows and random direction per beat,
  expected result computed inline in the testbench per the exact formula
  in the requirement (independently re-derived, not copy-pasted from the
  RTL).
- `test_border_forces_zero`: border bit asserted on some beats — expected
  output forced to 0 regardless of the compare result; `tuser(1)` still
  expected to read back `'1'` (passthrough).
- `test_full_throughput`: zero stall on both sides, `check_relation` on
  total elapsed time vs. beat count + the fixed 1-cycle latency + a small
  margin.
- Prefer non-blocking `push_axi_stream`/`check_axi_stream(...,
  blocking => false)` per `shared/Vunit.md` §12, since beats are pushed
  and checked interleaved in one process (mandatory per the project's own
  documented gotcha: blocking `check_axi_stream` combined with per-beat
  interleaving deadlocks on any nonzero pipeline latency).
- No separate Python reference model: the transform is cheap enough to
  compute inline in the testbench (a small `case` on the 2-bit direction
  plus one 11-bit `>=` comparison pair), following `canny_window3x3`'s and
  `axi_stream_join`'s own precedent of an inline VHDL model for
  structurally similar cases.

## Implementation Notes (vhfill)

Two `--@`-marked placeholders in `src/canny_nms.vhd`, both deliberately
wrong so the RED testbench run fails for the *expected* reason (not a
testbench-authoring bug):

1. The direction-to-neighbor-pair `case` statement maps `"00"` to
   `w_tm`/`w_bm` (a duplicate of `"10"`'s mapping) instead of the
   requirement's `w_ml`/`w_mr`.
2. The compare uses strict `>` instead of the requirement's non-strict
   `>=`, which will fail the dedicated tie-case checks.

Filled in below once `vhfill` completes.

(empty — to be filled in by `vhfill`)
