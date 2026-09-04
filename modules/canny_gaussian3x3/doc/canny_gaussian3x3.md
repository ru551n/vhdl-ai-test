# canny_gaussian3x3

## Purpose

3x3 approximate Gaussian smoothing of the raw pixel window, as a full
AXI4-Stream elastic stage (not windowing itself — consumes the window
produced by the upstream `canny_window3x3` instance). See
`modules/canny_gaussian3x3/doc/canny_gaussian3x3_req.md` and
`modules/canny_gaussian3x3/doc/canny_gaussian3x3_proposal.md`.

## Entity and architecture

Entity `canny_gaussian3x3`, architecture `rtl`.

## Generics

None (fixed 8-bit pixel width and fixed integer weights).

## Ports

| Name | Mode | Type/width | Description |
|---|---|---|---|
| `clk` | in | std_logic | single clock domain |
| `rst_n` | in | std_logic | synchronous active-low reset |
| `s_axis_tvalid` | in | std_logic | input valid |
| `s_axis_tready` | out | std_logic | backpressure to producer |
| `s_axis_tdata` | in | std_logic_vector(71 downto 0) | 3x3 raw-pixel window, row-major MSB-to-LSB: `w_tl`=71:64, `w_tm`=63:56, `w_tr`=55:48, `w_ml`=47:40, `w_mm`=39:32, `w_mr`=31:24, `w_bl`=23:16, `w_bm`=15:8, `w_br`=7:0 |
| `s_axis_tuser` | in | std_logic_vector(1 downto 0) | bit0=SOF, bit1=border |
| `s_axis_tlast` | in | std_logic | EOL |
| `m_axis_tvalid` | out | std_logic | output valid |
| `m_axis_tready` | in | std_logic | backpressure from consumer |
| `m_axis_tdata` | out | std_logic_vector(7 downto 0) | smoothed pixel |
| `m_axis_tuser` | out | std_logic_vector(1 downto 0) | passthrough unchanged |
| `m_axis_tlast` | out | std_logic | passthrough unchanged |

## Clocking and reset

Single clock (`clk`); `rst_n` is present on the entity per the
requirement's fixed port list but is **not wired to anything internal**.

**Known limitation, confirmed against
`hdl-modules:modules/common/src/handshake_pipeline.vhd`:** the wrapped
`common.handshake_pipeline` instance — the only sequential state this
module owns — has no reset port at all. Its internal state starts at
its VHDL default (`wait_for_input_valid`, `output_valid = '0'`,
`input_ready = '1'`) at elaboration/power-up, which is the same
externally-observable "empty, ready" state a reset would produce, so a
single `rst_n` pulse applied once before any beat is accepted (the
pattern used by every existing testbench in this repo, e.g.
`tb_canny_window3x3`'s `rst_n_gen`) behaves correctly. A mid-stream
reset would not flush an in-flight skid-buffered beat — out of scope,
same precedent as `canny_window3x3`'s un-resettable line-buffer FIFOs
(see `modules/canny_window3x3/doc/canny_window3x3.md` "Clocking and
reset").

## Interfaces/protocols

Full AXI4-Stream elastic handshake on both sides per `shared/Axi4.md`,
entirely delegated to the wrapped `common.handshake_pipeline` instance:
`m_axis_tvalid`/`s_axis_tready` are that instance's own registered
`output_valid`/`input_ready` (never combinationally dependent on
`m_axis_tready`/`s_axis_tvalid` respectively). No loss/no duplication
inherited directly from the wrapped module's own verified behavior.

## Functional behavior

`m_axis_tdata = (tl + 2*tm + tr + 2*ml + 4*mm + 2*mr + bl + 2*bm + br) / 16`,
using integer weights `[1,2,1;2,4,2;1,2,1]` (sum = 16, exact power-of-two
divide via right-shift, no rounding beyond truncation), computed
combinationally from the 9 taps unpacked from `s_axis_tdata`. Forced to
`x"00"` when `s_axis_tuser(1)` (border) is `'1'`. `m_axis_tuser(1)`
passes `s_axis_tuser(1)` through unchanged — this module does not
compute border itself, it only reacts to it by zeroing its numeric
output. `s_axis_tuser`/`s_axis_tlast` ride through the same
`handshake_pipeline` instance as extra data-bit lanes packed alongside
the computed pixel (`gaussian_result & s_axis_tuser & s_axis_tlast`),
so they can never desync from `m_axis_tdata`.

## Timing/latency

Latency: 1 accepted input beat before the corresponding output beat is
registered (`common.handshake_pipeline`'s default `full_throughput =>
true` skid-aside-buffer mode). Throughput: 1:1 with `s_axis`, sustained
at full rate when neither side stalls.

## Registers/configuration

None (no run-time configuration registers; the only register is
`handshake_pipeline`'s own internal skid-buffer state).

## Dependencies

- `common.handshake_pipeline` (hdl-modules,
  `modules/common/src/handshake_pipeline.vhd`), instantiated once with
  default generics (`full_throughput => true`, `pipeline_control_signals
  => true`, `pipeline_data_signals => true`) and `data_width => 11`
  (8 pixel bits + 2 tuser bits + 1 tlast bit).

## Implementation notes

- The 9 taps are unpacked from `s_axis_tdata` by fixed bit-slicing (no
  registers): matches the requirement's port table exactly.
- Weighted-sum accumulator is `unsigned(11 downto 0)` (worst case
  `16*255=4080 < 4096`); the `/16` divide is an exact 4-bit right-slice,
  no rounding.
- `input_data <= gaussian_result & s_axis_tuser & s_axis_tlast` is the
  only payload handed to `handshake_pipeline` — `tuser`/`tlast` are
  extra data-bit lanes, not `handshake_pipeline`'s own separate
  `input_last`/`output_last` ports (left unconnected/defaulted), so they
  cannot desync from the computed pixel: same register, same enable
  condition.
- Border masking (`m_axis_tdata` forced to `0` when `s_axis_tuser(1) =
  '1'`) happens combinationally before the `handshake_pipeline` input,
  not after its output register.

## Verification notes

- Dedicated VUnit unit test under `modules/canny_gaussian3x3/test/`,
  using VUnit's raw `axi_stream_master`/`axi_stream_slave` VCs directly
  (not the hdl-modules `bfm.*` wrappers, since `s_axis_tuser` is 2 bits,
  which fails `bfm.axi_stream_master`'s `user_width mod 8 = 0`
  assertion).
- `stall_probability_percent` generic swept via
  `module_canny_gaussian3x3.py` (0 for the dedicated full-throughput
  test, nonzero otherwise).
- Random-data correctness test: expected weighted-sum result computed
  in the testbench from the exact integer formula in
  `canny_gaussian3x3_req.md`, independently of the RTL under test.
- Border-forces-zero-output test: random taps with
  `s_axis_tuser(1) = '1'` injected, expecting `m_axis_tdata = 0` and
  `m_axis_tuser(1)` unchanged passthrough.
- Full-throughput (zero stall) timing check via `check_relation`.
