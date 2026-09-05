# canny_gaussian3x3 — proposal

## Requirements summary

3x3 approximate Gaussian smoothing of the raw pixel window, per
`modules/canny/doc/canny_gaussian3x3_req.md`. A pure
combinational function of the 9 unpacked taps (fixed integer weights
`[1,2,1;2,4,2;1,2,1]`, sum = 16, exact power-of-two divide via
right-shift), wrapped in a single elastic registered stage reused from
hdl-modules' `common.handshake_pipeline` (thin wrapper, no hand-rolled
handshake/pipeline logic of its own). Forces `m_axis_tdata` to `0` when
`s_axis_tuser(1)` (border) is `'1'`; does not compute border itself,
only reacts to it. `s_axis_tuser`/`s_axis_tlast` ride through the same
`handshake_pipeline` instance as extra data-bit lanes so they can never
desync from `m_axis_tdata`.

## Interface (copied from the requirement's structural section)

No generics (fixed 8-bit pixel width and fixed integer weights).

Ports: `clk`, `s_axis_{tvalid,tready,tdata,tuser,tlast}`,
`m_axis_{tvalid,tready,tdata,tuser,tlast}` per the requirement's port
table. `s_axis_tdata` is 72 bits (3x3 raw-pixel window, packed row-major
MSB-to-LSB: `w_tl`=71:64 ... `w_br`=7:0). `s_axis_tuser`/`m_axis_tuser`
are 2 bits (bit0=SOF, bit1=border). `m_axis_tdata` is 8 bits (smoothed
pixel).

## Clock/reset behavior

Single clock, resetless, confirmed against
`hdl-modules:modules/common/src/handshake_pipeline.vhd`: the entity has
no reset port at all (only `clk` plus the input/output handshake
signals). At elaboration `handshake_pipeline`'s own internal state
starts at its VHDL default (`wait_for_input_valid`, `output_valid =
'0'`, `input_ready = '1'`), which is the same externally-observable
"empty, ready" state a runtime reset would produce, so no reset port is
needed, per `shared/TsfpgaCodingConventions.md`'s resetless-by-default
policy.

## Architecture and dataflow

Reuses hdl-modules `common.handshake_pipeline` (default generics:
`full_throughput => true`, `pipeline_control_signals => true`,
`pipeline_data_signals => true` — the full skid-aside-buffer mode, per
`shared/ReusableRTL.md`'s "instantiate an existing module unmodified"
preference, no generic overrides needed since the requirement doesn't
call for the lower-footprint/lower-throughput variants) as the *only*
piece of sequential logic in this module:

```
s_axis_tdata(71:0) --> unpack 9 taps --> gaussian_weighted_sum() --+
s_axis_tuser(1) (border) -------------------------------> mask ---+--> input_data(10:0) --> handshake_pipeline --> output_data(10:0) --> unpack --> m_axis_tdata/tuser/tlast
s_axis_tuser(1:0), s_axis_tlast ------------------------------------+
s_axis_tvalid/tready <-----------------------------------------------> input_valid/ready
m_axis_tvalid/tready <----------------------------------------------> output_valid/ready
```

- The 9 taps are unpacked from `s_axis_tdata` by fixed bit-slicing (no
  registers): `w_tl`=71:64, `w_tm`=63:56, `w_tr`=55:48, `w_ml`=47:40,
  `w_mm`=39:32, `w_mr`=31:24, `w_bl`=23:16, `w_bm`=15:8, `w_br`=7:0 —
  matches the requirement's port table exactly.
- A pure function computes the weighted sum
  `(tl + 2*tm + tr + 2*ml + 4*mm + 2*mr + bl + 2*bm + br) / 16` in
  `unsigned` arithmetic (9 taps, each 8 bits unsigned, so the widest
  intermediate sum is `<= 16 * 255 = 4080`, fits in 12 bits; the `/16`
  is an exact `srl 4`/slice, no rounding).
- The result is forced to `x"00"` when `s_axis_tuser(1) = '1'` (border),
  per requirement — combinationally, before packing into
  `handshake_pipeline`'s `input_data`.
- `input_data <= gaussian_result & s_axis_tuser & s_axis_tlast` (8 + 2 +
  1 = 11 bits) is the *only* payload handed to the single
  `handshake_pipeline` instance — `tuser`/`tlast` are extra data-bit
  lanes packed alongside the computed pixel, not a second parallel
  pipeline and not the entity's own dedicated (but separate)
  `input_last`/`output_last` ports, so they are guaranteed to reach
  `m_axis_tdata`'s corresponding beat in lockstep by construction (same
  register, same enable condition).
- `output_data` is unpacked the same way: `m_axis_tdata <=
  output_data(10 downto 3)`, `m_axis_tuser <= output_data(2 downto 1)`,
  `m_axis_tlast <= output_data(0)`.
- `s_axis_tready`/`m_axis_tvalid`/`s_axis_tvalid`/`m_axis_tready` map
  directly to `handshake_pipeline`'s `input_ready`/`output_valid`/
  `input_valid`/`output_ready` — no additional gating logic in this
  module at all.

## State machines

None. `handshake_pipeline`'s own internal skid-buffer FSM (opaque to
this module) is the only sequencing in the design.

## Algorithms

3x3 fixed-weight Gaussian-blur weighted sum (`[1,2,1;2,4,2;1,2,1]`,
`/16` via bit-slice), computed once per accepted input beat, wrapped by
`handshake_pipeline`'s existing skid-aside-buffer elastic register.

## Numeric types and widths

- The 9 taps: sliced as `std_logic_vector(7 downto 0)` from
  `s_axis_tdata`, converted to `unsigned` at the point they enter the
  weighted-sum function (arithmetic boundary, per `shared/CodingStyle.md`
  numeric-type gate) — `std_logic_vector` stays the opaque interface
  type, `unsigned` is used for every add/shift.
- Weighted-sum accumulator: `unsigned(11 downto 0)` (worst case
  `16*255=4080 < 4096`), right-shifted (`srl 4` equivalent via a
  constrained slice) down to `unsigned(7 downto 0)` for the truncated
  result — no rounding beyond truncation, per requirement.
- `handshake_pipeline`'s `data_width` generic: `11` (`8` pixel bits + `2`
  tuser bits + `1` tlast bit).

## Latency/throughput

Latency: 1 accepted input beat before the corresponding output beat is
registered (the `handshake_pipeline` skid-buffer's fixed 1-cycle
latency in `full_throughput => true` mode — see "test_full_throughput"
in the verification plan for the exact fencepost). Throughput: 1:1 with
`s_axis`, sustained at full rate when neither side stalls (skid-aside
buffer mode never inserts bubbles).

## Corner cases

- Border beat (`s_axis_tuser(1) = '1'`): `m_axis_tdata` forced to `0`;
  `m_axis_tuser(1)` still passes through `'1'` unchanged (this module
  does not recompute border, only reacts to it for the numeric output).
- All-zero/all-max taps: covered by the general weighted-sum formula,
  no special-casing needed (max sum `4080`, `/16 = 255`, fits in 8 bits
  with no overflow).
- Back-to-back beats at full throughput: `handshake_pipeline`'s
  `full_throughput => true` mode (default) guarantees no bubble
  insertion — see hdl-modules `handshake_pipeline.vhd`
  `choose_mode`/`full_throughput and pipeline_data_signals and
  pipeline_control_signals` branch (skid-aside buffer).

## Selected patterns (`shared/DesignPatterns.md`)

"Ready/valid elastic stage" pattern via direct reuse of
`common.handshake_pipeline` — no hand-rolled FSM/counter, per
`shared/ReusableRTL.md`'s reuse-before-authoring rule: this module
supplies only the combinational function and the packed data width, the
elastic register and handshake logic are entirely the wrapped module's.

## AXI4/AXI4-Stream protocol decisions (`shared/Axi4.md`)

Full AXI4-Stream backpressure on both sides, entirely delegated to
`handshake_pipeline`: `m_axis_tvalid` is `handshake_pipeline`'s own
registered `output_valid` (never combinationally dependent on
`m_axis_tready`); `s_axis_tready` is `handshake_pipeline`'s own
`input_ready` (a registered skid-flag in `full_throughput => true`
mode, not combinationally derived from `s_axis_tvalid`). No loss/no
duplication is inherited directly from the wrapped module's own
verified behavior (see hdl-modules `tb_handshake_bfm`/`tb_handshake_merger`
precedent already relied upon by this project's `axi_stream_join`).

## Verification plan

- Dedicated VUnit unit test under `modules/canny/test/`,
  using VUnit's raw `axi_stream_master`/`axi_stream_slave` VCs directly
  (not the hdl-modules `bfm.*` wrappers, since `s_axis_tuser` is 2 bits,
  which fails `bfm.axi_stream_master`'s `user_width mod 8 = 0`
  assertion).
- `stall_probability_percent` generic swept via
  `module_canny.py` (0 for the dedicated full-throughput
  test, nonzero otherwise), non-blocking `push_axi_stream`/
  `check_axi_stream(..., blocking => false)`.
- Random-data correctness test: expected weighted-sum result computed
  in the testbench from the exact integer formula in
  `canny_gaussian3x3_req.md`, independently of the RTL under test.
- Border-forces-zero-output test: random taps with
  `s_axis_tuser(1) = '1'` injected, expecting `m_axis_tdata = 0` and
  `m_axis_tuser(1)` unchanged passthrough.
- Full-throughput (zero stall) timing check via `check_relation`.

## Implementation Notes (vhfill)

- `gaussian_weighted_sum`: widened the 9 taps to a 12-bit accumulator
  (`tap_width + 4`, worst case `16*255=4080` fits) before summing the
  weighted terms (duplicated `resize()` additions instead of multiplying by
  literal weights, since the weights are only 1/2/4), then did the exact
  `/16` divide via `shift_right(..., 4)` (truncating, no rounding) and
  resized back down to 8 bits.
- `masked_result`: a `when/else` mux on `s_axis_tuser(1)` (border), forcing
  `x"00"` at the border and passing through `gaussian_result` otherwise.
- GHDL rejected `return (others => '0');` in `gaussian_weighted_sum`'s
  unconstrained-return-type stub ("'others' choice not allowed for an
  aggregate in this context") -- had to use `return to_unsigned(0,
  tap_width);` instead for the RED-phase stub, since GHDL cannot resolve
  the aggregate's length from an unconstrained `return unsigned` alone.
- The RED-phase testbench (raw `vunit_lib.axi_stream_master`/
  `axi_stream_slave` VCs, `push_axi_stream`/`check_axi_stream(blocking =>
  false)`) initially reported a false "all passed" at the very first
  post-reset clock edge (`simulation stopped @35ns`) even against the
  all-zero RTL stub: `send()` to an unbounded-inbox VC actor does not block,
  so the whole test loop (300-500 iterations) raced through in delta
  cycles and reached `test_runner_cleanup`'s `core_pkg.stop` before the
  VCs ever drove/sampled a single real bus cycle. Fixed by adding
  `wait_until_idle(net, as_sync(axi_master_in))` and
  `wait_until_idle(net, as_sync(axi_slave_out))` (from `vunit_lib.sync_pkg`)
  right before `test_runner_cleanup` in every test case -- confirmed by the
  resulting realistic per-test simulation durations (4795-5055 ns for the
  300/500-word loops, 85 ns for the 3-case directed border test) and by
  re-confirming a real RED against the stub RTL before implementing.
