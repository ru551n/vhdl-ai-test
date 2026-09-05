# canny_hysteresis — proposal

## Requirements summary

Last stage of the canny pipeline: local (3x3) hysteresis over a
pre-classified window. A pixel is a final edge if it is `strong`, or
`weak` with at least one `strong` neighbor among its 8 window neighbors;
`border` forces `edge='0'` regardless. Drives the IP's top-level `m_axis`
directly, so `m_axis_tuser` is 1 bit (SOF only) — the incoming `border` bit
(`s_axis_tuser(1)`) is consumed here (it forces `edge='0'`) and is **not**
re-exposed, per `doc/canny_arch.md` "Border signal carrying (addendum, rev
2.1)". See `modules/canny/doc/canny_hysteresis_req.md` for the
full requirement.

## Interface (copied from the requirement's structural section)

No generics (fixed 2-bit classification width, fixed 8-bit output byte).

Ports: `clk`, `reset`, `s_axis_{tvalid,tready,tdata,tuser,tlast}`,
`m_axis_{tvalid,tready,tdata,tuser,tlast}`.
- `s_axis_tdata(17 downto 0)`: 3x3 classification window, row-major
  MSB-to-LSB, 2 bits/tap: `w_tl`=17:16, `w_tc`=15:14, `w_tr`=13:12,
  `w_ml`=11:10, `w_mm`=9:8, `w_mr`=7:6, `w_bl`=5:4, `w_bc`=3:2, `w_br`=1:0.
- `s_axis_tuser(1 downto 0)`: bit0=SOF, bit1=border (internal-link
  convention, `user_width=2`).
- `m_axis_tdata(7 downto 0)`: bit0=edge, bits 7:1='0'.
- `m_axis_tuser(0 downto 0)`: bit0=SOF passthrough only — top-level
  boundary convention, `user_width=1`, no border bit.
- `m_axis_tlast`: passthrough of `s_axis_tlast`.

## Clock/reset behavior

Single clock, synchronous active-high reset (`reset`).

`common.handshake_pipeline` (verified via `vhdl-rag-mcp` against
`hdl-modules:modules/common/src/handshake_pipeline.vhd`) has **no reset
port at all** — only `clk`. To still honor this module's own `reset`
contract (AXI4 rule 17: every interface signal settles to a deterministic
value during reset) without reimplementing `handshake_pipeline`'s internal
skid-buffer FSM, the wrapper adds a small amount of reset-gating glue
around the otherwise-unmodified instance:
- `input_valid` (into the pipeline) is gated with `reset` so no new beat is
  presented to the pipeline while in reset.
- `output_ready` (into the pipeline, i.e. what would normally just be
  `m_axis_tready`) is forced to `'1'` while `reset='1'` (`m_axis_tready or
  not reset`) — this force-drains any state the pipeline's internal skid
  buffer/output register may be holding from before the reset pulse. The
  pipeline's own FSM (`handshake_pipeline.vhd`, `full_handshake_throughput`/
  `wait_for_output_ready` states) fully empties (`output_valid` returns to
  `'0'`) within at most 2 cycles once both `input_valid='0'` and
  `output_ready='1'` hold simultaneously — well within any reset pulse of
  reasonable length (every testbench in this repo holds `reset='1'` for
  several cycles before the test body starts, e.g. `tb_axi_stream_join`'s
  `reset_gen`).
- `m_axis_tvalid` is additionally masked with `reset`
  (`pipeline_output_valid and not reset`) so no beat — stale or otherwise — is
  ever visible on `m_axis` while reset is asserted, even during the (at
  most 2-cycle) internal drain window above.
- `s_axis_tready` is masked with `reset` the same way, so the upstream
  producer is held off during reset too.

This is not a re-implementation of `handshake_pipeline`'s logic (still a
thin wrapper, per `shared/ReusableRTL.md`) — it is boundary-level reset
gating of the four signals crossing into/out of an otherwise-untouched,
unmodified instance.

## Architecture and dataflow

```
s_axis_tdata(17:0), s_axis_tuser(1:0) --> compute_edge (combinational fn)
                                            |
                                            v
                          edge_byte(7:0), sof(0:0)  -- packed as (sof & edge_byte), 9 bits
                                            |
                                            v
                          common.handshake_pipeline (data_width=9)
                          input_valid <= s_axis_tvalid and not reset
                          input_last  <= s_axis_tlast
                          output_ready <= m_axis_tready or reset
                                            |
                                            v
                          output_data(8)=sof, output_data(7:0)=edge_byte, output_last=tlast
                                            |
                                            v
              m_axis_tdata <= output_data(7:0); m_axis_tuser(0) <= output_data(8);
              m_axis_tlast <= output_last; m_axis_tvalid <= output_valid and not reset
```

`compute_edge` is a pure combinational function: unpacks the 9 2-bit taps
from `s_axis_tdata`, applies the hysteresis rule (see "Algorithms" below),
and forces `edge='0'` when `s_axis_tuser(1)` (border) is `'1'`. Its result,
concatenated with the passthrough SOF bit, is what `handshake_pipeline`
registers — the border bit itself is consumed at this combinational stage
and never carried into (or out of) the pipeline register, which is exactly
the 2-bit-in/1-bit-out `tuser` narrowing the requirement calls for.

`handshake_pipeline` is instantiated with its default generics
(`full_throughput => true`, `pipeline_control_signals => true`,
`pipeline_data_signals => true`) — the full skid-buffer mode — giving one
registered pipeline stage with full elastic backpressure and full
throughput, matching "a single elastic registered stage" per the
requirement and `canny_gaussian3x3`'s documented pattern.

## State machines

None owned by this module — `handshake_pipeline`'s internal skid-buffer FSM
is opaque/reused, not reimplemented.

## Algorithms

Classification codes (2 bits/tap): `"10"`=strong, `"01"`=weak, anything
else (`"00"`, `"11"`)=none/not-strong.

```
center       := w_mm
any_strong   := OR over {w_tl,w_tc,w_tr,w_ml,w_mr,w_bl,w_bc,w_br} = "10"
edge         := (center = "10") or (center = "01" and any_strong)
edge         := edge and not border   -- border forces edge='0'
```

`m_axis_tdata <= "0000000" & edge` (bit0=edge, bits 7:1='0').

## Numeric types and widths

- `s_axis_tdata`/`m_axis_tdata`: `std_logic_vector`, opaque packed
  bitfields — sliced with constant, locally-static indices only (no
  runtime/generic-conditioned bit-width indexing in this module, so no
  `if/generate`-vs-`when/else` GHDL bounds-check concern applies here,
  unlike `canny_window3x3`'s `user_width`-conditioned slicing).
- `handshake_pipeline`'s ports are `std_ulogic`/`std_ulogic_vector`; this
  module's ports are `std_logic`/`std_logic_vector` per project convention
  — direct port-map connection is legal (same base type family).

## Latency/throughput

One clock cycle of registered latency (the `handshake_pipeline` skid
buffer), one beat/cycle at full throughput once primed, matching
`handshake_pipeline`'s own documented full-throughput-mode contract.

## Corner cases

- `center=strong`: always `edge='1'` regardless of any neighbor
  (unless border).
- `center=weak`, no neighbor strong: `edge='0'`.
- `center=weak`, at least one neighbor strong: `edge='1'`.
- `center=none` ("00"/"11" other than the two defined codes): `edge='0'`
  regardless of neighbors.
- `border='1'`: `edge='0'` unconditionally, regardless of the window
  comparison above.
- Reset mid-stream: see "Clock/reset behavior" above (bounded drain via
  `output_ready` forcing).

## Selected patterns (`shared/DesignPatterns.md`)

"Ready/valid elastic stage" pattern's registered variant — a single
`common.handshake_pipeline` instance provides the one elastic register
stage, reused unmodified (thin wrapper around a purely combinational
compare function), matching every other internal canny-pipeline stage
(`canny_gaussian3x3`, `canny_sobel3x3`, `canny_nms`, `canny_threshold`) per
the requirement — this module additionally narrows `tuser` 2->1 bit at the
wrapper boundary (border consumed, not piped through the pipeline
register), unique to this stage.

## AXI4/AXI4-Stream protocol decisions (`shared/Axi4.md`)

Full AXI4-Stream backpressure on both `s_axis` and `m_axis`, delegated
entirely to `handshake_pipeline`'s own verified ready/valid contract
(`TVALID` not a function of `TREADY`, stable payload while
`TVALID='1'`/`TREADY='0'`, no loss/duplication). Reset-time determinism
(rule 17) is handled by the boundary-level gating described above, since
the reused primitive itself has no reset port.

## Verification plan

- Dedicated VUnit unit test under `modules/canny/test/`, using
  VUnit's raw `axi_stream_master`/`axi_stream_slave` VCs directly on both
  `s_axis` (2-bit tuser) and `m_axis` (1-bit tuser) — the hdl-modules
  `bfm.axi_stream_master`/`slave` wrapper's `user_width mod 8 = 0`
  restriction is violated by both widths, so raw VUnit VCs are used
  consistently on both sides (not a mix), per `shared/Vunit.md` §12's
  documented caveat.
- `stall_probability_percent : natural` generic, swept via
  `module_canny.py` (0 for the dedicated full-throughput test,
  nonzero otherwise), independently randomized `stall_config` on both
  sides per `shared/Vunit.md` §12's mandatory-backpressure rule.
- Directed cases: all 4 combinations of (center=strong/weak/none) x (a
  neighbor strong or not), per the exact rule in the requirement's
  functional description.
- Directed border case: border='1' forces `edge='0'` even when the window
  comparison alone would otherwise produce `edge='1'` (e.g.
  center=strong); also asserts the SOF bit (`m_axis_tuser(0)`) still passes
  through correctly on the same beat — `m_axis_tuser` being only 1 bit wide
  structurally guarantees border is not re-exposed, so this is a SOF
  passthrough check, not a border non-re-exposure check.
- Random-data correctness test: random 2-bit-per-tap windows and random
  border/SOF bits, expected `edge`/SOF computed inline in the testbench
  per the same rule (small enough to compute inline; no separate Python
  golden model needed for this module's simple combinational rule, per
  `shared/Vunit.md`'s "Python reference models" guidance to reserve that
  machinery for the canny-specific numeric stages / full-pipeline
  integration test).
- Full-throughput (zero-stall) timing check via `check_relation`.

## Implementation Notes (vhfill)

The two `--@`-marked stubs — `compute_edge`'s always-`'0'` result and the
four reset-gating signal assignments (all hardcoded `'0'`) — were filled in
exactly as designed above: `compute_edge` implements the center-strong /
center-weak-with-strong-neighbor rule with border forcing `'0'`, and the
reset-gating assigns `pipeline_input_valid <= s_axis_tvalid and not reset`,
`pipeline_output_ready <= m_axis_tready or reset`, `s_axis_tready <=
pipeline_input_ready and not reset`, `m_axis_tvalid <= pipeline_output_valid and
reset`. No deviation from this document's design was needed for the RTL
itself.

**RED-confirmation caught a testbench bug, not an RTL one**: the first
`run.py "canny_hysteresis*"` run reported all 4 tests **passing** even
though `canny_hysteresis.vhd` was still fully stubbed (`s_axis_tready` and
`m_axis_tvalid` hardcoded `'0'`, so the DUT could never accept or produce a
single beat). Root cause: the testbench's `main` process called
`test_runner_cleanup(runner)` immediately after queuing all
`push_axi_stream`/non-blocking `check_axi_stream` calls, with no
`wait_until_idle` on either VC. Since `push_axi_stream` and non-blocking
`check_axi_stream` only enqueue messages on the VCs' internal command
queues (they don't block until the transaction is actually driven/checked
on the interface), `test_runner_cleanup` ended the simulation at ~35 ns —
before the VCs had driven or checked anything — and VUnit reported a false
pass. Fixed by adding `wait_until_idle(net, as_sync(axi_master));` and
`wait_until_idle(net, as_sync(axi_slave));` right before
`test_runner_cleanup` in `test/tb_canny_hysteresis.vhd`, which requires
`use vunit_lib.sync_pkg.all;` (not pulled in by `vunit_context`, only by
`vc_context`, which this testbench doesn't use). After this fix, the RED
run correctly failed all 4 tests with a `test_runner_watchdog` timeout
(2 ms), confirming the stubs were genuinely exercised. **New gotcha for
future modules**: whenever `push_axi_stream`/non-blocking `check_axi_stream`
calls are queued in a test and nothing else in the process later blocks on
their completion, an explicit `wait_until_idle(net, as_sync(...))` on every
VC used in that test is required before `test_runner_cleanup`, or a fully
broken DUT can falsely pass.

After the testbench fix, GREEN was reached on the first `vhfill` pass with
no further RTL iteration needed: all 4 tests passed, and
`test_full_throughput` completed in ~5055 ns of simulation time for 500
beats at a 10 ns clock period with zero stall — consistent with sustaining
one output beat per clock cycle through the single-stage
`common.handshake_pipeline` skid buffer, well inside the `check_relation`
margin.
