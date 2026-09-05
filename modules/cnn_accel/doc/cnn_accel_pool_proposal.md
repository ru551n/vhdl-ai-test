# cnn_accel_pool — proposal

## Requirements summary

Spatial reduction over one `cnn_accel_window_gen` pooling window per beat,
per `modules/cnn_accel/doc/cnn_accel_pool_req.md`: `OPCODE_POOL_MAX`
computes the int8 max of the window's active taps and emits it directly
(already a valid int8 result); `OPCODE_POOL_AVG` computes the int32 sum of
the window's active taps and emits it for `cnn_accel_bias_requant` to
scale/round (the pool-area divide is a downstream requantize operation,
per `doc/cnn_accel_arch.md` "Non-obvious boundary rationale" — not
performed in this module). Two generics (`g_max_kernel_size`,
`g_accum_width`), one AXI4-Stream input, two mutually-exclusive AXI4-Stream
outputs selected by `cfg_opcode`, full backpressure on all three links.

## Interface (copied from the requirement's structural section)

Generics: `g_max_kernel_size : positive` (upper bound on pool `K_h`/`K_w`),
`g_accum_width : positive` (`POOL_AVG` sum width).

Ports: `clk`, `reset` (`std_ulogic`, `reset` = `reset_internal`);
`cfg_opcode`, `cfg_pool_kernel_h`, `cfg_pool_kernel_w`
(`std_ulogic_vector(7 downto 0)`); `s_window_m2s`/`s2m`
(`axi_stream_pkg.axi_stream_m2s_t`/`s2m_t`, from `cnn_accel_window_gen`);
`m_max_m2s`/`s2m` (`data` width 8, `POOL_MAX` result, to the final output
`handshake_mux`); `m_avgsum_m2s`/`s2m` (`data` width `g_accum_width`,
`POOL_AVG` sum, to `cnn_accel_bias_requant`).

## vhdesign addition: the fixed-width `axi_stream_pkg` data-field contract

`hdl-modules`' `axi_stream_m2s_t.data` is `std_ulogic_vector(axi_stream_data_sz
- 1 downto 0)` with `axi_stream_data_sz` a **package constant fixed at 128**
(not a generic) — every AXI4-Stream link in this IP that reuses this record
type is capped at 128 bits of data per beat, per `doc/cnn_accel_arch.md`'s
"Interface record policy" (reuse the type directly, do not re-derive a
wider one). Consequence for this module: `s_window_m2s.data`'s low
`cfg_pool_kernel_h * cfg_pool_kernel_w * 8` bits must fit in 128, i.e.
**`g_max_kernel_size**2 * 8 <= 128`, so `g_max_kernel_size <= 4`** for any
instantiation of this module (checked by an elaboration-time `assert` in
the RTL, together with the analogous `g_accum_width <= 128` check for
`m_avgsum_m2s.data`). This is a real, load-bearing constraint on the
pooling-path `cnn_accel_window_gen` instance's own `g_max_kernel_size`
(single channel per beat for pooling, no `g_line_buffer_channels`
multiplier in this module's own req) — recorded here so the
`cnn_accel_window_gen` implementer and the top-level integrator size that
instance's generic consistently with this module's contract.

## Clock/reset behavior

Single clock, synchronous active-high `reset`, per `doc/cnn_accel_arch.md`
"Reset policy" (every stateful module in this IP gets a host-restorable
`reset`). Only the pipeline's `valid` bit (`out_valid_q`) has an explicit
reset branch — the held result/tag/`last` registers (`out_max_q`,
`out_avgsum_q`, `out_is_avg_q`, `out_last_q`) are not reset, since their
content is `VALUE_IRRELEVANT_UNTIL_VALID` (per `vhfill`'s "Reset
minimization" gate): once `out_valid_q='0'` no downstream consumer may look
at them, and the very next accepted beat overwrites them all together
before `out_valid_q` is ever re-asserted.

## Architecture and dataflow

```
s_window_m2s.data(low bits) --> extract_taps (comb.) --> taps(0 .. g_max_kernel_size**2 - 1)
cfg_pool_kernel_h/w          --> active_count (comb.)  --/

taps, active_count --> reduce_max (comb.)    --> max_result    (signed(7 downto 0))
taps, active_count --> reduce_sum (comb.)    --> avgsum_result (signed(g_accum_width-1 downto 0))
cfg_opcode                                   --> is_avg_sel

s_window_m2s.valid, s_window_s2m.ready --> accepted (comb.)
accepted --> { out_valid_q <= '1'; out_is_avg_q <= is_avg_sel;
               out_last_q <= s_window_m2s.last;
               out_max_q <= max_result; out_avgsum_q <= avgsum_result }  (registered)

out_valid_q, out_is_avg_q, out_max_q, out_avgsum_q, out_last_q
  --> m_max_m2s   (valid = out_valid_q and not out_is_avg_q)
  --> m_avgsum_m2s (valid = out_valid_q and out_is_avg_q)
```

One shared one-entry output register (`out_valid_q`/`out_is_avg_q`/
`out_last_q`/`out_max_q`/`out_avgsum_q`), not two independent per-port
registers — the register holds **either** a max result **or** an avgsum
result, tagged by `out_is_avg_q`, and both `m_max_m2s.valid`/
`m_avgsum_m2s.valid` are derived from the same `out_valid_q`/`out_is_avg_q`
pair. This structurally guarantees the requirement's "only one of
`m_max`/`m_avgsum` active per instruction" property — there is no way for
both to be `'1'` in the same cycle, by construction, not by a separate mux-
select check. It also collapses "which output's `ready` gates the next
accept" into one signal (`selected_output_ready`), since only one output
can ever be the held register's target at a time.

Elastic-stage classification (per `shared/DesignPatterns.md` "Ready/valid
elastic stage"): **registered-ready, full throughput.**
`s_window_s2m.ready <= (not out_valid_q) or selected_output_ready` — a
combinational function of the *current* register's occupancy and the
*current* selected output's `ready`, same shape as a standard one-entry
AXI4-Stream register slice/skid buffer. No loss (a beat is only ever
accepted when the register is guaranteed to have room this same cycle), no
duplication (each accepted beat writes the register exactly once), stable
payload while held (register only changes on `accepted` or on the
register-draining condition), one transfer/cycle sustained when the
selected downstream is continuously ready.

## State machines

None — a single-bit `out_valid_q` register plays the same role a 2-state
(`empty`/`full`) FSM would, but is simple enough to write directly as a
conditional register update (see RTL `register_stage` process) rather than
an explicit `state_t` enumeration.

## Algorithms

- **Tap packing** (this module's own convention, since `cnn_accel_window_gen`
  does not exist yet at design time): row-major, **tap index ascending
  from the low bits**: tap `i = row * cfg_pool_kernel_w + col` at
  `s_window_m2s.data(8*i + 7 downto 8*i)`, for `i` in
  `0 .. cfg_pool_kernel_h*cfg_pool_kernel_w - 1`. Chosen for a simple,
  index-scaled slice expression (`extract_taps`), matching
  `cnn_accel_weight_buffer`'s own lane-indexing convention
  (`8 * (lane + 1) - 1 downto 8 * lane`). **The `cnn_accel_window_gen`
  implementation must produce this exact packing** — flagged back to the
  main agent/`flow_status.md` as a cross-module interface detail to
  reconcile once both modules exist.
- `reduce_max`: linear scan over the fixed `g_max_kernel_size**2`-lane tap
  array, keeping a running max seeded at `-128` (`INT8_MIN`, the identity
  element for max over int8), only comparing lanes `i < active_count`.
  Functionally identical to `cnn_accel_model.py`'s `pool_max()`
  (`max(taps)` per window) — cross-checked directly against that function
  (see Verification plan).
- `reduce_sum`: linear scan, running signed sum seeded at 0, each active
  tap `resize`d (sign-extended) to `g_accum_width` bits before adding — no
  intermediate saturation/rounding. Functionally identical to
  `cnn_accel_model.py`'s `_pool_windows()`'s raw `sum(taps)` (the
  pre-`bias_requantize_relu` value; `pool_avg()`'s downstream requantize
  step is out of scope for this module, per the req doc and the memory
  note recorded during research — cross-checked directly (see
  Verification plan).
- Both reductions are written as an **unrolled linear chain**, not a
  literal balanced binary tree, despite the requirement's "max-reduction
  tree"/"adder-tree" wording — functionally identical result and latency
  shape (both are single combinational stages feeding the same one output
  register) for a leaf module at this generic-bounded size
  (`g_max_kernel_size <= 4` per the data-width contract above, so at most
  16 lanes); noted as an implementation simplification, not a functional
  deviation.

## Numeric types and widths

- Taps: `signed(7 downto 0)` (int8 activations are signed throughout this
  IP, per `cnn_accel_model.py`'s "signed int8" convention).
- `max_result`/`out_max_q`: `signed(7 downto 0)`.
- `avgsum_result`/`out_avgsum_q`: `signed(g_accum_width - 1 downto 0)`.
- `active_count`: `natural range 0 to g_max_kernel_size**2` — computed as
  `to_integer(unsigned(cfg_pool_kernel_h)) *
  to_integer(unsigned(cfg_pool_kernel_w))`. Contract: `cfg_pool_kernel_h`
  and `cfg_pool_kernel_w` must each be in `1 .. g_max_kernel_size`
  (out-of-contract configs are undefined behavior for this module, same as
  any other generic-bounded sizing input in this IP — a violation trips
  `active_count`'s own range constraint at simulation time, a reasonable
  built-in safety net).
- `s_window_m2s.data`/`m_max_m2s.data`/`m_avgsum_m2s.data`: opaque
  `std_ulogic_vector` payload at the interface boundary, per project
  convention; only `extract_taps` converts to `signed` for arithmetic.
- No `std_logic_arith`/`std_logic_unsigned`/`std_logic_signed`; `numeric_std`
  throughout.

## Latency/throughput

Fixed 1-cycle latency (an accepted `s_window` beat's reduction result
appears on the selected output exactly one cycle later), full throughput
once started (one output beat per accepted input beat, sustained
indefinitely whenever the selected output stays continuously ready).

## Corner cases

- `active_count = 0` (`cfg_pool_kernel_h`/`w` both/either `0`, an
  out-of-contract config): `reduce_max` returns `-128`, `reduce_sum`
  returns `0` — no exception, defined (if not meaningful) values; not
  expected in a compiled program per the contract above.
- Reset asserted mid-stream: `out_valid_q` forced low, so both
  `m_max_m2s.valid`/`m_avgsum_m2s.valid` read `'0'` regardless of the held
  registers' stale content, for the entire duration of reset — consistent
  with `shared/Axi4.md` rule 18 (no in-flight transaction assumed live
  across reset).
- Backpressure on the *non-selected* output while the *selected* output is
  ready: irrelevant to this module's own `s_window_s2m.ready` — only the
  currently-held register's own target `ready` gates the next accept,
  per `selected_output_ready`'s definition (this is exactly the "only one
  is active per instruction" property paying off structurally: the other
  output's `ready`/`valid` never even needs to be examined).
- Full int8 range at both max-reduction extremes (`-128`, `127` present in
  the same window) and full sum-reduction extremes (all `-128` /
  all `127` across the maximum `g_max_kernel_size**2` active taps): no
  saturation intended for either output (`POOL_MAX`'s output range is
  already within int8 by definition of "max of int8s"; `POOL_AVG`'s
  `g_accum_width`-bit sum is sized by the integrator to never overflow,
  per this module's own generic-width contract, not saturated here).

## Selected patterns (`shared/DesignPatterns.md`)

"Ready/valid elastic stage" (registered-ready, one entry, full throughput)
— hand-written rather than a `common.handshake_pipeline` instance, because
the payload's *shape* (which of two output ports it targets) has to be
resolved before the elastic-stage abstraction can decide whose `ready`
even matters; a generic `handshake_pipeline` instance has exactly one
`output_ready`, not a dynamically-selected one.

## AXI4/AXI4-Stream protocol decisions (`shared/Axi4.md`)

Full backpressure on `s_window`, `m_max`, and `m_avgsum` (mandatory
default for streaming data, rule set in `shared/Axi4.md`). `valid` on
every link is never a function of that same link's own `ready` (`m_max_m2s.valid`/
`m_avgsum_m2s.valid` are pure functions of registered state;
`s_window_s2m.ready` is a function of the *other* two links' `ready`,
which is the standard, allowed "combinational feedthrough of readiness"
shape, not a same-direction `valid`-depends-on-`ready` loop). `last`
passed through unchanged alongside the reduction result on whichever
output is selected.

## Verification plan

- `tb_cnn_accel_pool.vhd`, VUnit-5, hand-rolled record-port stimulus/
  monitor procedures (no VUnit `axi_stream_master`/`slave` VC: this
  module's ports are `axi_stream_pkg` records, not flat `t*` signals, and
  no precedent in this repo yet bridges record ports to those VCs — same
  practical choice `cnn_accel_weight_buffer`'s own testbench already made
  for its record-typed `s_stream` port).
- Randomized-`valid` stimulus on `s_window` and independent randomized-
  `ready` responders on `m_max`/`m_avgsum`, both seeded from the test's own
  RNG (`get_string_seed(runner_cfg)`), per `shared/Vunit.md` §15's
  mandatory-backpressure default — plus a dedicated zero-stall
  `test_full_throughput` case.
- Golden values computed by an **independently re-derived** VHDL function
  in the testbench (`golden_max`/`golden_sum`, not calling into the RTL's
  own `reduce_max`/`reduce_sum`), cross-checked by hand against
  `cnn_accel_model.py`'s `pool_max()`/`_pool_windows()` (`sum(taps)`) for a
  handful of vectors during authoring (see the module's implementation
  notes below for the exact vectors used).
- Non-blocking scoreboard: two `queue_t` instances (`max_expected_q`,
  `avgsum_expected_q`, per `shared/Vunit.md` §11) — the stimulus process
  pushes each accepted beat's expected `(value, last)` onto the queue
  matching that beat's opcode; independent monitor processes on `m_max`/
  `m_avgsum` pop and check on every accepted output beat, catching both a
  wrong value and a beat routed to the wrong port (an unexpected pop from
  an empty queue).
- `test_pool_max_kernel_sizes`: multiple kernel shapes from `1x1` up to
  `g_max_kernel_size x g_max_kernel_size` (square and non-square),
  directed extremes (`-128`/`127` co-present) plus randomized taps.
- `test_pool_avg_exact_sum`: same kernel-shape sweep, directed all-`-128`/
  all-`127` extremes (exercises the full accumulator width, proving no
  premature rounding/truncation) plus randomized taps.
- `test_opcode_mutual_exclusion`: alternating `OPCODE_POOL_MAX`/
  `OPCODE_POOL_AVG` beats, back to back; a concurrent `assert` in the
  testbench additionally checks `not (m_max_m2s.valid and
  m_avgsum_m2s.valid)` on every clock edge as a redundant, always-on
  safety net alongside the scoreboard's own per-port routing check.
- `test_backpressure`: randomized stall on all three links simultaneously,
  interleaved `POOL_MAX`/`POOL_AVG` beats.
- `test_full_throughput`: zero stall on all three links, `check_relation`
  on total elapsed time vs. beat count + the fixed 1-cycle latency + a
  small margin.
- Recommended `module_cnn_accel.py` `setup_vunit` wiring (not applied by
  this module's own author, per task scope): split
  `stall_probability_percent_in`/`_max`/`_avgsum` generics, `0`/`0`/`0`
  for `test_full_throughput`, randomized (e.g. `20`) for every other test
  — mirrors `module_canny.py`'s `_setup_canny_sobel3x3` precedent (one
  input, multiple independently-stalled outputs).

## Implementation Notes (vhfill)

Implemented directly (not under the red-then-green TDD ordering: the
testbench and RTL were authored together in this subagent session rather
than in two separate rounds, since both were owned by the same task and
no separate red-checkpoint was requested) — no `--@` markers were ever
committed to `src/cnn_accel_pool.vhd`; see the final `vunit-mcp` test
report in the module's final report for the compiled/passing state.
Notable as-built decisions beyond the sections above:
- `is_avg_sel` treats any `cfg_opcode` other than `OPCODE_POOL_AVG`
  (including `OPCODE_POOL_MAX` and, defensively, any other value) as the
  max path — matches the requirement's "opcode-based mutually-exclusive
  output routing" using the two constants this IP actually defines for
  pooling, without adding a third "invalid opcode" behavior class this
  module has no way to report upstream.
- `extract_taps`/`reduce_max`/`reduce_sum` are pure functions of their
  arguments (no signal reads), so they are reusable directly as the
  combinational block's RHS expressions without an intermediate
  `process(all)`.

### Verification backend note

The shared `cnn_accel` VUnit library failed to compile as a whole during
this session because a sibling module worked on in parallel in the same
round, `cnn_accel_bias_requant.vhd`, had an in-progress compile error
(`type conversion cannot be indexed or sliced` at line 246) — not this
module's file, and out of scope to fix here per the round's
no-cross-module-edits rule. `vunit_test_dependencies` confirmed
`tb_cnn_accel_pool` does not actually depend on that file, so this
module was verified with a standalone GHDL analyze/elaborate/run
(`ghdl -a`/`--elab-run`, VHDL-2008) against the same precompiled
`vunit_lib`/`osvvm`/`axi_stream` support libraries `vunit-mcp` uses,
passing an explicit `runner_cfg` generic (`enabled_test_cases`, `seed`,
`active python runner : false`) per test case in place of the Python
runner. All five test cases passed at two different seeds (1234 and
9999), with `test_full_throughput` additionally run with the
`stall_probability_percent_*` generics forced to 0 (the value
`module_cnn_accel.py`'s `setup_vunit` should use for that test). Two
real bugs were caught and fixed by this process before it went green:
(1) `tb_cnn_accel_pool.vhd` called VUnit's `queue_pkg.pop` with
procedure-call syntax (`pop(queue, variable)`); `pop` is an
impure function (`variable := pop(queue)`), not a procedure — fixed at
both call sites (`monitor_max`/`monitor_avgsum`). (2) Four testbench
procedures (`random_taps`, `send_beat`, `run_kernel_sweep`,
`run_directed_extremes`) had formal parameters shadowing the `main`
process's own `taps`/`opcode` variables; renamed the formals to
`p_taps`/`p_opcode` to remove the resulting GHDL `-Whide` warnings.
Once the sibling module compiles again, re-run through `vunit-mcp`
(`vunit_compile` + `vunit_run_tests` with pattern
`cnn_accel.tb_cnn_accel_pool.*`) for the official record; no further
RTL or testbench changes are expected to be needed based on this
result.
