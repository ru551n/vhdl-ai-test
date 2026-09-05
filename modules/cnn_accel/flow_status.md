# cnn_accel — flow status

Owner: main agent (this file is the handoff contract; subagents report into
it via the main agent, they do not edit it themselves).

## IP

`cnn_accel` — CNN accelerator IP. See `doc/cnn_accel_arch.md` /
`doc/cnn_accel_req.md`.

## Phase status

| Phase | Status | Notes |
|---|---|---|
| 1. Architecture (`vharch`) | COMPLETE | `doc/cnn_accel_arch.md`, `doc/cnn_accel_req.md`, per-module `modules/cnn_accel/doc/*_req.md` (13 files) |
| 2/3. Design + TDD, leaf modules | COMPLETE | `cnn_accel_bias_requant`, `cnn_accel_pool`, `cnn_accel_window_gen`, `cnn_accel_weight_buffer` — all green, real `vunit-mcp` regression, see below |
| 2/3. Design + TDD, `cnn_accel_pe_array` | PENDING | depends on leaf modules' interfaces being stable |
| 2/3. Design + TDD, AXI-facing engines (`cnn_accel_axi_read_dma`, `cnn_accel_ofmap_dma`, `cnn_accel_csr`) | PENDING | |
| 2/3. Design + TDD, `cnn_accel_layer_ctrl` | PENDING | |
| 2/3. Design + TDD, `cnn_accel_sequencer` | PENDING | |
| 4. IP-level integration test | PENDING | full-program golden-model comparison vs. `cnn_accel_model.py` |
| 5. Regression | PENDING | full project regression via `vunit-mcp` once all modules green |
| 7. Synthesis | PENDING | `build_fpga.py` wiring for `cnn_accel` entities, once RTL exists |
| 8. Documentation | PENDING | `vhdoc` aggregation once modules are green |

## Affected modules (this round)

- `modules/cnn_accel/src/cnn_accel_bias_requant.vhd` + `test/tb_cnn_accel_bias_requant.vhd`
- `modules/cnn_accel/src/cnn_accel_pool.vhd` + `test/tb_cnn_accel_pool.vhd`
- `modules/cnn_accel/src/cnn_accel_window_gen.vhd` + `test/tb_cnn_accel_window_gen.vhd`
- `modules/cnn_accel/src/cnn_accel_weight_buffer.vhd` + `test/tb_cnn_accel_weight_buffer.vhd`

Each module's requirement (`modules/cnn_accel/doc/cnn_accel_<module>_req.md`)
is already written (hand-owned Functional Description section present) —
this round is `vhdesign` (proposal + final doc) → `vhfill` (RTL) →
`vhtestgen` (testbench) → `vhtestrun` (regression), per module.

Subagents were instructed NOT to touch `module_cnn_accel.py`,
`cnn_accel_pkg.vhd`, `run.py`, or any other module's files, to avoid
concurrent-edit conflicts on shared files. The main agent wires
`setup_vunit`/`add_vunit_config` for all four afterward, sequentially, and
re-runs the full project regression before committing.

## MCP availability

- `vunit-mcp`: available, healthy, used directly for all compile/run/log
  calls (`vunit_status`, `vunit_compile`, `vunit_run_tests`,
  `vunit_get_test_log`). Real regression evidence recorded below.
- `tsfpga-mcp`, `peeper-mcp`: available, not needed this round (no
  synthesis, no waveform debug required — failures were resolved from
  `vunit_get_test_log` alone).
- `corvidex-mcp`: **not connected in this session's maki host** (verified
  via `tool_search`, multiple queries, only `peeper-mcp`/`tsfpga-mcp`/
  `vunit-mcp` ever resolve). Fell back to grep/read/glob throughout, per
  `shared/McpToolPolicy.md`'s documented fallback rule. See memory note
  `mcp_availability_this_session.md`. Re-check at the start of future
  sessions.

## Blockers

- `axi_stream_pkg` GHDL synth workaround decision still pending user
  sign-off (unrelated to this round — simulation, not synthesis).

## Round 1 result (leaf modules)

Real regression evidence (`vunit-mcp`, GHDL, clean rebuild):
- `cnn_accel.*`: 25/25 passed
- full project (`*`): 58/58 passed

Two real bugs found and fixed by the orchestrator after subagent handoff
(both were GHDL-only failures the NVC-based subagent self-checks didn't
catch):
1. `tb_cnn_accel_weight_buffer.vhd`: `test_runner_watchdog` called inside
   the `main` process instead of as a concurrent statement (deadlock).
   Fixed by the orchestrator before delegating the remaining 2 RTL-looking
   failures (which turned out to be a testbench delta-cycle race, fixed
   by the weight_buffer subagent — not an RTL bug).
2. `tb_cnn_accel_bias_requant.vhd` (3 call sites): `rnd.RandInt(-(2**31),
   2**31-1)` — OSVVM's `RandInt` range-size computation
   (`Max - Min + 1`) overflows 32-bit `integer` at exactly this bound.
   Fixed by switching to `rnd.RandSigned(32)` (full-width random signed,
   no range-size arithmetic). Worth remembering for any future full-range
   32-bit `RandInt` use in this repo.

All four modules' final docs, proposals, RTL, and testbenches exist per
the "Affected modules" list above. `module_cnn_accel.py` now has
`setup_vunit` wiring `stall_probability_percent_*` generics (0 for each
module's dedicated full-throughput test, 20 otherwise) for
`cnn_accel_bias_requant`/`cnn_accel_pool` (the two modules whose
testbenches expose those generics); `cnn_accel_window_gen`/
`cnn_accel_weight_buffer` need no such wiring (their stall behavior, where
present, is hardcoded per test case, not generic-driven).

## Next recommended action

Commit atomically (one commit per module + one wiring commit), then start
Round 2: `cnn_accel_pe_array` (depends on these leaf modules' now-stable
interfaces).
