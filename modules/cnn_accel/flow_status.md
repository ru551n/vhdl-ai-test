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
| 2/3. Design + TDD, tiled dataflow (`window_gen` rework, `pe_array`, `weight_buffer`) | COMPLETE | Round 2, M3-M5. Blocking flaw found + fixed, see below |
| 3b. `conv_core` bit-exact vs `cnn_accel_model.py` | COMPLETE | M6b, commit `52adb45`. 73/73 VUnit, bit-exact vs Python vectors (decision D5) |
| 7a. `window_gen` block-RAM inference (M7, Option 3a) | COMPLETE | 23950 LUTs / 0 BRAM -> 2753 LUTs / 3 RAMB36. See "M7" below |
| 2/3. Design + TDD, AXI-facing engines (`cnn_accel_axi_read_dma`, `cnn_accel_ofmap_dma`, `cnn_accel_csr`) | PENDING | |
| 2/3. Design + TDD, `cnn_accel_layer_ctrl` | PENDING | |
| 2/3. Design + TDD, `cnn_accel_sequencer` | PENDING | |
| 4. IP-level integration test | PENDING | full-program golden-model comparison vs. `cnn_accel_model.py` |
| 5. Regression | PENDING | full project regression via `vunit-mcp` once all modules green |
| 7. Synthesis | IN PROGRESS | target ratified: **Xilinx Artix-7** (`chip='xilinx'`, `family='xc7'`). All 6 netlist builds pass; LUT limits still PROVISIONAL until a CI run |
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

## Round 2 (in progress) — blocking architectural flaw + fixes

Plan: `~/.local/state/maki/plans/cosmic-hip-cod.md` (decisions D1-D9).

### Blocking flaw found
The dataflow assumed ONE window beat carries all `K*K*C` taps of an output
pixel. Impossible: `axi_stream_data_sz = 128` bits = 16 int8 taps max, while
the target net needs 27 (layer 1) to 2304 (layer 9).
`cnn_accel_window_gen.vhd:204-207` tied `cfg_in_channels` to an *elaboration*
generic at `severity warning` only (silently wrong packing); no module could
carry a partial sum across beats. Fix: input-channel tiling with
`first_tile`/`last_tile` + partial-sum carry in `pe_array`.

### Done this round (real tool evidence)
- `pe_array.vhd` (uncommitted, never compiled) used `group`, a VHDL **reserved
  word** — it red-lined the WHOLE project regression, since tsfpga auto-scans
  `src/*.vhd`. Parked at `doc/reference/pe_array_v0_reference.vhd.txt`.
  Baseline restored: **58/58 GHDL**.
- `cnn_accel_model.py`: 5 semantic fixes (D2 saturate-on-both-paths, D3 int32
  accumulator contract, FC degenerate-form enforcement, zero-stride/degenerate
  -dim guards, explicit no-pool-padding). Now has a real test suite
  (`test_cnn_accel_model.py`) — **90 pytest passing**, including two
  independently-written naive references fuzzed over hundreds of seeded cases.
- `cnn_accel_pkg.vhd`: added `window_m2s_t`/`window_s2m_t`,
  `accum_m2s_t`/`accum_s2m_t` (VHDL-2008 unconstrained `data` element) +
  `window_data_width()`/`accum_data_width()`. Spike proved GHDL **and** NVC
  both support unconstrained record elements incl. array-of-record.
- `cnn_accel_bias_requant`: D2 applied — bypass path now saturates instead of
  wrapping; `test_bypass_wraparound` -> `test_bypass_saturates`; docs corrected.
- Test vectors exported to `test/vectors/` (10 cases, committed, deterministic).

### D9 (ratified, no code change needed)
An independent verifier claimed a bias-add overflow bug. **It was wrong**:
`cnn_accel_bias_requant.vhd:81,211` already uses a 33-bit sum
(`c_sum_width = g_accum_width + 1`), so int32+int32 cannot wrap and the model's
unbounded add is already bit-exact. Contract now documented on both sides with
boundary tests. Lesson: verifiers reasoning from hypothetical RTL must be
checked against the actual RTL.

### Current state
- GHDL full regression: **59/59 PASS**.
- pytest: **90/90 PASS**.
- NVC: 58/59 — the single failure is
  `axi_stream_join.tb_axi_stream_join.stall_probability_percent_0.test_full_throughput`,
  **pre-existing and unrelated** (module untouched, last commit `a929171`).
  Tracked, not blocking.

### Next recommended action (superseded — see M7 below)
Tiled-dataflow design proposal spanning `window_gen`/`pe_array`/
`weight_buffer`, then parallel implementation, then `conv_core`.
Note: `W*C` is roughly invariant across the target network (stride-2 halves W
and doubles C), so full-channel line buffers cost only ~2.5 KB/row — full
input-channel tiling per output pixel is cheap on Artix-7.

## M7 — `window_gen` block-RAM inference (Option 3a), 2026-09-06

Ratified in `doc/cnn_accel_window_gen_bram_proposal.md` §6 (Option 3a,
overriding the document's own Option 1 recommendation); implemented per its
§7.1. Uncommitted at time of writing.

### RTL change
`src/cnn_accel_window_gen.vhd`: row banks are now `g_max_kernel_size`
**physically separate** `bank_mem` signals, one per `gen_banks` generate
branch, each with a single decoded write (`wr_decode` process) and a single
**registered** read — the `memory_block` idiom already used by
`cnn_accel_weight_buffer.vhd`. Read side walks a registered `kc` column
counter over `cfg_kernel_w` cycles, reading all banks in parallel, packing
into a tap-assembly register; the window is still presented as ONE
`m_window_m2s` beat, so `cnn_accel_pkg` and `cnn_accel_pe_array` are
untouched. `window_valid` is now a registered valid-delay trailing the walk,
not a same-cycle function of `row_ready`.

Key finding: a single shared 2D array signal is ONE memory object to Yosys's
`memory_collect` once several banks' data must be live in the same cycle,
however statically-indexed each access site looks — it emitted zero `$mem_v2`
cells. Per-bank separate signals fixed that structurally.

Correctness follow-up found by the existing tests: the slower read side lets
the write side outrun the reader by `g_max_kernel_size` physical rows and
alias a bank still pending read. Fixed with a `write_freeze_i` backpressure
gate on `s_stream_s2m.ready` (not the out-of-scope double-buffering
mitigation).

### Measured (real tool results)
- VUnit GHDL, clean, `num_threads=12`: **73/73 PASS** (19.5 s).
- VUnit NVC, clean, `cnn_accel.*`: **40/40 PASS** (5.8 s).
- Netlist builds (local dev Yosys v0.68+182, `synth_xilinx -family xc7`),
  all 6 cnn_accel projects PASS against the new checkers:

  | Entity | LUTs | FFs | BRAM | DSP |
  |---|---|---|---|---|
  | `window_gen` (was 23950 CI / 0 BRAM) | 2 753 | 789 | 3 (3xRAMB36) | 9 |
  | `weight_buffer` | 181 | 59 | 72 (8xRAMB36 + 64xRAMB18) | 0 |
  | `pe_array` | 2 888 | 1 119 | 0 | 65 |
  | `bias_requant` | 4 658 | 66 | 0 | 32 |
  | **`conv_core`** (was 35056 CI / 72 BRAM) | **10 323** | **2 033** | **75** | **106** |

  FFs/BRAM/DSP are exactly additive over the four leaves (2033/75/106),
  confirming the numbers; LUTs come out 157 below the sum of the leaves
  (cross-boundary optimization).
- Throughput cost (accepted in DP3): +5 cycles/window within a row
  (standalone `test_full_throughput` 585 ns -> 2135 ns, 27% of old in
  isolation); integrated behind `pe_array`'s 12-cycle/window budget that
  dilutes to ~70%, i.e. the upper end of the ratified 70-75% band.

### Open
- LUT checker limits (`window_gen` <12000, `conv_core` <24000) are
  **PROVISIONAL** local-Yosys guard rails. Project standing rule: LUT limits
  must be re-baselined from a real CI run. FFs/BRAM/DSP are trusted directly
  (version-insensitive); `window_gen`'s BRAM checker is `EqualTo(3)` so a
  silent regression back to 0 BRAM fails CI.
- `.maki/mcp.toml`: `TSFPGA_MCP_PROJECT_PYTHON` repointed at this repo's own
  `.venv` — tsfpga-mcp's venv no longer has tsfpga installed, which made
  every `tsfpga_project_build` fail with `No module named
  tsfpga.build_project_list`. Takes effect after an MCP restart; this
  session's builds were run by invoking `build_fpga.py` directly.

### Next recommended action
M8 `cnn_accel_layer_ctrl` (incl. the D6 output-channel tile loop), then M9
DMAs, per the milestone table in `~/.local/state/maki/plans/cosmic-hip-cod.md`.
