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
| 7b. `weight_buffer` single-buffer + streaming rework | COMPLETE | 72 BRAM -> 15 BRAM. See "M7b" below |
| 7c. `conv_core` re-measured (Yosys + Vivado) | COMPLETE | **72 -> 18 BRAM** end to end. See "M7c" below |
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

### Next recommended action (superseded — see M7b below)
M8 `cnn_accel_layer_ctrl` (incl. the D6 output-channel tile loop), then M9
DMAs, per the milestone table in `~/.local/state/maki/plans/cosmic-hip-cod.md`.

## M7b — `weight_buffer` single-buffer + streaming rework, 2026-09-06

With `window_gen` fixed (M7), `cnn_accel_weight_buffer` became the entire
remaining block-RAM cost of `conv_core`: 72 of its 75 BRAM. Root causes,
both structural rather than sizing:

1. **Per-lane byte write enables.** The fill path decoded a write enable per
   8-bit lane over `g_pe_rows * g_pe_cols` = 64 lanes. Yosys could not keep
   that as one wide memory, so it fragmented the weight region into 64
   separate 1024x8 memories — one RAMB18 each.
2. **A bias memory sized like the weight memory.** The bias region inherited
   `g_weight_buffer_depth`, costing 8 RAMB36 of which
   `cnn_accel_bias_requant` only ever reads row 0.
3. **Double buffering.** Two A/B banks doubled a region that, per decision
   D6, does not need to be resident at all: weights are streamed from DDR4
   once per output-channel pass, and DDR4 bandwidth is not the constraint —
   on-chip footprint is.

### RTL change
`src/cnn_accel_weight_buffer.vhd`:
- **Single-buffered.** `fill_bank_sel`/`read_bank_sel` are gone, replaced by
  a `fill_start` pulse that rewinds the fill address. One region, filled per
  output-channel pass.
- **Whole-row writes.** The fill stream is accumulated in a row-assembly
  register (`weight_row_assemble_q` / `bias_row_assemble_q`) and committed as
  a single full-width write, so each region is one memory object to Yosys
  instead of 64.
- **Prefetch FIFO.** A shallow `fifo.fifo` (hdl-modules, `g_fill_fifo_depth`
  default 32) decouples the fill stream from the row-assembly commit so the
  producer is not stalled mid-row.
- **Separate bias depth.** New `g_bias_buffer_depth` generic (default 8,
  forwarded unmodified by `cnn_accel_conv_core`), sized by
  `ceil(out_channels/g_pe_rows)` rather than by the weight region.

`_WEIGHT_BUFFER_DEPTH` is now pinned to **288** = `K^2 * ceil(C_max/8)` =
`9*32`, the target backbone's worst layer (3x3x256, layer 9), replacing the
old arbitrary 512.

### Measured (real tool results)
- VUnit GHDL, full project: **73/73 PASS**. NVC: **40/40 PASS**
  (`cnn_accel.*`; the one project-wide NVC failure is
  `axi_stream_join.tb_axi_stream_join`'s full-throughput check, pre-existing
  and unrelated — that module is untouched and passes on GHDL).
- `weight_buffer` netlist build (local dev Yosys v0.68+182):

  | | LUTs | FFs | BRAM | DSP |
  |---|---|---|---|---|
  | before | 175 | 59 | 72 (8xRAMB36 + 64xRAMB18) | 0 |
  | after | 881 | 1093 | **15** (15xRAMB18) | 0 |

  The FF/LUT increase is the accepted trade: 768 bits of row-assembly
  registers plus the depth-32 prefetch FIFO's distributed RAM (RAM32M, too
  shallow for Yosys to pick block RAM) buy the 57-BRAM reduction.

### Open
- **Requirement doc conflict, deliberately left for the user.**
  `doc/cnn_accel_weight_buffer_req.md`'s Responsibility/Generics/Ports/
  Protocols sections are updated, but its hand-owned Functional Description
  still describes the old double-buffered A/B design. That section is
  user-owned; it is flagged here rather than rewritten.

## M7c — `conv_core` re-measured, and the Vivado backend, 2026-09-07

### Result: the BRAM goal is met

| `conv_core` | LUTs | FFs | BRAM | DSP |
|---|---|---|---|---|
| pre-M7 (CI Yosys) | 35 056 | 1 443 | **72** | 105 |
| post-M7+M7b (local dev Yosys) | 11 116 | 3 066 | **18** (3xRAMB36 + 15xRAMB18) | 106 |
| post-M7+M7b (Vivado, `xc7a200tfbg484-2`, OOC) | 15 358 | 2 824 | **14xRAMB36 + 2xRAMB18** | 36 |

**BRAM 72 -> 18, a 75% cut**, and Vivado independently agrees (15
RAMB36-equivalents). Both are comfortably inside the ~11 (single-buffered)
/ ~19 (double-buffered) target band.

Leaf additivity held exactly again, which is why the FF/BRAM/DSP checkers
are re-baselined straight from the local Yosys run:

| | window_gen | weight_buffer | pe_array | bias_requant | sum | measured |
|---|---|---|---|---|---|---|
| DSP | 9 | 0 | 65 | 32 | 106 | **106** |
| BRAM | 3 | 15 | 0 | 0 | 18 | **18** |
| FF | 789 | 1093 | 1118 | 66 | 3066 | **3066** |
| LUT | 2753 | 881 | 2891 | 4658 | 11183 | 11116 |

(Leaf figures are the *local* measurements from the same session, not the CI
baselines in each leaf's own comment — `pe_array` reads 2891/1118 locally
against its 3474/1119 CI baseline. LUTs come out 67 under the sum from
cross-boundary optimization; FFs, BRAM and DSP are exact.)

### Bug caught by this build (would have broken CI)
The M7b commit (`2d78d59`) left `conv_core`'s `Ffs(LessThan(2200))` checker
in place while M7b added ~1030 FFs to `weight_buffer`, which `conv_core`
inherits. The build **failed** on `Got 3066, expected < 2200`. Checkers now
read `Ffs(LessThan(3200))` and `BlockRams(LessThan(20))` (was
`LessThan(80)`). Lesson: a leaf-level resource trade must be re-checked
against every composition entity above it in the same pass, not deferred.

### Yosys vs Vivado: not noise, structural differences
- **DSP 36 vs 106.** Vivado packs the 8x8 MAC array two 8-bit multiplies
  per DSP48E1 (32 DSPs for 64 lanes, from its own DSP report); Yosys does
  not. `xc7a200t` has 740 DSPs, so this is headroom, not a problem.
- **LUT 15 358 vs 11 116** is the other side of that same trade.
- **LUTRAM 0 vs 49xRAM32M.** Vivado puts `weight_buffer`'s prefetch FIFO in
  block RAM; Yosys uses distributed RAM.

Neither tool's number belongs in the other's limit. Yosys stays the
CI-gating backend; Vivado is the vendor-accurate cross-check.

### Vivado backend wiring (new)
- `ghdl_yosys_env.resolve_vivado_path()`: env `VIVADO_PATH`, then `PATH`,
  then newest `/opt/xilinx/*/Vivado/bin/vivado` (this machine: 2026.1, not
  on `PATH`).
- `module_cnn_accel.py` registers `cnn_accel_conv_core_vivado`
  (`VivadoNetlistProject`, part `xc7a200tfbg484-2`) **only when that
  resolver returns a path**. The guard is mandatory, not defensive: CI's
  `ru551n/hdl-docker` image has no Vivado and the `synthesize` job runs
  `build_fpga.py --netlist-builds` with **no filter**, so an unconditional
  Vivado project breaks CI.
- That project carries **no** `build_result_checkers`, since Yosys- and
  Vivado-derived limits are not interchangeable.
- Speed note: Vivado did `conv_core` in **50 s**; Yosys takes **~18 min**
  (1098 s). Prefer Vivado for the large composition entities.
- Target part raised `xc7a100t` -> `xc7a200t` for the real 320x320 network.
  Package/speed grade are arbitrary for an out-of-context netlist build
  (no I/O, no timing closure); revisit when a board is chosen.

### Open
- LUT limits (`window_gen` <12000, `conv_core` <24000) are still
  **PROVISIONAL** local-Yosys guard rails. Composed CI estimate for
  `conv_core` is ~19 400. Tighten from a real CI run.
- `pe_array` DSP packing: Vivado gets 64 MAC lanes into 32 DSP48E1s, Yosys
  into 65. Not acted on — DSP is not the scarce resource on `xc7a200t`.

### Next recommended action
M8 `cnn_accel_layer_ctrl` (incl. the D6 output-channel tile loop), then M9
DMAs, per the milestone table in
`~/.local/state/maki/plans/cosmic-hip-cod.md`.
