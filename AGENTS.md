# VHDL RTL project instructions

Use VHDL-2008 as the portable production baseline. Use VHDL-2019 only when the project explicitly opts in and every active tool in the flow is verified for the constructs used.

Core conventions:
- `ieee.std_logic_1164` and `ieee.numeric_std`
- no `std_logic_arith`, `std_logic_unsigned`, or `std_logic_signed`
- use `unsigned` / `signed` for arithmetic
- resetless-by-default (declaration initial values); when a runtime-restorable reset is needed, use synchronous active-high `reset = '1'` (see `shared/TsfpgaCodingConventions.md`)
- use `rising_edge(clk)`
- architecture name `a` (RTL) / `tb` (testbench) unless the project says otherwise (see `shared/TsfpgaCodingConventions.md`)
- prefer direct entity instantiation
- preserve hand-owned requirement sections
- `--@` marks unfinished design-direction code and must be removed once implemented
- AXI4/AXI4-Stream interfaces follow `shared/Axi4.md` (handshake stability, 4 KiB burst boundary, same-ID ordering, no stream beat loss)
- VUnit-5 is the default verification framework; VUnit API rules (phases, gate locks, seeds, `check_pkg`) are authoritative in `shared/Vunit.md`

Use the project skills under `.maki/skills/` for non-trivial RTL work.

Preferred workflow:
`vharch` → `vhdesign` → `vhfill` → `vhtestgen` → `vhtestrun` → `vhdebug` as needed → `vhsynth` → `vhdoc`.
Direct VUnit authoring/repair/migration (`run.py`, testbenches, VUnit 4→5) uses the `vhunit` skill with `shared/Vunit.md`.

Environment:
- Python dependencies come from `requirements.txt` (simulation: the `ru551n/vunit` fork, which carries the `--wave` flag that headless waveform recording needs) and `requirements-synth.txt` (netlist synthesis: released VUnit 4.7.1 — see that file for why the two cannot share one env). Only `hdl-modules` is a git submodule.
- `.venv` runs `run.py`; `.venv-synth` runs `build_fpga.py`. vunit-mcp creates and activates `.venv` on its own. tsfpga-mcp would do the same, so point it at the synthesis env explicitly: `TSFPGA_MCP_PROJECT_PYTHON=<worktree>/.venv-synth/bin/python`, created with `python3 -m venv .venv-synth && .venv-synth/bin/pip install -r requirements-synth.txt`. Neither server installs a VUnit or tsfpga of its own, so every tool answer comes from these pins.
- Enable the hook once per clone: `git config core.hooksPath .githooks` — it populates `hdl-modules` on every checkout and in every new worktree.

Several agents on one clone:
- give each agent its own `git worktree` and start its MCP servers with that worktree as cwd; `vunit_out`, the venvs, caches and the git index are then disjoint with no configuration
- without that, concurrent `run.py` invocations clobber a shared `vunit_out` — set a per-agent `VUNIT_MCP_OUTPUT_DIR` instead
- never run `git worktree add` against a branch another agent has checked out

Tool policy:
- prefer `corvidex-mcp` for semantic VHDL/docs/source retrieval
- prefer `vunit-mcp` for compile, test discovery, regressions, logs and waveform paths
- prefer `peeper-mcp` for waveform measurements/debug
- prefer `tsfpga-mcp` for this project's netlist and top-level builds (`tsfpga_project_*`, which drive `build_fpga.py`); it no longer synthesizes standalone entities
- fall back to local tools only when the relevant MCP server is unavailable, unhealthy, or unsuitable
- never claim compile/test/waveform/synthesis/timing/power success without a real tool result

Subagent use:
- for long multi-module flows (e.g. a full IP through `vhflow`), delegate self-contained phases to subagents
- `flow_status.md` is the handoff contract; the main agent owns it and the final report
- a subagent's summary is not evidence — verify the real artifacts and tool results before treating a phase as done

Maki defers large MCP toolsets behind `tool_search`; search for the relevant server/tool rather than assuming every MCP tool is already loaded.
