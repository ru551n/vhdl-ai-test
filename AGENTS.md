# VHDL RTL project instructions

Use VHDL-2008 as the portable production baseline. Use VHDL-2019 only when the project explicitly opts in and every active tool in the flow is verified for the constructs used.

Core conventions:
- `ieee.std_logic_1164` and `ieee.numeric_std`
- no `std_logic_arith`, `std_logic_unsigned`, or `std_logic_signed`
- use `unsigned` / `signed` for arithmetic
- synchronous active-low reset by default: `rst_n = '0'`
- use `rising_edge(clk)`
- architecture name `rtl` unless the project says otherwise
- prefer direct entity instantiation
- preserve hand-owned requirement sections
- `--@` marks unfinished design-direction code and must be removed once implemented
- AXI4/AXI4-Stream interfaces follow `shared/Axi4.md` (handshake stability, 4 KiB burst boundary, same-ID ordering, no stream beat loss)
- VUnit-5 is the default verification framework; VUnit API rules (phases, gate locks, seeds, `check_pkg`) are authoritative in `shared/Vunit.md`

Use the project skills under `.maki/skills/` for non-trivial RTL work.

Preferred workflow:
`vharch` → `vhdesign` → `vhfill` → `vhtestgen` → `vhtestrun` → `vhdebug` as needed → `vhsynth` → `vhdoc`.
Direct VUnit authoring/repair/migration (`run.py`, testbenches, VUnit 4→5) uses the `vhunit` skill with `shared/Vunit.md`.

Tool policy:
- prefer `vhdl-rag-mcp` for semantic VHDL/docs/source retrieval
- prefer `vunit-mcp` for compile, test discovery, regressions, logs and waveform paths
- prefer `waver-mcp` for waveform measurements/debug
- prefer `tsfpga-mcp` for portable VHDL synthesis/resource summaries
- fall back to local tools only when the relevant MCP server is unavailable, unhealthy, or unsuitable
- never claim compile/test/waveform/synthesis/timing/power success without a real tool result

Subagent use:
- for long multi-module flows (e.g. a full IP through `vhflow`), delegate self-contained phases to subagents
- `flow_status.md` is the handoff contract; the main agent owns it and the final report
- a subagent's summary is not evidence — verify the real artifacts and tool results before treating a phase as done

Maki defers large MCP toolsets behind `tool_search`; search for the relevant server/tool rather than assuming every MCP tool is already loaded.
