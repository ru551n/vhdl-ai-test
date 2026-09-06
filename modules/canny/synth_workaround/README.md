# `axi_stream_pkg` synth-only workaround

## Why this exists

`canny_top` (via `dir_fifo_inst : entity axi_stream.axi_stream_fifo`) depends on
`hdl-modules/modules/axi_stream/src/axi_stream_pkg.vhd`. Its `to_slv`/
`to_axi_stream_m2s` functions marshal the `user` field using a **dynamic-width
slice**:

```vhdl
-- axi_stream_pkg.vhd:127
result(hi downto lo) := data.user(user_width - 1 downto 0);
-- axi_stream_pkg.vhd:154
result.user(user_width - 1 downto 0) := data(hi + offset downto lo + offset);
```

`user_width` is a generic, so GHDL's synthesis backend (`ghdl-yosys-plugin`)
cannot elaborate this for netlist synthesis:

```
axi_stream_pkg.vhd:127:11:error: cannot extract same variable part for dynamic slice
```

This is a real, still-open GHDL limitation
([ghdl/ghdl#2658](https://github.com/ghdl/ghdl/issues/2658)), **not** a bug in
this project's RTL, and it does **not** affect simulation — GHDL's normal
analysis/simulation flow handles the dynamic slice fine, which is why every
VUnit testbench in this project (including `axi_stream_fifo`'s own upstream
tests) passes without issue. It only trips the separate synthesis-oriented
backend used by `tsfpga_synthesize`/`build_fpga.py --netlist-builds`.

## What's here

`axi_stream/src/axi_stream_pkg.vhd` is a **synth-only** copy of the upstream
file, with only the two affected statements rewritten as a fixed-bound
(`0 to axi_stream_user_sz - 1`) bit-by-bit loop with single-bit dynamic
indexing (guarded by `if i < user_width`), which avoids the dynamic-width
slice entirely while remaining functionally identical for every valid
`user_width`. Everything else in the file (record types, other functions,
constants) is byte-for-byte identical to upstream.

`axi_stream/src/axi_stream_fifo.vhd` is an **unmodified** copy of the
upstream file — included only because `tsfpga.module.get_modules()` treats
`synth_workaround/axi_stream/` as a full replacement for the `axi_stream`
library when staged for this build (see `module_canny.py`'s
`get_build_projects()`), so all of that library's synthesizable sources must
be present here. **If hdl-modules bumps `axi_stream_fifo.vhd` (or adds new
`axi_stream` sources needed by `canny_top`), this copy must be manually
re-synced** — nothing currently detects staleness automatically.

`tb_axi_stream_pkg_equiv.vhd` is a standalone GHDL-simulatable equivalence
check (not part of the project's regular VUnit suite — it compiles the
original and patched `axi_stream_pkg` into two differently-named libraries,
`orig_pkg`/`patched_pkg`, and cross-checks every combination) proving the
patched functions are bit-exact equivalent to upstream across
`data_width in {1, 2, 8, 11, 16, 128}` and every `user_width in 0 .. 16`,
with varied/random data patterns. To re-run manually:

```bash
ghdl -a --std=08 --work=orig_pkg hdl-modules/modules/axi_stream/src/axi_stream_pkg.vhd
ghdl -a --std=08 --work=patched_pkg modules/canny/synth_workaround/axi_stream/src/axi_stream_pkg.vhd
ghdl -a --std=08 modules/canny/synth_workaround/tb_axi_stream_pkg_equiv.vhd
ghdl -r --std=08 tb_axi_stream_pkg_equiv
```

## Policy note

Substituting a locally-patched copy of third-party (`hdl-modules`) code is a
deliberate exception to this project's default "reuse third-party code
unmodified" policy. It was explicitly approved by the user for permanent use
in `canny_top`'s netlist build only — this file is never used for simulation
or verification of this project, only for `build_fpga.py --netlist-builds
canny_top` / `tsfpga_project_build`.
