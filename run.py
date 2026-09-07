#!/usr/bin/env python3
"""VUnit run script for the vhdl-ai-test project.

Discovers every module under ``modules/`` via ``tsfpga.module.get_modules()``
(one VHDL library per module folder, tests included), plus hdl-modules' own
modules as a dependency-only ``modules_no_test`` set (its own testbenches are
not re-run here) so ``common``/``fifo``/``bfm``/``dma_axi_write_simple``
resolve for the reuse decisions in ``doc/cnn_accel_arch.md``. Kept as
``run.py`` (not ``build.py``) since ``vunit-mcp`` targets ``run.py`` by
default (``VUNIT_MCP_RUN_SCRIPT``).
"""

from pathlib import Path

from tsfpga.module import get_modules
from vunit import VUnit

ROOT = Path(__file__).resolve().parent

vu = VUnit.from_argv()
vu.add_vhdl_builtins()
# hdl-modules' bfm module (stall_bfm_pkg.vhd) needs osvvm's RandomPType;
# VUnit ships/compiles it but it is not part of add_vhdl_builtins().
vu.add_osvvm()
# hdl-modules' bfm module also needs VUnit's verification-component packages
# (com_types_pkg, bus_master_pkg, memory_pkg, axi_slave_pkg, axi_read_slave,
# axi_write_slave, sync_pkg, com_pkg, ...) -- not part of add_vhdl_builtins().
vu.add_verification_components()

# This project's own modules (one library per module folder, tests included).
modules = get_modules(modules_folder=ROOT / "modules")

# hdl-modules, dependency-only: source/sim files added, its own testbenches
# are not (that is hdl-modules' own concern, not re-run in this project).
# ``hard_fifo`` is Xilinx-unisim-only (fifo36e2_wrapper.vhd needs a real
# unisim library GHDL doesn't provide) and irrelevant to this project's reuse
# set -- excluded so it doesn't block compiling everything else.
modules_no_test = get_modules(
    modules_folder=ROOT / "hdl-modules" / "modules",
    names_avoid={"hard_fifo"},
)

for module in modules + modules_no_test:
    vunit_library = vu.add_library(module.library_name, allow_duplicate=True)
    simulate_this_module = module not in modules_no_test

    for hdl_file in module.get_simulation_files(include_tests=simulate_this_module):
        vunit_library.add_source_file(hdl_file.path)

    if simulate_this_module:
        module.setup_vunit(vunit_proj=vu)

vu.main()
