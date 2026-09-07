#!/usr/bin/env python3
"""tsfpga build script (netlist-only, Yosys+GHDL) for vhdl-ai-test.

Mirrors tsfpga's own ``tsfpga/examples/build_fpga.py`` and tsfpga-mcp's
``tests/fixture_project/build_fpga.py``, but drives this project's own
``modules/`` tree. Every synthesizable entity under ``modules/`` that wants
a netlist build defines ``get_build_projects()`` on its own ``Module``
class (``tsfpga.module.BaseModule``), returning one or more
``tsfpga.yosys.project.YosysNetlistBuild`` instances -- see
``modules/cnn_accel/module_cnn_accel.py`` for the worked example.
``hdl-modules`` is used as
a dependency only from within each of those methods (no netlist builds of
its own get registered here, unlike ``run.py``'s simulation-only
``modules_no_test`` set, which is unrelated to this file).

Run via tsfpga-mcp in "project mode" (``tsfpga_project_list_builds`` /
``tsfpga_project_build``), which drives this script as a subprocess
exactly as a human would from a terminal -- see ``.maki/mcp.toml`` for the
``TSFPGA_MCP_PROJECT_*`` environment this needs (an interpreter with
tsfpga's Yosys/GHDL netlist-build support, i.e. ``tsfpga.yosys.project``,
plus ``vunit_hdl`` -- not yet part of the stable ``tsfpga`` release this
project otherwise uses for ``run.py``/simulation).
"""

from __future__ import annotations

import sys
from pathlib import Path

from tsfpga.build_project_list import BuildProjectList, get_build_projects
from tsfpga.examples.build_fpga_utils import arguments, setup_and_run
from tsfpga.module import get_modules

ROOT = Path(__file__).resolve().parent


def main() -> None:
    args = arguments(default_temp_dir=ROOT / "generated")
    modules = get_modules(modules_folder=ROOT / "modules")
    project_list = BuildProjectList(
        projects=get_build_projects(
            modules=modules,
            project_filters=args.project_filters,
            include_netlist_not_full_builds=args.netlist_builds,
        ),
        no_color=args.no_color,
    )

    sys.exit(
        setup_and_run(
            modules=modules,
            project_list=project_list,
            args=args,
            collect_artifacts_function=None,
        )
    )


if __name__ == "__main__":
    main()
