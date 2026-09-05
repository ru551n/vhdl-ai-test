from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from tsfpga.module import BaseModule, get_modules

if TYPE_CHECKING:
    from vunit.ui import VUnit


class Module(BaseModule):
    def get_build_projects(self) -> list:
        # Local import: tsfpga.yosys.project needs a tsfpga build with Yosys
        # netlist-build support (not yet in the stable release this
        # project's run.py/VUnit flow uses), so this must not be imported
        # at module load time -- only build_fpga.py (run under
        # TSFPGA_MCP_PROJECT_PYTHON, see .maki/mcp.toml) ever calls this
        # method.
        from tsfpga.yosys.project import YosysNetlistBuild

        modules = get_modules(
            modules_folder=self.path.parent, names_include={self.name}
        ) + get_modules(
            modules_folder=self.path.parent.parent / "hdl-modules" / "modules",
            names_include={"common"},
        )
        ghdl_plugin_path = os.environ.get("TSFPGA_MCP_GHDL_PLUGIN")
        ghdl_prefix = os.environ.get("TSFPGA_MCP_GHDL_PREFIX")

        return [
            YosysNetlistBuild(
                name="axi_stream_join",
                modules=modules,
                top="axi_stream_join",
                generics={"data_width_a": 8, "data_width_b": 8},
                ghdl_plugin_path=Path(ghdl_plugin_path) if ghdl_plugin_path else None,
                ghdl_prefix=Path(ghdl_prefix) if ghdl_prefix else None,
                defined_at=Path(__file__),
            )
        ]

    def setup_vunit(self, vunit_proj: VUnit, **kwargs) -> None:
        tb = vunit_proj.library(self.library_name).test_bench("tb_axi_stream_join")

        for test in tb.get_tests():
            # Zero stall only for the dedicated full-throughput test; randomized
            # backpressure otherwise. Matches hdl-modules' own
            # modules/common/module_common.py precedent for tb_handshake_merger/
            # tb_handshake_splitter.
            stall_probability_percent = 0 if "full_throughput" in test.name else 20

            self.add_vunit_config(
                test=test,
                generics={"stall_probability_percent": stall_probability_percent},
            )
