from __future__ import annotations

from typing import TYPE_CHECKING

from tsfpga.module import BaseModule

if TYPE_CHECKING:
    from vunit.ui import VUnit


class Module(BaseModule):
    def setup_vunit(self, vunit_proj: VUnit, **kwargs) -> None:
        tb = vunit_proj.library(self.library_name).test_bench("tb_canny_nms")

        for test in tb.get_tests():
            # Zero stall only for the dedicated full-throughput test; randomized
            # backpressure otherwise. Matches module_axi_stream_join.py's and
            # module_canny_window3x3.py's own precedent (and hdl-modules'
            # modules/common/module_common.py).
            stall_probability_percent = 0 if "full_throughput" in test.name else 20

            self.add_vunit_config(
                test=test,
                generics={"stall_probability_percent": stall_probability_percent},
            )
