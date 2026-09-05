from __future__ import annotations

from tsfpga.module import BaseModule
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vunit.ui import VUnit


class Module(BaseModule):
    """`get_build_projects` deferred until the top-level entity (or a
    synthesizable leaf) exists; see `module_canny.py`/`module_axi_stream_join.py`
    for the lazy-`tsfpga.yosys.project`-import pattern to reuse then."""

    def setup_vunit(self, vunit_proj: VUnit, **kwargs) -> None:
        library = vunit_proj.library(self.library_name)

        self._setup_cnn_accel_bias_requant(library)
        self._setup_cnn_accel_pool(library)

    def _setup_cnn_accel_bias_requant(self, library) -> None:
        tb = library.test_bench("tb_cnn_accel_bias_requant")

        for test in tb.get_tests():
            # Zero stall on both links only for the dedicated
            # full-throughput test (its check_relation timing check requires
            # back-to-back beats); randomized independent per-link
            # backpressure otherwise. Matches module_canny.py's
            # `_setup_canny_threshold` precedent.
            stall = 0 if "full_throughput" in test.name else 20

            self.add_vunit_config(
                test=test,
                generics={
                    "stall_probability_percent_in": stall,
                    "stall_probability_percent_out": stall,
                },
            )

    def _setup_cnn_accel_pool(self, library) -> None:
        tb = library.test_bench("tb_cnn_accel_pool")

        for test in tb.get_tests():
            # Zero stall on all three links only for the dedicated
            # full-throughput test; randomized independent per-link
            # backpressure otherwise. Matches module_canny.py's
            # `_setup_canny_threshold` precedent.
            stall = 0 if "full_throughput" in test.name else 20

            self.add_vunit_config(
                test=test,
                generics={
                    "stall_probability_percent_in": stall,
                    "stall_probability_percent_max": stall,
                    "stall_probability_percent_avgsum": stall,
                },
            )
