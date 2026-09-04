from __future__ import annotations

from typing import TYPE_CHECKING

from tsfpga.module import BaseModule

if TYPE_CHECKING:
    from vunit.ui import VUnit


class Module(BaseModule):
    def setup_vunit(self, vunit_proj: VUnit, **kwargs) -> None:
        tb = vunit_proj.library(self.library_name).test_bench("tb_canny_sobel3x3")

        for test in tb.get_tests():
            # Zero stall on all three handshakes only for the dedicated
            # full-throughput test (its check_relation timing check requires
            # back-to-back beats). All other tests use independent, asymmetric
            # stall probabilities on the input and on the two output forks so
            # that the forks are exercised stalling differently from each
            # other, which is what actually stresses handshake_splitter's
            # sticky per-output "already transacted" bookkeeping.
            if "full_throughput" in test.name:
                configs = [(0, 0, 0)]
            else:
                # Two asymmetric configs, with the slower fork swapped between
                # mag and dir, so both "mag lags dir" and "dir lags mag"
                # orderings of handshake_splitter's joint acceptance are
                # exercised, in addition to input-side stalling.
                configs = [(10, 10, 30), (10, 30, 10)]

            for stall_in, stall_mag, stall_dir in configs:
                self.add_vunit_config(
                    test=test,
                    name=f"in{stall_in}_mag{stall_mag}_dir{stall_dir}",
                    generics={
                        "stall_probability_percent_in": stall_in,
                        "stall_probability_percent_mag": stall_mag,
                        "stall_probability_percent_dir": stall_dir,
                    },
                )
