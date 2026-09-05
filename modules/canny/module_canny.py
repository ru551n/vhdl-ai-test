from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import TYPE_CHECKING

from tsfpga.module import BaseModule

from canny_model import canny_pipeline

if TYPE_CHECKING:
    from vunit.ui import VUnit

# Kept modest so the four chained canny_window3x3 stages' fill latency
# (2*img_width+2 cycles each) plus the axi_stream_fifo elasticity buffer
# still simulate quickly, but large enough that the pipeline's 4-pixel-wide
# forced-zero border ring (doc/canny_arch.md "Growing border") does not
# swallow the entire frame -- a width/height <= 8 leaves no interior pixel
# at all (every position is within 4 taps of an edge), which would make
# "expected" all-zero and the golden-model comparison vacuous.
_IMG_WIDTH = 20
_IMG_HEIGHT = 16
_THRESH_LOW = 40
_THRESH_HIGH = 90


def _write_csv(path: Path, rows: list[list[int]]) -> None:
    with path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(rows)


def _read_csv(path: Path) -> list[list[int]]:
    with path.open() as csv_file:
        return [[int(value) for value in row] for row in csv.reader(csv_file) if row]


class Module(BaseModule):
    def setup_vunit(self, vunit_proj: VUnit, **kwargs) -> None:
        library = vunit_proj.library(self.library_name)

        self._setup_canny_window3x3(library)
        self._setup_canny_gaussian3x3(library)
        self._setup_canny_sobel3x3(library)
        self._setup_canny_nms(library)
        self._setup_canny_threshold(library)
        self._setup_canny_hysteresis(library)
        self._setup_canny_top(library)

    def _setup_canny_window3x3(self, library) -> None:
        tb = library.test_bench("tb_canny_window3x3")

        for test in tb.get_tests():
            # Zero stall only for the dedicated full-throughput test; randomized
            # backpressure otherwise. Matches module_axi_stream_join.py's own
            # precedent (and hdl-modules' modules/common/module_common.py).
            stall_probability_percent = 0 if "full_throughput" in test.name else 20

            self.add_vunit_config(
                test=test,
                generics={"stall_probability_percent": stall_probability_percent},
            )

    def _setup_canny_gaussian3x3(self, library) -> None:
        tb = library.test_bench("tb_canny_gaussian3x3")

        for test in tb.get_tests():
            # Zero stall only for the dedicated full-throughput test; randomized
            # backpressure otherwise. Matches module_canny.py's own
            # precedent (and hdl-modules' modules/common/module_common.py).
            stall_probability_percent = 0 if "full_throughput" in test.name else 20

            self.add_vunit_config(
                test=test,
                generics={"stall_probability_percent": stall_probability_percent},
            )

    def _setup_canny_sobel3x3(self, library) -> None:
        tb = library.test_bench("tb_canny_sobel3x3")

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

    def _setup_canny_nms(self, library) -> None:
        tb = library.test_bench("tb_canny_nms")

        for test in tb.get_tests():
            # Zero stall only for the dedicated full-throughput test; randomized
            # backpressure otherwise. Matches module_axi_stream_join.py's and
            # module_canny.py's own precedent (and hdl-modules'
            # modules/common/module_common.py).
            stall_probability_percent = 0 if "full_throughput" in test.name else 20

            self.add_vunit_config(
                test=test,
                generics={"stall_probability_percent": stall_probability_percent},
            )

    def _setup_canny_threshold(self, library) -> None:
        tb = library.test_bench("tb_canny_threshold")

        for test in tb.get_tests():
            # Zero stall only for the dedicated full-throughput test; randomized
            # backpressure otherwise. Matches module_axi_stream_join.py's own
            # precedent (and hdl-modules' modules/common/module_common.py).
            stall_probability_percent = 0 if "full_throughput" in test.name else 20

            self.add_vunit_config(
                test=test,
                generics={"stall_probability_percent": stall_probability_percent},
            )

    def _setup_canny_hysteresis(self, library) -> None:
        tb = library.test_bench("tb_canny_hysteresis")

        for test in tb.get_tests():
            # Zero stall only for the dedicated full-throughput test; randomized
            # backpressure otherwise. Matches module_axi_stream_join.py's own
            # precedent (and hdl-modules' modules/common/module_common.py).
            stall_probability_percent = 0 if "full_throughput" in test.name else 20

            self.add_vunit_config(
                test=test,
                generics={"stall_probability_percent": stall_probability_percent},
            )

    def _setup_canny_top(self, library) -> None:
        tb = library.test_bench("tb_canny_top")

        for test in tb.get_tests():
            # Directed zero-stall config plus a randomized-backpressure
            # config, per shared/Vunit.md's mandatory-backpressure rule --
            # same VHDL test case, same golden-model comparison, only the
            # AXI4-Stream VCs' stall_config differs between configs.
            for stall_name, stall_probability_percent in (
                ("zero_stall", 0),
                ("random_stall", 20),
            ):
                # Bind stall_name as a default arg (not a closure over the
                # loop variable) so each config's pre_config regenerates its
                # own independent stimulus/expected pair.
                def pre_config(
                    output_path: str,
                    seed: int,
                    _stall_name: str = stall_name,
                ) -> bool:
                    width, height = _IMG_WIDTH, _IMG_HEIGHT
                    rnd = random.Random(f"{seed}-{_stall_name}")
                    pixels = [rnd.randrange(0, 256) for _ in range(width * height)]

                    result = canny_pipeline(
                        pixels, width, height, _THRESH_LOW, _THRESH_HIGH
                    )

                    stim_rows = [pixels[row * width : (row + 1) * width] for row in range(height)]
                    expected_rows = [
                        result.edges[row * width : (row + 1) * width] for row in range(height)
                    ]

                    out_dir = Path(output_path)
                    _write_csv(out_dir / "stimulus.csv", stim_rows)
                    _write_csv(out_dir / "expected.csv", expected_rows)
                    return True

                def post_check(output_path: str) -> bool:
                    out_dir = Path(output_path)
                    expected_rows = _read_csv(out_dir / "expected.csv")
                    result_rows = _read_csv(out_dir / "result.csv")

                    if expected_rows == result_rows:
                        return True

                    ok = True
                    for row, (expected_row, result_row) in enumerate(
                        zip(expected_rows, result_rows)
                    ):
                        for col, (expected_val, result_val) in enumerate(
                            zip(expected_row, result_row)
                        ):
                            if expected_val != result_val:
                                ok = False
                                print(
                                    "canny_top golden-model mismatch at "
                                    f"row={row} col={col}: "
                                    f"expected={expected_val} actual={result_val}"
                                )
                    return ok

                self.add_vunit_config(
                    test=test,
                    name=stall_name,
                    generics={
                        "stall_probability_percent": stall_probability_percent,
                        "img_width": _IMG_WIDTH,
                        "img_height": _IMG_HEIGHT,
                        "thresh_low": _THRESH_LOW,
                        "thresh_high": _THRESH_HIGH,
                    },
                    pre_config=pre_config,
                    post_check=post_check,
                )
