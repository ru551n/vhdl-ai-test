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
# (2*g_img_width+2 cycles each) plus the axi_stream_fifo elasticity buffer
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
        tb = vunit_proj.library(self.library_name).test_bench("tb_canny_top")

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
                        "g_img_width": _IMG_WIDTH,
                        "g_img_height": _IMG_HEIGHT,
                        "g_thresh_low": _THRESH_LOW,
                        "g_thresh_high": _THRESH_HIGH,
                    },
                    pre_config=pre_config,
                    post_check=post_check,
                )
