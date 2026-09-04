"""Python golden/reference model for the streaming Canny edge detector IP.

Implements, function-by-function, the exact same integer approximations
documented in each module's requirement file (`modules/<name>/doc/<name>_req.md`)
and in `doc/canny_arch.md` — not a higher-precision "more correct" version.
This is the single source of expected-behavior truth for both per-module
VUnit unit tests (import the individual stage function) and the full-pipeline
integration test (`canny_pipeline`), per `shared/Vunit.md` ("Python reference
models (golden models)").

Frame representation: a flat `list[int]`, raster order (row-major,
`index = row * width + col`), matching the AXI4-Stream beat order the DUT
sees (`TLAST` = end-of-line, `TUSER(0)` = start-of-frame per
`doc/canny_arch.md` "Frame framing convention").

Border convention: every `canny_window3x3` instance dilates the border ring
by 1 pixel (own edge-of-frame test OR'd with an OR-reduction of all 9
incoming border taps). Four instances (W1..W4) in the pipeline -> the final
4-pixel-wide border ring around every frame reads as a forced-zero/no-edge
result (`doc/canny_arch.md` "Growing border (corrected, rev 2.2)").
"""

from __future__ import annotations

from typing import NamedTuple

Window = tuple[int, int, int, int, int, int, int, int, int]
"""3x3 window taps, row-major: (tl, tm, tr, ml, mm, mr, bl, bm, br)."""

DIR_0 = 0  # "00" - 0 deg (horizontal gradient / vertical edge)
DIR_45 = 1  # "01" - 45 deg
DIR_90 = 2  # "10" - 90 deg (vertical gradient / horizontal edge)
DIR_135 = 3  # "11" - 135 deg

CLASS_NONE = 0  # "00"
CLASS_WEAK = 1  # "01"
CLASS_STRONG = 2  # "10"


class WindowResult(NamedTuple):
    windows: list[Window]
    border: list[int]


def sliding_window_3x3(
    values: list[int], width: int, height: int, border_in: list[int] | None = None
) -> WindowResult:
    """Golden model of `canny_window3x3`.

    `border_in`: incoming per-position border bit, same raster order as
    `values`. `None` models `g_user_width=1` (first instance in the chain,
    no border marked yet upstream) -> treated as constant 0 for all 9 taps,
    per `modules/canny_window3x3/doc/canny_window3x3_req.md`.

    Out-of-bounds taps (before row 0 / after the last row / before col 0 /
    after the last col) read as 0, for both `values` and `border_in` — a
    position with any out-of-bounds tap is always at the frame edge itself,
    so its own edge-of-frame test already forces `border_out=1` regardless
    of the out-of-bounds tap's border value (see module docstring).
    """
    n = width * height
    if len(values) != n:
        raise ValueError(f"values length {len(values)} != width*height {n}")
    if border_in is None:
        border_in = [0] * n
    elif len(border_in) != n:
        raise ValueError(f"border_in length {len(border_in)} != width*height {n}")

    def sample(arr: list[int], row: int, col: int) -> int:
        if row < 0 or row >= height or col < 0 or col >= width:
            return 0
        return arr[row * width + col]

    windows: list[Window] = []
    border_out: list[int] = []
    for row in range(height):
        for col in range(width):
            windows.append(
                (
                    sample(values, row - 1, col - 1),
                    sample(values, row - 1, col),
                    sample(values, row - 1, col + 1),
                    sample(values, row, col - 1),
                    sample(values, row, col),
                    sample(values, row, col + 1),
                    sample(values, row + 1, col - 1),
                    sample(values, row + 1, col),
                    sample(values, row + 1, col + 1),
                )
            )
            own_edge = row == 0 or row == height - 1 or col == 0 or col == width - 1
            neighbor_border = any(
                sample(border_in, r, c)
                for r in (row - 1, row, row + 1)
                for c in (col - 1, col, col + 1)
            )
            border_out.append(1 if (own_edge or neighbor_border) else 0)
    return WindowResult(windows, border_out)


def gaussian3x3(window: Window, border: int) -> int:
    """Golden model of `canny_gaussian3x3`.

    Weights [1,2,1;2,4,2;1,2,1] (sum 16), truncating integer divide.
    """
    tl, tm, tr, ml, mm, mr, bl, bm, br = window
    if border:
        return 0
    return (tl + 2 * tm + tr + 2 * ml + 4 * mm + 2 * mr + bl + 2 * bm + br) // 16


class SobelResult(NamedTuple):
    magnitude: int
    direction: int


def sobel3x3(window: Window, border: int) -> SobelResult:
    """Golden model of `canny_sobel3x3` (both forks: L1 magnitude + 4-sector
    direction). Border forces magnitude=0, direction="00"/DIR_0.
    """
    tl, tm, tr, ml, mm, mr, bl, bm, br = window
    if border:
        return SobelResult(0, DIR_0)

    gx = (tr + 2 * mr + br) - (tl + 2 * ml + bl)
    gy = (bl + 2 * bm + br) - (tl + 2 * tm + tr)
    magnitude = abs(gx) + abs(gy)

    ax, ay = abs(gx), abs(gy)
    if ay <= (ax >> 1):
        direction = DIR_0
    elif ax <= (ay >> 1):
        direction = DIR_90
    else:
        same_sign = (gx < 0) == (gy < 0)
        direction = DIR_45 if same_sign else DIR_135

    return SobelResult(magnitude, direction)


def nms(mag_window: Window, direction: int, border: int) -> int:
    """Golden model of `canny_nms`. Ties resolve to "kept" (non-strict >=)."""
    tl, tm, tr, ml, mm, mr, bl, bm, br = mag_window
    if border:
        return 0

    if direction == DIR_0:
        a, b = ml, mr
    elif direction == DIR_45:
        a, b = tr, bl
    elif direction == DIR_90:
        a, b = tm, bm
    else:  # DIR_135
        a, b = tl, br

    return mm if (mm >= a and mm >= b) else 0


def threshold(magnitude: int, thresh_low: int, thresh_high: int, border: int) -> int:
    """Golden model of `canny_threshold`."""
    if border:
        return CLASS_NONE
    if magnitude >= thresh_high:
        return CLASS_STRONG
    if magnitude >= thresh_low:
        return CLASS_WEAK
    return CLASS_NONE


def hysteresis(class_window: Window, border: int) -> int:
    """Golden model of `canny_hysteresis`. Returns the final edge bit (0/1)."""
    tl, tm, tr, ml, mm, mr, bl, bm, br = class_window
    if border:
        return 0
    others = (tl, tm, tr, ml, mr, bl, bm, br)
    if mm == CLASS_STRONG:
        return 1
    if mm == CLASS_WEAK and any(tap == CLASS_STRONG for tap in others):
        return 1
    return 0


class CannyResult(NamedTuple):
    edges: list[int]
    """Final edge bit per position, raster order, same length as the input frame."""


def canny_pipeline(
    pixels: list[int], width: int, height: int, thresh_low: int, thresh_high: int
) -> CannyResult:
    """Full-pipeline golden model, stage order matching `canny_top`:

    s_axis -> W1 -> gaussian -> W2 -> sobel -+-> W3 -> join(a) -> nms -> threshold -> W4 -> hysteresis -> m_axis
                                              +----------------------------> join(b) --^
    """
    w1 = sliding_window_3x3(pixels, width, height, border_in=None)
    smoothed = [gaussian3x3(w, b) for w, b in zip(w1.windows, w1.border)]

    w2 = sliding_window_3x3(smoothed, width, height, border_in=w1.border)
    sobel_out = [sobel3x3(w, b) for w, b in zip(w2.windows, w2.border)]
    magnitudes = [s.magnitude for s in sobel_out]
    directions = [s.direction for s in sobel_out]
    # sobel passes border through unchanged, identically to both forks.
    sobel_border = w2.border

    w3 = sliding_window_3x3(magnitudes, width, height, border_in=sobel_border)
    # axi_stream_join's border = OR(w3.border, direction fork's border); the
    # direction fork's border is sobel_border unchanged, which w3's own
    # dilate already OR's in via its center tap, so this is a no-op OR.
    joined_border = [a | b for a, b in zip(w3.border, sobel_border)]

    suppressed = [
        nms(w, d, b) for w, d, b in zip(w3.windows, directions, joined_border)
    ]
    classified = [
        threshold(m, thresh_low, thresh_high, b)
        for m, b in zip(suppressed, joined_border)
    ]

    w4 = sliding_window_3x3(classified, width, height, border_in=joined_border)
    edges = [hysteresis(w, b) for w, b in zip(w4.windows, w4.border)]

    return CannyResult(edges)
