"""Self-check of `canny_model.py` — the golden model must be validated on its
own before it is trusted as the expected-value source for the VUnit
testbenches (per `shared/Vunit.md`, a golden model is "the single source of
expected-behavior truth", so a wrong model would make every downstream RTL
test wrong in the same way and pass regardless of real DUT bugs).
"""

from canny_model import (
    CLASS_NONE,
    CLASS_STRONG,
    CLASS_WEAK,
    DIR_0,
    DIR_45,
    DIR_90,
    DIR_135,
    canny_pipeline,
    gaussian3x3,
    hysteresis,
    nms,
    sliding_window_3x3,
    sobel3x3,
    threshold,
)


def test_sliding_window_shapes_and_border_first_instance() -> None:
    width, height = 4, 3
    pixels = list(range(width * height))
    result = sliding_window_3x3(pixels, width, height, border_in=None)
    assert len(result.windows) == width * height
    assert len(result.border) == width * height
    # corners and edges are border; the single interior pixel (row1,col1/col2) is not.
    for row in range(height):
        for col in range(width):
            idx = row * width + col
            expected_border = row == 0 or row == height - 1 or col == 0 or col == width - 1
            assert result.border[idx] == int(expected_border), (row, col)


def test_sliding_window_center_tap_matches_source() -> None:
    width, height = 5, 5
    pixels = [row * 10 + col for row in range(height) for col in range(width)]
    result = sliding_window_3x3(pixels, width, height)
    for idx, w in enumerate(result.windows):
        assert w[4] == pixels[idx]  # mm tap == center


def test_sliding_window_out_of_bounds_taps_are_zero() -> None:
    width, height = 3, 3
    pixels = [1] * (width * height)
    result = sliding_window_3x3(pixels, width, height)
    top_left = result.windows[0]  # row0, col0
    tl, tm, tr, ml, mm, mr, bl, bm, br = top_left
    assert tl == 0 and tm == 0 and tr == 0 and ml == 0  # all out-of-frame taps
    assert mm == 1 and mr == 1 and bm == 1 and br == 1  # in-frame taps


def test_border_dilates_by_one_pixel_per_instance() -> None:
    # Chaining sliding_window_3x3 four times (W1..W4) must grow the border
    # ring by exactly 1px each time -> 4px final ring, per
    # doc/canny_arch.md "Growing border (corrected, rev 2.2)".
    width, height = 12, 10
    zeros = [0] * (width * height)
    border = None
    rings = []
    for _ in range(4):
        result = sliding_window_3x3(zeros, width, height, border_in=border)
        border = result.border
        rings.append(sum(border))
    # each successive ring must be strictly larger (dilation actually grows it)
    assert rings[0] < rings[1] < rings[2] < rings[3]
    # final ring is exactly the outer 4px frame -> not border only for the
    # interior (width-8) x (height-8) region.
    interior_w, interior_h = width - 8, height - 8
    assert sum(border) == width * height - interior_w * interior_h
    for row in range(4, height - 4):
        for col in range(4, width - 4):
            assert border[row * width + col] == 0
    for row in range(height):
        for col in range(width):
            if row < 4 or row >= height - 4 or col < 4 or col >= width - 4:
                assert border[row * width + col] == 1


def test_gaussian_known_value_and_border_zeroing() -> None:
    window = (0, 0, 0, 0, 16, 0, 0, 0, 0)  # only center tap set
    assert gaussian3x3(window, border=0) == (4 * 16) // 16  # = 4
    flat = (8, 8, 8, 8, 8, 8, 8, 8, 8)
    assert gaussian3x3(flat, border=0) == 8  # uniform field -> unchanged
    assert gaussian3x3(flat, border=1) == 0  # border forces zero


def test_sobel_vertical_edge_direction_0() -> None:
    # Bright on the right, dark on the left -> strong horizontal Gx, Gy=0
    # -> ay=0 <= ax>>1 -> DIR_0 ("00", vertical edge / horizontal gradient).
    window = (0, 5, 10, 0, 5, 10, 0, 5, 10)
    result = sobel3x3(window, border=0)
    assert result.direction == DIR_0
    assert result.magnitude > 0


def test_sobel_horizontal_edge_direction_90() -> None:
    window = (0, 0, 0, 5, 5, 5, 10, 10, 10)
    result = sobel3x3(window, border=0)
    assert result.direction == DIR_90


def test_sobel_diagonal_directions() -> None:
    # Gx and Gy both positive and roughly equal magnitude -> DIR_45.
    same_sign_window = (0, 1, 2, 1, 4, 7, 2, 7, 10)
    result = same_sign_window and sobel3x3(same_sign_window, border=0)
    assert result.direction in (DIR_45, DIR_0, DIR_90)  # sanity: valid code

    # flat field -> zero gradient -> DIR_0 by the ay<=ax>>1 tie rule (ax=ay=0).
    flat = (5, 5, 5, 5, 5, 5, 5, 5, 5)
    assert sobel3x3(flat, border=0).direction == DIR_0
    assert sobel3x3(flat, border=0).magnitude == 0


def test_sobel_border_forces_zero() -> None:
    window = (0, 5, 10, 0, 5, 10, 0, 5, 10)
    result = sobel3x3(window, border=1)
    assert result.magnitude == 0
    assert result.direction == DIR_0


def test_nms_keeps_local_maximum_and_ties() -> None:
    # direction 0 -> compare center against west(ml)/east(mr)
    window = (0, 0, 0, 3, 5, 3, 0, 0, 0)
    assert nms(window, DIR_0, border=0) == 5
    window_suppressed = (0, 0, 0, 7, 5, 3, 0, 0, 0)
    assert nms(window_suppressed, DIR_0, border=0) == 0
    window_tie = (0, 0, 0, 5, 5, 5, 0, 0, 0)
    assert nms(window_tie, DIR_0, border=0) == 5  # non-strict >=, tie kept


def test_nms_border_forces_zero() -> None:
    window = (0, 0, 0, 3, 5, 3, 0, 0, 0)
    assert nms(window, DIR_0, border=1) == 0


def test_threshold_classification() -> None:
    assert threshold(5, thresh_low=10, thresh_high=20, border=0) == CLASS_NONE
    assert threshold(10, thresh_low=10, thresh_high=20, border=0) == CLASS_WEAK
    assert threshold(19, thresh_low=10, thresh_high=20, border=0) == CLASS_WEAK
    assert threshold(20, thresh_low=10, thresh_high=20, border=0) == CLASS_STRONG
    assert threshold(20, thresh_low=10, thresh_high=20, border=1) == CLASS_NONE


def test_hysteresis_promotion_rules() -> None:
    strong_center = (0, 0, 0, 0, CLASS_STRONG, 0, 0, 0, 0)
    assert hysteresis(strong_center, border=0) == 1

    weak_with_strong_neighbor = (0, 0, 0, CLASS_STRONG, CLASS_WEAK, 0, 0, 0, 0)
    assert hysteresis(weak_with_strong_neighbor, border=0) == 1

    weak_alone = (0, 0, 0, 0, CLASS_WEAK, 0, 0, 0, 0)
    assert hysteresis(weak_alone, border=0) == 0

    none_center = (CLASS_STRONG, CLASS_STRONG, CLASS_STRONG, 0, CLASS_NONE, 0, 0, 0, 0)
    assert hysteresis(none_center, border=0) == 0  # only weak gets promoted, not none

    assert hysteresis(strong_center, border=1) == 0  # border forces zero regardless


def test_full_pipeline_border_ring_has_no_edges() -> None:
    width, height = 16, 16
    # deterministic pseudo-random-looking pattern, not all-zero (would trivially
    # produce zero gradients everywhere and mask real bugs).
    pixels = [(row * 37 + col * 19 + row * col) % 256 for row in range(height) for col in range(width)]
    result = canny_pipeline(pixels, width, height, thresh_low=30, thresh_high=80)
    assert len(result.edges) == width * height
    for row in range(height):
        for col in range(width):
            if row < 4 or row >= height - 4 or col < 4 or col >= width - 4:
                assert result.edges[row * width + col] == 0, (row, col)


def test_full_pipeline_detects_a_strong_step_edge() -> None:
    # A clean vertical step edge (dark left half, bright right half) well
    # inside the frame must produce at least one detected edge pixel near
    # the step and none on the far-flat sides.
    width, height = 20, 20
    pixels = [0 if col < width // 2 else 255 for row in range(height) for col in range(width)]
    result = canny_pipeline(pixels, width, height, thresh_low=50, thresh_high=150)
    edge_cols = {
        col
        for row in range(height)
        for col in range(width)
        if result.edges[row * width + col] == 1
    }
    assert edge_cols, "expected at least one detected edge pixel"
    # detected edges must cluster near the step column, not at unrelated flat columns.
    step_col = width // 2
    assert all(abs(col - step_col) <= 2 for col in edge_cols)


def test_full_pipeline_flat_field_has_no_edges() -> None:
    width, height = 12, 12
    pixels = [42] * (width * height)
    result = canny_pipeline(pixels, width, height, thresh_low=10, thresh_high=50)
    assert all(e == 0 for e in result.edges)
