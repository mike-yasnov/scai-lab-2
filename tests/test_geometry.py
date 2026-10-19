import numpy as np
import pytest

from tracker.geometry import (box_iou, check_quad, corner_error, project, quad_bbox, quad_iou, reference_corners,
                              visible_fraction)

SIZE = (640, 360)
REFERENCE = reference_corners(*SIZE)


def quad(*points):
    return np.array(points, dtype=np.float64)


def test_reference_corners_with_roi():
    np.testing.assert_allclose(reference_corners(640, 360, (10, 20, 100, 50)), [[10, 20], [110, 20], [110, 70], [10, 70]])


def test_identity_projection():
    np.testing.assert_allclose(project(np.eye(3), REFERENCE), REFERENCE)


def test_valid_quad_passes():
    ok, reason = check_quad(quad((100, 50), (400, 70), (380, 300), (120, 280)), SIZE, REFERENCE)
    assert ok, reason


@pytest.mark.parametrize("name, points, reason", [
    ("bowtie", [(100, 50), (400, 300), (400, 50), (100, 300)], None),
    ("point", [(200, 200)] * 4, None),
    ("tiny", [(100, 100), (105, 100), (105, 103), (100, 103)], "area"),
    ("huge", [(-5000, -5000), (5000, -5000), (5000, 5000), (-5000, 5000)], "area"),
    ("mirrored", [(100, 50), (100, 300), (400, 300), (400, 50)], "flipped"),
    ("sharp_angles", [(100, 100), (400, 100), (520, 140), (220, 140)], None),
    ("uneven_sides", [(100, 100), (500, 100), (320, 180), (280, 180)], None),
])
def test_bad_quads_rejected(name, points, reason):
    ok, why = check_quad(quad(*points), SIZE, REFERENCE)
    assert not ok, name
    if reason:
        assert why == reason


def test_nan_rejected():
    assert check_quad(np.full((4, 2), np.nan), SIZE, REFERENCE) == (False, "nonfinite")


def test_corner_error():
    assert corner_error(REFERENCE, REFERENCE + [3, 4]) == pytest.approx(5.0)


def test_box_iou():
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert box_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)
    assert box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_quad_iou_and_bbox():
    square = quad((0, 0), (10, 0), (10, 10), (0, 10))
    shifted = square + [5, 0]
    assert quad_iou(square, shifted) == pytest.approx(1 / 3, abs=1e-6)
    assert quad_bbox(shifted) == (5.0, 0.0, 15.0, 10.0)


def test_visible_fraction():
    half_out = quad((-100, 0), (100, 0), (100, 100), (-100, 100))
    assert visible_fraction(half_out, SIZE) == pytest.approx(0.5)
