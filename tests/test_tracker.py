import cv2
import numpy as np
import pytest

from conftest import HEIGHT, WIDTH, homography, make_texture, warp
from tracker import PlanarTracker
from tracker.features import match
from tracker.geometry import corner_error, project, reference_corners
from tracker.video import VideoSink, iter_frames

REFERENCE = reference_corners(WIDTH, HEIGHT)


@pytest.mark.parametrize("detector, tolerance", [("sift", 1.5), ("orb", 3.0), ("akaze", 3.0)])
def test_recovers_known_homography(texture, detector, tolerance):
    h = homography(angle_deg=15, scale=0.6, tx=20, ty=-10, px=1e-4, py=-5e-5)
    tracker = PlanarTracker(detector, use_flow=False, max_coast=0)
    tracker.init(texture)
    result = tracker.update(warp(texture, h))
    assert result.mode == "det"
    assert corner_error(result.quad, project(h, REFERENCE)) <= tolerance


def smooth_sequence(steps=20):
    return [homography(angle_deg=0.5 * i, scale=1.0 - 0.015 * i, tx=2 * i, ty=-i, px=2e-6 * i) for i in range(1, steps + 1)]


def test_sequence_with_detection(texture):
    tracker = PlanarTracker("sift")
    tracker.init(texture)
    for h in smooth_sequence():
        result = tracker.update(warp(texture, h))
        assert result.mode in ("det", "klt")
        assert corner_error(result.quad, project(h, REFERENCE)) < 2.0


def test_sequence_with_flow_only(texture):
    tracker = PlanarTracker("sift", use_detection=False)
    tracker.init(texture)
    for h in smooth_sequence():
        result = tracker.update(warp(texture, h))
        assert result.mode == "klt"
        assert corner_error(result.quad, project(h, REFERENCE)) < 2.0


def test_heavy_blur_gives_lost_not_garbage(texture):
    tracker = PlanarTracker("sift", max_coast=0)
    tracker.init(texture)
    blurred = cv2.GaussianBlur(warp(texture, homography(scale=0.7)), (0, 0), 20)
    result = tracker.update(blurred)
    assert result.mode == "lost"
    assert result.quad is None


def test_noise_frame_gives_lost(texture):
    tracker = PlanarTracker("sift", max_coast=0)
    tracker.init(texture)
    noise = np.random.default_rng(1).integers(0, 256, texture.shape, dtype=np.uint8)
    assert tracker.update(noise).mode == "lost"


def test_prediction_then_lost(texture):
    tracker = PlanarTracker("sift", max_coast=3)
    tracker.init(texture)
    for h in smooth_sequence(3):
        tracker.update(warp(texture, h))
    empty = np.full_like(texture, 40)
    modes = [tracker.update(empty).mode for _ in range(5)]
    assert modes == ["pred", "pred", "pred", "lost", "lost"]


def test_relock_after_loss(texture):
    tracker = PlanarTracker("sift", max_coast=0)
    tracker.init(texture)
    tracker.update(np.full_like(texture, 40))
    h = homography(angle_deg=5, scale=0.8)
    result = tracker.update(warp(texture, h))
    assert result.mode == "det"
    assert corner_error(result.quad, project(h, REFERENCE)) < 2.0


def test_one_to_one_matching():
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    reference = np.array([[0.0] * 8, [100.0] * 8], dtype=np.float32)
    frame = np.array([[0.1 * i] * 8 for i in range(10)], dtype=np.float32)
    pairs = match(matcher, frame, reference, 0.75)
    assert len(pairs) == 1
    assert pairs[0, 1] == 0


def test_deterministic(texture):
    runs = []
    for _ in range(2):
        tracker = PlanarTracker("sift")
        tracker.init(texture)
        runs.append([tracker.update(warp(texture, h)).quad for h in smooth_sequence(5)])
    for a, b in zip(*runs):
        np.testing.assert_array_equal(a, b)


def test_roi_reference(texture):
    tracker = PlanarTracker("sift", use_flow=False)
    first = tracker.init(texture, roi=(100, 50, 400, 250))
    np.testing.assert_allclose(first.quad, reference_corners(WIDTH, HEIGHT, (100, 50, 400, 250)))
    h = homography(angle_deg=8, scale=0.8)
    result = tracker.update(warp(texture, h))
    expected = project(h, reference_corners(WIDTH, HEIGHT, (100, 50, 400, 250)))
    assert corner_error(result.quad, expected) < 2.0


def test_video_sink_roundtrip(tmp_path):
    path = tmp_path / "clip.mp4"
    frame = make_texture(3)
    with VideoSink(path, 30, (WIDTH, HEIGHT)) as sink:
        for _ in range(5):
            sink.write(frame)
    frames = list(iter_frames(path))
    assert len(frames) == 5
    assert frames[0].shape == (HEIGHT, WIDTH, 3)
