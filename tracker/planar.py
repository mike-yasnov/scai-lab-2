import time
from dataclasses import dataclass

import cv2
import numpy as np

from .features import detect, make_detector, match
from .flow import RESEED_BELOW, propagate, seed_points
from .geometry import check_quad, corner_error, project, reference_corners, signed_area, visible_fraction

MIN_INLIER_RATIO = 0.25
STRONG_INLIERS = 40
STRONG_RATIO = 0.5
AGREE_PX = 15.0
AGREE_FRACTION = 0.05
COAST_DECAY = 0.9
MIN_VISIBLE = 0.1
MAGSAC_THRESHOLD = 3.0


@dataclass
class TrackResult:
    quad: np.ndarray | None
    mode: str
    n_keypoints: int = 0
    n_matches: int = 0
    n_inliers: int = 0
    n_flow: int = 0
    ms: float = 0.0


def to_gray(frame):
    return frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


class PlanarTracker:
    def __init__(self, detector="sift", use_flow=True, max_coast=30, ratio=0.75, min_inliers=20, use_detection=True):
        self.detector_name = detector
        self.detector, norm = make_detector(detector)
        self.matcher = cv2.BFMatcher(norm)
        self.use_flow = use_flow
        self.use_detection = use_detection
        self.max_coast = max_coast
        self.ratio = ratio
        self.min_inliers = min_inliers

    def init(self, frame, roi=None):
        cv2.setRNGSeed(0)
        gray = to_gray(frame)
        height, width = gray.shape
        self.frame_size = (width, height)
        self.reference = reference_corners(width, height, roi)
        mask = None
        if roi is not None:
            x, y, w, h = roi
            mask = np.zeros(gray.shape, dtype=np.uint8)
            mask[y:y + h, x:x + w] = 255
        self.ref_points, self.ref_descriptors = detect(self.detector, gray, mask)
        self.homography = np.eye(3)
        self.quad = self.reference.copy()
        self.velocity = np.zeros_like(self.quad)
        self.mode = "det"
        self.coast = 0
        self.prev_gray = gray
        self.flow_points = seed_points(gray, self.quad) if self.use_flow else None
        count = len(self.ref_points)
        return TrackResult(self.quad.copy(), "det", count, count, count, self._flow_count())

    def update(self, frame):
        start = time.perf_counter()
        gray = to_gray(frame)
        detection, stats = self._detect(gray) if self.use_detection else (None, (0, 0, 0))
        flow = self._flow(gray)
        chosen = None
        if detection is not None:
            homography, inliers, ratio = detection
            strong = inliers >= STRONG_INLIERS and ratio >= STRONG_RATIO
            if strong or flow is None or self._agrees(homography, flow[0]):
                chosen = (homography, "det", None)
        if chosen is None and flow is not None:
            chosen = (flow[0], "klt", flow[1])

        if chosen is not None:
            homography, mode, flow_points = chosen
            quad = project(homography, self.reference)
            self.velocity = quad - self.quad if self.mode in ("det", "klt") else np.zeros_like(quad)
            self.homography, self.quad, self.mode, self.coast = homography, quad, mode, 0
            if self.use_flow:
                if mode == "det" or flow_points is None or len(flow_points) < RESEED_BELOW:
                    self.flow_points = seed_points(gray, quad)
                else:
                    self.flow_points = flow_points.reshape(-1, 1, 2)
        else:
            self._coast_or_lose()
            self.flow_points = None

        self.prev_gray = gray
        quad = None if self.mode == "lost" else self.quad.copy()
        ms = (time.perf_counter() - start) * 1000.0
        return TrackResult(quad, self.mode, stats[0], stats[1], stats[2], self._flow_count(), ms)

    def _detect(self, gray):
        points, descriptors = detect(self.detector, gray)
        pairs = match(self.matcher, descriptors, self.ref_descriptors, self.ratio)
        stats = [len(points), len(pairs), 0]
        if len(pairs) < self.min_inliers:
            return None, stats
        source = self.ref_points[pairs[:, 1]]
        target = points[pairs[:, 0]]
        homography, mask = cv2.findHomography(source, target, cv2.USAC_MAGSAC, MAGSAC_THRESHOLD,
                                              maxIters=2000, confidence=0.999)
        if homography is None or mask is None:
            return None, stats
        inlier_mask = mask.ravel() == 1
        inliers = int(inlier_mask.sum())
        stats[2] = inliers
        unique_targets = len(np.unique(np.round(target[inlier_mask]), axis=0))
        if inliers < self.min_inliers or unique_targets < self.min_inliers or inliers / len(pairs) < MIN_INLIER_RATIO:
            return None, stats
        if not check_quad(project(homography, self.reference), self.frame_size, self.reference)[0]:
            return None, stats
        return (homography, inliers, inliers / len(pairs)), stats

    def _flow(self, gray):
        if not self.use_flow or self.mode not in ("det", "klt") or self.flow_points is None:
            return None
        step, points = propagate(self.prev_gray, gray, self.flow_points)
        if step is None:
            return None
        homography = step @ self.homography
        if not check_quad(project(homography, self.reference), self.frame_size, self.reference)[0]:
            return None
        return homography, points

    def _agrees(self, homography_a, homography_b):
        quad_a = project(homography_a, self.reference)
        quad_b = project(homography_b, self.reference)
        tolerance = max(AGREE_PX, AGREE_FRACTION * np.sqrt(abs(signed_area(quad_b))))
        return corner_error(quad_a, quad_b) <= tolerance

    def _coast_or_lose(self):
        if self.mode == "lost" or self.coast >= self.max_coast:
            self.mode = "lost"
            return
        self.coast += 1
        quad = self.quad + self.velocity * COAST_DECAY ** self.coast
        if not check_quad(quad, self.frame_size, self.reference)[0] or visible_fraction(quad, self.frame_size) < MIN_VISIBLE:
            self.mode = "lost"
            return
        self.quad = quad
        self.mode = "pred"

    def _flow_count(self):
        return 0 if self.flow_points is None else len(self.flow_points)
