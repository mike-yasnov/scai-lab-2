import cv2
import numpy as np

LK_PARAMS = {
    "winSize": (21, 21),
    "maxLevel": 3,
    "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
}
MAX_CORNERS = 800
MIN_FLOW_POINTS = 10
QUALITY_LEVEL = 0.001
MIN_DISTANCE = 3
MAX_FB_ERROR = 1.0
RANSAC_THRESHOLD = 2.0
RESEED_BELOW = 300
SHRINK = 2


def seed_points(gray, quad):
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(quad).astype(np.int32), 255)
    mask = cv2.erode(mask, np.ones((2 * SHRINK + 1, 2 * SHRINK + 1), np.uint8))
    if not mask.any():
        return None
    points = cv2.goodFeaturesToTrack(gray, MAX_CORNERS, QUALITY_LEVEL, MIN_DISTANCE, mask=mask)
    if points is None or len(points) < MIN_FLOW_POINTS:
        return None
    return points.astype(np.float32)


def propagate(prev_gray, gray, points):
    if points is None or len(points) < MIN_FLOW_POINTS:
        return None, None
    forward, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, points, None, **LK_PARAMS)
    backward, status_back, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, forward, None, **LK_PARAMS)
    fb_error = np.linalg.norm((points - backward).reshape(-1, 2), axis=1)
    good = (status.ravel() == 1) & (status_back.ravel() == 1) & (fb_error < MAX_FB_ERROR)
    if good.sum() < MIN_FLOW_POINTS:
        return None, None
    step, inliers = cv2.findHomography(points[good], forward[good], cv2.RANSAC, RANSAC_THRESHOLD)
    if step is None or inliers is None or inliers.sum() < MIN_FLOW_POINTS:
        return None, None
    return step, forward[good][inliers.ravel() == 1]
