import cv2
import numpy as np

DETECTORS = ("sift", "orb", "akaze")


def make_detector(name):
    if name == "sift":
        return cv2.SIFT_create(), cv2.NORM_L2
    if name == "orb":
        return cv2.ORB_create(nfeatures=3000), cv2.NORM_HAMMING
    if name == "akaze":
        return cv2.AKAZE_create(threshold=1e-4), cv2.NORM_HAMMING
    raise ValueError(f"Неизвестный детектор {name!r}, доступны: {', '.join(DETECTORS)}")


def detect(detector, gray, mask=None):
    keypoints, descriptors = detector.detectAndCompute(gray, mask)
    points = np.array([kp.pt for kp in keypoints], dtype=np.float64).reshape(-1, 2)
    return points, descriptors


def match(matcher, frame_descriptors, reference_descriptors, ratio):
    if frame_descriptors is None or reference_descriptors is None:
        return np.empty((0, 2), dtype=int)
    if len(frame_descriptors) < 2 or len(reference_descriptors) < 2:
        return np.empty((0, 2), dtype=int)
    best = {}
    for pair in matcher.knnMatch(frame_descriptors, reference_descriptors, k=2):
        if len(pair) < 2:
            continue
        first, second = pair
        if first.distance >= ratio * second.distance:
            continue
        current = best.get(first.trainIdx)
        if current is None or first.distance < current.distance:
            best[first.trainIdx] = first
    return np.array([[m.queryIdx, m.trainIdx] for m in best.values()], dtype=int).reshape(-1, 2)
