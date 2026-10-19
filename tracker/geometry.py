import cv2
import numpy as np

MIN_AREA_FRACTION = 0.002
MAX_AREA_FRACTION = 10.0
MAX_SIDE_RATIO = 3.0
MIN_ASPECT_RATIO = 0.25
MAX_ASPECT_RATIO = 4.0
MIN_ANGLE = 30.0
MAX_ANGLE = 150.0


def reference_corners(width, height, roi=None):
    x, y, w, h = roi if roi is not None else (0, 0, width, height)
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float64)


def project(homography, corners):
    points = np.asarray(corners, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(points, homography).reshape(-1, 2)


def signed_area(quad):
    x, y = quad[:, 0], quad[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def side_lengths(quad):
    return np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)


def interior_angles(quad):
    angles = []
    for i in range(4):
        a = quad[i - 1] - quad[i]
        b = quad[(i + 1) % 4] - quad[i]
        cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        angles.append(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
    return np.array(angles)


def aspect_ratio(quad):
    sides = side_lengths(quad)
    return (sides[0] + sides[2]) / (sides[1] + sides[3])


def check_quad(quad, frame_size, reference):
    if quad is None or not np.all(np.isfinite(quad)):
        return False, "nonfinite"
    width, height = frame_size
    area = signed_area(quad)
    if area * signed_area(reference) <= 0:
        return False, "flipped"
    if not cv2.isContourConvex(quad.astype(np.float32).reshape(-1, 1, 2)):
        return False, "concave"
    if not MIN_AREA_FRACTION <= abs(area) / (width * height) <= MAX_AREA_FRACTION:
        return False, "area"
    sides = side_lengths(quad)
    if sides.min() < 1e-6:
        return False, "degenerate"
    if max(sides[0] / sides[2], sides[2] / sides[0], sides[1] / sides[3], sides[3] / sides[1]) > MAX_SIDE_RATIO:
        return False, "sides"
    if not MIN_ASPECT_RATIO <= aspect_ratio(quad) / aspect_ratio(reference) <= MAX_ASPECT_RATIO:
        return False, "aspect"
    angles = interior_angles(quad)
    if angles.min() < MIN_ANGLE or angles.max() > MAX_ANGLE:
        return False, "angles"
    return True, "ok"


def corner_error(quad_a, quad_b):
    return float(np.mean(np.linalg.norm(np.asarray(quad_a) - np.asarray(quad_b), axis=1)))


def quad_bbox(quad):
    return (float(quad[:, 0].min()), float(quad[:, 1].min()), float(quad[:, 0].max()), float(quad[:, 1].max()))


def clip_box(box, frame_size):
    width, height = frame_size
    x0, y0, x1, y1 = box
    return (min(max(x0, 0.0), width), min(max(y0, 0.0), height), min(max(x1, 0.0), width), min(max(y1, 0.0), height))


def box_iou(box_a, box_b):
    ix = max(0.0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]))
    iy = max(0.0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))
    inter = ix * iy
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def quad_iou(quad_a, quad_b):
    a = np.asarray(quad_a, dtype=np.float32)
    b = np.asarray(quad_b, dtype=np.float32)
    inter, _ = cv2.intersectConvexConvex(a, b)
    union = abs(signed_area(a.astype(np.float64))) + abs(signed_area(b.astype(np.float64))) - inter
    return float(inter / union) if union > 0 else 0.0


def visible_fraction(quad, frame_size):
    width, height = frame_size
    frame = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    area = abs(signed_area(np.asarray(quad, dtype=np.float64)))
    if area <= 0:
        return 0.0
    inter, _ = cv2.intersectConvexConvex(np.asarray(quad, dtype=np.float32), frame)
    return float(inter / area)
