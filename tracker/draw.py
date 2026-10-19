import cv2
import numpy as np

GREEN = (0, 200, 0)
YELLOW = (0, 215, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def put_label(image, text, org, scale=0.5, color=WHITE, background=BLACK):
    (w, h), baseline = cv2.getTextSize(text, FONT, scale, 1)
    height, width = image.shape[:2]
    x = int(min(max(org[0], 2), width - w - 4))
    y = int(min(max(org[1], h + 4), height - baseline - 2))
    cv2.rectangle(image, (x - 3, y - h - 4), (x + w + 3, y + baseline + 1), background, -1)
    cv2.putText(image, text, (x, y), FONT, scale, color, 1, cv2.LINE_AA)


def draw_quad(image, quad, color, label=None, thickness=2):
    points = np.round(quad).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(image, [points], True, color, thickness, cv2.LINE_AA)
    if label:
        corner = quad[np.argmin(quad[:, 0] + quad[:, 1])]
        put_label(image, label, (corner[0] + 4, corner[1] - 6), color=BLACK, background=color)


def draw_result(frame, result, label, method, index, total, fps):
    image = frame.copy()
    if result.quad is not None:
        color = YELLOW if result.mode == "pred" else GREEN
        draw_quad(image, result.quad, color, label)
    if result.mode == "klt":
        points = f"flow {result.n_flow}"
    elif result.mode == "yolo":
        points = f"conf {result.n_inliers / 100:.2f}"
    else:
        points = f"inliers {result.n_inliers}"
    speed = f"{fps:.0f} fps" if fps else "- fps"
    status = f"frame {index}/{total} | {method} | {result.mode} | {points} | {speed}"
    put_label(image, status, (8, image.shape[0] - 8), scale=0.45)
    return image
