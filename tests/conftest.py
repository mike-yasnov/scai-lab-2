import cv2
import numpy as np
import pytest

WIDTH, HEIGHT = 640, 360


def make_texture(seed=0):
    rng = np.random.default_rng(seed)
    image = cv2.GaussianBlur(rng.integers(0, 256, (HEIGHT, WIDTH, 3), dtype=np.uint8), (0, 0), 2)
    for _ in range(60):
        color = tuple(int(c) for c in rng.integers(0, 256, 3))
        x, y = int(rng.integers(0, WIDTH)), int(rng.integers(0, HEIGHT))
        if rng.random() < 0.5:
            cv2.rectangle(image, (x, y), (x + int(rng.integers(10, 80)), y + int(rng.integers(10, 80))), color, -1)
        else:
            cv2.circle(image, (x, y), int(rng.integers(5, 40)), color, -1)
    for i in range(6):
        cv2.putText(image, f"TEXT {i * 17}", (20 + 90 * i, 40 + 50 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return image


def homography(angle_deg=0.0, scale=1.0, tx=0.0, ty=0.0, px=0.0, py=0.0):
    center = np.array([[1, 0, -WIDTH / 2], [0, 1, -HEIGHT / 2], [0, 0, 1]], dtype=np.float64)
    a = np.radians(angle_deg)
    rotate = np.array([[scale * np.cos(a), -scale * np.sin(a), 0], [scale * np.sin(a), scale * np.cos(a), 0], [0, 0, 1]])
    perspective = np.array([[1, 0, 0], [0, 1, 0], [px, py, 1]], dtype=np.float64)
    back = np.array([[1, 0, WIDTH / 2 + tx], [0, 1, HEIGHT / 2 + ty], [0, 0, 1]], dtype=np.float64)
    h = back @ perspective @ rotate @ center
    return h / h[2, 2]


def warp(image, h):
    return cv2.warpPerspective(image, h, (WIDTH, HEIGHT), borderMode=cv2.BORDER_CONSTANT, borderValue=(40, 40, 40))


@pytest.fixture(scope="session")
def texture():
    return make_texture()
