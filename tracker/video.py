import shutil
import subprocess
from pathlib import Path

import cv2


def resize_to_fit(frame, max_size):
    height, width = frame.shape[:2]
    if not max_size or max(height, width) <= max_size:
        return frame
    scale = max_size / max(height, width)
    new_width = round(width * scale / 2) * 2
    new_height = round(height * scale / 2) * 2
    return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)


def open_video(path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise OSError(f"Не удалось открыть видео {path}")
    return capture


def video_fps(path):
    capture = open_video(path)
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    return fps if fps and fps > 0 else 30.0


def iter_frames(path, max_size=None):
    capture = open_video(path)
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                return
            yield resize_to_fit(frame, max_size)
    finally:
        capture.release()


def frame_count_hint(path):
    capture = open_video(path)
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return count if count > 0 else "?"


def count_frames(path):
    return sum(1 for _ in iter_frames(path))


class VideoSink:
    def __init__(self, path, fps, size):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        width, height = size
        self.process = None
        self.writer = None
        if shutil.which("ffmpeg") and self.path.suffix.lower() == ".mp4":
            command = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", f"{fps:.3f}", "-i", "-",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(self.path),
            ]
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE)
        else:
            fourcc = cv2.VideoWriter_fourcc(*("mp4v" if self.path.suffix.lower() == ".mp4" else "MJPG"))
            self.writer = cv2.VideoWriter(str(self.path), fourcc, fps, (width, height))

    def write(self, frame):
        if self.process is not None:
            self.process.stdin.write(frame.tobytes())
        else:
            self.writer.write(frame)

    def close(self):
        if self.process is not None:
            self.process.stdin.close()
            self.process.wait()
        elif self.writer is not None:
            self.writer.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
