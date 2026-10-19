import time
from pathlib import Path

import torch
from ultralytics import YOLOWorld

from tracker.geometry import box_iou

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "models" / "yolov8s-worldv2.pt"
TRACKER_CONFIG = Path(__file__).resolve().parent / "bytetrack_lowconf.yaml"
CONF = 0.01
REASSIGN_IOU = 0.3
MAX_MISSING = 15


def pick_device(name="auto"):
    if name != "auto":
        return name
    return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def load_model(prompt):
    WEIGHTS.parent.mkdir(exist_ok=True)
    model = YOLOWorld(str(WEIGHTS))
    model.set_classes([prompt])
    return model


class YoloWorldTracker:
    def __init__(self, prompt="poster", device="auto"):
        self.model = load_model(prompt)
        self.device = pick_device(device)
        self.target = None
        self.last_box = None
        self.switches = 0
        self.missing = 0

    def update(self, frame):
        start = time.perf_counter()
        result = self.model.track(frame, persist=True, tracker=str(TRACKER_CONFIG), conf=CONF, imgsz=640,
                                  device=self.device, verbose=False)[0]
        boxes = result.boxes
        candidates = []
        if boxes is not None and len(boxes) and boxes.id is not None:
            for box, track_id, conf in zip(boxes.xyxy.cpu().numpy(), boxes.id.cpu().numpy(), boxes.conf.cpu().numpy()):
                candidates.append((tuple(float(v) for v in box), int(track_id), float(conf)))
        chosen = next((c for c in candidates if c[1] == self.target), None)
        if chosen is None and candidates:
            near = []
            if self.last_box is not None:
                near = [c for c in candidates if box_iou(c[0], self.last_box) >= REASSIGN_IOU]
            if near:
                chosen = max(near, key=lambda c: box_iou(c[0], self.last_box))
            elif self.last_box is None or self.missing > MAX_MISSING:
                chosen = max(candidates, key=lambda c: c[2])
            if chosen is not None:
                if self.target is not None:
                    self.switches += 1
                self.target = chosen[1]
        if chosen is not None:
            self.last_box = chosen[0]
            self.missing = 0
        else:
            self.missing += 1
        ms = (time.perf_counter() - start) * 1000.0
        return (chosen[0] if chosen else None), (chosen[2] if chosen else 0.0), ms
