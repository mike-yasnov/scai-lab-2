import argparse
import collections
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

from tracker import DETECTORS, PlanarTracker, draw_result
from tracker.video import VideoSink, frame_count_hint, iter_frames, video_fps

WINDOW = "Keypoint tracker"


def parse_roi(text):
    if text is None or text == "select":
        return text
    parts = text.split(",")
    if len(parts) != 4 or not all(p.strip().isdigit() for p in parts):
        raise ValueError(f"ROI задаётся как x,y,w,h или select, получено {text!r}")
    x, y, w, h = (int(p) for p in parts)
    if w <= 0 or h <= 0:
        raise ValueError(f"У ROI ширина и высота должны быть положительными, получено {text!r}")
    return x, y, w, h


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Трекинг плоского объекта по ключевым точкам")
    parser.add_argument("video", help="видео, первый кадр которого — крупный план объекта")
    parser.add_argument("-o", "--output", help="куда сохранить видео с рамкой (.mp4 или .avi), по умолчанию output/<имя>_<детектор>.mp4")
    parser.add_argument("-d", "--detector", default="sift", choices=DETECTORS, help="детектор ключевых точек (по умолчанию sift)")
    parser.add_argument("--no-klt", action="store_true", help="не продлевать рамку оптическим потоком между детекциями")
    parser.add_argument("--max-coast", type=int, default=30, help="сколько кадров рисовать прогноз рамки после потери (по умолчанию 30, 0 — не рисовать)")
    parser.add_argument("--ratio", type=float, default=0.75, help="порог теста Лоу (по умолчанию 0.75)")
    parser.add_argument("--min-inliers", type=int, default=20, help="минимум инлаеров гомографии (по умолчанию 20)")
    parser.add_argument("--roi", help="область объекта на первом кадре: x,y,w,h или select")
    parser.add_argument("--label", default="object", help="подпись рамки (латиницей)")
    parser.add_argument("--max-size", type=int, default=960, help="уменьшать кадры, у которых длинная сторона больше этого значения (по умолчанию 960)")
    parser.add_argument("--csv", help="покадровый лог в CSV")
    parser.add_argument("--show", action="store_true", help="показывать кадры во время работы")
    parser.add_argument("--method", default="keypoints", choices=("keypoints", "yolo"),
                        help="keypoints — трекер по ключевым точкам, yolo — YOLO-World + ByteTrack (нужен requirements-neural.txt)")
    parser.add_argument("--prompt", default="poster", help="текстовый класс для YOLO-World (по умолчанию poster)")
    parser.add_argument("--device", default="auto", help="устройство для YOLO-World: mps, cuda, cpu или auto")
    return parser.parse_args(argv)


def check_args(args):
    if not Path(args.video).is_file():
        return f"файл {args.video} не найден"
    if not 0 < args.ratio < 1:
        return f"порог теста Лоу должен быть в интервале (0, 1), получено {args.ratio}"
    if args.min_inliers < 4:
        return f"для гомографии нужно не меньше 4 инлаеров, получено {args.min_inliers}"
    if args.max_coast < 0:
        return f"--max-coast не может быть отрицательным, получено {args.max_coast}"
    if args.max_size < 64:
        return f"--max-size слишком маленький: {args.max_size}"
    if args.output and Path(args.output).suffix.lower() not in (".mp4", ".avi"):
        return f"видео сохраняется в .mp4 или .avi, получено {args.output}"
    if args.method == "yolo" and (args.roi or args.no_klt or args.detector != "sift"):
        return "--roi, --no-klt и --detector относятся к трекеру по ключевым точкам, а не к --method yolo"
    if not args.label.isascii():
        return "подпись должна быть латиницей: шрифт OpenCV не рисует кириллицу"
    try:
        parse_roi(args.roi)
    except ValueError as exc:
        return str(exc)
    return None


def select_roi(frame):
    x, y, w, h = cv2.selectROI("Select object", frame, showCrosshair=False)
    cv2.destroyWindow("Select object")
    if w == 0 or h == 0:
        raise ValueError("область не выбрана")
    return int(x), int(y), int(w), int(h)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def result_row(index, result):
    row = {"frame": index, "mode": result.mode, "keypoints": result.n_keypoints, "matches": result.n_matches,
           "inliers": result.n_inliers, "flow_points": result.n_flow, "ms": round(result.ms, 3)}
    quad = result.quad if result.quad is not None else np.full((4, 2), np.nan)
    for i, (x, y) in enumerate(quad):
        row[f"x{i}"], row[f"y{i}"] = round(float(x), 2), round(float(y), 2)
    return row


class YoloAdapter:
    def __init__(self, prompt, device):
        from neural.yolo_world import YoloWorldTracker
        from tracker.planar import TrackResult
        self.tracker = YoloWorldTracker(prompt, device)
        self.result_type = TrackResult

    def init(self, frame, roi=None):
        return self.update(frame)

    def update(self, frame):
        box, conf, ms = self.tracker.update(frame)
        if box is None:
            return self.result_type(None, "lost", ms=ms)
        x0, y0, x1, y1 = box
        quad = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)
        return self.result_type(quad, "yolo", n_inliers=round(100 * conf), ms=ms)


def run(args):
    roi = parse_roi(args.roi)
    frames = iter_frames(args.video, args.max_size)
    first = next(frames, None)
    if first is None:
        raise ValueError(f"в видео {args.video} нет кадров")
    if roi == "select":
        roi = select_roi(first)
    if roi is not None:
        x, y, w, h = roi
        if x + w > first.shape[1] or y + h > first.shape[0]:
            raise ValueError(f"ROI {roi} выходит за кадр {first.shape[1]}x{first.shape[0]}")

    if args.method == "yolo":
        tracker = YoloAdapter(args.prompt, args.device)
        method = f"YOLO-World '{args.prompt}'"
        suffix = "yolo"
    else:
        tracker = PlanarTracker(args.detector, use_flow=not args.no_klt, max_coast=args.max_coast,
                                ratio=args.ratio, min_inliers=args.min_inliers)
        method = args.detector.upper() + ("" if args.no_klt else "+KLT")
        suffix = args.detector
    output = args.output or f"output/{Path(args.video).stem}_{suffix}.mp4"
    fps = video_fps(args.video)
    total = frame_count_hint(args.video)
    height, width = first.shape[:2]

    results = [tracker.init(first, roi)]
    rows = [result_row(0, results[0])]
    with VideoSink(output, fps, (width, height)) as sink:
        sink.write(draw_result(first, results[0], args.label, method, 0, total, 0))
        for index, frame in enumerate(frames, start=1):
            result = tracker.update(frame)
            results.append(result)
            rows.append(result_row(index, result))
            image = draw_result(frame, result, args.label, method, index, total, 1000.0 / max(result.ms, 1e-3))
            sink.write(image)
            if args.show:
                cv2.imshow(WINDOW, image)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord(" "):
                    cv2.waitKey(0)
    if args.show:
        cv2.destroyAllWindows()
    if args.csv:
        write_csv(args.csv, rows)

    modes = collections.Counter(r.mode for r in results)
    with_box = sum(1 for r in results if r.quad is not None)
    times = [r.ms for r in results[1:]] or [0.0]
    print(f"Кадров: {len(results)}, режимы: " + ", ".join(f"{m} {c}" for m, c in sorted(modes.items())))
    print(f"Рамка есть на {with_box} кадрах из {len(results)} ({100 * with_box / len(results):.1f} %), "
          f"трекер: {1000.0 / max(np.mean(times), 1e-3):.0f} кадров/с")
    print(f"Видео: {output}")


def main(argv=None):
    args = parse_args(argv)
    error = check_args(args)
    if error:
        sys.exit(f"Ошибка: {error}")
    try:
        run(args)
    except (OSError, ValueError, cv2.error) as exc:
        sys.exit(f"Ошибка: {exc}")


if __name__ == "__main__":
    main()
