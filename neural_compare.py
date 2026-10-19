import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

import plot_style
import matplotlib.pyplot as plt
from neural.yolo_world import CONF, YoloWorldTracker, load_model, pick_device
from tracker import PlanarTracker
from tracker.draw import GREEN, YELLOW, draw_quad, put_label
from tracker.geometry import box_iou, clip_box, quad_bbox
from tracker.video import iter_frames

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
VIDEOS = ["mona-lisa", "mona-lisa-blur", "mona-lisa-blur-extra-credit"]
PROMPTS = ["painting", "picture frame", "portrait painting", "poster", "Mona Lisa"]
SHOW = "mona-lisa-blur-extra-credit"
SHOW_FRAMES = [0, 20, 100, 250, 400, 520]
ORANGE = (52, 104, 235)
BLUR_DIFF = 5.0


def truth_boxes():
    frames = iter_frames(DATA / "mona-lisa.avi")
    tracker = PlanarTracker("sift", max_coast=0)
    quads = [tracker.init(next(frames)).quad] + [tracker.update(f).quad for f in frames]
    return quads, [clip_box(quad_bbox(q), (640, 360)) for q in quads]


def blur_frames(name):
    if name == "mona-lisa":
        return set()
    out = set()
    for i, (a, b) in enumerate(zip(iter_frames(DATA / "mona-lisa.avi"), iter_frames(DATA / f"{name}.avi"))):
        ga, gb = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if np.mean(np.abs(gb - ga)) - abs(gb.mean() - ga.mean()) > BLUR_DIFF:
            out.add(i)
    return out


def evaluate_prompts(device, boxes):
    frames = list(iter_frames(DATA / "mona-lisa.avi"))
    index = list(range(0, len(frames), 5))
    rows = []
    for prompt in PROMPTS:
        model = load_model(prompt)
        ious = []
        for i in index:
            result = model.predict(frames[i], conf=CONF, imgsz=640, device=device, verbose=False)[0]
            found = result.boxes.xyxy.cpu().numpy() if len(result.boxes) else []
            ious.append(max((box_iou(tuple(b), boxes[i]) for b in found), default=0.0))
        ious = np.array(ious)
        rows.append({"prompt": prompt, "recall50": 100 * float((ious >= 0.5).mean()),
                     "found": 100 * float((ious > 0).mean()), "mean_iou": float(ious.mean())})
        print(f"  {prompt:18} recall@0.5 {rows[-1]['recall50']:.1f} %")
    return rows


def run_yolo(name, prompt, device, keep=()):
    tracker = YoloWorldTracker(prompt, device)
    out, kept = [], {}
    for i, frame in enumerate(iter_frames(DATA / f"{name}.avi")):
        box, conf, ms = tracker.update(frame)
        out.append((box, conf, ms))
        if i in keep:
            kept[i] = frame
    return out, tracker.switches, kept


def run_keypoints(name):
    frames = iter_frames(DATA / f"{name}.avi")
    tracker = PlanarTracker("sift")
    return [tracker.init(next(frames))] + [tracker.update(f) for f in frames]


def summarize(name, yolo, switches, keypoints, boxes, blurred):
    n = len(yolo)
    ious = np.array([box_iou(b, boxes[i]) if b is not None else 0.0 for i, (b, _, _) in enumerate(yolo)])
    has = np.array([b is not None for b, _, _ in yolo])
    kp_ious = np.array([box_iou(clip_box(quad_bbox(r.quad), (640, 360)), boxes[i]) if r.quad is not None else 0.0
                        for i, r in enumerate(keypoints)])
    blur_idx = sorted(blurred)
    ms = np.array([m for _, _, m in yolo[1:]])
    return {
        "video": name,
        "yolo_box": 100 * float(has.mean()),
        "yolo_recall50": 100 * float((ious >= 0.5).mean()),
        "yolo_recall50_blur": 100 * float((ious[blur_idx] >= 0.5).mean()) if blur_idx else float("nan"),
        "yolo_mean_iou": float(ious[has].mean()) if has.any() else 0.0,
        "yolo_switches": switches,
        "yolo_fps": 1000.0 / float(np.median(ms)),
        "kp_recall50": 100 * float((kp_ious >= 0.5).mean()),
        "kp_recall50_blur": 100 * float((kp_ious[blur_idx] >= 0.5).mean()) if blur_idx else float("nan"),
        "kp_mean_iou": float(kp_ious.mean()),
        "frames": n,
    }


def plot_comparison(frames, yolo, keypoints, path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 5.9), layout="constrained")
    for ax, index in zip(axes.flat, SHOW_FRAMES):
        image = frames[index].copy()
        if keypoints[index].quad is not None:
            color = YELLOW if keypoints[index].mode == "pred" else GREEN
            draw_quad(image, keypoints[index].quad, color, "SIFT+KLT")
        box = yolo[index][0]
        if box is not None:
            x0, y0, x1, y1 = (int(round(v)) for v in box)
            cv2.rectangle(image, (x0, y0), (x1, y1), ORANGE, 2, cv2.LINE_AA)
            put_label(image, f"YOLO-World {yolo[index][1]:.2f}", (x0 + 4, y1 + 16), color=(0, 0, 0), background=ORANGE)
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.set_axis_off()
        ax.set_title(f"кадр {index}", fontsize=11)
    fig.suptitle("Зелёный: ключевые точки (жёлтый: прогноз), оранжевый: YOLO-World «poster» + ByteTrack",
                 fontsize=13, fontweight="bold")
    fig.savefig(path)
    plt.close(fig)


def fmt(x, digits=1):
    return "-" if isinstance(x, float) and np.isnan(x) else f"{x:.{digits}f}"



OWN_PROMPTS = ["red package", "package", "box", "poster", "packet"]


def own_video(path, roi, device):
    frames = list(iter_frames(path, 960))
    height, width = frames[0].shape[:2]
    tracker = PlanarTracker("sift")
    quads = [tracker.init(frames[0], roi).quad] + [tracker.update(f).quad for f in frames[1:]]
    reference = [clip_box(quad_bbox(q), (width, height)) for q in quads]
    rows = []
    for prompt in OWN_PROMPTS:
        yolo = YoloWorldTracker(prompt, device)
        ious, found, times = [], [], []
        for i, frame in enumerate(frames):
            box, _, ms = yolo.update(frame)
            found.append(box is not None)
            ious.append(box_iou(box, reference[i]) if box is not None else 0.0)
            times.append(ms)
        ious, found = np.array(ious), np.array(found)
        rows.append([prompt, fmt(100 * found.mean()), fmt(100 * (ious >= 0.5).mean()),
                     fmt(float(ious[found].mean()) if found.any() else 0.0, 2), yolo.switches,
                     fmt(1000.0 / float(np.median(times[1:])), 0)])
        print(f"  {prompt:12} IoU>=0.5 {rows[-1][2]} %")
    lines = ["## YOLO-World на своём видео", "",
             "Эталон: описанный прямоугольник рамки SIFT + KLT.", "",
             "| Подсказка | Рамка, % | IoU ≥ 0,5, % | Средний IoU | Смен трека | Кадров/с |", "|:---|---:|---:|---:|---:|---:|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    (RESULTS / "neural_own.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Сравнение YOLO-World + ByteTrack с трекером по ключевым точкам")
    parser.add_argument("--device", default="auto", help="mps, cuda, cpu или auto")
    parser.add_argument("--prompt", help="текстовый класс; по умолчанию выбирается лучший из списка")
    parser.add_argument("--own", help="своё видео: сравнить подсказки YOLO-World с трекером по точкам")
    parser.add_argument("--own-roi", help="область объекта на первом кадре своего видео: x,y,w,h")
    args = parser.parse_args(argv)
    if args.own:
        RESULTS.mkdir(exist_ok=True)
        roi = tuple(int(v) for v in args.own_roi.split(",")) if args.own_roi else None
        print("Своё видео")
        own_video(args.own, roi, pick_device(args.device))
        return
    if not (DATA / "mona-lisa.avi").exists():
        sys.exit("Ошибка: нет data/mona-lisa.avi, сначала запустите prepare_data.py")
    RESULTS.mkdir(exist_ok=True)
    plot_style.apply_style()
    device = pick_device(args.device)
    print(f"Устройство: {device}")
    quads, boxes = truth_boxes()

    print("Подбор текстового класса (каждый 5-й кадр mona-lisa.avi)")
    prompts = evaluate_prompts(device, boxes)
    prompt = args.prompt or max(prompts, key=lambda r: r["recall50"])["prompt"]
    print(f"Выбран класс «{prompt}»")

    rows = []
    for name in VIDEOS:
        keep = SHOW_FRAMES if name == SHOW else ()
        yolo, switches, kept = run_yolo(name, prompt, device, keep)
        keypoints = run_keypoints(name)
        rows.append(summarize(name, yolo, switches, keypoints, boxes, blur_frames(name)))
        r = rows[-1]
        print(f"  {name:28} YOLO: рамка {r['yolo_box']:.1f} %, IoU≥0.5 {r['yolo_recall50']:.1f} %, "
              f"смен ID {switches}, {r['yolo_fps']:.0f} кадров/с; ключевые точки: IoU≥0.5 {r['kp_recall50']:.1f} %")
        if name == SHOW:
            plot_comparison(kept, yolo, keypoints, RESULTS / "yolo_vs_keypoints.png")

    with open(RESULTS / "neural.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows([{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()} for r in rows])
    lines = ["## Подбор текстового класса YOLO-World", "",
             "| Класс | Рамка с IoU ≥ 0,5, % | Есть хоть одна рамка, % | Средний IoU |", "|:---|---:|---:|---:|"]
    lines += [f"| {r['prompt']} | {fmt(r['recall50'])} | {fmt(r['found'])} | {fmt(r['mean_iou'], 2)} |" for r in prompts]
    lines += ["", f"## YOLO-World «{prompt}» + ByteTrack против SIFT + KLT ({device})", "",
              "| Видео | YOLO: рамка, % | YOLO: IoU ≥ 0,5, % | YOLO на размытых, % | YOLO: средний IoU | Смен ID | YOLO, кадров/с | "
              "Точки: IoU ≥ 0,5, % | Точки на размытых, % |", "|:---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    lines += [f"| {r['video']} | {fmt(r['yolo_box'])} | {fmt(r['yolo_recall50'])} | {fmt(r['yolo_recall50_blur'])} | "
              f"{fmt(r['yolo_mean_iou'], 2)} | {r['yolo_switches']} | {fmt(r['yolo_fps'], 0)} | {fmt(r['kp_recall50'])} | "
              f"{fmt(r['kp_recall50_blur'])} |" for r in rows]
    (RESULTS / "neural.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Готово: {RESULTS / 'neural.md'}")


if __name__ == "__main__":
    main()
