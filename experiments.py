import argparse
import csv
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

import plot_style
import matplotlib.pyplot as plt
from plot_style import DETECTOR_STYLE, INK_SECONDARY, MODE_STYLE, MUTED, SERIES
from tracker import DETECTORS, PlanarTracker, draw_result
from tracker.geometry import corner_error, signed_area
from tracker.video import VideoSink, frame_count_hint, iter_frames, video_fps

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
OUTPUT = ROOT / "output"

BASE = "mona-lisa"
VIDEOS = ["mona-lisa", "mona-lisa-blur", "mona-lisa-blur-extra-credit"]
SAMPLE = "simple_track"
SAMPLE_SOURCE = "mona-lisa-blur-extra-credit"
SAMPLE_OFFSET = 1
VARIANTS = {
    "det": {"use_flow": False, "max_coast": 0},
    "klt": {"use_flow": True, "max_coast": 0},
    "klt+pred": {"use_flow": True, "max_coast": 30},
}
VARIANT_LABEL = {"det": "только детекция", "klt": "детекция + KLT", "klt+pred": "детекция + KLT + прогноз"}
BLUR_DIFF = 5.0
LABEL = "Mona Lisa"
MONTAGE_FRAMES = [0, 20, 100, 250, 400, 520]


def gray(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def sharpness(frame):
    return float(cv2.Laplacian(gray(frame), cv2.CV_64F).var())


def track(name, detector, variant, keep=()):
    frames = iter_frames(DATA / f"{name}.avi")
    tracker = PlanarTracker(detector, **VARIANTS[variant])
    first = next(frames)
    results = [tracker.init(first)]
    kept = {0: first} if 0 in keep else {}
    for index, frame in enumerate(frames, start=1):
        results.append(tracker.update(frame))
        if index in keep:
            kept[index] = frame
    return results, kept


def describe_videos():
    info, blur = {}, {}
    for name in VIDEOS:
        diffs, shifts, sharp, size = [], [], [], None
        for base, frame in zip(iter_frames(DATA / f"{BASE}.avi"), iter_frames(DATA / f"{name}.avi")):
            a, b = gray(base).astype(np.float32), gray(frame).astype(np.float32)
            diffs.append(float(np.mean(np.abs(b - a))))
            shifts.append(float(b.mean() - a.mean()))
            sharp.append(sharpness(frame))
            size = f"{frame.shape[1]}x{frame.shape[0]}"
        shift = float(np.median(shifts))
        blur[name] = {i for i, d in enumerate(diffs) if d - abs(shift) > BLUR_DIFF}
        info[name] = {"frames": len(diffs), "fps": round(video_fps(DATA / f"{name}.avi"), 2), "size": size,
                      "brightness_shift": round(shift, 2), "blur_segments": segments(sorted(blur[name])),
                      "sharpness": sharp}
    return info, blur


def segments(indices):
    runs = []
    for i in indices:
        if runs and i == runs[-1][1] + 1:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    return [tuple(r) for r in runs]


def metrics(results, truth, blur_frames, segs):
    n = len(results)
    has = np.array([r.quad is not None for r in results])
    sharp_idx = [i for i in range(n) if i not in blur_frames]
    blur_idx = sorted(blur_frames)
    errors, relative, false_boxes = {}, {}, 0
    for i, r in enumerate(results):
        if r.quad is None or truth[i] is None:
            continue
        err = corner_error(r.quad, truth[i])
        scale = np.sqrt(abs(signed_area(truth[i])))
        errors[i], relative[i] = err, 100 * err / scale
        if r.mode != "pred" and err > max(20.0, 0.1 * scale):
            false_boxes += 1
    sharp_err = [errors[i] for i in sharp_idx if i in errors]
    blur_err = [errors[i] for i in blur_idx if i in errors]
    delays = []
    for _, end in segs:
        later = [j for j in range(end + 1, n) if results[j].mode in ("det", "klt")]
        delays.append(later[0] - end if later else None)
    ms = [r.ms for r in results[1:]]
    return {
        "box_all": 100 * has.mean(),
        "box_sharp": 100 * has[sharp_idx].mean() if sharp_idx else float("nan"),
        "box_blur": 100 * has[blur_idx].mean() if blur_idx else float("nan"),
        "det_share": 100 * np.mean([r.mode == "det" for r in results]),
        "err_median": float(np.median(sharp_err)) if sharp_err else float("nan"),
        "err_p95": float(np.percentile(sharp_err, 95)) if sharp_err else float("nan"),
        "rel_median": float(np.median([relative[i] for i in sharp_idx if i in relative])) if sharp_err else float("nan"),
        "blur_err_median": float(np.median(blur_err)) if blur_err else float("nan"),
        "blur_err_max": float(np.max(blur_err)) if blur_err else float("nan"),
        "false_boxes": false_boxes,
        "reacquire": "/".join("-" if d is None else str(d) for d in delays) if delays else "",
        "fps": 1000.0 / float(np.mean(ms)),
        "errors": errors,
    }


def jitter(results):
    quads = [r.quad for r in results]
    values = [np.mean(np.linalg.norm(quads[i + 1] - 2 * quads[i] + quads[i - 1], axis=1))
              for i in range(1, len(quads) - 1) if all(q is not None for q in quads[i - 1:i + 2])]
    return float(np.mean(values))


def green_mask(sample, source):
    hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)
    green = (np.abs(hsv[..., 0].astype(int) - 60) <= 12) & (hsv[..., 1] > 120) & (hsv[..., 2] > 100)
    changed = np.abs(sample.astype(int) - source.astype(int)).sum(axis=2) > 60
    return green & changed


def compare_with_sample(results):
    distances, both, sample_only, ours_only = [], 0, 0, 0
    source = iter_frames(DATA / f"{SAMPLE_SOURCE}.avi")
    for _ in range(SAMPLE_OFFSET):
        next(source)
    for i, (sample, frame) in enumerate(zip(iter_frames(DATA / f"{SAMPLE}.avi"), source)):
        j = i + SAMPLE_OFFSET
        mask = green_mask(sample, frame)
        sample_box = mask.sum() > 50
        ours = results[j].quad if results[j].mode in ("det", "klt") else None
        if sample_box and ours is not None:
            both += 1
            contour = np.full(mask.shape, 255, np.uint8)
            cv2.polylines(contour, [np.round(ours).astype(np.int32).reshape(-1, 1, 2)], True, 0, 1)
            distance = cv2.distanceTransform(contour, cv2.DIST_L2, 3)
            distances.append(float(np.median(distance[mask])))
        elif sample_box:
            sample_only += 1
        elif ours is not None:
            ours_only += 1
    return {"median_px": float(np.median(distances)), "p95_px": float(np.percentile(distances, 95)),
            "both": both, "sample_only": sample_only, "ours_only": ours_only}


def fmt(x, digits=1):
    if isinstance(x, str):
        return x
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{x:.{digits}f}"


def md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|:---|" + "---:|" * (len(header) - 1)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def plot_montage(name, path):
    results, frames = track(name, "sift", "klt+pred", keep=MONTAGE_FRAMES)
    fig, axes = plt.subplots(2, 3, figsize=(15, 5.9), layout="constrained")
    for ax, index in zip(axes.flat, MONTAGE_FRAMES):
        image = draw_result(frames[index], results[index], LABEL, "SIFT+KLT", index, len(results), 0)
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.set_axis_off()
        ax.set_title(f"кадр {index}: {MODE_STYLE[results[index].mode]['label']}", fontsize=11)
    fig.suptitle("SIFT + KLT + прогноз на mona-lisa-blur-extra-credit", fontsize=13, fontweight="bold")
    fig.savefig(path)
    plt.close(fig)


def plot_timeline(results, sharp, blur_segs, path):
    n = len(results)
    x = np.arange(n)
    fig, axes = plt.subplots(3, 1, figsize=(11, 7.2), sharex=True, layout="constrained",
                             gridspec_kw={"height_ratios": [3, 2, 0.5]})
    inliers = np.array([r.n_inliers if r.mode == "det" else np.nan for r in results], dtype=float)
    flow = np.array([r.n_flow if r.mode == "klt" else np.nan for r in results], dtype=float)
    axes[0].plot(x, inliers, ".", color=MODE_STYLE["det"]["color"], markersize=4, label="инлаеры детекции")
    axes[0].plot(x, flow, ".", color=MODE_STYLE["klt"]["color"], markersize=3, label="точки KLT")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("число точек")
    axes[0].legend(loc="lower center", markerscale=3)
    area = [100 * abs(signed_area(r.quad)) / (640 * 360) if r.quad is not None else np.nan for r in results]
    axes[1].plot(x, sharp, color=INK_SECONDARY, linewidth=1.2, label="резкость кадра")
    axes[1].plot(x, area, color=SERIES[6], linewidth=1.2, label="площадь рамки, % кадра")
    axes[1].set_yscale("log")
    axes[1].legend(loc="lower right")
    for i, r in enumerate(results):
        axes[2].axvspan(i - 0.5, i + 0.5, color=MODE_STYLE[r.mode]["color"], linewidth=0)
    axes[2].set_yticks([])
    axes[2].set_xlabel("кадр")
    handles = [plt.Rectangle((0, 0), 1, 1, color=MODE_STYLE[m]["color"]) for m in MODE_STYLE]
    axes[2].legend(handles, [MODE_STYLE[m]["label"] for m in MODE_STYLE], loc="upper center",
                   bbox_to_anchor=(0.5, -1.6), ncol=4)
    for ax in axes[:2]:
        for a, b in blur_segs:
            ax.axvspan(a - 0.5, b + 0.5, color=MUTED, alpha=0.15, linewidth=0)
    axes[0].set_title("Что происходит по кадрам (mona-lisa-blur-extra-credit, SIFT); серым — размытые участки",
                      loc="left")
    fig.savefig(path)
    plt.close(fig)


def plot_blur_error(errors_by_variant, blur_segs, path):
    windows = [(max(a - 8, 0), b + 12) for a, b in blur_segs]
    fig, axes = plt.subplots(1, len(windows), figsize=(5.5 * len(windows), 3.8), layout="constrained", sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (lo, hi), (a, b) in zip(axes, windows, blur_segs):
        ax.axvspan(a - 0.5, b + 0.5, color=MUTED, alpha=0.15, linewidth=0)
        for variant, color, marker in (("klt+pred", SERIES[3], "o"), ("klt", SERIES[0], "s")):
            errors = errors_by_variant[variant]
            frames = [i for i in range(lo, hi + 1) if i in errors]
            ax.plot(frames, [errors[i] for i in frames], marker=marker, linestyle="none", markersize=5,
                    color=color, label=VARIANT_LABEL[variant])
        ax.set_xlabel("кадр")
        ax.set_title(f"кадры {lo}–{hi}", loc="left")
    axes[0].set_ylabel("ошибка углов рамки, px")
    axes[-1].legend(loc="upper right")
    fig.suptitle("Ошибка рамки вокруг размытых участков (серым), SIFT", fontsize=12, fontweight="bold")
    fig.savefig(path)
    plt.close(fig)


def render_demo(name, gif_path):
    video = OUTPUT / f"{name}_sift_demo.mp4"
    frames = iter_frames(DATA / f"{name}.avi")
    tracker = PlanarTracker("sift")
    first = next(frames)
    with VideoSink(video, video_fps(DATA / f"{name}.avi"), (first.shape[1], first.shape[0])) as sink:
        result = tracker.init(first)
        sink.write(draw_result(first, result, LABEL, "SIFT+KLT", 0, 558, 0))
        for index, frame in enumerate(frames, start=1):
            result = tracker.update(frame)
            sink.write(draw_result(frame, result, LABEL, "SIFT+KLT", index, 558, 1000 / max(result.ms, 1e-3)))
    if shutil.which("ffmpeg"):
        filters = "fps=10,scale=320:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=64[p];[b][p]paletteuse=dither=bayer"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-t", "10", "-i", str(video), "-vf", filters, str(gif_path)],
                       check=True)



OWN_FRAMES = [0, 110, 150, 200, 240, 280, 330, 420]


def own_video(path, roi, label):
    frames = iter_frames(path, 960)
    tracker = PlanarTracker("sift")
    first = next(frames)
    results = [tracker.init(first, roi)]
    kept = {0: first}
    video = OUTPUT / "own_demo.mp4"
    total = frame_count_hint(path)
    with VideoSink(video, video_fps(path), (first.shape[1], first.shape[0])) as sink:
        sink.write(draw_result(first, results[0], label, "SIFT+KLT", 0, total, 0))
        for index, frame in enumerate(frames, start=1):
            results.append(tracker.update(frame))
            sink.write(draw_result(frame, results[-1], label, "SIFT+KLT", index, total, 1000 / max(results[-1].ms, 1e-3)))
            if index in OWN_FRAMES:
                kept[index] = frame
    if shutil.which("ffmpeg"):
        filters = "fps=6,scale=180:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=64[p];[b][p]paletteuse=dither=bayer"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-vf", filters,
                        str(RESULTS / "own_demo.gif")], check=True)
    shown = [i for i in OWN_FRAMES if i in kept]
    fig, axes = plt.subplots(1, len(shown), figsize=(2.2 * len(shown), 4.6), layout="constrained")
    for ax, index in zip(np.atleast_1d(axes), shown):
        image = draw_result(kept[index], results[index], label, "SIFT+KLT", index, len(results), 0)
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.set_axis_off()
        ax.set_title(f"кадр {index}: {MODE_STYLE[results[index].mode]['label']}", fontsize=9)
    fig.suptitle("Своё видео: SIFT + KLT", fontsize=13, fontweight="bold")
    fig.savefig(RESULTS / "own_montage.png")
    plt.close(fig)
    counts = {m: sum(r.mode == m for r in results) for m in MODE_STYLE}
    fps = 1000.0 / float(np.mean([r.ms for r in results[1:]]))
    height, width = first.shape[:2]
    lines = ["## Своё видео", "",
             f"{Path(path).name}: {len(results)} кадров, обработка в {width}x{height}, эталон ROI {roi}.", "",
             md_table(["Рамка, % кадров", "Детекция, %", "Оптический поток, %", "Прогноз, %", "Потерян, %", "Кадров/с"],
                      [[fmt(100 * (len(results) - counts["lost"]) / len(results))]
                       + [fmt(100 * counts[m] / len(results)) for m in ("det", "klt", "pred", "lost")] + [fmt(fps, 0)]]), ""]
    (RESULTS / "own.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Своё видео: " + ", ".join(f"{m} {c}" for m, c in counts.items()) + f", {fps:.0f} кадров/с")


def environment():
    try:
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    except OSError:
        cpu = platform.processor()
    return {"cpu": cpu or platform.processor(), "python": platform.python_version(), "opencv": cv2.__version__,
            "numpy": np.__version__, "os": platform.platform()}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Эксперименты с трекером по ключевым точкам")
    parser.add_argument("--no-gif", action="store_true", help="не рендерить демо-видео и GIF")
    parser.add_argument("--own", help="своё видео: построить монтаж и таблицу results/own.*")
    parser.add_argument("--own-roi", help="область объекта на первом кадре своего видео: x,y,w,h")
    parser.add_argument("--own-label", default="object", help="подпись рамки для своего видео")
    parser.add_argument("--own-only", action="store_true", help="обработать только своё видео")
    args = parser.parse_args(argv)
    if args.own:
        RESULTS.mkdir(exist_ok=True)
        plot_style.apply_style()
        roi = tuple(int(v) for v in args.own_roi.split(",")) if args.own_roi else None
        own_video(args.own, roi, args.own_label)
        if args.own_only:
            return
    if not (DATA / f"{BASE}.avi").exists():
        sys.exit("Ошибка: нет data/mona-lisa.avi, сначала запустите prepare_data.py")
    RESULTS.mkdir(exist_ok=True)
    plot_style.apply_style()
    env = environment()
    (RESULTS / "environment.json").write_text(json.dumps(env, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Видео и размытые участки")
    info, blur = describe_videos()
    print("Псевдо-разметка: SIFT + KLT на резком видео")
    truth_results, _ = track(BASE, "sift", "klt")
    truth = [r.quad for r in truth_results]
    print(f"  рамка на {sum(q is not None for q in truth)} кадрах из {len(truth)}, дрожание {jitter(truth_results):.2f} px")

    rows, by_key = [], {}
    for name in VIDEOS:
        segs = info[name]["blur_segments"]
        for detector in DETECTORS:
            for variant in VARIANTS:
                results, _ = track(name, detector, variant)
                m = metrics(results, truth, blur[name], segs)
                by_key[(name, detector, variant)] = (results, m)
                rows.append({"video": name, "detector": detector, "variant": variant,
                             **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items() if k != "errors"}})
                print(f"  {name:28} {detector:5} {variant:8} рамка {m['box_all']:5.1f} %  "
                      f"ошибка {fmt(m['err_median'], 2)} px  ложных {m['false_boxes']}  {m['fps']:.0f} fps")
    with open(RESULTS / "runs.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    sample = compare_with_sample(by_key[(SAMPLE_SOURCE, "sift", "klt")][0])
    print(f"Сравнение с образцом: медиана {sample['median_px']:.2f} px, p95 {sample['p95_px']:.2f} px")

    extra = SAMPLE_SOURCE
    plot_montage(extra, RESULTS / "montage.png")
    plot_timeline(by_key[(extra, "sift", "klt+pred")][0], info[extra]["sharpness"], info[extra]["blur_segments"],
                  RESULTS / "timeline.png")
    plot_blur_error({v: by_key[(extra, "sift", v)][1]["errors"] for v in ("klt", "klt+pred")},
                    info[extra]["blur_segments"], RESULTS / "blur_error.png")
    if not args.no_gif:
        render_demo(extra, RESULTS / "demo.gif")

    write_summary(env, info, by_key, sample, jitter(truth_results))
    print(f"Готово: {RESULTS}")


def write_summary(env, info, by_key, sample, base_jitter):
    out = ["## Стенд", "", f"{env['cpu']}, Python {env['python']}, OpenCV {env['opencv']}, NumPy {env['numpy']}", ""]
    out += ["## Видео", "", md_table(
        ["Видео", "Кадров", "Размер", "Сдвиг яркости", "Размытые кадры"],
        [[n, i["frames"], i["size"], fmt(i["brightness_shift"], 2),
          ", ".join(f"{a}–{b}" for a, b in i["blur_segments"]) or "нет"] for n, i in info.items()]), ""]
    out += ["## SIFT + KLT + прогноз", "", md_table(
        ["Видео", "Рамка, % кадров", "На размытых, %", "Детекция, % кадров", "Ошибка, медиана px", "p95 px",
         "Ошибка на размытых, медиана px", "Ложных рамок", "Повторный захват, кадров", "Кадров/с"],
        [[n, fmt(m["box_all"]), fmt(m["box_blur"]), fmt(m["det_share"]), fmt(m["err_median"], 2), fmt(m["err_p95"], 2),
          fmt(m["blur_err_median"]), m["false_boxes"], m["reacquire"] or "-", fmt(m["fps"], 0)]
         for n in VIDEOS for m in [by_key[(n, "sift", "klt+pred")][1]]]), ""]
    rows = []
    for detector in DETECTORS:
        for variant in VARIANTS:
            m = by_key[(SAMPLE_SOURCE, detector, variant)][1]
            rows.append([DETECTOR_STYLE[detector]["label"], VARIANT_LABEL[variant], fmt(m["box_sharp"]), fmt(m["box_blur"]),
                         fmt(m["err_median"], 2), fmt(m["err_p95"], 2), m["false_boxes"], fmt(m["fps"], 0)])
    out += ["## Детекторы и варианты (mona-lisa-blur-extra-credit)", "", md_table(
        ["Детектор", "Вариант", "Рамка на резких, %", "Рамка на размытых, %", "Ошибка, медиана px", "p95 px",
         "Ложных рамок", "Кадров/с"], rows), ""]
    out += ["## Сравнение с образцом simple_track.avi", "",
            f"Медианное расстояние от зелёной рамки образца до нашей (SIFT + KLT): {sample['median_px']:.2f} px, "
            f"95-й перцентиль {sample['p95_px']:.2f} px. Рамка есть у обоих на {sample['both']} кадрах, "
            f"только у образца на {sample['sample_only']}, только у нас на {sample['ours_only']}.", "",
            f"Дрожание псевдо-разметки (средняя вторая разность углов): {base_jitter:.2f} px.", ""]
    (RESULTS / "summary.md").write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    main()
