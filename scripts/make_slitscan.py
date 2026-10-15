"""Slit-scan ("photo-finish") images of the counting lines for manual reference counts.

For every frame the pixel column at the counting line (channel rows only) is taken
and the columns are placed side by side, so the image shows the line over time:
x axis = frame, y axis = position across the channel. Every piece that crosses the
line leaves one streak, so pieces can be counted on a single image instead of by
watching the video. The image is independent of any detector or tracker.

At L1 the true count is N, which tests the manual slit-scan count itself; at L2 the
count gives the reference N_L2 for the outlet count and the trapping efficiency.

    python scripts/make_slitscan.py
Writes {work_dir}/slitscan/{video}_{L1|L2}.jpg and a CSV template.
"""
from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trackwood import config  # noqa: E402
from trackwood.video import list_videos, parse_experiment  # noqa: E402

Y0, Y1 = 0.24, 0.84          # channel rows (fraction of frame height), as in the count kit
LINES = {"L1": 0.20, "L2": 0.80}
HALF = 1                     # columns averaged on each side of the line (3-px slit)
TICK_EVERY_S = 1.0


def scan(args) -> str:
    path, out_dir = args
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 119.88
    cols = {k: [] for k in LINES}
    while True:
        ok, f = cap.read()
        if not ok:
            break
        H, W = f.shape[:2]
        y0, y1 = int(H * Y0), int(H * Y1)
        for k, p in LINES.items():
            x = int(W * p)
            cols[k].append(f[y0:y1, x - HALF:x + HALF + 1].mean(axis=1).astype(np.uint8))
    cap.release()
    for k, c in cols.items():
        img = np.stack(c, axis=1)  # rows x frames x 3
        # time ticks every second along the top and bottom edges, label every 5 s
        img = cv2.copyMakeBorder(img, 40, 40, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        n = img.shape[1]
        for i, t in enumerate(np.arange(0, n / fps, TICK_EVERY_S)):
            x = int(round(t * fps))
            cv2.line(img, (x, 0), (x, 18), (0, 0, 0), 2)
            cv2.line(img, (x, img.shape[0] - 18), (x, img.shape[0] - 1), (0, 0, 0), 2)
            if i % 5 == 0:
                cv2.putText(img, f"{t:.0f} s", (x + 4, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        cv2.imwrite(str(out_dir / f"{path.stem}_{k}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return path.stem


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    paths = config.load_paths()
    out = paths["work_dir"] / "slitscan"
    out.mkdir(parents=True, exist_ok=True)
    videos = list_videos(paths["videos_dir"])
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for v in ex.map(scan, [(p, out) for p in videos]):
            print(v, flush=True)
    with open(out / "slitscan_counts_template.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(["video", "N_released", "count_L1", "count_L2", "counter_initials", "notes"])
        for p in videos:
            w.writerow([p.stem, parse_experiment(p.stem).n_released, "", "", "", ""])
    print(f"slit-scans -> {out}")


if __name__ == "__main__":
    main()
