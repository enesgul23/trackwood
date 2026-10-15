"""Candidate streaks on the slit-scan images, shown as numbered contact sheets for a
component-wise visual count.

Background = temporal median of each row; candidates = connected regions brighter
than the background. Clear noise (faint, flat blobs from bed texture and surface
glints) is dropped: a candidate is kept when the 90th percentile of its brightness
excess is >= 70, or >= 45 with a height of at least 12 rows. Kept candidates are
shown enlarged with everything outside the candidate faded, so a neighbour's stick
is not counted twice. Candidates that would have to be shrunk on a tile are written
as separate images at full resolution or larger. The counter writes how many sticks
each candidate holds (0 for noise, 2 or more for merged streaks). An overview image
with numbered boxes shows any stick left without a box.

    python scripts/make_slitscan_sheets.py --videos 1040 3040 --lines L1
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trackwood import config  # noqa: E402

THR = 15          # brightness above the row median (grey levels) for a candidate pixel
MIN_AREA = 40     # px
TILE = 200        # px, tile size on the sheet
COLS, ROWS = 5, 4
PAD = 14          # px around each component


def candidates(img: np.ndarray):
    g = img.astype(np.float32)
    score = (g - np.median(g, axis=1, keepdims=True)).mean(axis=2)
    m = (score > THR).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    comps = []
    for i in range(1, n):
        x, y, w, h, a = (int(v) for v in st[i, :5])
        if a < MIN_AREA:
            continue
        p90 = float(np.percentile(score[lab == i], 90))
        if p90 >= 70 or (p90 >= 45 and h >= 12):
            comps.append((x, y, w, h, a, i, p90))
    comps.sort(key=lambda c: (c[0], c[1]))  # by time, then position
    return comps, lab


def render(img, c, lab, scale):
    x, y, w, h, _, li, _ = c
    H, W = img.shape[:2]
    x0, y0, x1, y1 = max(0, x - PAD), max(0, y - PAD), min(W, x + w + PAD), min(H, y + h + PAD)
    crop = img[y0:y1, x0:x1].astype(np.float32)
    own = cv2.dilate((lab[y0:y1, x0:x1] == li).astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    crop[~own] = 0.35 * crop[~own] + 0.65 * 255  # fade everything outside this candidate
    crop = crop.astype(np.uint8)
    return cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)


def tile(img, c, k, lab):
    x, y, w, h = c[:4]
    avail = TILE - 24
    s = min(3.0, avail / (max(w, h) + 2 * PAD))
    crop = render(img, c, lab, s)
    t = np.full((TILE, TILE, 3), 255, np.uint8)
    oy, ox = 22 + (avail - crop.shape[0]) // 2, (TILE - crop.shape[1]) // 2
    t[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
    cv2.putText(t, str(k), (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)
    cv2.rectangle(t, (0, 0), (TILE - 1, TILE - 1), (180, 180, 180), 1)
    return t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="*")
    ap.add_argument("--lines", nargs="*", default=["L1", "L2"])
    args = ap.parse_args()
    paths = config.load_paths()
    src = paths["work_dir"] / "slitscan"
    out = src / "sheets"
    out.mkdir(exist_ok=True)
    for old in out.glob("*"):
        if args.videos is None or old.name.split("_")[0] in set(args.videos):
            if old.name.split("_")[1] in args.lines:
                old.unlink()
    vids = args.videos or sorted({p.stem.split("_")[0] for p in src.glob("*_L1.jpg") if p.stem.split("_")[0].isdigit()},
                                 key=lambda v: (len(v), v))
    for vid in vids:
        for ln in args.lines:
            img = cv2.imread(str(src / f"{vid}_{ln}.jpg"))[40:-40]
            comps, lab = candidates(img)
            small, big = [], []
            for k, c in enumerate(comps, 1):
                (big if (TILE - 24) / (max(c[2], c[3]) + 2 * PAD) < 0.95 else small).append((k, c))
            per = COLS * ROWS
            for s in range(0, len(small), per):
                tiles = [tile(img, c, k, lab) for k, c in small[s:s + per]]
                tiles += [np.full((TILE, TILE, 3), 255, np.uint8)] * (per - len(tiles))
                sheet = np.vstack([np.hstack(tiles[r * COLS:(r + 1) * COLS]) for r in range(ROWS)])
                cv2.imwrite(str(out / f"{vid}_{ln}_sheet{s // per + 1}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
            for k, c in big:
                scale = max(1.0, min(2.0, 700 / (max(c[2], c[3]) + 2 * PAD)))
                im = render(img, c, lab, scale)
                im = cv2.copyMakeBorder(im, 30, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
                cv2.putText(im, f"#{k}", (4, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)
                cv2.imwrite(str(out / f"{vid}_{ln}_big{k:03d}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
            ov = img.copy()
            for k, c in enumerate(comps, 1):
                x, y, w, h = c[:4]
                cv2.rectangle(ov, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(ov, str(k), (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imwrite(str(out / f"{vid}_{ln}_overview.jpg"), ov, [cv2.IMWRITE_JPEG_QUALITY, 90])
            with open(out / f"{vid}_{ln}_components.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["id", "frame_start", "y", "frames", "height", "area", "p90_excess", "shown_as"])
                for k, c in enumerate(comps, 1):
                    w.writerow([k, *c[:5], round(c[6], 1), "big" if any(k == b for b, _ in big) else "tile"])
            print(vid, ln, len(comps), "candidates,", len(big), "large", flush=True)


if __name__ == "__main__":
    main()
