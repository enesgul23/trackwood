"""Full-resolution tiles of the active part of the slit-scan images, for counting
streaks by eye without double counting.

The active period is found on the slit-scan itself (columns with more bright pixels
than the quiet background), not from any detector. It is cut into tiles of TILE_W
frames; each tile shows CTX frames of the previous tile on its left, shaded grey.
Rule: count a streak in the tile where its left (earliest) end lies outside the grey
strip; a streak whose left end lies in the grey strip was counted in the previous tile.

    python scripts/make_slitscan_tiles.py --videos 5040 12040 --lines L1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trackwood import config  # noqa: E402

TILE_W, CTX, MARGIN = 240, 40, 90   # frames


def active_range(img: np.ndarray) -> tuple[int, int]:
    g = img.astype(np.float32)
    score = (g - np.median(g, axis=1, keepdims=True)).mean(axis=2)
    act = (score > 45).sum(axis=0).astype(float)
    act = np.convolve(act, np.ones(15) / 15, mode="same")
    base = np.median(act)
    mad = np.median(np.abs(act - base)) + 1e-6
    on = np.where(act > base + 8 * mad + 3)[0]
    if on.size == 0:
        return 0, img.shape[1]
    return max(0, int(on[0]) - MARGIN), min(img.shape[1], int(on[-1]) + MARGIN)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="*")
    ap.add_argument("--lines", nargs="*", default=["L1", "L2"])
    args = ap.parse_args()
    paths = config.load_paths()
    src = paths["work_dir"] / "slitscan"
    out = src / "tiles"
    out.mkdir(exist_ok=True)
    vids = args.videos or sorted({p.stem.split("_")[0] for p in src.glob("*_L1.jpg") if p.stem.split("_")[0].isdigit()},
                                 key=lambda v: (len(v), v))
    for vid in vids:
        for ln in args.lines:
            img = cv2.imread(str(src / f"{vid}_{ln}.jpg"))[40:-40]
            a, b = active_range(img)
            n = 0
            for s in range(a, b, TILE_W):
                lo = max(0, s - CTX)
                t = img[:, lo:min(b, s + TILE_W)].copy()
                ctx_w = s - lo
                if ctx_w > 0:
                    t[:, :ctx_w] = (0.45 * t[:, :ctx_w] + 0.55 * 160).astype(np.uint8)
                    cv2.line(t, (ctx_w, 0), (ctx_w, t.shape[0] - 1), (0, 0, 255), 1)
                head = np.full((36, t.shape[1], 3), 255, np.uint8)
                n += 1
                cv2.putText(head, f"{vid} {ln} tile {n}  frames {s}-{min(b, s + TILE_W)}", (4, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                cv2.imwrite(str(out / f"{vid}_{ln}_t{n:02d}.jpg"), np.vstack([head, t]), [cv2.IMWRITE_JPEG_QUALITY, 94])
            print(vid, ln, f"active frames {a}-{b}", n, "tiles", flush=True)


if __name__ == "__main__":
    main()
