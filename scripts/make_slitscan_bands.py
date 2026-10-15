"""Enlarged bands of a slit-scan window for counting streaks by eye.

The window [start, end) (frames) is split into time chunks of at most --chunk frames
and each chunk into --bands horizontal bands, enlarged --scale times. Frame numbers
(top axis) and channel rows (left axis) are drawn outside the image, so every counted
streak can be recorded by its (frame, row) position and checked later on the marked
slit-scan (scripts/mark_slitscan_counts.py).

Counting rules that keep each stick in exactly one image: a streak belongs to the
chunk holding its left (earliest) end and to the band holding its top end. With
--ctx, the end of the previous chunk is shown greyed left of a red line; streaks
whose left end lies in the grey strip were counted in the previous image. --rows
restricts the images to the channel rows that hold pieces.

    python scripts/make_slitscan_bands.py 1040 L2 1100 1300 --bands 2
    python scripts/make_slitscan_bands.py 8040 L2 1100 1700 --chunk 120 --bands 1 --rows 680 1230 --scale 2.5 --ctx 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trackwood import config  # noqa: E402

AX_TOP, AX_LEFT = 34, 46   # px of axis margin


def axes(band: np.ndarray, f0: int, r0: int, scale: float, title: str) -> np.ndarray:
    h, w = band.shape[:2]
    out = np.full((h + AX_TOP, w + AX_LEFT, 3), 255, np.uint8)
    out[AX_TOP:, AX_LEFT:] = band
    cv2.putText(out, title, (AX_LEFT + 4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    f1 = f0 + int(round(w / scale))
    for f in range((f0 + 9) // 10 * 10, f1, 10):
        x = AX_LEFT + int(round((f - f0) * scale))
        big = f % 50 == 0
        cv2.line(out, (x, AX_TOP - (10 if big else 4)), (x, AX_TOP - 1), (0, 0, 200), 1)
        if big:
            cv2.putText(out, str(f), (x - 14, AX_TOP - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 200), 1)
    r1 = r0 + int(round(h / scale))
    for r in range((r0 + 24) // 25 * 25, r1, 25):
        y = AX_TOP + int(round((r - r0) * scale))
        big = r % 50 == 0
        cv2.line(out, (AX_LEFT - (10 if big else 4), y), (AX_LEFT - 1, y), (0, 0, 200), 1)
        if big:
            cv2.putText(out, str(r), (2, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 200), 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("line", choices=["L1", "L2"])
    ap.add_argument("start", type=int)
    ap.add_argument("end", type=int)
    ap.add_argument("--bands", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=380, help="frames per image")
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--rows", type=int, nargs=2, help="only these channel rows (r0 r1)")
    ap.add_argument("--ctx", type=int, default=0,
                    help="frames of the previous chunk shown greyed on the left of each image")
    args = ap.parse_args()
    paths = config.load_paths()
    src = paths["work_dir"] / "slitscan"
    out = src / "bands"
    out.mkdir(exist_ok=True)
    for old in out.glob(f"{args.video}_{args.line}_*"):
        old.unlink()
    img = cv2.imread(str(src / f"{args.video}_{args.line}.jpg"))[40:-40]
    R0, R1 = args.rows if args.rows else (0, img.shape[0])
    H = R1 - R0
    for c, a in enumerate(range(args.start, args.end, args.chunk), 1):
        b = min(args.end, a + args.chunk)
        lo = max(0, a - args.ctx)
        for i in range(args.bands):
            y0, y1 = R0 + i * H // args.bands, R0 + (i + 1) * H // args.bands
            seg = img[y0:y1, lo:b].copy()
            if a > lo:  # context strip: counted in the previous image
                seg[:, :a - lo] = (0.45 * seg[:, :a - lo] + 0.55 * 160).astype(np.uint8)
            band = cv2.resize(seg, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_CUBIC)
            if a > lo:
                x = int(round((a - lo) * args.scale))
                cv2.line(band, (x, 0), (x, band.shape[0] - 1), (0, 0, 255), 1)
            title = f"{args.video} {args.line} frames {a}-{b} rows {y0}-{y1}"
            cv2.imwrite(str(out / f"{args.video}_{args.line}_c{c}_b{i + 1}.jpg"),
                        axes(band, lo, y0, args.scale, title), [cv2.IMWRITE_JPEG_QUALITY, 93])
    print(sorted(p.name for p in out.glob(f"{args.video}_{args.line}_*")))


if __name__ == "__main__":
    main()
