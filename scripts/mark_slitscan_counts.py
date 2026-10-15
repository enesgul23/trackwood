"""Mark every counted piece on the slit-scan images and total the visual counts.

Reads results/slitscan_points.csv (one row per counted piece: video, line, piece,
frame, row, certainty, note), draws a numbered marker at each (frame, row) on the
slit-scan and writes {work_dir}/slitscan/marked/{video}_{line}.jpg, so a second
person can check the count on one image: every streak should carry exactly one
marker. Totals per video and line go to results/slitscan_counts.csv. Points whose note
starts with "audit:" were added after the comparison with the detector crossings
(scripts/match_L2_crossings.py) and confirmed on the video; the blind count (first
pass, before any comparison) is reported separately.

    python scripts/mark_slitscan_counts.py
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trackwood import config  # noqa: E402
from trackwood.video import parse_experiment  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MARGIN = 120  # frames shown around the marked pieces


def main() -> None:
    paths = config.load_paths()
    src = paths["work_dir"] / "slitscan"
    out = src / "marked"
    out.mkdir(exist_ok=True)
    pts = defaultdict(list)
    with open(ROOT / "results" / "slitscan_points.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            pts[(r["video"], r["line"])].append(r)
    rows = []
    for (vid, ln), ps in sorted(pts.items(), key=lambda k: (k[0][1], len(k[0][0]), k[0][0])):
        img = cv2.imread(str(src / f"{vid}_{ln}.jpg"))[40:-40]
        fr = [int(p["frame"]) for p in ps]
        a, b = max(0, min(fr) - MARGIN), min(img.shape[1], max(fr) + MARGIN)
        im = cv2.resize(img[:, a:b], None, fx=2, fy=1, interpolation=cv2.INTER_CUBIC)
        for p in ps:
            x, y = (int(p["frame"]) - a) * 2, int(p["row"])
            col = (0, 0, 255) if p["certainty"] == "sure" else (255, 0, 255)
            cv2.circle(im, (x, y), 9, col, 2)
            cv2.putText(im, p["piece"], (x + 10, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
        head = np.full((30, im.shape[1], 3), 255, np.uint8)
        sure = sum(p["certainty"] == "sure" for p in ps)
        cv2.putText(head, f"{vid} {ln} frames {a}-{b} (x2 in time): {sure} sure + {len(ps) - sure} uncertain",
                    (4, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.imwrite(str(out / f"{vid}_{ln}.jpg"), np.vstack([head, im]), [cv2.IMWRITE_JPEG_QUALITY, 92])
        audit = [p for p in ps if p["note"].startswith("audit:")]
        rows.append({"video": vid, "line": ln, "N_released": parse_experiment(vid).n_released,
                     "count_sure": sure, "count_uncertain": len(ps) - sure, "count": len(ps),
                     "added_in_audit": len(audit),
                     "blind_sure": sure - sum(p["certainty"] == "sure" for p in audit),
                     "blind_uncertain": len(ps) - sure - sum(p["certainty"] != "sure" for p in audit)})
    with open(ROOT / "results" / "slitscan_counts.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
