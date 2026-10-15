"""Pixel-to-metre calibration from the channel width (B = 1.2 m).

The bed-wall junctions are located on an empty-channel frame (largest intensity
step in a search window above and below the bed) every 5 % of the frame width; a
straight line W_bed(x) is fitted. The camera is not exactly centred, so the bed
width grows from about 1020 px (left) to 1070 px (right).

Floating wood lies at the water surface, which is closer to the camera than the
bed; its apparent width is W_s = k * W_bed. The extreme box edges of the detected
wood near the walls lie at most about 3 % outside the bed lines, so k = 1.015 with a
rectangular half-width of 0.015 (Type B, u(k)/k = 0.87 %). Scale at the surface:
s(x) = B / (k * W_bed(x)) in m/px.

    python scripts/calibrate.py --video 1040 --frame 100
Writes results/calibration.json.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trackwood import config  # noqa: E402

CHANNEL_WIDTH_M = 1.2
K_SURFACE, K_HALF = 1.015, 0.015
TOP_WIN, BOT_WIN = (650, 760), (1700, 1800)   # search windows (px) for the bed-wall junctions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="1040")
    ap.add_argument("--frame", type=int, default=100, help="a frame before any wood enters")
    args = ap.parse_args()
    paths = config.load_paths()
    cap = cv2.VideoCapture(str(paths["videos_dir"] / f"{args.video}.MP4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, f = cap.read()
    cap.release()
    assert ok
    g = cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(float), (0, 0), 3)
    H, W = g.shape
    xs, widths, tops, bots = [], [], [], []
    for xp in range(5, 100, 5):
        x = int(W * xp / 100)
        d = np.diff(g[:, x - 30:x + 30].mean(1))
        top = TOP_WIN[0] + int(np.argmax(np.abs(d[TOP_WIN[0]:TOP_WIN[1]])))
        bot = BOT_WIN[0] + int(np.argmax(np.abs(d[BOT_WIN[0]:BOT_WIN[1]])))
        xs.append(x); tops.append(top); bots.append(bot); widths.append(bot - top)
    xs, widths = np.array(xs, float), np.array(widths, float)
    a, b = np.polyfit(xs, widths, 1)
    resid = widths - (a * xs + b)
    u_line_px = float(resid.std(ddof=2))
    out = {
        "channel_width_m": CHANNEL_WIDTH_M,
        "frame": f"{args.video}#{args.frame}", "frame_size_px": [W, H],
        "bed_lines_px": [{"x": int(x), "top": int(t), "bottom": int(bo)} for x, t, bo in zip(xs, tops, bots)],
        "bed_width_fit": {"slope_px_per_px": float(a), "intercept_px": float(b), "resid_sd_px": u_line_px},
        "surface_factor_k": K_SURFACE, "surface_factor_halfwidth": K_HALF,
        "u_rel_k": K_HALF / math.sqrt(3) / K_SURFACE,
        "note": "s(x) = channel_width_m / (k * (slope * x + intercept)) [m/px] at the water surface",
    }
    for name, xp in (("L1", 0.20), ("L2", 0.80)):
        x = W * xp
        wb = a * x + b
        s = CHANNEL_WIDTH_M / (K_SURFACE * wb)
        u_rel = math.sqrt(out["u_rel_k"] ** 2 + (u_line_px / wb) ** 2)
        out[f"scale_at_{name}"] = {"x_px": x, "bed_width_px": wb, "m_per_px": s, "mm_per_px": s * 1000,
                                   "u_rel": u_rel}
        print(f"{name}: bed width {wb:.1f} px, scale {s * 1000:.4f} mm/px (u_rel {u_rel * 100:.2f} %)")
    (config.REPO / "results" / "calibration.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
