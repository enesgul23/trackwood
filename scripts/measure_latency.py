"""Detector latency on a real 4K frame (batch 1, FP32, adapter resize included).

Run on an otherwise idle machine:
    python scripts/measure_latency.py --detectors rfdetr_base_legacy yolo11m_random
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("YOLO_OFFLINE", "true")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trackwood import config  # noqa: E402
from trackwood.detectors import build_detector  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detectors", nargs="+", required=True)
    ap.add_argument("--video", default="8040", help="video providing the test frame")
    ap.add_argument("--frame", type=int, default=1200)
    ap.add_argument("--runs", type=int, default=200)
    args = ap.parse_args()

    import torch
    paths = config.load_paths()
    specs = config.load_detectors(paths)
    cap = cv2.VideoCapture(str(paths["videos_dir"] / f"{args.video}.MP4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    assert ok, "could not read the test frame"
    out = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "frame": f"{args.video}#{args.frame}",
           "frame_size": list(frame.shape[:2]), "runs": args.runs, "detectors": {}}
    for name in args.detectors:
        spec = specs[name]
        spec.min_conf = 0.25
        det = build_detector(spec)
        for _ in range(30):
            det(frame)
        t = []
        for _ in range(args.runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            det(frame)
            torch.cuda.synchronize()
            t.append((time.perf_counter() - t0) * 1000)
        t = np.array(t)
        out["detectors"][name] = {"median_ms": float(np.median(t)), "p95_ms": float(np.percentile(t, 95)),
                                  "fps_at_median": float(1000 / np.median(t))}
        print(f"{name:22s} median {np.median(t):6.1f} ms  p95 {np.percentile(t, 95):6.1f} ms", flush=True)
        del det
        torch.cuda.empty_cache()
    (config.REPO / "results" / "latency.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
