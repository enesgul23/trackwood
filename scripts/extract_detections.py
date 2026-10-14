"""Stage 1: run a detector over every frame of the experiment videos and store the
raw detections (boxes, scores, classes) at a low confidence threshold.

    python scripts/extract_detections.py --detector rfdetr_base_legacy
    python scripts/extract_detections.py --detector yolo11n_legacy --videos 1040 1060

Output: {work_dir}/detections/{detector}/{video_id}.npz
"""
from __future__ import annotations

import argparse
import hashlib
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trackwood import config, detfile  # noqa: E402
from trackwood.detectors import build_detector  # noqa: E402
from trackwood.video import iter_frames, list_videos, probe  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def versions() -> dict:
    out = {"python": platform.python_version()}
    for mod in ("torch", "ultralytics", "rfdetr", "supervision", "cv2", "numpy"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            pass
    try:
        import torch
        out["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", required=True)
    ap.add_argument("--videos", nargs="*", help="video ids (default: all in videos_dir)")
    ap.add_argument("--max-frames", type=int, default=0, help="debug: stop after n frames")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--tag", default="", help="suffix of the output folder (e.g. _repeat)")
    args = ap.parse_args()

    paths = config.load_paths()
    spec = config.load_detectors(paths)[args.detector]
    videos = list_videos(paths["videos_dir"])
    if args.videos:
        videos = [v for v in videos if v.stem in set(args.videos)]
    out_dir = paths["work_dir"] / "detections" / f"{spec.name}{args.tag}"
    detector = build_detector(spec)
    weights_hash = sha256(spec.weights)
    env = versions()
    print(f"{spec.name}: {len(videos)} videos -> {out_dir}", flush=True)

    for vp in videos:
        out = out_dir / f"{vp.stem}.npz"
        if out.exists() and not args.overwrite and not args.max_frames:
            print(f"  skip {vp.stem} (exists)", flush=True)
            continue
        info = probe(vp)
        boxes, scores, classes, det_ms = [], [], [], []
        t0 = time.perf_counter()
        for i, frame in enumerate(iter_frames(vp)):
            if args.max_frames and i >= args.max_frames:
                break
            t = time.perf_counter()
            b, s, c = detector(frame)
            det_ms.append((time.perf_counter() - t) * 1000)
            boxes.append(b)
            scores.append(s)
            classes.append(c)
        wall = time.perf_counter() - t0
        det_ms = np.array(det_ms)
        meta = {
            "video": vp.name, "video_id": vp.stem, "width": info.width, "height": info.height,
            "fps": info.fps, "n_frames_header": info.n_frames, "n_frames": len(scores),
            "detector": spec.name, "kind": spec.kind, "weights": spec.weights.name,
            "weights_sha256": weights_hash, "min_conf": spec.min_conf, "imgsz": spec.imgsz,
            "yolo_nms_iou": spec.yolo_nms_iou if spec.kind == "yolo" else None,
            "resolution": spec.resolution if spec.kind == "rfdetr" else None,
            "max_det": spec.max_det, "wall_s": wall,
            "det_ms_mean": float(det_ms.mean()) if det_ms.size else None,
            "det_ms_median": float(np.median(det_ms)) if det_ms.size else None,
            "det_ms_p95": float(np.percentile(det_ms, 95)) if det_ms.size else None,
            "env": env,
        }
        if args.max_frames:
            out = out.with_name(f"{vp.stem}_debug{args.max_frames}.npz")
        detfile.save(out, boxes, scores, classes, meta)
        n_det = sum(len(s) for s in scores)
        print(f"  {vp.stem}: {len(scores)} frames, {n_det} detections, {wall:.0f} s "
              f"({len(scores) / wall:.1f} fps end-to-end, detector {meta['det_ms_median']:.1f} ms median)",
              flush=True)


if __name__ == "__main__":
    main()
