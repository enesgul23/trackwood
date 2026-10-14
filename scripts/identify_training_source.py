"""Find which experiment video the detector training images were taken from.

Dataset images (Roboflow export, 896 x 896 stretched) and video frames are reduced
to normalised 64 x 64 grey thumbnails; for a set of probe images the closest frame
of every candidate video is found. A source video shows near-zero distance with a
consistent frame mapping.

    python scripts/identify_training_source.py --videos-glob "*.MP4" --stride 1
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trackwood import config  # noqa: E402

SIZE = 64


def thumb(img: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    t = cv2.resize(g, (SIZE, SIZE), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    return (t - t.mean()) / (t.std() + 1e-6)


def scan_video(args) -> dict:
    path, probes, stride = args
    cap = cv2.VideoCapture(str(path))
    best_d = np.full(len(probes), np.inf)
    best_f = np.full(len(probes), -1)
    f = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if f % stride == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            d = np.abs(probes - thumb(frame)).mean(axis=1)
            better = d < best_d
            best_d[better] = d[better]
            best_f[better] = f
        f += 1
    cap.release()
    return {"video": str(path), "n_frames": f, "best_d": best_d.tolist(), "best_f": best_f.tolist()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-dir", type=Path)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--n-probes", type=int, default=12)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", type=Path, default=config.REPO / "results" / "training_source.json")
    args = ap.parse_args()

    paths = config.load_paths()
    coco = paths["coco_dir"]
    imgs = sorted(p for p in coco.glob("*/*.jpg"))
    idx = [int(re.search(r"frame_(\d+)", p.name).group(1)) for p in imgs]
    order = np.argsort(idx)
    sel = order[np.linspace(0, len(order) - 1, args.n_probes).astype(int)]
    probe_names = [imgs[i].name for i in sel]
    probe_idx = [idx[i] for i in sel]
    probes = np.stack([thumb(cv2.imread(str(imgs[i]))) for i in sel])

    vdir = args.videos_dir or paths["videos_dir"]
    pattern = "**/*" if args.recursive else "*"
    videos = sorted(p for p in vdir.glob(pattern) if p.suffix.lower() in {".mp4", ".avi", ".mov"})
    print(f"{len(videos)} videos, {len(probes)} probe images", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        res = list(ex.map(scan_video, [(v, probes, args.stride) for v in videos]))
    res.sort(key=lambda r: float(np.median(r["best_d"])))
    for r in res[:10]:
        print(f"{Path(r['video']).name:30s} median d = {np.median(r['best_d']):.3f}  "
              f"frames {r['best_f']}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"probe_images": probe_names, "probe_frame_index": probe_idx,
                   "stride": args.stride, "results": res}, f, indent=1)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
