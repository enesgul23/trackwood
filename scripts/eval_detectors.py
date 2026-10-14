"""Detector-level evaluation on the held-out test images.

For each detector and split: COCO AP (maxDets 1/10/500, the scenes hold up to 132
pieces), precision / recall / F1 at the operating point used for counting
(confidence 0.25, class-agnostic NMS IoU 0.6), the F1-confidence curve, the
precision-recall curve at IoU 0.5, and single-image latency on the GPU.

    python scripts/eval_detectors.py --detectors rfdetr_base_legacy yolo11n_legacy --splits random blocked
"""
from __future__ import annotations

import argparse
import contextlib
import io
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
from trackwood.pipeline import nms  # noqa: E402

OP_CONF, OP_NMS = 0.25, 0.6


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0]); iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2]); iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None] - inter + 1e-9)


def match(pred: np.ndarray, scores: np.ndarray, gt: np.ndarray, thr: float = 0.5) -> np.ndarray:
    """Greedy matching by score; returns a true-positive flag per prediction."""
    tp = np.zeros(len(pred), bool)
    if len(pred) == 0 or len(gt) == 0:
        return tp
    iou = box_iou(pred, gt)
    used = np.zeros(len(gt), bool)
    for i in np.argsort(-scores):
        j = np.argmax(np.where(used, -1, iou[i]))
        if iou[i, j] >= thr and not used[j]:
            used[j] = True
            tp[i] = True
    return tp


def evaluate(spec, coco_split_dir: Path, latency_runs: int = 50) -> dict:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    spec.min_conf = 0.001
    det = build_detector(spec)
    ann_file = coco_split_dir / "_annotations.coco.json"
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(ann_file))
    results, per_img = [], []
    for img_id, info in gt.imgs.items():
        img = cv2.imread(str(coco_split_dir / info["file_name"]))
        b, s, _ = det(img)
        g = np.array([[x, y, x + w, y + h] for x, y, w, h in
                      (a["bbox"] for a in gt.loadAnns(gt.getAnnIds(imgIds=img_id)))], float).reshape(-1, 4)
        per_img.append((b, s, g))
        for (x1, y1, x2, y2), sc in zip(b.tolist(), s.tolist()):
            results.append({"image_id": img_id, "category_id": 1, "bbox": [x1, y1, x2 - x1, y2 - y1], "score": sc})
    out = {}
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(results)
        ev = COCOeval(gt, dt, "bbox")
        ev.params.maxDets = [1, 10, 500]
        ev.evaluate(); ev.accumulate()

    def ap(iou=None, area="all", md=500):
        # pycocotools.summarize hard-codes maxDets = 100, so compute from the arrays
        p = ev.params
        a = p.areaRngLbl.index(area); m = p.maxDets.index(md)
        s = ev.eval["precision"][:, :, :, a, m]
        if iou is not None:
            s = s[np.isclose(p.iouThrs, iou)]
        s = s[s > -1]
        return float(s.mean()) if s.size else float("nan")

    def ar(area="all", md=500):
        p = ev.params
        r = ev.eval["recall"][:, :, p.areaRngLbl.index(area), p.maxDets.index(md)]
        r = r[r > -1]
        return float(r.mean()) if r.size else float("nan")

    out["coco"] = {"AP50_95": ap(), "AP50": ap(0.5), "AP75": ap(0.75), "AP_small": ap(area="small"),
                   "AP_medium": ap(area="medium"), "AP_large": ap(area="large"),
                   "AR1": ar(md=1), "AR10": ar(md=10), "AR500": ar(), "AR_small": ar("small"),
                   "AR_medium": ar("medium"), "AR_large": ar("large")}
    # precision-recall at IoU 0.5 (all areas, maxDets 500)
    prec = ev.eval["precision"][0, :, 0, 0, -1]
    out["pr_iou50"] = {"recall": np.linspace(0, 1, 101).tolist(), "precision": prec.tolist()}

    # operating point and F1-confidence curve, with and without class-agnostic NMS
    n_gt = sum(len(g) for _, _, g in per_img)
    for tag, use_nms in (("raw", False), ("nms0.6", True)):
        all_s, all_tp = [], []
        for b, s, g in per_img:
            if use_nms and len(b):
                k = nms(b, s, OP_NMS)
                b, s = b[k], s[k]
            all_s.append(s)
            all_tp.append(match(b, s, g))
        s = np.concatenate(all_s); tp = np.concatenate(all_tp)
        curve = []
        for t in np.round(np.arange(0.05, 0.951, 0.05), 2):
            keep = s >= t
            P = tp[keep].sum() / max(keep.sum(), 1); R = tp[keep].sum() / n_gt
            curve.append({"conf": float(t), "precision": float(P), "recall": float(R),
                          "f1": float(2 * P * R / (P + R)) if P + R else 0.0})
        op = next(c for c in curve if abs(c["conf"] - OP_CONF) < 1e-9)
        out[f"operating_point_{tag}"] = op
        out[f"f1_conf_curve_{tag}"] = curve
        out[f"best_f1_{tag}"] = max(curve, key=lambda c: c["f1"])
    out["n_images"] = len(per_img)
    out["n_gt_boxes"] = int(n_gt)

    # latency on a 4K-sized input (the detector adapter includes its resize)
    frame = cv2.resize(cv2.imread(str(coco_split_dir / gt.imgs[next(iter(gt.imgs))]["file_name"])), (3840, 2160))
    for _ in range(10):
        det(frame)
    import torch
    times = []
    for _ in range(latency_runs):
        torch.cuda.synchronize(); t = time.perf_counter()
        det(frame)
        torch.cuda.synchronize(); times.append((time.perf_counter() - t) * 1000)
    out["latency_ms_4k_frame"] = {"median": float(np.median(times)), "p95": float(np.percentile(times, 95))}
    try:
        m = det.model.model if hasattr(det.model, "model") else det.model
        out["n_parameters"] = int(sum(p.numel() for p in m.parameters()))
    except Exception:
        try:
            out["n_parameters"] = int(sum(p.numel() for p in det.model.model.model.parameters()))
        except Exception:
            out["n_parameters"] = None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detectors", nargs="+", required=True)
    ap.add_argument("--splits", nargs="+", default=["random", "blocked"])
    args = ap.parse_args()
    paths = config.load_paths()
    specs = config.load_detectors(paths)
    out_path = config.REPO / "results" / "detector_eval.json"
    allres = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    for name in args.detectors:
        for split in args.splits:
            d = paths["work_dir"] / "datasets" / f"coco_{split}" / "test"
            r = evaluate(specs[name], d)
            allres[f"{name}__{split}"] = r
            print(f"{name:22s} {split:8s} AP50:95 {r['coco']['AP50_95']:.3f} AP50 {r['coco']['AP50']:.3f} "
                  f"op(raw) P {r['operating_point_raw']['precision']:.3f} R {r['operating_point_raw']['recall']:.3f} "
                  f"op(nms) P {r['operating_point_nms0.6']['precision']:.3f} R {r['operating_point_nms0.6']['recall']:.3f} "
                  f"lat {r['latency_ms_4k_frame']['median']:.1f} ms", flush=True)
            out_path.write_text(json.dumps(allres, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
