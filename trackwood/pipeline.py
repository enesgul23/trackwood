"""Stage 2: confidence filtering, NMS, tracking and line counting for one video."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .detfile import Detections
from .tracking import LineCounter, Tracker


@dataclass(frozen=True)
class CountConfig:
    conf: float = 0.25            # operating confidence threshold
    nms_iou: float | None = None  # extra class-agnostic NMS; None = detector output as is
    trk_iou: float = 0.3          # minimum IoU for a detection-track match
    min_hits: int = 3             # updates before a track is confirmed
    max_age_s: float = 2.0        # seconds a track may coast without detections
    min_area: float = 0.0         # px^2
    lines_pct: tuple = (20.0, 80.0)
    sweep_pct: tuple = ()         # extra counting lines for the line-position study
    end_window_s: float = 0.5     # window at the video end for the storage estimate
    roi_margin_rel: float | None = None  # keep box centres on the water surface (see surface_band); None = off

    def key(self) -> str:
        d = asdict(self)
        if d["roi_margin_rel"] is None:  # keys of configurations without ROI are unchanged
            del d["roi_margin_rel"]
        txt = json.dumps(d, sort_keys=True)
        nms = "none" if self.nms_iou is None else f"{self.nms_iou:g}"
        short = (f"c{self.conf:g}_n{nms}_t{self.trk_iou:g}_h{self.min_hits}"
                 f"_a{self.max_age_s:g}_m{self.min_area:g}")
        if self.roi_margin_rel is not None:
            short += f"_r{self.roi_margin_rel:g}"
        return f"{short}_{hashlib.sha1(txt.encode()).hexdigest()[:6]}"


def surface_band(margin_rel: float) -> tuple[np.ndarray, np.ndarray]:
    """Linear far-wall (top) and near-wall (bottom) image rows of the water surface.

    The bed-wall junction lines of the calibration (results/calibration.json) are
    fitted as y = a + b x and widened on each side by margin_rel times the local
    bed width. Box centres outside the band lie on the glass side walls, where the
    detectors see reflections of pieces floating along the walls.
    """
    cal = json.loads((Path(__file__).resolve().parents[1] / "results" / "calibration.json").read_text(encoding="utf-8"))
    x = np.array([p["x"] for p in cal["bed_lines_px"]], float)
    top = np.polyfit(x, [p["top"] for p in cal["bed_lines_px"]], 1)
    bot = np.polyfit(x, [p["bottom"] for p in cal["bed_lines_px"]], 1)
    width = bot - top
    return top - margin_rel * width, bot + margin_rel * width


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """Greedy class-agnostic non-maximum suppression; returns kept indices."""
    if len(boxes) == 0:
        return np.zeros(0, dtype=int)
    order = np.argsort(-scores, kind="stable")
    x1, y1, x2, y2 = boxes.T
    area = (x2 - x1) * (y2 - y1)
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        w = np.maximum(0.0, np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]))
        h = np.maximum(0.0, np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]))
        inter = w * h
        iou = inter / (area[i] + area[rest] - inter + 1e-9)
        order = rest[iou <= iou_thr]
    return np.array(keep, dtype=int)


def postprocess(boxes: np.ndarray, scores: np.ndarray, cfg: CountConfig, band=None) -> list[tuple]:
    keep = scores >= cfg.conf
    if band is not None and len(boxes):
        cx, cy = (boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2
        keep &= (cy >= np.polyval(band[0], cx)) & (cy <= np.polyval(band[1], cx))
    boxes, scores = boxes[keep], scores[keep]
    if cfg.nms_iou is not None and len(boxes):
        k = nms(boxes, scores, cfg.nms_iou)
        boxes, scores = boxes[k], scores[k]
    out = []
    for x1, y1, x2, y2 in boxes.tolist():
        if x2 <= x1 or y2 <= y1:
            continue
        if cfg.min_area and (x2 - x1) * (y2 - y1) < cfg.min_area:
            continue
        out.append((int(x1), int(y1), int(x2), int(y2)))  # integer boxes, as in the original code
    return out


def dedup_count(ids: set, traj: dict, dmax: float, min_overlap: int = 10) -> int:
    """Number of distinct objects among the given tracks after merging tracks that
    move together (median centre distance <= dmax px over >= min_overlap shared frames)."""
    ids = sorted(ids)
    parent = {i: i for i in ids}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    arr = {i: np.asarray(traj[i]) for i in ids}
    for n, a in enumerate(ids):
        fa = arr[a]
        for b in ids[n + 1:]:
            fb = arr[b]
            if fa[-1, 0] < fb[0, 0] or fb[-1, 0] < fa[0, 0]:
                continue
            _, ia, ib = np.intersect1d(fa[:, 0], fb[:, 0], return_indices=True)
            if ia.size < min_overlap:
                continue
            if np.median(np.hypot(fa[ia, 1] - fb[ib, 1], fa[ia, 2] - fb[ib, 2])) <= dmax:
                parent[find(a)] = find(b)
    return len({find(i) for i in ids})


def run(dets: Detections, cfg: CountConfig, keep_tracks: bool = False) -> dict:
    t0 = time.perf_counter()
    meta = dets.meta
    width, fps = int(meta["width"]), float(meta["fps"])
    tracker = Tracker(cfg.trk_iou, int(fps * cfg.max_age_s), cfg.min_hits)
    main_lines = [LineCounter(f"L{i + 1}", int(width * p / 100.0)) for i, p in enumerate(cfg.lines_pct)]
    sweep_lines = [LineCounter(f"S{p:g}", int(width * p / 100.0)) for p in cfg.sweep_pct]
    lines = main_lines + sweep_lines
    x_l1, x_l2 = main_lines[0].x, main_lines[-1].x
    band = surface_band(cfg.roi_margin_rel) if cfg.roi_margin_rel is not None else None

    traj: dict[int, list] = {}      # track id -> [(frame, cx, cy)] for every update
    zone_counts = np.zeros((dets.n_frames, 3), dtype=np.int16)  # upstream, between, downstream
    n_det_total = 0
    for f in range(dets.n_frames):
        b, s, _ = dets.frame(f)
        boxes = postprocess(b, s, cfg, band)
        n_det_total += len(boxes)
        if boxes:
            cx = np.array([(x1 + x2) / 2 for x1, _, x2, _ in boxes])
            zone_counts[f] = ((cx < x_l1).sum(), ((cx >= x_l1) & (cx < x_l2)).sum(), (cx >= x_l2).sum())
        tracker.update(boxes)
        for t in tracker.updated + tracker.born:
            h = t.history[-1]
            traj.setdefault(t.id, []).append(h)
            for ln in lines:
                ln.observe(t)

    # tracks confirmed at any time; Tracker marks ever_confirmed on live tracks only,
    # so recompute from the trajectory length (hits = number of points)
    confirmed = {tid for tid, p in traj.items() if len(p) >= cfg.min_hits}
    w = max(1, int(round(cfg.end_window_s * fps)))
    end = np.median(zone_counts[-w:], axis=0)

    def line_summary(ln: LineCounter, with_dedup: bool) -> dict:
        fixed_conf = ln.fixed_ids & confirmed
        dirs = {e[2]: e[4] for e in ln.events if e[0] == "fixed"}
        out = {
            "x": ln.x,
            "legacy": len(ln.legacy_ids),
            "fixed": len(ln.fixed_ids),
            "fixed_confirmed": len(fixed_conf),
            "fixed_confirmed_lr": sum(1 for i in fixed_conf if dirs.get(i, 0) > 0),
            "fixed_confirmed_rl": sum(1 for i in fixed_conf if dirs.get(i, 0) < 0),
        }
        if with_dedup:
            for d in (15, 30):
                out[f"fixed_confirmed_dedup{d}"] = dedup_count(fixed_conf, traj, d)
        return out

    res = {
        "video_id": meta["video_id"], "detector": meta["detector"], "config": asdict(cfg),
        "config_key": cfg.key(), "n_frames": dets.n_frames, "fps": fps,
        "n_detections": int(n_det_total), "n_tracks": len(traj), "n_confirmed": len(confirmed),
        "track_len_median": float(np.median([len(p) for p in traj.values()])) if traj else 0.0,
        "lines": {ln.line_id: line_summary(ln, ln in main_lines) for ln in lines},
        "end_zone_counts": {"upstream": float(end[0]), "between": float(end[1]), "downstream": float(end[2])},
        "runtime_s": time.perf_counter() - t0,
    }
    if keep_tracks:
        res["_traj"] = traj
        res["_events"] = {ln.line_id: ln.events for ln in main_lines}
        res["_zone_counts"] = zone_counts
    return res
