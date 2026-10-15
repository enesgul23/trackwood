"""Match the detector crossings of L2 to the visually counted pieces on the slit-scan.

For every run, the confirmed-track crossings of L2 (corrected rule, nominal
configuration) are taken with their frame and the row of the box centre, and are
assigned one-to-one to the visual points in results/slitscan_points.csv (Hungarian
assignment, cost = sqrt((df / 15)^2 + (drow / 40)^2), pairs allowed up to MAX_COST and
gated before the assignment; a visual point marks some position on the streak, so the
tolerance is a streak size).
Unassigned detector crossings are extra counts (double counts or pieces missed by
the visual count); unassigned visual points are pieces the detector did not count.
Unassigned detector crossings whose box centre lies on the glass side walls (outside
the water-surface band of the ROI) are flagged: they are reflections of pieces that
float along the walls. Both kinds are drawn on the slit-scan for inspection:
{work_dir}/slitscan/matched/{detector}_{grid}_{video}.jpg.

    python scripts/match_L2_crossings.py --detectors rfdetr_base_legacy yolo11m_random --grids nominal nominal_roi
Writes results/l2_matching.csv (per run) and results/l2_unmatched.csv (per event).
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from trackwood import config, detfile  # noqa: E402
from trackwood.pipeline import run, surface_band  # noqa: E402
from run_counting import load_grid  # noqa: E402

Y0 = 0.24          # first channel row of the slit-scan (fraction of frame height), as in make_slitscan.py
SF, SR, MAX_COST = 15.0, 40.0, 2.5
GATE_PENALTY = 1e6  # cost of a forbidden pair; larger than any sum of allowed costs
MARGIN = 60


def crossings(det: str, vid: str, grid: str, line: str = "L2") -> tuple[list[tuple[int, float, int, int]], dict]:
    paths = config.load_paths()
    cfg = load_grid(grid)[0]
    cfg = cfg.__class__(**{**cfg.__dict__, "sweep_pct": ()})
    d = detfile.load(paths["work_dir"] / "detections" / det / f"{vid}.npz")
    r = run(d, cfg, keep_tracks=True)
    h = int(d.meta["height"])
    confirmed = {tid for tid, p in r["_traj"].items() if len(p) >= cfg.min_hits}
    ev = [(int(e[1]), float(e[3]) - int(h * Y0), int(e[2]), int(e[4]))
          for e in r["_events"][line] if e[0] == "fixed" and e[2] in confirmed]
    return sorted(ev), r


def match(task) -> tuple[dict, list[dict]]:
    det, vid, pts, grid = task
    ev, _ = crossings(det, vid, grid)
    band = surface_band(0.015)
    xl2 = 0.8 * 3840
    P = np.array([(p[0], p[1]) for p in pts], float).reshape(-1, 2)
    E = np.array([(e[0], e[1]) for e in ev], float).reshape(-1, 2)
    cost = np.hypot((P[:, None, 0] - E[None, :, 0]) / SF, (P[:, None, 1] - E[None, :, 1]) / SR)
    # Gate before assigning: pairs beyond MAX_COST are forbidden, so the assignment first
    # maximises the number of matches within the tolerance and only then minimises their
    # cost (assigning on the raw cost and gating afterwards lets one far pair displace
    # several near ones).
    gated = np.where(cost <= MAX_COST, cost, GATE_PENALTY)
    ri, ci = linear_sum_assignment(gated) if cost.size else (np.array([], int), np.array([], int))
    ok = cost[ri, ci] <= MAX_COST
    mp, me = set(ri[ok]), set(ci[ok])
    events = []
    for i in range(len(pts)):
        if i not in mp:
            events.append({"detector": det, "grid": grid, "video": vid, "kind": "visual_only", "frame": int(P[i, 0]),
                           "row": int(P[i, 1]), "track_id": "", "direction": "", "piece": pts[i][2], "on_wall_glass": ""})
    for j in range(len(ev)):
        if j not in me:
            y = ev[j][1] + int(2160 * Y0)
            glass = not (np.polyval(band[0], xl2) <= y <= np.polyval(band[1], xl2))
            events.append({"detector": det, "grid": grid, "video": vid, "kind": "detector_only", "frame": ev[j][0],
                           "row": int(round(ev[j][1])), "track_id": ev[j][2], "direction": ev[j][3], "piece": "",
                           "on_wall_glass": int(glass)})
    draw(det, vid, P, E, ri[ok], ci[ok], mp, me, pts, grid)
    summ = {"detector": det, "grid": grid, "video": vid, "visual": len(pts), "detector_crossings": len(ev),
            "matched": int(ok.sum()), "visual_only": len(pts) - len(mp), "detector_only": len(ev) - len(me),
            "detector_only_on_wall_glass": sum(1 for e in events if e["kind"] == "detector_only" and e["on_wall_glass"])}
    return summ, events


def draw(det, vid, P, E, ri, ci, mp, me, pts, grid) -> None:
    paths = config.load_paths()
    src = paths["work_dir"] / "slitscan"
    img = cv2.imread(str(src / f"{vid}_L2.jpg"))[40:-40]
    allf = np.concatenate([P[:, 0], E[:, 0]]) if E.size else P[:, 0]
    a, b = int(max(0, allf.min() - MARGIN)), int(min(img.shape[1], allf.max() + MARGIN))
    im = cv2.resize(img[:, a:b], None, fx=2, fy=1, interpolation=cv2.INTER_CUBIC)
    X = lambda f: int(round((f - a) * 2))  # noqa: E731
    for i, j in zip(ri, ci):
        cv2.line(im, (X(P[i, 0]), int(P[i, 1])), (X(E[j, 0]), int(E[j, 1])), (0, 200, 0), 1)
    for i, (f, r) in enumerate(P):
        c = (0, 0, 255) if i in mp else (255, 0, 255)
        cv2.circle(im, (X(f), int(r)), 8 if i in mp else 12, c, 2)
        cv2.putText(im, str(pts[i][2]), (X(f) + 9, int(r) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)
    for j, (f, r) in enumerate(E):
        c = (255, 120, 0) if j in me else (0, 255, 255)
        s = 6 if j in me else 10
        cv2.drawMarker(im, (X(f), int(r)), c, cv2.MARKER_TILTED_CROSS if j in me else cv2.MARKER_SQUARE, 2 * s, 2)
    head = np.full((30, im.shape[1], 3), 255, np.uint8)
    cv2.putText(head, f"{det} {vid} L2: red = visual (magenta unmatched), blue x = detector crossing, "
                      f"yellow square = unmatched detector crossing", (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    out = src / "matched"
    out.mkdir(exist_ok=True)
    cv2.imwrite(str(out / f"{det}_{grid}_{vid}.jpg"), np.vstack([head, im]), [cv2.IMWRITE_JPEG_QUALITY, 92])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detectors", nargs="+", required=True)
    ap.add_argument("--grids", nargs="+", default=["nominal", "nominal_roi"])
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    pts = defaultdict(list)
    with open(config.REPO / "results" / "slitscan_points.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["line"] == "L2" and r["certainty"] != "rejected":
                pts[r["video"]].append((int(r["frame"]), int(r["row"]), int(r["piece"])))
    tasks = [(d, v, p, g) for g in args.grids for d in args.detectors
             for v, p in sorted(pts.items(), key=lambda kv: (len(kv[0]), kv[0]))]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        res = list(ex.map(match, tasks))
    summ = [s for s, _ in res]
    events = [e for _, ev in res for e in ev]
    for name, rows in (("l2_matching.csv", summ), ("l2_unmatched.csv", events)):
        with open(config.REPO / "results" / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    tot = defaultdict(lambda: defaultdict(int))
    for s in summ:
        for k in ("visual", "detector_crossings", "matched", "visual_only", "detector_only", "detector_only_on_wall_glass"):
            tot[(s["detector"], s["grid"])][k] += s[k]
    for k, v in tot.items():
        print(k, dict(v))


if __name__ == "__main__":
    main()
