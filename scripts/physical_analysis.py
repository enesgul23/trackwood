"""Physical quantities at the inlet line and validation of the dead-zone model.

For every confirmed track that crosses L1 (corrected rule, nominal configuration):
  * step d: advance of the track centre per tracker update around the line (px);
  * surface velocity v (m/s): slope of x against time for the centre updates within
    200 px of the line, times the scale at L1 (results/calibration.json);
  * whether the original rule counted it, and whether any centre update fell inside
    the original dead zone |x - x_L| < 1 px (zone width w = 2 px).

Dead-zone model: with update positions of uniform phase relative to the line and a
constant step d, the probability that some update falls inside a zone of width w is
P = min(1, w / d). In physical units d = v / (f s), so P = min(1, w s f / v): the loss
grows with the frame rate f and the spatial resolution 1/s and falls with speed v.

Per run: median and inter-quartile range of v, passage duration T90 (5th-95th
percentile of the crossing times), mean wood flux 0.9 C / T90 and peak 1-s flux.

    python scripts/physical_analysis.py --detectors rfdetr_base_legacy yolo11m_random yolo11n_legacy
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from trackwood import config, detfile  # noqa: E402
from trackwood.pipeline import run  # noqa: E402
from trackwood.video import parse_experiment  # noqa: E402
from run_counting import load_grid  # noqa: E402

W_ZONE = 2.0          # px, width of the original dead zone (|x - L| < 1)
VEL_WINDOW = 200.0    # px around the line used for the velocity fit
TRAINING_VIDEO = "20040"


def analyse(args) -> list[dict]:
    det, vid, scale_l1 = args
    paths = config.load_paths()
    cfg = [c for c in load_grid("nominal")][0]
    cfg = cfg.__class__(**{**cfg.__dict__, "sweep_pct": ()})
    d = detfile.load(paths["work_dir"] / "detections" / det / f"{vid}.npz")
    r = run(d, cfg, keep_tracks=True)
    fps = r["fps"]
    L = r["lines"]["L1"]["x"]
    traj = r["_traj"]
    ev = r["_events"]["L1"]
    legacy_ids = {e[2] for e in ev if e[0] == "legacy"}
    fixed = {e[2]: e for e in ev if e[0] == "fixed"}
    rows = []
    for tid, e in fixed.items():
        pts = np.asarray(traj[tid], float)
        if len(pts) < cfg.min_hits:
            continue  # not a confirmed track
        f_, x_ = pts[:, 0], pts[:, 1]
        direction = e[4]
        xs = x_ * direction  # make the motion positive
        Ls = L * direction
        after = np.where(xs >= Ls + 1)[0]
        j = next((k for k in after if np.any(xs[:k] <= Ls - 1)), None)
        if j is None:
            continue
        i = int(np.where(xs[:j] <= Ls - 1)[0][-1])
        step = (xs[j] - xs[i]) / (j - i)
        in_zone = bool(np.any(np.abs(x_[i:j + 1] - L) < 1))
        near = np.abs(x_ - L) <= VEL_WINDOW
        v = np.nan
        if near.sum() >= 3:
            slope = np.polyfit(f_[near], x_[near], 1)[0]  # px per frame
            v = abs(slope) * fps * scale_l1
        ex = parse_experiment(vid)
        rows.append({"detector": det, "video": vid, "N": ex.n_released, "Q": ex.discharge_code,
                     "track_id": tid, "frame": int(e[1]), "t_s": round(e[1] / fps, 4),
                     "direction": int(direction), "step_px": round(float(step), 3),
                     "updates_between": int(j - i), "v_m_s": round(float(v), 4) if np.isfinite(v) else "",
                     "in_dead_zone": int(in_zone), "counted_by_original_rule": int(tid in legacy_ids),
                     "p_miss_model": round(min(1.0, W_ZONE / step), 4) if step > 0 else 1.0})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detectors", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    paths = config.load_paths()
    cal = json.loads((config.REPO / "results" / "calibration.json").read_text(encoding="utf-8"))
    s1 = cal["scale_at_L1"]["m_per_px"]
    tasks = []
    for det in args.detectors:
        for p in sorted((paths["work_dir"] / "detections" / det).glob("*.npz")):
            if "_debug" not in p.stem:
                tasks.append((det, p.stem, s1))
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = [r for part in ex.map(analyse, tasks) for r in part]
    res = config.REPO / "results"
    with open(res / "crossings_L1.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # per run summary
    summ = []
    for det in args.detectors:
        for vid in sorted({r["video"] for r in rows if r["detector"] == det}, key=lambda v: (len(v), v)):
            sel = [r for r in rows if r["detector"] == det and r["video"] == vid]
            t = np.sort([r["t_s"] for r in sel])
            v = np.array([r["v_m_s"] for r in sel if r["v_m_s"] != ""], float)
            t5, t95 = np.percentile(t, [5, 95])
            T90 = max(t95 - t5, 1e-9)
            peak = max(int(((t >= a) & (t < a + 1.0)).sum()) for a in t)
            p = np.array([r["p_miss_model"] for r in sel])
            summ.append({"detector": det, "video": vid, "N": sel[0]["N"], "Q": sel[0]["Q"], "C_L1": len(sel),
                         "v_median_m_s": round(float(np.median(v)), 4), "v_q25": round(float(np.percentile(v, 25)), 4),
                         "v_q75": round(float(np.percentile(v, 75)), 4), "T90_s": round(float(T90), 3),
                         "flux_mean_per_s": round(0.9 * len(sel) / T90, 3), "flux_peak_1s": peak,
                         "loss_observed": round(1 - np.mean([r["counted_by_original_rule"] for r in sel]), 4),
                         "in_zone_fraction": round(float(np.mean([r["in_dead_zone"] for r in sel])), 4),
                         "loss_predicted": round(float(p.mean()), 4)})
    with open(res / "physical_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0]))
        w.writeheader()
        w.writerows(summ)

    # dead-zone model validation (all detectors pooled and per detector)
    def validate(sel):
        d = np.array([r["step_px"] for r in sel])
        z = np.array([r["in_dead_zone"] for r in sel])
        m = np.array([1 - r["counted_by_original_rule"] for r in sel])
        p = np.array([r["p_miss_model"] for r in sel])
        edges = [0, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 8, 12, 1e9]
        bins = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            k = (d >= lo) & (d < hi)
            if k.sum() >= 5:
                n = int(k.sum())
                ph = z[k].mean()
                bins.append({"d_lo": lo, "d_hi": hi if hi < 1e9 else None, "n": n,
                             "d_median": float(np.median(d[k])),
                             "observed_in_zone": float(ph), "observed_missed": float(m[k].mean()),
                             "ci95_half": float(1.96 * np.sqrt(ph * (1 - ph) / n)),
                             "predicted": float(p[k].mean())})
        return {"n_crossings": int(len(sel)), "observed_loss": float(m.mean()), "predicted_loss": float(p.mean()),
                "in_zone_equals_missed": float(np.mean(z == m)),
                "brier_model": float(np.mean((p - z) ** 2)), "brier_constant": float(np.mean((z.mean() - z) ** 2)),
                "median_step_px": float(np.median(d)), "bins": bins}
    out = {"zone_width_px": W_ZONE, "scale_L1_m_per_px": s1,
           "zone_width_m": W_ZONE * s1, "frame_rate_hz": 119.88,
           "pooled": validate(rows),
           "by_detector": {det: validate([r for r in rows if r["detector"] == det]) for det in args.detectors}}
    pv = np.array([[r["loss_predicted"], r["loss_observed"]] for r in summ])
    out["per_run_predicted_vs_observed"] = {
        "n_runs": len(pv), "mean_abs_diff": float(np.abs(pv[:, 0] - pv[:, 1]).mean()),
        "pearson_r": float(np.corrcoef(pv[:, 0], pv[:, 1])[0, 1])}
    (res / "deadzone_model.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("zone_width_m", "per_run_predicted_vs_observed")}, indent=1))
    print(json.dumps({k: v for k, v in out["pooled"].items() if k != "bins"}, indent=1))
    for b in out["pooled"]["bins"]:
        print(b)


if __name__ == "__main__":
    main()
