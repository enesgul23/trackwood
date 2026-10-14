"""Collect stage-2 runs and compute counting accuracy, sensitivity, mass-balance
closure, trapping efficiency and the uncertainty budget.

    python scripts/summarize_counting.py --detectors yolo11n_legacy rfdetr_base_legacy

Writes small CSV/JSON tables to results/ (versioned with the code).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from trackwood import config, stats  # noqa: E402
from trackwood.pipeline import CountConfig  # noqa: E402
from trackwood.video import parse_experiment  # noqa: E402

TRAINING_VIDEO = "20040"  # source of all detector training images (results/training_source.json)
SUBSETS = {"all16": lambda v: True, "indep15": lambda v: v != TRAINING_VIDEO}
ESTIMATORS = {
    "L1_legacy": ("L1", "legacy"),
    "L1_fixed": ("L1", "fixed"),
    "L1_fixed_confirmed": ("L1", "fixed_confirmed"),
    "L1_dedup15": ("L1", "fixed_confirmed_dedup15"),
    "L1_dedup30": ("L1", "fixed_confirmed_dedup30"),
}
PRIMARY = "L1_fixed_confirmed"
from run_counting import load_grid  # noqa: E402

NOMINAL = load_grid("sensitivity")[0].__class__(**{**load_grid("nominal")[0].__dict__, "sweep_pct": ()})
NOMINAL_SWEEP = load_grid("nominal")[0]
TUNING_KEYS = {c.key() for c in load_grid("tuning")}
# Type B ranges for the uncertainty budget (one-at-a-time, around the nominal value)
TYPE_B_RANGES = {"conf": (0.2, 0.3), "nms_iou": (0.5, 0.7), "trk_iou": (0.2, 0.4), "max_age_s": (1.0, 4.0)}


def load_runs(work: Path, detectors: list[str]) -> list[dict]:
    rows = []
    folders = list(detectors) + [f"{d}_repeat" for d in detectors if (work / "counting" / f"{d}_repeat").exists()]
    for det in folders:
        for p in (work / "counting" / det).glob("*/*.json"):
            with open(p, encoding="utf-8") as f:
                rows.append(json.load(f))
    return rows


def flat(r: dict) -> dict:
    ex = parse_experiment(r["video_id"])
    c = r["config"]
    out = {"detector": r["detector"], "config_key": r["config_key"], "video": r["video_id"],
           "N": ex.n_released, "Q": ex.discharge_code,
           "conf": c["conf"], "nms_iou": c["nms_iou"], "trk_iou": c["trk_iou"], "min_hits": c["min_hits"],
           "max_age_s": c["max_age_s"], "n_detections": r["n_detections"], "n_tracks": r["n_tracks"],
           "n_confirmed": r["n_confirmed"]}
    for ln in ("L1", "L2"):
        for k, v in r["lines"][ln].items():
            if k != "x":
                out[f"{ln}_{k}"] = v
    for k, v in r["end_zone_counts"].items():
        out[f"end_{k}"] = v
    return out


def key_of(c: CountConfig) -> str:
    return c.key()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detectors", nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=config.REPO / "results")
    args = ap.parse_args()
    paths = config.load_paths()
    runs = load_runs(paths["work_dir"], args.detectors)
    args.out.mkdir(exist_ok=True)
    flat_rows = [flat(r) for r in runs]
    write_csv(args.out / "counting_runs.csv", sorted(flat_rows, key=lambda r: (r["detector"], r["config_key"], r["N"], r["Q"])))

    by = defaultdict(dict)  # (detector, config_key) -> video -> row
    cfg_of = {}
    for r, fr in zip(runs, flat_rows):
        by[(fr["detector"], fr["config_key"])][fr["video"]] = fr
        cfg_of[fr["config_key"]] = r["config"]

    # accuracy for every detector x config x estimator x subset
    acc_rows = []
    for (det, ck), vids in by.items():
        for sub, keep in SUBSETS.items():
            vv = sorted((v for v in vids if keep(v)), key=lambda v: (vids[v]["N"], vids[v]["Q"]))
            if len(vv) < 3:
                continue
            N = np.array([vids[v]["N"] for v in vv], float)
            for est, (ln, rule) in ESTIMATORS.items():
                col = f"{ln}_{rule}"
                if col not in vids[vv[0]]:
                    continue
                C = np.array([vids[v][col] for v in vv], float)
                row = {"detector": det, "config_key": ck, "subset": sub, "estimator": est,
                       **{k: cfg_of[ck][k] for k in ("conf", "nms_iou", "trk_iou", "min_hits", "max_age_s")}}
                row.update(stats.agreement(C, N))
                row.update(stats.bootstrap_ci(C, N))
                for kind, aff in (("ratio", False), ("affine", True)):
                    s, _ = stats.loocv_calibration(C, N, affine=aff)
                    row.update({f"loocv_{kind}_{k}": v for k, v in s.items()})
                acc_rows.append(row)
    write_csv(args.out / "counting_accuracy.csv", acc_rows)

    sens_key = key_of(NOMINAL)
    nominal_key = key_of(NOMINAL_SWEEP)
    summary = {"nominal_config": cfg_of.get(nominal_key), "nominal_key": nominal_key, "detectors": {}}
    sens_rows, closure_rows, trap_rows = [], [], []
    for det in args.detectors:
        nom = by.get((det, nominal_key))
        if not nom:
            continue
        dsum = {}
        # one-at-a-time sensitivity of the primary estimator (independent videos)
        for (d2, ck), vids in by.items():
            if d2 != det:
                continue
            c = cfg_of[ck]
            diffs = [p for p in ("conf", "nms_iou", "trk_iou", "min_hits", "max_age_s")
                     if c[p] != getattr(NOMINAL, p)]
            if len(diffs) > 1 or tuple(c["lines_pct"]) != NOMINAL.lines_pct or tuple(c["sweep_pct"]):
                continue
            param = diffs[0] if diffs else "nominal"
            vv = [v for v in vids if v != TRAINING_VIDEO]
            N = np.array([vids[v]["N"] for v in vv], float)
            C = np.array([vids[v][f"L1_fixed_confirmed"] for v in vv], float)
            a = stats.agreement(C, N)
            sens_rows.append({"detector": det, "param": param, "value": c[param] if diffs else None,
                              "eta_total": a["eta_total"], "mape_pct": a["mape_pct"], "mpe_pct": a["mpe_pct"],
                              "eta_cv_pct": a["eta_cv_pct"], "config_key": ck})
        # Type B components: half range of eta_total over the plausible range, rectangular
        type_b = {}
        for p, (lo, hi) in TYPE_B_RANGES.items():
            sel = [r for r in sens_rows if r["detector"] == det and r["param"] in (p, "nominal")]
            vals = [r["eta_total"] for r in sel
                    if r["param"] == "nominal" or (r["value"] is not None and lo <= r["value"] <= hi)]
            if len(vals) >= 2:
                eta_nom = next(r["eta_total"] for r in sel if r["param"] == "nominal")
                type_b[f"u_{p}"] = (max(vals) - min(vals)) / 2 / np.sqrt(3) / eta_nom
        vv = [v for v in nom if v != TRAINING_VIDEO]
        N = np.array([nom[v]["N"] for v in vv], float)
        eta = np.array([nom[v]["L1_fixed_confirmed"] for v in vv], float) / N
        # repeatability: second extraction of the same runs (GPU inference repeated)
        core_b = {}
        rep = by.get((f"{det}_repeat", nominal_key))
        if rep:
            diffs = [(rep[v]["L1_fixed_confirmed"] - nom[v]["L1_fixed_confirmed"]) / nom[v]["N"] for v in rep if v in nom]
            dsum["rerun_repeatability"] = {"videos": sorted(rep), "rel_diff_pct": [round(100 * d, 2) for d in diffs]}
            core_b["u_rerun"] = float(np.sqrt(np.mean(np.square(diffs))) / np.sqrt(2))
        dsum["uncertainty_budget_indep15"] = stats.uncertainty_budget(eta, core_b)
        dsum["processing_sensitivity_typeB_rel"] = {k: float(v) for k, v in type_b.items()}
        dsum["uncertainty_budget_with_processing_indep15"] = stats.uncertainty_budget(eta, {**core_b, **type_b})
        # nested leave-one-run-out tuning over the factorial grid (independent runs only)
        tuning = {ck: vids for (d2, ck), vids in by.items() if d2 == det and ck in TUNING_KEYS}
        if tuning:
            vv = sorted({v for vids in tuning.values() for v in vids if v != TRAINING_VIDEO},
                        key=lambda v: (parse_experiment(v).n_released, parse_experiment(v).discharge_code))
            tuning = {ck: vids for ck, vids in tuning.items() if all(v in vids for v in vv)}
            Nt = np.array([parse_experiment(v).n_released for v in vv], float)
            table = {ck: np.array([vids[v]["L1_fixed_confirmed"] for v in vv], float) for ck, vids in tuning.items()}
            dsum["nested_loocv_tuning_indep15"] = {"n_configs": len(table), **stats.nested_loocv_select(table, Nt)}
        # mass-balance closure and trapping efficiency at the nominal configuration
        for v, r in sorted(nom.items(), key=lambda kv: (kv[1]["N"], kv[1]["Q"])):
            l1, l2 = r["L1_fixed_confirmed"], r["L2_fixed_confirmed"]
            closure_rows.append({"detector": det, "video": v, "N": r["N"], "Q": r["Q"], "L1": l1, "L2": l2,
                                 "stored_by_counts": l1 - l2, "between_lines_at_end": r["end_between"],
                                 "upstream_at_end": r["end_upstream"],
                                 "closure_error": (l1 - l2) - r["end_between"]})
            trap_rows.append({"detector": det, "video": v, "N": r["N"], "Q": r["Q"], "L1": l1, "L2": l2,
                              "R": 1 - l2 / l1 if l1 else None, "R_vs_N": 1 - l2 / r["N"]})
        summary["detectors"][det] = dsum
    write_csv(args.out / "sensitivity_oat.csv", sens_rows)
    write_csv(args.out / "mass_balance_closure.csv", closure_rows)
    write_csv(args.out / "trapping_efficiency.csv", trap_rows)

    # paired method comparisons at the nominal configuration
    comps = {}
    dets = [d for d in args.detectors if (d, nominal_key) in by]
    for i, a in enumerate(dets):
        for b in dets[i + 1:]:
            for sub, keep in SUBSETS.items():
                vv = sorted(v for v in by[(a, nominal_key)] if keep(v) and v in by[(b, nominal_key)])
                N = np.array([by[(a, nominal_key)][v]["N"] for v in vv], float)
                Ca = np.array([by[(a, nominal_key)][v][f"L1_fixed_confirmed"] for v in vv], float)
                Cb = np.array([by[(b, nominal_key)][v][f"L1_fixed_confirmed"] for v in vv], float)
                comps[f"{a}__vs__{b}__{sub}"] = stats.paired_comparison(Ca, Cb, N)
    summary["paired_comparisons_L1_fixed_confirmed"] = comps

    # line-position study at the nominal configuration
    sweep_rows = []
    for det in dets:
        for p in sorted((paths["work_dir"] / "counting" / det / nominal_key).glob("*.json")):
            with open(p, encoding="utf-8") as f:
                r = json.load(f)
            for lid, d in r["lines"].items():
                pct = 20.0 if lid == "L1" else 80.0 if lid == "L2" else float(lid[1:])
                sweep_rows.append({"detector": det, "video": r["video_id"], "N": parse_experiment(r["video_id"]).n_released,
                                   "line_pct": pct, "count": d["fixed_confirmed"],
                                   "count_lr": d["fixed_confirmed_lr"], "count_rl": d["fixed_confirmed_rl"],
                                   "count_legacy": d["legacy"]})
    write_csv(args.out / "line_position_sweep.csv", sweep_rows)
    with open(args.out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print(f"{len(runs)} runs summarised -> {args.out}")


if __name__ == "__main__":
    main()
