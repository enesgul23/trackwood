"""Reference counts at the outlet line L2 from the slit-scan images, and validation of
the detector counts at L2 against them.

The number of pieces passing L2 is not known from the experiment (pieces trapped at
the spur were not counted), so a reference is counted visually on the slit-scan
images (scripts/make_slitscan.py): every piece that crosses the line leaves one
streak, and each counted streak is recorded with its (frame, row) position in
results/slitscan_points.csv (marked images: scripts/mark_slitscan_counts.py).
Unclear streaks were checked on the video frames and are flagged "uncertain".

The counting procedure is validated at L1, where the true count is N:
results/slitscan_counts.csv (point-marked runs) and results/slitscan_band_counts.csv
(earlier runs counted on band images without point marks).

Reference value and standard uncertainty per run:
    C_ref = n_sure + n_uncertain / 2
    u(C_ref) = sqrt( (n_uncertain / sqrt(12))^2 + (u_rel * C_ref)^2 )
where each uncertain streak is a rectangular 0/1 term and u_rel is the RMS relative
error of the same procedure at L1. Trapping efficiency R_ref = 1 - C_ref / N with
u(R_ref) = u(C_ref) / N (N is exact).

    python scripts/validate_L2.py
Detector counts are evaluated at the nominal configuration and with the water-surface
ROI (grid "nominal_roi"), at L1 against N and at L2 against the adjudicated and the
blind reference.

Writes results/l2_reference.csv and results/l2_validation.json.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trackwood import config, stats  # noqa: E402
from trackwood.video import parse_experiment  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_counting import GRIDS as GRID_FILE, load_grid  # noqa: E402

GRIDS = json.loads(GRID_FILE.read_text(encoding="utf-8"))

RES = config.REPO / "results"
TRAINING_VIDEO = "20040"
COUNTER = ("Claude (AI-assisted visual count); every mark checked on the marked images "
           "by one of the authors, no changes (2026-10-15)")


def read(name: str) -> list[dict]:
    with open(RES / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def order(v: str) -> tuple[int, int]:
    e = parse_experiment(v)
    return e.n_released, e.discharge_code


def main() -> None:
    counts = read("slitscan_counts.csv")
    band = read("slitscan_band_counts.csv") if (RES / "slitscan_band_counts.csv").exists() else []

    # 1. validation of the visual procedure at L1 (true count = N)
    val = []
    for r in [*(dict(r, method="point marks") for r in counts if r["line"] == "L1"), *(r for r in band if r["line"] == "L1")]:
        n, s, u = int(r["N_released"]), int(r["count_sure"]), int(r["count_uncertain"])
        c = s + u / 2
        val.append({"video": r["video"], "N": n, "count_sure": s, "count_uncertain": u, "count": c,
                    "error": c - n, "rel_error_pct": (c - n) / n * 100, "method": r["method"]})
    val.sort(key=lambda r: order(r["video"]))
    rel = np.array([r["rel_error_pct"] / 100 for r in val])
    u_rel = float(math.sqrt(np.mean(rel ** 2))) if rel.size else 0.0

    # 2. reference counts at L2 (adjudicated, and blind = first pass before the detector audit)
    nom_key = json.loads((RES / "summary.json").read_text(encoding="utf-8"))["nominal_key"]
    configs = {"nominal": nom_key}
    roi = [c.key() for c in load_grid("nominal_roi")] if "nominal_roi" in GRIDS else []
    if roi:
        configs["nominal_roi"] = roi[0]
    margins = {c.roi_margin_rel: c.key() for c in load_grid("roi_margin")} if "roi_margin" in GRIDS else {}
    det = {}
    for r in read("counting_runs.csv"):
        if not r["detector"].endswith("_repeat") and r["config_key"] in {*configs.values(), *margins.values()}:
            det[(r["config_key"], r["detector"], r["video"])] = r
    detectors = sorted({d for ck, d, _ in det if ck in configs.values()})
    rows = []
    for r in sorted((r for r in counts if r["line"] == "L2"), key=lambda r: order(r["video"])):
        v = r["video"]
        e = parse_experiment(v)
        s_, u = int(r["count_sure"]), int(r["count_uncertain"])
        c = s_ + u / 2
        cb = int(r.get("blind_sure", s_)) + int(r.get("blind_uncertain", u)) / 2
        uc = math.sqrt((u / math.sqrt(12)) ** 2 + (u_rel * c) ** 2)
        row = {"video": v, "N": e.n_released, "Q": e.discharge_code, "L2_sure": s_, "L2_uncertain": u,
               "L2_ref": c, "u_L2_ref": round(uc, 3), "L2_ref_blind": cb, "trapped_ref": e.n_released - c,
               "R_ref": round(1 - c / e.n_released, 4), "u_R_ref": round(uc / e.n_released, 4)}
        for cname, ck in configs.items():
            for d in detectors:
                if (ck, d, v) in det:
                    sfx = "" if cname == "nominal" else "_roi"
                    row[f"{d}__L2{sfx}"] = int(det[(ck, d, v)]["L2_fixed_confirmed"])
                    if cname == "nominal":
                        row[f"{d}__L2_legacy"] = int(det[(ck, d, v)]["L2_legacy"])
        rows.append(row)
    with open(RES / "l2_reference.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)

    # 3. detector counts at L2 against the reference, and at L1 against N, for each configuration
    out = {"counter": COUNTER, "configs": configs,
           "procedure_validation_L1": {"runs": val, "n_runs": len(val), "n_pieces": int(sum(r["N"] for r in val)),
                                        "u_rel_count_rms": u_rel,
                                        "max_abs_error_pieces": float(max(abs(r["error"]) for r in val))},
           "reference_L2": {"n_runs": len(rows), "n_pieces_counted": float(sum(r["L2_ref"] for r in rows)),
                            "n_pieces_blind": float(sum(r["L2_ref_blind"] for r in rows)),
                            "n_uncertain": int(sum(r["L2_uncertain"] for r in rows)),
                            "n_added_in_audit": int(sum(int(c.get("added_in_audit", 0)) for c in counts if c["line"] == "L2"))},
           "detectors": {}}
    for sub, keep in {"indep15": lambda v: v != TRAINING_VIDEO, "all16": lambda v: True}.items():
        sel = [r for r in rows if keep(r["video"])]
        ref = np.array([r["L2_ref"] for r in sel], float)
        N = np.array([r["N"] for r in sel], float)
        R = 1 - ref / N
        out["reference_L2"][sub] = {
            "R_ref_pooled": float(1 - ref.sum() / N.sum()),
            "R_ref_mean": float(R.mean()), "R_ref_sd": float(R.std(ddof=1)),
            "R_ref_by_Q": {str(q): {"mean": float(R[[r["Q"] == q for r in sel]].mean()),
                                    "sd": float(R[[r["Q"] == q for r in sel]].std(ddof=1)),
                                    "n": int(sum(r["Q"] == q for r in sel))} for q in (40, 60)},
            "R_ref_vs_N_spearman": {"rho": float(sps.spearmanr(N, R).statistic), "p": float(sps.spearmanr(N, R).pvalue)},
            "R_ref_Q40_vs_Q60_mannwhitney_p": float(sps.mannwhitneyu(
                R[[r["Q"] == 40 for r in sel]], R[[r["Q"] == 60 for r in sel]]).pvalue),
        }
        for d in detectors:
            for cname, ck in configs.items():
                vv = [r["video"] for r in sel]
                if not all((ck, d, v) in det for v in vv):
                    continue
                o = out["detectors"].setdefault(d, {}).setdefault(sub, {}).setdefault(cname, {})
                l1 = np.array([float(det[(ck, d, v)]["L1_fixed_confirmed"]) for v in vv])
                l2 = np.array([float(det[(ck, d, v)]["L2_fixed_confirmed"]) for v in vv])
                blind = np.array([r["L2_ref_blind"] for r in sel], float)
                o["L1_vs_N"] = stats.agreement(l1, N)
                o["L2_vs_ref"] = stats.agreement(l2, ref)
                o["L2_vs_ref"]["R_bias_pct_points"] = float(np.mean(ref / N - l2 / N) * 100)
                o["L2_vs_ref_blind"] = stats.agreement(l2, blind)
                if cname == "nominal":
                    leg = np.array([float(det[(ck, d, v)]["L2_legacy"]) for v in vv])
                    o["L2_legacy_rule_vs_ref"] = stats.agreement(leg, ref)
    # 4. sensitivity to the ROI margin and paired comparisons at L2 (independent runs)
    sel = [r for r in rows if r["video"] != TRAINING_VIDEO]
    vv = [r["video"] for r in sel]
    N = np.array([r["N"] for r in sel], float)
    ref = np.array([r["L2_ref"] for r in sel], float)
    sens = []
    for d in detectors:
        for m, ck in sorted(margins.items()):
            if all((ck, d, v) in det for v in vv):
                l1 = np.array([float(det[(ck, d, v)]["L1_fixed_confirmed"]) for v in vv])
                l2 = np.array([float(det[(ck, d, v)]["L2_fixed_confirmed"]) for v in vv])
                a1, a2 = stats.agreement(l1, N), stats.agreement(l2, ref)
                sens.append({"detector": d, "roi_margin_rel": m, "L1_mape_pct": a1["mape_pct"], "L1_bias": a1["bias"],
                             "L2_mape_pct": a2["mape_pct"], "L2_bias": a2["bias"]})
    out["roi_margin_sensitivity_indep15"] = sens
    comps = {}
    for cname, ck in configs.items():
        have = [d for d in detectors if all((ck, d, v) in det for v in vv)]
        for i, a in enumerate(have):
            for b in have[i + 1:]:
                Ca = np.array([float(det[(ck, a, v)]["L2_fixed_confirmed"]) for v in vv])
                Cb = np.array([float(det[(ck, b, v)]["L2_fixed_confirmed"]) for v in vv])
                comps[f"{cname}__{a}__vs__{b}"] = stats.paired_comparison(Ca, Cb, ref)
    out["paired_comparisons_L2_indep15"] = comps

    # 5. event-level agreement from scripts/match_L2_crossings.py (all 16 runs)
    if (RES / "l2_matching.csv").exists():
        ev = {}
        for r in read("l2_matching.csv"):
            t = ev.setdefault(f"{r['detector']}__{r['grid']}", {k: 0 for k in (
                "visual", "detector_crossings", "matched", "visual_only", "detector_only", "detector_only_on_wall_glass")})
            for k in t:
                t[k] += int(r[k])
        for t in ev.values():
            t["recall"] = t["matched"] / t["visual"]
            t["precision"] = t["matched"] / t["detector_crossings"]
            t["extra_not_on_glass"] = t["detector_only"] - t["detector_only_on_wall_glass"]
        out["event_matching_L2_all16"] = ev

    (RES / "l2_validation.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"L1 validation: {len(val)} runs, {sum(r['N'] for r in val)} pieces, u_rel = {u_rel * 100:.2f} %")
    for r in rows:
        print(f"{r['video']:>6} N={r['N']:>3} L2_ref={r['L2_ref']:>6.1f} ±{r['u_L2_ref']:.2f}  R_ref={r['R_ref']:.3f}")
    for d, v in out["detectors"].items():
        for cname, o in v["indep15"].items():
            print(f"{d:<20} {cname:<12} L1 MAPE {o['L1_vs_N']['mape_pct']:4.1f} % bias {o['L1_vs_N']['bias']:+5.2f} | "
                  f"L2 MAPE {o['L2_vs_ref']['mape_pct']:4.1f} % bias {o['L2_vs_ref']['bias']:+5.2f} "
                  f"(blind ref {o['L2_vs_ref_blind']['mape_pct']:4.1f} %)")


if __name__ == "__main__":
    main()
