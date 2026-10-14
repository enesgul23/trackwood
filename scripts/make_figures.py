"""Manuscript figures from the tables in results/.

    python scripts/make_figures.py --detectors rfdetr_base_legacy yolo11n_legacy

Figures are written to figures/ as vector PDF and 600 dpi PNG, sized for Elsevier
single (90 mm), 1.5 (140 mm) and double (190 mm) column widths.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trackwood import config  # noqa: E402

MM = 1 / 25.4
RES = config.REPO / "results"
FIG = config.REPO / "figures"
TRAINING_VIDEO = "20040"
LABEL = {"rfdetr_base_legacy": "RF-DETR (original)", "yolo11n_legacy": "YOLO11n (original)",
         "rfdetr_base_random": "RF-DETR (retrained)", "rfdetr_base_blocked": "RF-DETR (blocked split)",
         "yolo11n_random": "YOLO11n (same data)", "yolo11s_random": "YOLO11s (same data)",
         "yolo11m_random": "YOLO11m (same data)", "yolo11m_blocked": "YOLO11m (blocked split)"}
STYLE = {  # detector -> (colour, marker)
    "rfdetr_base_legacy": ("#1f5fbf", "o"), "rfdetr_base_random": ("#6a3d9a", "D"),
    "rfdetr_base_blocked": ("#56b4e9", "v"), "yolo11n_legacy": ("#d9822b", "s"),
    "yolo11n_random": ("#e6b422", "^"), "yolo11s_random": ("#b15928", "P"),
    "yolo11m_random": ("#2a9d4b", "X"), "yolo11m_blocked": ("#8fd18f", "<"),
}


def col(d: str) -> str:
    return STYLE.get(d, ("0.3", "o"))[0]


def mk(d: str) -> str:
    return STYLE.get(d, ("0.3", "o"))[1]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.0,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})


def family(det: str) -> str:
    return "rfdetr" if det.startswith("rfdetr") else "yolo"


def read_csv(name: str) -> list[dict]:
    with open(RES / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save(fig, name: str) -> None:
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=600)
    plt.close(fig)


def nominal_rows(detectors: list[str]) -> dict[str, list[dict]]:
    key = json.loads((RES / "summary.json").read_text(encoding="utf-8"))["nominal_key"]
    rows = defaultdict(list)
    for r in read_csv("counting_runs.csv"):
        if r["config_key"] == key and r["detector"] in detectors:
            rows[r["detector"]].append(r)
    for d in rows:
        rows[d].sort(key=lambda r: (int(r["N"]), int(r["Q"])))
    return rows


def fig_counts_vs_n(rows, detectors):
    fig, ax = plt.subplots(figsize=(90 * MM, 85 * MM))
    lim = 230
    ax.plot([0, lim], [0, lim], color="0.4", lw=0.8, ls="--", zorder=1)
    ax.text(185, 200, "1:1", ha="right", va="bottom", color="0.4", fontsize=7)
    for d in detectors:
        fam = family(d)
        N = np.array([int(r["N"]) for r in rows[d]])
        C = np.array([int(r["L1_fixed_confirmed"]) for r in rows[d]])
        legacy = np.array([int(r["L1_legacy"]) for r in rows[d]])
        train = np.array([r["video"] == TRAINING_VIDEO for r in rows[d]])
        ax.scatter(N, legacy, s=10, marker=mk(d), facecolor="none", edgecolor=col(d), alpha=0.5,
                   lw=0.6, zorder=2, label=f"{LABEL.get(d, d)}, original counting rule")
        ax.scatter(N, C, s=16, marker=mk(d), color=col(d), zorder=3,
                   label=f"{LABEL.get(d, d)}, corrected pipeline")
        ax.scatter(N[train], C[train], s=46, marker="o", facecolor="none", edgecolor="k", lw=0.6, zorder=4)
        if d == detectors[0] and train.any():
            target = (float(N[train][0]), float(C[train][0]))
    ax.scatter([], [], s=46, marker="o", facecolor="none", edgecolor="k", lw=0.6,
               label=f"run {TRAINING_VIDEO} (training images)")
    ax.set(xlim=(0, lim), ylim=(0, lim), xlabel="Pieces released, $N$", ylabel="Pieces counted at L1, $C$")
    ax.set_aspect("equal")
    ax.legend(frameon=False, loc="upper left", fontsize=6.5)
    save(fig, "fig_counts_vs_N")


def fig_bland_altman(rows, detectors):
    fig, axes = plt.subplots(1, len(detectors), figsize=(190 * MM, 65 * MM), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, d in zip(axes, detectors):
        fam = family(d)
        N = np.array([int(r["N"]) for r in rows[d]], float)
        C = np.array([int(r["L1_fixed_confirmed"]) for r in rows[d]], float)
        indep = np.array([r["video"] != TRAINING_VIDEO for r in rows[d]])
        rel = (C - N) / N * 100
        m, s = rel[indep].mean(), rel[indep].std(ddof=1)
        ax.axhline(0, color="0.6", lw=0.6)
        ax.axhline(m, color=col(d), lw=1.0)
        for y in (m - 1.96 * s, m + 1.96 * s):
            ax.axhline(y, color=col(d), lw=0.8, ls="--")
        ax.scatter(N[indep], rel[indep], s=16, marker=mk(d), color=col(d), zorder=3)
        ax.scatter(N[~indep], rel[~indep], s=16, marker=mk(d), facecolor="white", edgecolor=col(d), zorder=3)
        ax.text(1.0, m, f" mean {m:+.1f} %", transform=ax.get_yaxis_transform(), va="center", fontsize=6.5, color=col(d))
        ax.text(1.0, m + 1.96 * s, f" +1.96 s {m + 1.96 * s:+.1f} %", transform=ax.get_yaxis_transform(), va="center", fontsize=6.5, color=col(d))
        ax.text(1.0, m - 1.96 * s, f" -1.96 s {m - 1.96 * s:+.1f} %", transform=ax.get_yaxis_transform(), va="center", fontsize=6.5, color=col(d))
        ax.set(xscale="log", xlabel="Pieces released, $N$", title=LABEL.get(d, d))
        ax.set_xticks([10, 20, 50, 100, 200])
        ax.set_xticklabels(["10", "20", "50", "100", "200"])
        ax.minorticks_off()
    axes[0].set_ylabel("Relative error $(C - N)/N$ (%)")
    fig.subplots_adjust(wspace=0.45)
    save(fig, "fig_bland_altman")


def fig_sensitivity(detectors):
    rows = read_csv("sensitivity_oat.csv")
    nominal = json.loads((RES / "summary.json").read_text(encoding="utf-8"))["nominal_config"]
    params = [("conf", "Confidence threshold"), ("nms_iou", "NMS IoU"), ("trk_iou", "Tracker IoU gate"),
              ("min_hits", "Hits to confirm"), ("max_age_s", "Coasting time (s)")]
    fig, axes = plt.subplots(1, 5, figsize=(190 * MM, 50 * MM), sharey=True)
    for ax, (p, lab) in zip(axes, params):
        for d in detectors:
            fam = family(d)
            pts = {}
            for r in rows:
                if r["detector"] != d:
                    continue
                if r["param"] == "nominal":
                    pts[float(nominal[p])] = float(r["eta_total"])
                elif r["param"] == p:
                    v = np.nan if r["value"] in ("", "None") else float(r["value"])
                    pts[v] = float(r["eta_total"])
            xs = sorted(x for x in pts if not np.isnan(x))
            ax.plot(xs, [pts[x] for x in xs], marker=mk(d), ms=3, color=col(d), label=LABEL.get(d, d))
            ax.scatter([float(nominal[p])], [pts[float(nominal[p])]], s=30, facecolor="none", edgecolor="k", lw=0.7, zorder=4)
            nan_keys = [x for x in pts if np.isnan(x)]
            if nan_keys:
                ax.scatter([max(xs) + 0.1], [pts[nan_keys[0]]], marker="x", color=col(d), s=14)
        ax.axhline(1.0, color="0.6", lw=0.6, ls="--")
        ax.set_xlabel(lab)
        if p == "min_hits":
            ax.set_xticks([1, 3, 5, 8])
    axes[0].set_ylabel(r"$\sum C\,/\sum N$ at L1")
    axes[0].legend(frameon=False, loc="upper right")
    axes[1].text(0.98, 0.02, "x: no NMS", transform=axes[1].transAxes, ha="right", va="bottom", fontsize=6)
    axes[2].text(0.5, 1.03, "black circle: nominal setting", transform=axes[2].transAxes, ha="center", fontsize=6)
    fig.subplots_adjust(wspace=0.15)
    save(fig, "fig_sensitivity")


def fig_line_profile(detectors):
    rows = read_csv("line_position_sweep.csv")
    fig, ax = plt.subplots(figsize=(140 * MM, 60 * MM))
    ax.axvspan(40, 66, color="0.9", zorder=0)
    ax.text(53, 1.27, "permeable spur", ha="center", fontsize=6.5, color="0.35")
    for d in detectors:
        fam = family(d)
        sel = [r for r in rows if r["detector"] == d and r["video"] != TRAINING_VIDEO]
        by = defaultdict(lambda: [0, 0])
        for r in sel:
            by[float(r["line_pct"])][0] += int(r["count"])
            by[float(r["line_pct"])][1] += int(r["N"])
        xs = sorted(by)
        ax.plot(xs, [by[x][0] / by[x][1] for x in xs], marker=mk(d), ms=3, color=col(d), label=LABEL.get(d, d))
    for x, lab in ((20, "L1"), (80, "L2")):
        ax.axvline(x, color="0.3", lw=0.6, ls=":")
        ax.text(x, 1.27, lab, ha="center", fontsize=6.5)
    ax.axhline(1.0, color="0.6", lw=0.6, ls="--")
    ax.set(xlim=(0, 100), ylim=(0, 1.35), xlabel="Counting line position (% of frame width, flow left to right)",
           ylabel=r"$\sum C / \sum N$ (15 independent runs)")
    ax.legend(frameon=False, loc="lower left")
    save(fig, "fig_line_profile")


def fig_trapping(detectors):
    rows = read_csv("trapping_efficiency.csv")
    fig, ax = plt.subplots(figsize=(90 * MM, 65 * MM))
    for d in detectors:
        fam = family(d)
        sel = [r for r in rows if r["detector"] == d]
        N = np.array([int(r["N"]) for r in sel])
        R = np.array([float(r["R"]) for r in sel])
        ax.scatter(N * (0.97 if fam == "rfdetr" else 1.03), R, s=16, marker=mk(d), color=col(d), label=LABEL.get(d, d))
    ax.set(xscale="log", xlabel="Pieces released, $N$", ylabel=r"Trapping efficiency $R = 1 - C_{L2}/C_{L1}$", ylim=(0, 1))
    ax.set_xticks([10, 20, 30, 50, 80, 120, 150, 200])
    ax.set_xticklabels(["10", "20", "30", "50", "80", "120", "150", "200"])
    ax.legend(frameon=False)
    save(fig, "fig_trapping_efficiency")


def fig_forest(detectors):
    """Bias with bootstrap 95 % CI and Bland-Altman limits for every detector."""
    key = json.loads((RES / "summary.json").read_text(encoding="utf-8"))["nominal_key"]
    acc = {r["detector"]: r for r in read_csv("counting_accuracy.csv")
           if r["config_key"] == key and r["subset"] == "indep15" and r["estimator"] == "L1_fixed_confirmed"}
    dets = [d for d in detectors if d in acc]
    fig, ax = plt.subplots(figsize=(140 * MM, (18 + 9 * len(dets)) * MM))
    for i, d in enumerate(dets[::-1]):
        r = acc[d]
        lo, hi = json.loads(r["mpe_pct_ci95"])
        ax.plot([float(r["ba_loa_low_pct"]), float(r["ba_loa_high_pct"])], [i, i], color=col(d), lw=0.8, alpha=0.5)
        ax.plot([lo, hi], [i, i], color=col(d), lw=2.4, solid_capstyle="butt")
        ax.scatter([float(r["mpe_pct"])], [i], marker=mk(d), color=col(d), s=24, zorder=3, edgecolor="white", lw=0.4)
        ax.text(1.02, i, f"MAPE {float(r['mape_pct']):.1f} %", transform=ax.get_yaxis_transform(), va="center", fontsize=6.5)
    ax.axvline(0, color="0.5", lw=0.6)
    ax.set_yticks(range(len(dets)))
    ax.set_yticklabels([LABEL.get(d, d) for d in dets[::-1]])
    ax.set_xlabel("Relative error of the L1 count, (C - N)/N (%)\n"
                  "marker: mean; thick bar: 95 % CI of the mean; thin bar: limits of agreement")
    ax.set_ylim(-0.6, len(dets) - 0.4)
    save(fig, "fig_model_comparison")


def fig_ap_vs_mape(detectors):
    key = json.loads((RES / "summary.json").read_text(encoding="utf-8"))["nominal_key"]
    acc = {r["detector"]: r for r in read_csv("counting_accuracy.csv")
           if r["config_key"] == key and r["subset"] == "indep15" and r["estimator"] == "L1_fixed_confirmed"}
    ev = json.loads((RES / "detector_eval.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(190 * MM, 70 * MM), sharey=True)
    # a model is scored only on test images it never saw: random-split models on the
    # random test set, blocked-split models on the blocked test set; the original YOLO
    # (trained on another dataset) is valid on both
    valid = {"random": lambda d: d.endswith("_random") or d in ("rfdetr_base_legacy", "yolo11n_legacy"),
             "blocked": lambda d: d.endswith("_blocked") or d == "yolo11n_legacy"}
    for ax, split in zip(axes, ("random", "blocked")):
        for d in detectors:
            e = ev.get(f"{d}__{split}")
            if not e or d not in acc or not valid[split](d):
                continue
            ax.scatter(e["coco"]["AP50_95"], float(acc[d]["mape_pct"]), marker=mk(d), color=col(d), s=30, label=LABEL.get(d, d))
        ax.set(xlabel=f"AP@[.50:.95], {split} test split", title=f"{split} split")
    axes[0].set_ylabel("Counting MAPE at L1, 15 runs (%)")
    handles = {}
    for ax in axes:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    fig.legend(handles.values(), handles.keys(), frameon=False, fontsize=6.5, ncol=4,
               loc="upper center", bbox_to_anchor=(0.5, 0.0))
    save(fig, "fig_ap_vs_counting_error")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", nargs="+", required=True, help="detectors shown in the detailed figures")
    ap.add_argument("--all", nargs="*", default=[], help="detectors in the comparison figures")
    args = ap.parse_args()
    rows = nominal_rows(args.main)
    dets = [d for d in args.main if d in rows]
    fig_counts_vs_n(rows, dets)
    fig_bland_altman(rows, dets)
    fig_sensitivity(dets)
    fig_line_profile(dets)
    fig_trapping(dets)
    if args.all:
        fig_forest(args.all)
        if (RES / "detector_eval.json").exists():
            fig_ap_vs_mape(args.all)
    print(f"figures -> {FIG}")


if __name__ == "__main__":
    main()
