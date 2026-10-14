"""Agreement statistics, calibration and uncertainty evaluation for count data.

Terminology follows the VIM: the measurand is the number of wood pieces passing
the inlet line L1 during an experiment; the reference value is the number of
pieces released, N, counted by hand before each run.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats as sps

RNG_SEED = 42


def lins_ccc(x: np.ndarray, y: np.ndarray) -> float:
    mx, my = x.mean(), y.mean()
    sxy = ((x - mx) * (y - my)).mean()
    return float(2 * sxy / (x.var() + y.var() + (mx - my) ** 2))


def agreement(C: np.ndarray, N: np.ndarray) -> dict:
    """Error statistics of counts C against reference N (one value per video)."""
    C = np.asarray(C, float)
    N = np.asarray(N, float)
    e = C - N
    rel = e / N
    eta = C / N
    reg = sps.linregress(N, C)
    sd_rel = rel.std(ddof=1)
    return {
        "n_videos": int(N.size),
        "sum_count": float(C.sum()), "sum_N": float(N.sum()),
        "eta_total": float(C.sum() / N.sum()),
        "eta_mean": float(eta.mean()), "eta_sd": float(eta.std(ddof=1)),
        "eta_cv_pct": float(eta.std(ddof=1) / eta.mean() * 100),
        "eta_min": float(eta.min()), "eta_max": float(eta.max()),
        "bias": float(e.mean()), "mae": float(np.abs(e).mean()),
        "rmse": float(math.sqrt((e ** 2).mean())),
        "mape_pct": float(np.abs(rel).mean() * 100), "mpe_pct": float(rel.mean() * 100),
        "ols_slope": float(reg.slope), "ols_intercept": float(reg.intercept),
        "r2": float(reg.rvalue ** 2), "lins_ccc": lins_ccc(N, C),
        "ba_mean_pct": float(rel.mean() * 100), "ba_sd_pct": float(sd_rel * 100),
        "ba_loa_low_pct": float((rel.mean() - 1.96 * sd_rel) * 100),
        "ba_loa_high_pct": float((rel.mean() + 1.96 * sd_rel) * 100),
        "eta_vs_N_spearman_rho": float(sps.spearmanr(N, eta).statistic) if N.size > 2 else float("nan"),
        "eta_vs_N_spearman_p": float(sps.spearmanr(N, eta).pvalue) if N.size > 2 else float("nan"),
    }


def bootstrap_ci(C: np.ndarray, N: np.ndarray, n_boot: int = 10000, seed: int = RNG_SEED) -> dict:
    """Percentile 95 % intervals, resampling videos with replacement."""
    rng = np.random.default_rng(seed)
    C = np.asarray(C, float)
    N = np.asarray(N, float)
    idx = rng.integers(0, N.size, size=(n_boot, N.size))
    Cb, Nb = C[idx], N[idx]
    e = Cb - Nb
    stats_ = {
        "bias": e.mean(1), "mae": np.abs(e).mean(1),
        "mape_pct": (np.abs(e) / Nb).mean(1) * 100, "mpe_pct": (e / Nb).mean(1) * 100,
        "eta_total": Cb.sum(1) / Nb.sum(1),
    }
    return {f"{k}_ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            for k, v in stats_.items()}


def loocv_calibration(C: np.ndarray, N: np.ndarray, affine: bool = False) -> tuple[dict, np.ndarray]:
    """Leave-one-video-out calibration. Ratio: k = sum C / sum N over the other
    videos, N_hat = C / k. Affine: C = a N + b fitted on the other videos,
    N_hat = (C - b) / a. Returns error statistics and the N_hat values."""
    C = np.asarray(C, float)
    N = np.asarray(N, float)
    Nhat = np.empty_like(N)
    for i in range(N.size):
        m = np.arange(N.size) != i
        if affine:
            a, b = np.polyfit(N[m], C[m], 1)
            Nhat[i] = (C[i] - b) / a
        else:
            Nhat[i] = C[i] / (C[m].sum() / N[m].sum())
    e = Nhat - N
    rel = e / N
    sd = rel.std(ddof=1)
    return {
        "bias": float(e.mean()), "mae": float(np.abs(e).mean()),
        "rmse": float(math.sqrt((e ** 2).mean())),
        "mape_pct": float(np.abs(rel).mean() * 100), "mpe_pct": float(rel.mean() * 100),
        "sd_pct": float(sd * 100),
        "loa_low_pct": float((rel.mean() - 1.96 * sd) * 100),
        "loa_high_pct": float((rel.mean() + 1.96 * sd) * 100),
        "max_abs_rel_pct": float(np.abs(rel).max() * 100),
    }, Nhat


def paired_comparison(Ca: np.ndarray, Cb: np.ndarray, N: np.ndarray) -> dict:
    """Wilcoxon signed-rank test on absolute relative errors of two methods."""
    ea = np.abs(np.asarray(Ca, float) - N) / N
    eb = np.abs(np.asarray(Cb, float) - N) / N
    out = {"a_better": int((ea < eb).sum()), "b_better": int((ea > eb).sum()), "ties": int((ea == eb).sum()),
           "median_abs_rel_a_pct": float(np.median(ea) * 100), "median_abs_rel_b_pct": float(np.median(eb) * 100)}
    if np.any(ea != eb):
        w = sps.wilcoxon(ea, eb)
        out.update(statistic=float(w.statistic), p=float(w.pvalue))
    return out


def uncertainty_budget(eta: np.ndarray, type_b: dict[str, float]) -> dict:
    """Relative standard uncertainty of a calibrated count N_hat = C / k_bar.

    Components (relative, 1 sigma):
      u_repeat : between-experiment scatter of the counting efficiency (Type A),
                 s(eta) / mean(eta), nu = n - 1
      u_k      : uncertainty of the calibration factor k_bar, s(eta) / (sqrt(n) mean(eta))
      type_b   : further components, e.g. half-width / sqrt(3) of the efficiency
                 change over a plausible range of a processing parameter
    Combined by the law of propagation of uncertainty (uncorrelated inputs);
    effective degrees of freedom by Welch-Satterthwaite (Type B taken as infinite);
    expanded uncertainty with k from the t distribution at 95 %.
    """
    eta = np.asarray(eta, float)
    n = eta.size
    m, s = eta.mean(), eta.std(ddof=1)
    comps = {"u_repeat": s / m, "u_k": s / (math.sqrt(n) * m)}
    comps.update(type_b)
    uc = math.sqrt(sum(v ** 2 for v in comps.values()))
    type_a = comps["u_repeat"] ** 2 + comps["u_k"] ** 2
    nu_eff = uc ** 4 / (type_a ** 2 / (n - 1)) if type_a > 0 else float("inf")
    k95 = float(sps.t.ppf(0.975, nu_eff)) if math.isfinite(nu_eff) else 1.96
    return {"components_rel": {k: float(v) for k, v in comps.items()}, "u_c_rel": float(uc),
            "nu_eff": float(nu_eff), "k95": k95, "U95_rel": float(k95 * uc)}


def nested_loocv_select(table: dict[str, np.ndarray], N: np.ndarray) -> dict:
    """Nested leave-one-run-out choice of the processing configuration.

    table maps a configuration key to its counts on the runs (same order as N).
    For each held-out run the configuration is chosen on the remaining runs only:
      raw        - smallest mean absolute relative error of the raw counts;
      calibrated - smallest standard deviation of the efficiency C/N (precision),
                   with the ratio calibration factor k = sum C / sum N of the
                   remaining runs applied to the held-out run.
    Returns the held-out errors and how often each configuration was chosen.
    """
    N = np.asarray(N, float)
    keys = list(table)
    M = np.array([np.asarray(table[k], float) for k in keys])  # configs x runs
    out = {}
    for mode in ("raw", "calibrated"):
        rel, chosen = [], []
        for i in range(N.size):
            m = np.arange(N.size) != i
            if mode == "raw":
                crit = np.abs(M[:, m] / N[m] - 1).mean(1)
            else:
                crit = (M[:, m] / N[m]).std(1, ddof=1) / (M[:, m] / N[m]).mean(1)
            j = int(np.argmin(crit))
            chosen.append(keys[j])
            if mode == "raw":
                est = M[j, i]
            else:
                est = M[j, i] / (M[j, m].sum() / N[m].sum())
            rel.append((est - N[i]) / N[i])
        rel = np.array(rel)
        counts = {k: chosen.count(k) for k in set(chosen)}
        out[mode] = {"mape_pct": float(np.abs(rel).mean() * 100), "mpe_pct": float(rel.mean() * 100),
                     "sd_pct": float(rel.std(ddof=1) * 100),
                     "loa_low_pct": float((rel.mean() - 1.96 * rel.std(ddof=1)) * 100),
                     "loa_high_pct": float((rel.mean() + 1.96 * rel.std(ddof=1)) * 100),
                     "max_abs_rel_pct": float(np.abs(rel).max() * 100),
                     "held_out_rel_error_pct": (rel * 100).round(2).tolist(),
                     "chosen_configs": dict(sorted(counts.items(), key=lambda kv: -kv[1]))}
    return out
