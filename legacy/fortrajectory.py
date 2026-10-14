# -*- coding: utf-8 -*-

from __future__ import annotations
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.collections import LineCollection
from scipy.signal import savgol_filter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes   # NEW

# ------------------------------------------------------------------------------
# --- TEMEL AYARLAR ------------------------------------------------------------
# ------------------------------------------------------------------------------
output_dir              = Path("results%1")
trajectory_file         = output_dir / "analysis_results_excel" / "1bx216_trajectory_detailed.csv"
crossings_file          = output_dir / "analysis_results_excel" / "1bx216_line_crossings.csv"

plot_dir                = output_dir / "analysis_plots_all_v3"
plot_dir.mkdir(parents=True, exist_ok=True)

VIDEO_WIDTH             = 3_840
line_spec               = "v:20%,v:80%"

APPLY_SMOOTHING         = True
SAVGOL_WINDOW           = 11
SAVGOL_POLY             = 3
MIN_POINTS_SMOOTH       = 15
INVERT_Y_AXIS           = True

SELECT_METHOD           = "longest"
MAX_TRACKS_TO_PLOT      = 250

sns.set_theme(context="notebook", style="whitegrid", font_scale=1.0)

# ------------------------------------------------------------------------------
# --- VERİ ---------------------------------------------------------------------
# ------------------------------------------------------------------------------
print(f"Yörünge verisi yükleniyor: {trajectory_file}")
numeric_cols = [
    'estimated_speed', 'kalman_area', 'kalman_vx',
    'kalman_vy', 'kalman_vs', 'bbox_area'
]
traj_df = pd.read_csv(trajectory_file, dtype={c: float for c in numeric_cols})
traj_df.sort_values(["track_id", "frame"], inplace=True)
traj_df.dropna(subset=["cx", "cy", "frame", "track_id"], inplace=True)
print(f"Toplam {traj_df['track_id'].nunique()} iz – {len(traj_df)} satır.")

try:
    cross_df = pd.read_csv(crossings_file)
except FileNotFoundError:
    cross_df = pd.DataFrame()

def parse_virtual_lines(spec: str, w: int) -> List[int]:
    xs = []
    for tok in (t.strip() for t in spec.split(',') if t.strip()):
        axis, val = tok.split(':')
        if axis.lower() == 'v':
            perc = '%' in val
            v = float(val.replace('%', ''))
            xs.append(int(w * v / 100) if perc else int(v))
    return xs

ref_lines_x = parse_virtual_lines(line_spec, VIDEO_WIDTH)

# ------------------------------------------------------------------------------
# --- METRİK -------------------------------------------------------------------
# ------------------------------------------------------------------------------
def compute_track_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for tid, g in df.groupby("track_id"):
        cx, cy, fr = g["cx"].values, g["cy"].values, g["frame"].values
        if len(cx) < 2: continue
        step = np.hypot(np.diff(cx), np.diff(cy))
        rows.append({
            "track_id": tid,
            "path_length": step.sum(),
            "duration": fr[-1] - fr[0] + 1,
            "mean_step": step.mean(),
        })
    return pd.DataFrame(rows)

track_metrics_df = compute_track_metrics(traj_df)

# ------------------------------------------------------------------------------
# --- GRAFİK FONKSİYONLARI -----------------------------------------------------
# ------------------------------------------------------------------------------
def _choose_tracks(df: pd.DataFrame,
                   metrics: pd.DataFrame,
                   method: str,
                   top_n: Optional[int]) -> pd.DataFrame:
    if (top_n is None) or (df['track_id'].nunique() <= top_n):
        return df.copy()
    if method == "longest":
        ids = metrics.nlargest(top_n, "path_length")['track_id']
    else:  # random
        ids = np.random.choice(df['track_id'].unique(), top_n, replace=False)
    return df[df["track_id"].isin(ids)]

# ..............................................................................
def plot_detailed_trajectories(df: pd.DataFrame,
                               metrics_df: pd.DataFrame,
                               save_path: Path,
                               base_title: str,
                               ref_lines: List[int] | None = None,
                               selection_method: str = "longest",
                               max_tracks: int | None = None,
                               apply_smoothing: bool = True):
    if df.empty:
        return

    plot_df = _choose_tracks(df, metrics_df, selection_method, max_tracks)
    sel_txt = f"(En Uzun {max_tracks})" if max_tracks else ""

    # 🔸 Fig yüksekliği artırıldı, y-eksenine daha fazla alan
    fig, ax = plt.subplots(figsize=(30, 12))

    # Referans dikey çizgiler
    colors = ['lime', 'cyan', 'magenta', 'yellow']
    for i, x in enumerate(ref_lines or []):
        ax.axvline(x, color=colors[i % len(colors)],
                   linestyle='--', lw=2, label=f"L{i+1} (x={x})")

    # Renk gradyanı (kare numarası)
    min_fr, max_fr = df['frame'].min(), df['frame'].max()
    norm = plt.Normalize(min_fr, max_fr)

    for i, (tid, g) in enumerate(plot_df.groupby('track_id')):
        x, y, fr = g['cx'].values, g['cy'].values, g['frame'].values

        if apply_smoothing and len(x) >= MIN_POINTS_SMOOTH:
            win = min(SAVGOL_WINDOW, len(x) - (len(x)+1)%2)
            poly = min(SAVGOL_POLY, win-1)
            x = savgol_filter(x, win, poly)
            y = savgol_filter(y, win, poly)

        pts = np.vstack([x, y]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap='viridis', norm=norm, lw=1.6, alpha=0.85)
        lc.set_array(fr[:-1])
        ax.add_collection(lc)

        ax.scatter(x[0],  y[0],  s=25, c='blue', zorder=3,
                   label='Başlangıç' if i == 0 else "")
        ax.scatter(x[-1], y[-1], s=25, c='red',  zorder=3,
                   label='Bitiş'     if i == 0 else "")

    # 🔸 COLORBAR GRAFİK İÇİNE (sağ), tam y-yüksekliğinde
    cax = inset_axes(ax,
                     width="2.5%", height="100%",
                     loc='center right',
                     bbox_to_anchor=(0.02, 0, 1, 1),
                     bbox_transform=ax.transAxes,
                     borderpad=0)
    cb = fig.colorbar(lc, cax=cax)
    cb.set_label("Kare Numarası")

    ax.set_title(f"Detaylı Nesne Yörüngeleri: {base_title} {sel_txt}")
    ax.set_xlabel("X Koordinatı (piksel)")
    ax.set_ylabel("Y Koordinatı (piksel)")
    if INVERT_Y_AXIS:
        ax.invert_yaxis()

    # 🔸 Artık eşit ölçek yok – y-eksenine rahatlık
    ax.set_aspect('auto')
    ax.margins(0.05, 0.10)            # y-marj %10

    # Kategorik lejant sol-üstte
    h, l = ax.get_legend_handles_labels()
    by_lbl = {lab: han for han, lab in zip(h, l) if lab}
    if by_lbl:
        ax.legend(by_lbl.values(), by_lbl.keys(),
                  title="Lejant",
                  title_fontsize='small',
                  fontsize='x-small',
                  loc='upper left',
                  bbox_to_anchor=(0.01, 0.99),
                  framealpha=0.9)

    plt.grid(True, linestyle=":", alpha=0.6)
    plt.savefig(save_path, dpi=320, bbox_inches='tight')
    print(f"  ➜ {save_path.name} kaydedildi.")
    plt.close()

# ------------------------------------------------------------------------------
# --- (DİĞER GRAFİK FONKSİYONLARI – AYNI) --------------------------------------
# ------------------------------------------------------------------------------
def plot_speed_distribution(df, save_path):
    if df.empty or 'estimated_speed' not in df.columns:
        return
    plt.figure(figsize=(8,6))
    sns.histplot(df['estimated_speed'].dropna(), bins=50, kde=True)
    plt.title("Nesne Hız Dağılımı")
    plt.xlabel("Tahmini Hız (piksel/kare)")
    plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.close()

def plot_area_distribution(df, save_path):
    if df.empty or 'bbox_area' not in df.columns:
        return
    plt.figure(figsize=(8,6))
    sns.histplot(df['bbox_area'].dropna(), bins=50, kde=True)
    plt.title("Nesne Alan Dağılımı")
    plt.xlabel("BBox Alanı (piksel²)")
    plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.close()

def plot_cumulative_crossings(cross, save_path):
    if cross.empty: return
    cross = cross.sort_values('frame').copy()
    cross['cumulative'] = cross.groupby('line_id').cumcount() + 1
    plt.figure(figsize=(10,6))
    sns.lineplot(data=cross, x='frame', y='cumulative',
                 hue='line_id', marker='o', markersize=4)
    plt.title("Kümülatif Çizgi Geçişleri")
    plt.xlabel("Kare"); plt.ylabel("Toplam")
    plt.legend(title="Çizgi ID"); plt.grid(True)
    plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.close()

def plot_crossing_area_distribution(cross, save_path):
    if cross.empty or 'bbox_area_at_crossing' not in cross.columns:
        return
    plt.figure(figsize=(8,6))
    sns.boxplot(data=cross, x='line_id', y='bbox_area_at_crossing')
    plt.title("Geçiş Anı Alan Dağılımı")
    plt.xlabel("Çizgi ID"); plt.ylabel("Alan (piksel²)")
    plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.close()

# ------------------------------------------------------------------------------
# --- ANA ÇALIŞTIRMA -----------------------------------------------------------
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    vid_name = trajectory_file.stem.replace("_trajectory_detailed", "")

    # 1) Detaylı yörünge grafiği (GÜNCELLENDİ)
    plot_detailed_trajectories(
        traj_df, track_metrics_df,
        save_path   = plot_dir / f"{vid_name}_trajectories_detailed_top{MAX_TRACKS_TO_PLOT}.png",
        base_title  = vid_name,
        ref_lines   = ref_lines_x,
        selection_method = SELECT_METHOD,
        max_tracks  = MAX_TRACKS_TO_PLOT,
        apply_smoothing = APPLY_SMOOTHING,
    )

    # 2) Diğer grafikler – değişmedi
    plot_speed_distribution(
        traj_df, plot_dir / f"{vid_name}_speed_distribution.png")
    plot_area_distribution(
        traj_df, plot_dir / f"{vid_name}_area_distribution.png")
    plot_cumulative_crossings(
        cross_df, plot_dir / f"{vid_name}_cumulative_crossings.png")
    plot_crossing_area_distribution(
        cross_df, plot_dir / f"{vid_name}_crossing_area_box.png")

    print(f"\n✅ Tüm grafikler '{plot_dir.resolve()}' dizinine kaydedildi.")
