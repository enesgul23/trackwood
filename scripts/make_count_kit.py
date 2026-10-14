"""Prepare images for a manual count of the pieces trapped between L1 and L2 at the
end of each run (reference for the outlet count N_L2 = N - trapped).

For every video the last frame is cropped to the channel between the counting lines
and split into two full-resolution tiles; a CSV template is written for the counts.

    python scripts/make_count_kit.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trackwood import config  # noqa: E402
from trackwood.video import list_videos, parse_experiment  # noqa: E402

Y0, Y1 = 0.24, 0.84   # channel bed between the side walls, including the wall shadow (fraction of height)


def last_frame(path: Path):
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n - 10))
    frame = None
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frame = f
    cap.release()
    return frame


def main() -> None:
    paths = config.load_paths()
    out = paths["work_dir"] / "count_kit"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for vp in list_videos(paths["videos_dir"]):
        ex = parse_experiment(vp.stem)
        f = last_frame(vp)
        H, W = f.shape[:2]
        x1, x2 = int(W * 0.20), int(W * 0.80)
        crop = f[int(H * Y0):int(H * Y1), x1:x2].copy()
        h, w = crop.shape[:2]
        for name, tile in (("A_left", crop[:, : w // 2 + 100]), ("B_right", crop[:, w // 2 - 100:])):
            cv2.imwrite(str(out / f"{vp.stem}_{name}.jpg"), tile, [cv2.IMWRITE_JPEG_QUALITY, 95])
        overview = f.copy()
        for x in (x1, x2):
            cv2.line(overview, (x, 0), (x, H), (0, 255, 255), 6)
        cv2.putText(overview, f"{vp.stem}  N = {ex.n_released}  last frame", (60, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 8)
        cv2.imwrite(str(out / f"{vp.stem}_overview.jpg"), cv2.resize(overview, (1920, 1080)),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        rows.append({"video": vp.stem, "N_released": ex.n_released,
                     "trapped_between_L1_L2": "", "upstream_of_L1": "", "counter_initials": "", "notes": ""})
        print(vp.stem, flush=True)
    with open(out / "trapped_counts_template.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter=";")
        w.writeheader()
        w.writerows(rows)
    (out / "README.txt").write_text(
        "Elle sayim seti (son kare)\n"
        "Her video icin: {video}_overview.jpg sari cizgiler L1 (%20) ve L2 (%80);\n"
        "{video}_A_left.jpg ve {video}_B_right.jpg iki hat arasindaki kanalin tam cozunurluklu yarilari\n"
        "(ortada 100 piksellik ortusme var; ayni cubugu iki kez saymayin).\n"
        "trapped_counts_template.csv: trapped_between_L1_L2 = video sonunda L1 ile L2 arasinda kalan cubuk sayisi;\n"
        "upstream_of_L1 = L1'in solunda kalan cubuk sayisi (genelde 0). Mumkunse iki kisi bagimsiz saysin.\n",
        encoding="utf-8")
    print(f"kit -> {out}")


if __name__ == "__main__":
    main()
