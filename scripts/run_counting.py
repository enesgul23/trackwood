"""Stage 2 driver: run tracking and counting for detectors x configurations x videos.

    python scripts/run_counting.py --detectors yolo11n_legacy rfdetr_base_legacy --grid legacy
    python scripts/run_counting.py --detectors rfdetr_base_legacy --grid sensitivity --workers 30

Configurations are defined in configs/grids.json. Each run writes
{work_dir}/counting/{detector}/{config_key}/{video_id}.json; existing outputs are reused.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trackwood import config, detfile  # noqa: E402
from trackwood.pipeline import CountConfig, run  # noqa: E402

GRIDS = config.REPO / "configs" / "grids.json"


def load_grid(name: str) -> list[CountConfig]:
    with open(GRIDS, encoding="utf-8") as f:
        grids = json.load(f)
    g = grids[name]
    base = dict(g.get("base", {}))
    out = []
    if not g.get("vary"):
        out.append(CountConfig(**_tuples(base)))
    for param, values in g.get("vary", {}).items():
        for v in values:
            d = dict(base)
            d[param] = v
            out.append(CountConfig(**_tuples(d)))
    for extra in g.get("extra", []):
        d = dict(base)
        d.update(extra)
        out.append(CountConfig(**_tuples(d)))
    uniq = {c.key(): c for c in out}
    return list(uniq.values())


def _tuples(d: dict) -> dict:
    return {k: tuple(v) if isinstance(v, list) else v for k, v in d.items()}


@lru_cache(maxsize=4)
def _dets(path: str):
    return detfile.load(Path(path))


def _task(det_path: str, cfg: CountConfig, out_path: str) -> str:
    res = run(_dets(det_path), cfg)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f)
    return f"{res['detector']} {res['video_id']} {cfg.key()} {res['runtime_s']:.0f}s"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detectors", nargs="+", required=True)
    ap.add_argument("--grid", required=True)
    ap.add_argument("--videos", nargs="*")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    paths = config.load_paths()
    cfgs = load_grid(args.grid)
    tasks = []
    for det in args.detectors:
        det_dir = paths["work_dir"] / "detections" / det
        vids = sorted(p.stem for p in det_dir.glob("*.npz") if "_debug" not in p.stem)
        if args.videos:
            vids = [v for v in vids if v in set(args.videos)]
        for vid in vids:  # group by video so each worker reuses the loaded detections
            for cfg in cfgs:
                out = paths["work_dir"] / "counting" / det / cfg.key() / f"{vid}.json"
                if out.exists() and not args.overwrite:
                    continue
                tasks.append((str(det_dir / f"{vid}.npz"), cfg, str(out)))
    print(f"grid {args.grid}: {len(cfgs)} configs, {len(tasks)} runs to do", flush=True)
    if not tasks:
        return
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_task, *t) for t in tasks]
        for i, fu in enumerate(as_completed(futs), 1):
            print(f"[{i}/{len(futs)}] {fu.result()}", flush=True)


if __name__ == "__main__":
    main()
