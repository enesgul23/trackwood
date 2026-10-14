"""Train a YOLO11 detector on the same images and split as RF-DETR.

    python scripts/train_yolo.py --model yolo11m --split random

Runs offline (no downloads): the AMP self-test, which would fetch an extra model,
is replaced by a constant True (AMP is known to work on the training GPU).
The best checkpoint is copied to {weights_dir}/{model}_{split}_best.pt.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("YOLO_OFFLINE", "true")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trackwood import config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="yolo11n | yolo11s | yolo11m")
    ap.add_argument("--split", default="random", help="random | blocked")
    ap.add_argument("--imgsz", type=int, default=896)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2, help="data-loader processes (each loads torch; keep low when virtual memory is tight)")
    args = ap.parse_args()

    import ultralytics.engine.trainer as trainer_mod
    trainer_mod.check_amp = lambda model: True
    from ultralytics import YOLO

    paths = config.load_paths()
    data = paths["work_dir"] / "datasets" / f"yolo_{args.split}" / "data.yaml"
    project = paths["work_dir"] / "train" / "yolo"
    name = f"{args.model}_{args.split}"
    model = YOLO(str(paths["weights_dir"] / f"{args.model}.pt"))
    model.train(data=str(data), imgsz=args.imgsz, epochs=args.epochs, patience=args.patience,
                batch=args.batch, seed=args.seed, deterministic=True, workers=args.workers, device=0,
                project=str(project), name=name, exist_ok=True, plots=False, verbose=True)
    best = project / name / "weights" / "best.pt"
    dst = paths["weights_dir"] / f"{name}_best.pt"
    shutil.copy2(best, dst)
    res = YOLO(str(dst)).val(data=str(data), imgsz=args.imgsz, split="test", batch=8, plots=False,
                             project=str(project), name=f"{name}_test", exist_ok=True, verbose=False)
    summary = {"model": args.model, "split": args.split, "imgsz": args.imgsz,
               "test_map50_95": float(res.box.map), "test_map50": float(res.box.map50),
               "test_precision": float(res.box.mp), "test_recall": float(res.box.mr),
               "weights": dst.name}
    with open(project / name / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
