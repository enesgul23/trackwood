"""Train RF-DETR Base on one of the prepared splits.

    python scripts/train_rfdetr.py --split blocked --epochs 60

Differences from the original training (TubitakFloodTraditional/train_rfdetr.py):
the COCO-pretrained weights are loaded from disk (no download); batch 2 with
8 gradient-accumulation steps keeps the effective batch at 16 while fitting in
24 GB (the original batch 8 at the 1176 px multi-scale size spilled into shared
memory); the learning rate drops 10x at 75 % of the epochs; early stopping on the
EMA validation mAP. The best (total) checkpoint is copied to
{weights_dir}/rfdetr_base_{split}_best.pt.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trackwood import config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="blocked")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--resolution", type=int, default=896)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from rfdetr import RFDETRBase

    paths = config.load_paths()
    data = paths["work_dir"] / "datasets" / f"coco_{args.split}"
    out = paths["work_dir"] / "train" / "rfdetr" / f"rfdetr_base_{args.split}"
    out.mkdir(parents=True, exist_ok=True)
    model = RFDETRBase(pretrain_weights=str(paths["weights_dir"] / "rf-detr-base.pth"),
                       resolution=args.resolution)
    model.train(dataset_dir=str(data), output_dir=str(out), epochs=args.epochs,
                batch_size=args.batch, grad_accum_steps=args.accum, lr=args.lr,
                lr_drop=int(round(0.75 * args.epochs)), resolution=args.resolution,
                early_stopping=True, early_stopping_patience=args.patience,
                early_stopping_use_ema=True, seed=args.seed, tensorboard=False,
                num_workers=2, device="cuda")
    for name in ("checkpoint_best_total.pth", "checkpoint_best_ema.pth", "checkpoint_best_regular.pth"):
        if (out / name).exists():
            shutil.copy2(out / name, paths["weights_dir"] / f"rfdetr_base_{args.split}_best.pt")
            print(f"copied {name}")
            break


if __name__ == "__main__":
    main()
