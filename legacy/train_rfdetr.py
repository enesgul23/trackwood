from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch
from rfdetr import RFDETRBase


def _assert_coco_layout(dataset_dir: Path) -> None:
    required_splits = ["train", "valid", "test"]
    missing = []
    for split in required_splits:
        split_dir = dataset_dir / split
        ann_path = split_dir / "_annotations.coco.json"
        if not split_dir.is_dir():
            missing.append(str(split_dir))
        elif not ann_path.is_file():
            missing.append(str(ann_path))
    if missing:
        joined = "\n  - " + "\n  - ".join(missing)
        raise FileNotFoundError(
            "COCO dataset layout is incomplete. Missing:\n" + joined
        )


def _find_best_checkpoint(output_dir: Path) -> Path | None:
    preferred = [
        output_dir / "checkpoint_best_total.pth",
        output_dir / "checkpoint_best.pth",
        output_dir / "best.pth",
    ]
    for path in preferred:
        if path.is_file():
            return path

    candidates = sorted(
        output_dir.glob("*.pth"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def main() -> None:
    repo_dir = Path(__file__).resolve().parent
    default_dataset = repo_dir / "FloodTubitakTraditional.v3i.coco"
    default_output = repo_dir / "runs" / "rfdetr_base_896"
    default_save = repo_dir / "rfdetr_base_896_best.pt"

    parser = argparse.ArgumentParser(
        description="Train RF-DETR on a COCO-formatted dataset."
    )
    parser.add_argument("--dataset-dir", type=Path, default=default_dataset)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--save-pt", type=Path, default=default_save)
    parser.add_argument("--resolution", type=int, default=896)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if not args.dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset not found: {args.dataset_dir}")
    _assert_coco_layout(args.dataset_dir)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Install a CUDA-enabled PyTorch build."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.save_pt.parent.mkdir(parents=True, exist_ok=True)

    model = RFDETRBase()
    model.train(
        dataset_dir=str(args.dataset_dir),
        output_dir=str(args.output_dir),
        device=args.device,
        resolution=args.resolution,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
    )

    best_ckpt = _find_best_checkpoint(args.output_dir)
    if best_ckpt is None:
        raise FileNotFoundError(
            f"No checkpoint found in output dir: {args.output_dir}"
        )

    shutil.copy2(best_ckpt, args.save_pt)
    print(f"Saved best checkpoint to: {args.save_pt}")


if __name__ == "__main__":
    main()
