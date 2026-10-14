"""Build the detector training splits from the Roboflow COCO export.

All 1000 annotated images are frames 0-1998 (every second frame) of video 20040
(see results/training_source.json). Two splits are produced:

* random: the original Roboflow 700/200/100 split (adjacent frames end up in
  different subsets, so its test scores are optimistic);
* blocked: frames are grouped into 100-frame blocks; blocks are assigned
  round-robin to train/val/test and images within `guard` frames of a block of a
  different subset are dropped, so no test image has a training image nearby.

Outputs (under {work_dir}/datasets/): coco_{split}/{train,valid,test}/ with
_annotations.coco.json (RF-DETR layout) and yolo_{split}/ with images/, labels/
and data.yaml (Ultralytics layout). One class: wood.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trackwood import config  # noqa: E402

BLOCK = 100
TEST_BLOCKS = {3, 9, 15}
VAL_BLOCKS = {5, 12, 18}


def frame_index(name: str) -> int:
    return int(re.search(r"frame_(\d+)", name).group(1))


def load_all(coco_dir: Path):
    images, anns = {}, defaultdict(list)
    for split in ("train", "valid", "test"):
        with open(coco_dir / split / "_annotations.coco.json", encoding="utf-8") as f:
            d = json.load(f)
        idmap = {}
        for im in d["images"]:
            key = im["file_name"]
            images[key] = {**im, "src": coco_dir / split / key, "orig_split": split}
            idmap[im["id"]] = key
        for a in d["annotations"]:
            anns[idmap[a["image_id"]]].append(a)
    return images, anns


def blocked_split(names: list[str], guard: int) -> dict[str, str]:
    def subset(fi: int) -> str:
        b = fi // BLOCK
        return "test" if b in TEST_BLOCKS else "valid" if b in VAL_BLOCKS else "train"

    out = {}
    for n in names:
        fi = frame_index(n)
        s = subset(fi)
        near = {subset(max(0, fi - guard)), subset(fi + guard)}
        if len(near | {s}) > 1:
            continue  # within guard frames of another subset
        out[n] = s
    return out


def write_coco(dst: Path, assign: dict, images: dict, anns: dict) -> None:
    for split in ("train", "valid", "test"):
        d = dst / split
        d.mkdir(parents=True, exist_ok=True)
        ims, aa = [], []
        for i, name in enumerate(sorted(n for n, s in assign.items() if s == split)):
            im = images[name]
            shutil.copy2(im["src"], d / name)
            ims.append({k: im[k] for k in ("file_name", "height", "width")} | {"id": i})
            for a in anns[name]:
                aa.append({"id": len(aa), "image_id": i, "category_id": 1, "bbox": a["bbox"],
                           "area": a["bbox"][2] * a["bbox"][3], "iscrowd": 0, "segmentation": []})
        cats = [{"id": 0, "name": "wood", "supercategory": "none"},
                {"id": 1, "name": "wood", "supercategory": "wood"}]  # Roboflow layout expected by rfdetr
        with open(d / "_annotations.coco.json", "w", encoding="utf-8") as f:
            json.dump({"images": ims, "annotations": aa, "categories": cats}, f)


def write_yolo(dst: Path, coco_dst: Path) -> None:
    for split, ysplit in (("train", "train"), ("valid", "val"), ("test", "test")):
        with open(coco_dst / split / "_annotations.coco.json", encoding="utf-8") as f:
            d = json.load(f)
        (dst / "images" / ysplit).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / ysplit).mkdir(parents=True, exist_ok=True)
        by_img = defaultdict(list)
        for a in d["annotations"]:
            by_img[a["image_id"]].append(a["bbox"])
        for im in d["images"]:
            shutil.copy2(coco_dst / split / im["file_name"], dst / "images" / ysplit / im["file_name"])
            W, H = im["width"], im["height"]
            lines = [f"0 {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}"
                     for x, y, w, h in by_img[im["id"]] if w > 0 and h > 0]
            (dst / "labels" / ysplit / (Path(im["file_name"]).stem + ".txt")).write_text("\n".join(lines))
    (dst / "data.yaml").write_text(
        f"path: {dst.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: wood\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--guard", type=int, default=6, help="frames kept free around subset boundaries")
    args = ap.parse_args()
    paths = config.load_paths()
    images, anns = load_all(paths["coco_dir"])
    root = paths["work_dir"] / "datasets"
    summary = {}
    for name, assign in (("random", {n: im["orig_split"] for n, im in images.items()}),
                         ("blocked", blocked_split(list(images), args.guard))):
        coco_dst = root / f"coco_{name}"
        if coco_dst.exists():
            shutil.rmtree(coco_dst)
        write_coco(coco_dst, assign, images, anns)
        yolo_dst = root / f"yolo_{name}"
        if yolo_dst.exists():
            shutil.rmtree(yolo_dst)
        write_yolo(yolo_dst, coco_dst)
        summary[name] = {s: {"images": sum(1 for v in assign.values() if v == s),
                             "boxes": sum(len(anns[n]) for n, v in assign.items() if v == s)}
                         for s in ("train", "valid", "test")}
        print(name, summary[name])
    out = config.REPO / "results" / "dataset_splits.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"block_frames": BLOCK, "test_blocks": sorted(TEST_BLOCKS), "val_blocks": sorted(VAL_BLOCKS),
                   "guard_frames": args.guard, "splits": summary}, f, indent=1)


if __name__ == "__main__":
    main()
