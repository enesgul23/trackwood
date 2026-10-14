"""Storage of per-frame raw detections (one compressed .npz per video and model)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Detections:
    boxes: np.ndarray    # float32 [M, 4], xyxy in frame pixels
    scores: np.ndarray   # float32 [M]
    classes: np.ndarray  # int16 [M]
    ptr: np.ndarray      # int64 [F + 1]; frame f owns rows ptr[f]:ptr[f + 1]
    meta: dict

    @property
    def n_frames(self) -> int:
        return len(self.ptr) - 1

    def frame(self, f: int):
        a, b = self.ptr[f], self.ptr[f + 1]
        return self.boxes[a:b], self.scores[a:b], self.classes[a:b]


def save(path: Path, boxes: list, scores: list, classes: list, meta: dict) -> None:
    counts = np.array([len(s) for s in scores], dtype=np.int64)
    ptr = np.concatenate([[0], np.cumsum(counts)])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp,
        boxes=np.concatenate(boxes).astype(np.float32) if boxes else np.zeros((0, 4), np.float32),
        scores=np.concatenate(scores).astype(np.float32) if scores else np.zeros(0, np.float32),
        classes=np.concatenate(classes).astype(np.int16) if classes else np.zeros(0, np.int16),
        ptr=ptr,
        meta=np.array(json.dumps(meta)),
    )
    tmp.replace(path)


def load(path: Path) -> Detections:
    z = np.load(path, allow_pickle=False)
    return Detections(z["boxes"], z["scores"], z["classes"], z["ptr"], json.loads(str(z["meta"])))
