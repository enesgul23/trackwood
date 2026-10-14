"""Detector adapters with a common interface.

Each adapter maps one BGR frame to (boxes xyxy float32 [n, 4], scores float32 [n],
classes int16 [n]) in frame pixel coordinates. Adapters run at a low confidence
threshold; the operating threshold is applied later, so a single extraction serves
every threshold studied (filtering the low-threshold output at t gives exactly the
output at t, because both YOLO NMS and RF-DETR top-k selection act before the
threshold).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass
class DetectorSpec:
    name: str                # identifier used in output paths, e.g. "rfdetr_base_v3"
    kind: str                # "yolo" or "rfdetr"
    weights: Path
    min_conf: float = 0.05   # extraction threshold
    imgsz: int | None = None  # YOLO inference size; None keeps the checkpoint default
    yolo_nms_iou: float = 0.6  # YOLO built-in NMS (legacy pipeline used 0.6)
    max_det: int = 300
    stretch: int | None = None  # YOLO: resize the frame to stretch x stretch first (training geometry)
    resolution: int = 896    # RF-DETR input resolution
    extra: dict[str, Any] = field(default_factory=dict)


class YoloAdapter:
    def __init__(self, spec: DetectorSpec, device: str | int = 0):
        from ultralytics import YOLO
        self.spec = spec
        self.model = YOLO(str(spec.weights))
        self.device = device

    def __call__(self, frame_bgr: np.ndarray):
        kw = dict(conf=self.spec.min_conf, iou=self.spec.yolo_nms_iou,
                  max_det=self.spec.max_det, verbose=False, device=self.device)
        if self.spec.imgsz:
            kw["imgsz"] = self.spec.imgsz
        img = frame_bgr
        if self.spec.stretch:
            # training images are full frames stretched to a square; reproduce that geometry
            s = int(self.spec.stretch)
            img = cv2.resize(frame_bgr, (s, s), interpolation=cv2.INTER_AREA)
        r = self.model.predict(img, **kw)[0]
        b = r.boxes
        xyxy = b.xyxy.cpu().numpy().astype(np.float32)
        if self.spec.stretch:
            h, w = frame_bgr.shape[:2]
            xyxy *= np.array([w / s, h / s, w / s, h / s], dtype=np.float32)
        return (xyxy, b.conf.cpu().numpy().astype(np.float32), b.cls.cpu().numpy().astype(np.int16))


class RfdetrAdapter:
    def __init__(self, spec: DetectorSpec, device: str = "cuda"):
        from rfdetr import RFDETRBase
        self.spec = spec
        self.model = RFDETRBase(pretrain_weights=str(spec.weights),
                                resolution=int(spec.resolution), device=device)

    def __call__(self, frame_bgr: np.ndarray):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        d = self.model.predict(rgb, threshold=self.spec.min_conf)
        if isinstance(d, list):
            d = d[0]
        if d.xyxy is None or len(d.xyxy) == 0:
            return (np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int16))
        return (np.asarray(d.xyxy, np.float32), np.asarray(d.confidence, np.float32),
                np.asarray(d.class_id, np.int16))


def build_detector(spec: DetectorSpec):
    if spec.kind == "yolo":
        return YoloAdapter(spec)
    if spec.kind == "rfdetr":
        return RfdetrAdapter(spec)
    raise ValueError(f"Unknown detector kind: {spec.kind}")
