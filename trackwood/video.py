"""Video access and experiment metadata."""
from __future__ import annotations

import queue
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: float
    n_frames: int


@dataclass(frozen=True)
class Experiment:
    """Experiment encoded in the file name as {N}{Q}: N pieces of wood released,
    Q the two-digit discharge code (e.g. 1040 -> N = 10, Q = 40)."""
    video_id: str
    n_released: int
    discharge_code: int


def parse_experiment(video_id: str) -> Experiment:
    m = re.fullmatch(r"(\d+)(\d{2})", video_id)
    if not m:
        raise ValueError(f"Video id does not follow the {{N}}{{Q}} pattern: {video_id}")
    return Experiment(video_id, int(m.group(1)), int(m.group(2)))


def list_videos(folder: Path) -> list[Path]:
    return sorted((p for p in folder.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS),
                  key=lambda p: (len(p.stem), p.stem))


def probe(path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    info = VideoInfo(path, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                     float(cap.get(cv2.CAP_PROP_FPS) or 30.0),
                     int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.release()
    return info


def iter_frames(path: Path, prefetch: int = 8) -> Iterator[np.ndarray]:
    """Yield BGR frames in order; decoding runs in a background thread so that it
    overlaps with GPU inference."""
    q: queue.Queue = queue.Queue(maxsize=prefetch)
    stop = object()

    def reader() -> None:
        cap = cv2.VideoCapture(str(path))
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                q.put(frame)
        finally:
            cap.release()
            q.put(stop)

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    while True:
        item = q.get()
        if item is stop:
            break
        yield item
    t.join()
