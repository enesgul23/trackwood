"""Kalman/IoU multi-object tracker and virtual-line counting.

The tracker reproduces the original TrackWood implementation (main_eg12.py /
main_eg_13rfdetr.py) operation for operation: same state vector, matrices,
integer box conversion, Hungarian matching on 1 - IoU, track birth, ageing and
confirmation rules. Two crossing rules are evaluated side by side:

* "legacy": the original rule. A point within 1 px of the line gets side 0 and a
  crossing needs side(prev) * side(curr) < 0 for two consecutive centre updates,
  so a track whose centre lands within +-1 px of the line is never counted.
* "fixed": points on the line are skipped; a crossing is a change of the last
  non-zero side (hysteresis-free sign change). Each track counts at most once per
  line, as in the original.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


# ------------------------------------------------------------------ geometry --- #

def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.empty((len(a), len(b)))
    area1 = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area2 = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    ix1 = np.maximum(a[:, 0, None], b[:, 0])
    iy1 = np.maximum(a[:, 1, None], b[:, 1])
    ix2 = np.minimum(a[:, 2, None], b[:, 2])
    iy2 = np.minimum(a[:, 3, None], b[:, 3])
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    return inter / (area1[:, None] + area2 - inter + 1e-6)


def to_state(b) -> np.ndarray:
    x1, y1, x2, y2 = b
    w, h = x2 - x1, y2 - y1
    return np.array([x1 + w / 2, y1 + h / 2, w * h, w / (h + 1e-6)], dtype=float)


def from_state(x: np.ndarray) -> tuple[tuple[int, int, int, int], float]:
    cx, cy, s, r = (float(v) for v in np.ravel(x)[:4])
    s = max(s, 1e-6)
    r = max(r, 1e-6)
    w = math.sqrt(s * r)
    h = s / w
    return (int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2)), float(s)


# ------------------------------------------------------------------- tracker --- #

class Track:
    def __init__(self, tid: int, box, frame: int, history_len: int = 150):
        self.id = tid
        kf = KalmanFilter(dim_x=7, dim_z=4)
        kf.F = np.array([[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1],
                         [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 0, 1]], float)
        kf.H = np.array([[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]], float)
        kf.R *= 10.0
        kf.Q[4:, 4:] *= 0.01
        kf.Q[:4, :4] *= 0.01
        kf.P[4:, 4:] *= 1000.0
        kf.P *= 10.0
        kf.x[:4] = to_state(box).reshape(4, 1)
        kf.x[4:] = 0
        self.kf = kf
        self.hits = 1
        self.misses = 0
        self.start_frame = frame
        self.history: list[tuple[int, float, float]] = [(frame, float(kf.x[0, 0]), float(kf.x[1, 0]))]
        self.history_len = history_len
        self.ever_confirmed = False
        self.n_history_total = 1  # history points ever appended (history itself is trimmed)

    def predict(self):
        self.kf.predict()
        self.misses += 1
        return from_state(self.kf.x)[0]

    def update(self, box, frame: int) -> None:
        self.kf.update(to_state(box).reshape(4, 1))
        self.hits += 1
        self.misses = 0
        self.history.append((frame, float(self.kf.x[0, 0]), float(self.kf.x[1, 0])))
        self.n_history_total += 1
        if len(self.history) > self.history_len:
            self.history.pop(0)

    @property
    def center(self) -> tuple[float, float]:
        return float(self.kf.x[0, 0]), float(self.kf.x[1, 0])


class Tracker:
    def __init__(self, iou_threshold: float, max_age: int, min_hits: int):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks: list[Track] = []
        self.frame_count = 0
        self._next_id = 1
        self.born: list[Track] = []  # tracks created in the last update

    def update(self, dets: list[tuple[int, int, int, int]]) -> list[Track]:
        self.frame_count += 1
        pred = np.array([t.predict() for t in self.tracks]) if self.tracks else np.empty((0, 4))
        det = np.array(dets) if dets else np.empty((0, 4))
        unmatched_t = list(range(len(pred)))
        unmatched_d = list(range(len(det)))
        matched = []
        if len(pred) and len(det):
            cost = 1.0 - iou_matrix(pred, det)
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if cost[r, c] < 1.0 - self.iou_threshold:
                    matched.append((r, c))
                    unmatched_t.remove(r)
                    unmatched_d.remove(c)
        updated = []
        for r, c in matched:
            self.tracks[r].update(dets[c], self.frame_count)
            updated.append(self.tracks[r])
        self.born = []
        for c in unmatched_d:
            t = Track(self._next_id, dets[c], self.frame_count)
            self._next_id += 1
            self.tracks.append(t)
            self.born.append(t)
        survivors, active = [], []
        for t in self.tracks:
            if t.misses <= self.max_age:
                survivors.append(t)
                if t.hits >= self.min_hits and t.misses == 0:
                    active.append(t)
                    t.ever_confirmed = True
        self.tracks = survivors
        self.updated = updated
        return active


# ------------------------------------------------------------------ counting --- #

def _side(x: float, line_x: int) -> int:
    v = x - line_x
    return 0 if abs(v) < 1 else (1 if v > 0 else -1)


@dataclass
class LineCounter:
    """Counts crossings of a vertical line with both rules."""
    line_id: str
    x: int
    legacy_ids: set = field(default_factory=set)
    fixed_ids: set = field(default_factory=set)
    last_side: dict = field(default_factory=dict)        # track id -> last non-zero side
    last_checked: dict = field(default_factory=dict)     # track id -> len(history) when last seen
    events: list = field(default_factory=list)           # (rule, frame, track_id, cy, direction)

    def observe(self, t: Track) -> None:
        """Call once per frame for every live track (as the original code did)."""
        h = t.history
        # legacy rule: last two history points, every frame
        if len(h) >= 2 and t.id not in self.legacy_ids:
            fp, xp, _ = h[-2]
            fc, xc, yc = h[-1]
            if _side(xc, self.x) * _side(xp, self.x) < 0:
                self.legacy_ids.add(t.id)
                self.events.append(("legacy", fc, t.id, yc, 1 if xc > xp else -1))
        # fixed rule: process every history point once, skipping points on the line
        n_seen = self.last_checked.get(t.id, 0)
        total = t.n_history_total
        new = total - n_seen
        if new <= 0:
            return
        for f, xc, yc in h[-min(new, len(h)):]:
            s = _side(xc, self.x)
            if s == 0:
                continue
            prev = self.last_side.get(t.id, 0)
            if prev != 0 and s != prev and t.id not in self.fixed_ids:
                self.fixed_ids.add(t.id)
                self.events.append(("fixed", f, t.id, yc, s))
            self.last_side[t.id] = s
        self.last_checked[t.id] = total
