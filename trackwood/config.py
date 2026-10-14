"""Local paths and detector registry."""
from __future__ import annotations

import json
from pathlib import Path

from .detectors import DetectorSpec

REPO = Path(__file__).resolve().parents[1]


def load_paths(path: Path | None = None) -> dict[str, Path]:
    """Read configs/paths.local.json (machine specific, not versioned); see
    configs/paths.example.json for the expected keys."""
    path = path or REPO / "configs" / "paths.local.json"
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {k: Path(v) for k, v in raw.items()}


def load_detectors(paths: dict[str, Path], path: Path | None = None) -> dict[str, DetectorSpec]:
    path = path or REPO / "configs" / "detectors.json"
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    specs = {}
    for name, d in raw.items():
        d = dict(d)
        weights = paths["weights_dir"] / d.pop("weights")
        specs[name] = DetectorSpec(name=name, weights=weights, **d)
    return specs
