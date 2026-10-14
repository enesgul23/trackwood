"""TrackWood: video-based counting of floating wood in open-channel experiments.

Two-stage pipeline:
  1. detection (GPU, once per video and model): raw boxes and scores per frame,
     extracted at a low confidence threshold and stored in a compressed file;
  2. tracking and counting (CPU): confidence filtering, non-maximum suppression,
     Kalman/IoU tracking and virtual-line counting, re-run cheaply for every
     configuration studied.
"""

__version__ = "0.2.0"
