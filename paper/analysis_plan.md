# Analysis plan

This file fixes the design of the evaluation before the final numbers are produced.
Changes are recorded in the git history.

## Measurand and reference

* **Measurand.** Number of wood pieces crossing the inlet line L1 (20 % of the frame
  width, upstream of the permeable spur) during one run.
* **Reference value.** N, the number of pieces counted by hand and released in the
  run (encoded in the file name). All N pieces enter the field of view and must cross
  L1, so N is the true value of the measurand. N is treated as exact (u(N) = 0).
* **Measurement procedure.** Detector (RF-DETR Base or YOLO11) → confidence filter →
  class-agnostic NMS → Kalman/IoU tracker → crossing of L1 by a confirmed track
  (corrected rule, see README).

## Runs

16 runs: N ∈ {10, 20, 30, 50, 80, 120, 150, 200} × discharge code Q ∈ {40, 60}, one
run per combination. Run 20040 supplied all detector training images; the primary
statistics use the **15 independent runs**, and results for all 16 are reported as a
secondary view.

## Primary endpoint

For each detector at the **nominal configuration fixed a priori** (confidence 0.25,
NMS IoU 0.6, tracker IoU gate 0.3, 3 hits to confirm, 2 s coasting, legacy tracker
settings otherwise):

* relative error (C − N)/N per run; mean (bias), mean absolute value (MAPE), standard
  deviation, Bland–Altman 95 % limits of agreement;
* counting efficiency η = ΣC/ΣN with a 95 % bootstrap interval (resampling runs).

Comparison of the detectors: paired Wilcoxon signed-rank test on |C − N|/N over the
15 runs.

## Secondary analyses

1. **Original vs corrected counting rule** (dead zone at the line) for both detectors.
2. **Calibrated measurement.** Leave-one-run-out ratio (C/k) and affine
   ((C − b)/a) calibration; errors of the calibrated counts on the held-out runs.
3. **Uncertainty budget (GUM).** Relative standard uncertainty of a calibrated count:
   between-run scatter of η (Type A, ν = n − 1), uncertainty of the calibration factor,
   re-run repeatability of the GPU inference; combined by the law of propagation,
   ν_eff by Welch–Satterthwaite, expanded uncertainty at 95 %.
4. **Robustness.** One-at-a-time variation of each processing parameter around the
   nominal value; reported as sensitivity, and as optional Type B components.
5. **Tuned procedure without optimism.** Nested leave-one-run-out choice of the
   processing configuration over a factorial grid (confidence × tracker IoU × NMS IoU);
   the held-out run never influences the choice.
6. **Line-position profile.** η for lines every 5 % of the frame width; the decrease
   downstream of the spur is wood retention, not counting error.
7. **Outlet count and trapping efficiency** R = 1 − C_L2 / C_L1. Without a reference for
   L2, the methods are compared with each other; if the pieces left between the lines at
   the end of each run are counted by hand (kit: `scripts/make_count_kit.py`),
   N_L2 = N − trapped becomes the reference for L2 and R.
8. **Detector-level metrics** on held-out images: COCO AP (maxDets 500), precision,
   recall and F1 at the operating point, PR and F1–confidence curves, latency. Two
   splits: the original random split (adjacent frames in train and test, optimistic) and
   a blocked split (100-frame blocks with 6-frame guards).
9. **Fair detector comparison.** YOLO11n/s/m retrained on exactly the RF-DETR training
   images with the same square-stretch geometry; RF-DETR retrained with a converging
   schedule; both families also trained on the blocked split.
10. **Physical scale.** Channel width B = 1.2 m against the bed-wall junctions located
    every 5 % of the frame width; surface factor k = 1.015 ± 0.015 (rectangular) from
    the extreme wood box edges; u(s)/s ≈ 0.9 %.
11. **Velocity and flux at L1.** Surface velocity of each crossing piece from a linear
    fit of its centre within 200 px of the line; per run: median, IQR, passage duration
    T90, mean flux 0.9 C / T90 and peak 1-s flux. Inter-method agreement of the
    velocities.
12. **Dead-zone model.** P(miss) = min(1, w/d) with w = 2 px and d the centre advance per
    tracker update at the line, tested per crossing (binned observed loss with 95 %
    intervals) and per run (predicted vs observed loss); generalised to
    P = min(1, w s f / v).
13. **Reference for L2 (slit-scan count).** Pieces passing L2 counted on slit-scan
    images of the line (`scripts/make_slitscan.py`). Each counted streak is recorded with
    its (frame, row) position (`results/slitscan_points.csv`). Unclear streaks are
    checked on the video, and uncertain pieces count ½ with a rectangular
    uncertainty. The procedure is validated at L1, where the truth is N. The
    combined standard uncertainty adds the RMS relative error at L1. The blind
    first-pass count is reported with the adjudicated one (after the comparison with
    the detector crossings, every change confirmed on the video). This gives
    R_ref = 1 − N_L2/N with u(R) = u(N_L2)/N.
    **Status:** counted by an AI assistant; one of the authors checked the marked images
    (2026-10-15): every certain mark was confirmed without changes, and of the six
    uncertain marks (four in 20040, two in 20060) five were confirmed and one was
    rejected, so no uncertain pieces remain. A second independent count by another
    person, for the inter-counter agreement, is still open.
14. **Validation at L2.** Detector counts against the reference (agreement statistics,
    paired Wilcoxon with Holm correction). One-to-one matching of crossing events to
    counted pieces (Hungarian assignment in frame and row, with the distance gate applied
    before the assignment) gives event-level recall and precision and classifies the
    errors: misses, extra counts, and reflections on the glass walls.
15. **Water-surface ROI (post hoc, physically defined).** Detections are kept only when
    their box centre lies between the bed-wall junction lines widened by 1.5 % of the
    bed width (k + U(k) from item 10). This was motivated by the reflections found in
    item 14 and is not tuned on counts. It is reported next to the a-priori nominal
    configuration, at L1 and L2, with the margin sensitivity (0, 0.75, 1.5, 3 and 5 %).
