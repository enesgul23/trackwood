# Derived results

All files are produced by the scripts; do not edit by hand.

| File | Content | Script |
| --- | --- | --- |
| `training_source.json` | Match of the detector training images to video frames (all from run 20040, frames 0–1998) | `identify_training_source.py` |
| `dataset_splits.json` | Image and box counts of the random and blocked detector splits | `prepare_datasets.py` |
| `detector_eval.json` | COCO AP, operating-point precision/recall/F1, PR and F1–confidence curves, latency per detector and split | `eval_detectors.py` |
| `counting_runs.csv` | One row per detector × processing configuration × run: L1/L2 counts with the original and corrected rules, track statistics, wood between the lines at the end | `run_counting.py`, `summarize_counting.py` |
| `counting_accuracy.csv` | Agreement with N per detector × configuration × count estimator × run subset (all 16 runs, 15 independent runs): bias, MAPE, Bland–Altman limits, CCC, bootstrap 95 % intervals, leave-one-run-out calibration | `summarize_counting.py` |
| `sensitivity_oat.csv` | Counting efficiency when one processing parameter is varied around the nominal value | `summarize_counting.py` |
| `line_position_sweep.csv` | Counts for lines every 5 % of the frame width (nominal configuration) | `summarize_counting.py` |
| `trapping_efficiency.csv` | R = 1 − C_L2 / C_L1 per run and detector | `summarize_counting.py` |
| `mass_balance_closure.csv` | C_L1 − C_L2 against the detections between the lines at the end of each run | `summarize_counting.py` |
| `summary.json` | Nominal configuration, paired detector comparisons, uncertainty budgets, re-run repeatability, nested leave-one-run-out tuning | `summarize_counting.py` |

Count estimators: `L1_legacy` (original crossing rule), `L1_fixed` (corrected rule, all
tracks), `L1_fixed_confirmed` (corrected rule, confirmed tracks; primary),
`L1_dedup15/30` (additionally merging tracks that move together within 15/30 px).
