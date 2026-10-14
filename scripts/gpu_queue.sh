#!/usr/bin/env bash
# Sequential GPU job queue (training, then detection extraction for the new models).
# Usage: bash scripts/gpu_queue.sh <python> <log_dir>
set -u
PY="$1"
LOG="$2"
cd "$(dirname "$0")/.."
export YOLO_OFFLINE=true PYTHONWARNINGS=ignore
mkdir -p "$LOG"

run() {  # run <name> <cmd...>; skip if a .done marker exists
  local name="$1"; shift
  if [ -f "$LOG/$name.done" ]; then echo "skip $name"; return; fi
  echo "$(date '+%F %T') start $name"
  if "$@" > "$LOG/$name.log" 2>&1; then
    touch "$LOG/$name.done"; echo "$(date '+%F %T') done  $name"
  else
    echo "$(date '+%F %T') FAIL  $name (see $LOG/$name.log)"
  fi
}

for m in yolo11n yolo11s yolo11m; do
  run "train_${m}_random" "$PY" scripts/train_yolo.py --model "$m" --split random
done
for m in yolo11n yolo11s yolo11m; do
  run "extract_${m}_random" "$PY" scripts/extract_detections.py --detector "${m}_random"
done
run train_rfdetr_random  "$PY" scripts/train_rfdetr.py --split random --epochs 60
run extract_rfdetr_random "$PY" scripts/extract_detections.py --detector rfdetr_base_random
run train_rfdetr_blocked "$PY" scripts/train_rfdetr.py --split blocked --epochs 60
run train_yolo11m_blocked "$PY" scripts/train_yolo.py --model yolo11m --split blocked
run extract_rfdetr_blocked "$PY" scripts/extract_detections.py --detector rfdetr_base_blocked
run extract_yolo11m_blocked "$PY" scripts/extract_detections.py --detector yolo11m_blocked
echo "$(date '+%F %T') queue finished"
