#!/bin/bash
# Overnight continuation of the room_kf study: many more torch trials.
# Usage: run_rustuna_room_kf_ep2.sh <state_dir> [n_trials]
set -u

STATE_DIR="$1"
TRIALS="${2:-300}"
REPO=/home/ubuntu/ai-assistant-workspace/onnx_image_processing
cd "$REPO" || exit 1
PY="$REPO/.venv/bin/python"

echo $$ > "$STATE_DIR/pid"
: > "$STATE_DIR/rcs"
: > "$STATE_DIR/task.log"

export TORCH_THREADS=4
echo "[$(date +%H:%M:%S)] START extra ${TRIALS} trials (room_kf, torch)" >> "$STATE_DIR/task.log"
"$PY" eval/rustuna_tune_loop.py --seq room --fix-odom-ref kf --fix-kf-mode motion \
  --n-trials "$TRIALS" --tune-iterations 10 --matcher torch \
  --storage eval/results/rustuna_room_kf.db --study-name room_kf \
  --out eval/results/rustuna_tune_loop_room_kf.json >> "$STATE_DIR/task.log" 2>&1
echo "study rc=$?" >> "$STATE_DIR/rcs"

echo "[$(date +%H:%M:%S)] END study" >> "$STATE_DIR/task.log"
echo 0 > "$STATE_DIR/exit"
