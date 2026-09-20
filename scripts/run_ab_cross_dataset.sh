#!/bin/bash
# Cross-dataset adaptive-gate A/B with the sparse solver.
# Resumes sequences already present in eval/results/ab_cross_dataset.json.
# Usage: run_ab_cross_dataset.sh <state_dir> [seq_csv]
set -u

STATE_DIR="$1"
SEQ_CSV="${2:-}"
REPO=/home/ubuntu/ai-assistant-workspace/onnx_image_processing
cd "$REPO" || exit 1
PY="$REPO/.venv/bin/python"

mkdir -p "$STATE_DIR"
echo $$ > "$STATE_DIR/pid"
: > "$STATE_DIR/log"
: > "$STATE_DIR/rcs"

ARGS=(--resume)
[ -n "$SEQ_CSV" ] && ARGS+=(--seq "$SEQ_CSV")

echo "[$(date +%H:%M:%S)] cross-dataset A/B (sparse solver) ${ARGS[*]}" >> "$STATE_DIR/log"
TORCH_THREADS=4 "$PY" scripts/ab_cross_dataset.py "${ARGS[@]}" >> "$STATE_DIR/log" 2>&1
echo "ab rc=$?" >> "$STATE_DIR/rcs"

echo "[$(date +%H:%M:%S)] DONE" >> "$STATE_DIR/log"
echo 0 > "$STATE_DIR/exit"
