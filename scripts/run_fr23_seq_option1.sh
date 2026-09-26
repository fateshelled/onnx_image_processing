#!/bin/bash
# Resume the fr2/fr3 stride vs adaptive-gate vs seq option-1 comparison.
# Sequences already present in eval/results/fr23_seq_option1.json are skipped.
# Usage: run_fr23_seq_option1.sh <state_dir> [mem_kb]
set -u

STATE_DIR="$1"
MEM_KB="${2:-25165824}"
REPO=/home/ubuntu/ai-assistant-workspace/onnx_image_processing
cd "$REPO" || exit 1
PY="$REPO/.venv/bin/python"

mkdir -p "$STATE_DIR"
echo $$ > "$STATE_DIR/pid"
: > "$STATE_DIR/log"
: > "$STATE_DIR/rcs"

ulimit -v "$MEM_KB"
echo "[$(date +%H:%M:%S)] fr23 seq option1 resume (ulimit -v ${MEM_KB}KB)" >> "$STATE_DIR/log"
TORCH_THREADS=4 "$PY" scripts/fr23_seq_option1.py >> "$STATE_DIR/log" 2>&1
rc=$?
echo "fr23 rc=$rc" >> "$STATE_DIR/rcs"

echo "[$(date +%H:%M:%S)] DONE rc=$rc" >> "$STATE_DIR/log"
echo "$rc" > "$STATE_DIR/exit"
