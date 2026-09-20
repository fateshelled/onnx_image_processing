#!/bin/bash
# Wait for the TUM fr2/fr3 download to finish, build feature caches for the
# fr1/fr2/fr3 sequences, then run the cross-dataset adaptive-gate check.
# Usage: run_cross_dataset_eval.sh <state_dir>
set -u

STATE_DIR="$1"
REPO=/home/ubuntu/ai-assistant-workspace/onnx_image_processing
DL_STATE=/home/ubuntu/datasets/_dl_state
cd "$REPO" || exit 1
PY="$REPO/.venv/bin/python"

echo $$ > "$STATE_DIR/pid"
: > "$STATE_DIR/log"
: > "$STATE_DIR/rcs"

SEQ="360 xyz freiburg2_desk freiburg2_xyz freiburg2_rpy freiburg3_long_office_household freiburg3_sitting_xyz"

echo "[$(date +%H:%M:%S)] waiting for download ($DL_STATE/exit)" >> "$STATE_DIR/log"
for _ in $(seq 1 720); do
  [ -f "$DL_STATE/exit" ] && break
  sleep 60
done
echo "[$(date +%H:%M:%S)] download wait done (rcs: $(cat "$DL_STATE/rcs" 2>/dev/null | tr '\n' ' '))" >> "$STATE_DIR/log"

for s in $SEQ; do
  if [ ! -f "eval/results/tune_cache_loop/$s.npz" ]; then
    echo "[$(date +%H:%M:%S)] build-cache $s" >> "$STATE_DIR/log"
    "$PY" eval/rustuna_tune_loop.py --build-cache --seq "$s" --auto-intrinsics \
      >> "$STATE_DIR/log" 2>&1
    echo "build $s rc=$?" >> "$STATE_DIR/rcs"
  fi
done

echo "[$(date +%H:%M:%S)] cross-dataset A/B" >> "$STATE_DIR/log"
TORCH_THREADS=4 "$PY" scripts/ab_cross_dataset.py >> "$STATE_DIR/log" 2>&1
echo "ab rc=$?" >> "$STATE_DIR/rcs"

echo "[$(date +%H:%M:%S)] DONE" >> "$STATE_DIR/log"
echo 0 > "$STATE_DIR/exit"
