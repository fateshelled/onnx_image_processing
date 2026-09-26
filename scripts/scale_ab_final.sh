#!/usr/bin/env bash
set -u
cd /home/ubuntu/ai-assistant-workspace/onnx_image_processing
OUT=eval/results/online_graph_ab/scale_ab_final.log
: > "$OUT"
for kf in 0 1; do
  for s in desk desk2 room; do
    .venv/bin/python scripts/quick_eval_scale.py --seq "$s" --scale-kf "$kf" >> "$OUT" 2>&1
  done
done
