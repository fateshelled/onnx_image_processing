#!/usr/bin/env bash
# scale_kf (sequential Kalman-filter scale estimation) on/off A/B.
# Two configs: "default" (SEQ_OPT1_DEFAULTS) and "tuned" (tune_online.json best).
set -u
cd /home/ubuntu/ai-assistant-workspace/onnx_image_processing
ulimit -v 18874368
export TORCH_THREADS=4

OUT=eval/results/online_graph_ab/scale_kf_ab.log
: > "$OUT"
TUNED=eval/results/online_graph_ab/scale_ab_tuned_params.json

.venv/bin/python - "$TUNED" <<'PY'
import json, sys
p = json.load(open("eval/results/tune_online.json"))["best_params"]
p.pop("scale_kf", None)  # toggled by --scale-kf, not by overrides
json.dump(p, open(sys.argv[1], "w"))
print("tuned overrides written:", sys.argv[1], flush=True)
PY

for cfg in default tuned; do
  if [ "$cfg" = default ]; then OV='{}'; else OV="$(cat "$TUNED")"; fi
  for kf in 0 1; do
    for s in desk desk2 room; do
      echo "### cfg=$cfg scale_kf=$kf seq=$s" >> "$OUT"
      .venv/bin/python scripts/quick_eval_scale.py --seq "$s" --scale-kf "$kf" \
        --overrides "$OV" >> "$OUT" 2>&1
      echo "rc=$? cfg=$cfg scale_kf=$kf seq=$s" >> "$OUT"
    done
  done
done
echo "ALL DONE" >> "$OUT"
