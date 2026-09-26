#!/usr/bin/env bash
# One-factor ablation around online_bounded_kf trial 5 (scale_kf off):
# which knob drives desk 0.183 / desk2 0.215 / room 0.316?
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ulimit -v 18874368
export TORCH_THREADS=4
OUT=eval/results/online_graph_ab/t5_ablation.log
: > "$OUT"
T5=$(cat eval/results/online_graph_ab/t5_params.json)
mk() { .venv/bin/python - "$1" "$T5" "$2" <<'PY'
import json, sys
ep = json.loads(sys.argv[2]); over = json.loads(sys.argv[3]); ep.update(over)
json.dump(ep, open(sys.argv[1], "w"))
PY
}
run() { # tag overrides
  mk eval/results/online_graph_ab/t5ov.json "$2"
  for s in desk desk2 room; do
    echo "### $1 $s" >> "$OUT"
    .venv/bin/python scripts/quick_eval_scale.py --seq "$s" --scale-kf 0 \
      --overrides "$(cat eval/results/online_graph_ab/t5ov.json)" >> "$OUT" 2>&1
  done
}
run t5                 '{}'
run t5_kf3             '{"max_keyframes": 3}'
run t5_gop0            '{"global_opt_period": 0}'
run t5_kf3_gop0        '{"max_keyframes": 3, "global_opt_period": 0}'
echo "ALL DONE" >> "$OUT"
