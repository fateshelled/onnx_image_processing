#!/bin/bash
# odom-ref kf evaluation: per-frame matching against the last keyframe,
# alone and combined with loop closure + per-edge scale.
# Usage: run_kfref_ab.sh <state_dir>
set -u

STATE_DIR="$1"
REPO=/home/ubuntu/ai-assistant-workspace/onnx_image_processing
cd "$REPO" || exit 1
PY="$REPO/.venv/bin/python"

echo $$ > "$STATE_DIR/pid"
: > "$STATE_DIR/rcs"
: > "$STATE_DIR/task.log"

run() {  # run <cfg> <seq> <extra args...>
  local cfg="$1"; shift
  local seq="$1"; shift
  echo "[$(date +%H:%M:%S)] START $cfg $seq" >> "$STATE_DIR/task.log"
  "$PY" eval/eval_tum_vo.py vo \
    --model eval/pyramid_k512_l2_wd.onnx \
    --seq "$seq" --stride 2 --method magsac --threshold 1.4 --dbin 0.1 \
    --out "eval/results/kfref_${cfg}_${seq}.json" "$@" \
    >> "$STATE_DIR/task.log" 2>&1
  echo "$cfg $seq rc=$?" >> "$STATE_DIR/rcs"
  echo "[$(date +%H:%M:%S)] END   $cfg $seq" >> "$STATE_DIR/task.log"
}

for seq in desk desk2 room; do
  run plain "$seq" --odom-ref kf --kf-rot-thresh 30
  run loop "$seq" --odom-ref kf --kf-rot-thresh 30 \
      --loop-closure --loop-window 40 --loop-min-gap 30 --loop-temporal-k 1 \
      --edge-scale --scale-prior-sigma 0.5
done

cat > "$STATE_DIR/compare.py" << 'EOF_CMP'
import json
def load(p):
    try: return json.load(open(p))[0]
    except Exception: return None
base = {r["seq"]: r for r in json.load(open("eval/results/loop_baseline_candidateA.json"))}
print(f"{'seq':7} {'cfg':14} {'ATEmed':>8} {'metric':>8} {'ok':>6} {'nKF':>5} {'nloop':>6}")
for seq in ["desk", "desk2", "room"]:
    rows = [("baseline", base.get(seq)),
            ("decim_scale", load(f"eval/results/scale_all_{seq}.json")),
            ("kfref", load(f"eval/results/kfref_plain_{seq}.json")),
            ("kfref_loop", load(f"eval/results/kfref_loop_{seq}.json"))]
    for name, r in rows:
        if not r:
            print(f"{seq:7} {name:14} {'-':>8}")
            continue
        print(f"{seq:7} {name:14} {r['ATE_median']:8.3f} "
              f"{r.get('ATE_metric_median',0):8.3f} {r.get('ok_rate',0):6.3f} "
              f"{r.get('n_keyframes',0) or 0:5d} {r.get('n_loop',0):6d}")
EOF_CMP

"$PY" "$STATE_DIR/compare.py" > "$STATE_DIR/compare.txt" 2>&1
echo 0 > "$STATE_DIR/exit"
