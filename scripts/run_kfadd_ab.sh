#!/bin/bash
# Additive keyframe-edge evaluation (gate off): consecutive chain + per-frame
# edge to the last keyframe, alone and with loop closure + per-edge scale.
# Also emits a 5-way comparison (baseline/decim/motion/kfref/kfadd).
# Usage: run_kfadd_ab.sh <state_dir>
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
    --out "eval/results/kfadd_${cfg}_${seq}.json" "$@" \
    >> "$STATE_DIR/task.log" 2>&1
  echo "$cfg $seq rc=$?" >> "$STATE_DIR/rcs"
  echo "[$(date +%H:%M:%S)] END   $cfg $seq" >> "$STATE_DIR/task.log"
}

for seq in desk desk2 room; do
  run plain "$seq" --odom-ref kf --kf-rot-thresh 30 \
      --edge-scale --scale-prior-sigma 0.5
done
for seq in desk desk2 room; do
  run loop "$seq" --odom-ref kf --kf-rot-thresh 30 \
      --edge-scale --scale-prior-sigma 0.5 \
      --loop-closure --loop-window 40 --loop-min-gap 30 --loop-temporal-k 1
done

cat > "$STATE_DIR/compare.py" << 'EOF_CMP'
import json
def L(p):
    try: return json.load(open(p))[0]
    except Exception: return None
base = {r["seq"]: r for r in json.load(open("eval/results/loop_baseline_candidateA.json"))}
src = {
    "decim":  {s: f"eval/results/scale_all_{s}.json" for s in ["desk", "desk2", "room"]},
    "motion": {"desk": "eval/results/kfmotion_matched.json",
               "desk2": "eval/results/kfmotion_desk2.json",
               "room": "eval/results/kfmotion_room.json"},
    "kfref":  {s: f"eval/results/kfref_plain_{s}.json" for s in ["desk", "desk2", "room"]},
    "kfadd":  {s: f"eval/results/kfadd_plain_{s}.json" for s in ["desk", "desk2", "room"]},
    "kfadd_loop": {s: f"eval/results/kfadd_loop_{s}.json" for s in ["desk", "desk2", "room"]},
}
print(f"{'seq':6} {'cfg':11} {'ATEmed':>8} {'metric':>8} {'ok':>6} {'nKF':>4} {'nloop':>6}")
for seq in ["desk", "desk2", "room"]:
    rows = [("baseline", base.get(seq))]
    for cfg in ["decim", "motion", "kfref", "kfadd", "kfadd_loop"]:
        rows.append((cfg, L(src[cfg][seq])))
    for name, r in rows:
        if not r:
            print(f"{seq:6} {name:11} {'-':>8}")
            continue
        print(f"{seq:6} {name:11} {r['ATE_median']:8.3f} {r.get('ATE_metric_median',0):8.3f} "
              f"{r.get('ok_rate',0):6.3f} {str(r.get('n_keyframes') or '-'):>4} {r.get('n_loop',0):6d}")
EOF_CMP

"$PY" "$STATE_DIR/compare.py" > "$STATE_DIR/compare.txt" 2>&1
echo 0 > "$STATE_DIR/exit"
