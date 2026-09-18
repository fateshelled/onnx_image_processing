#!/bin/bash
# Motion-based keyframe criterion vs fixed decimation, with per-edge scale.
# Usage: run_kfmotion_ab.sh <state_dir>
set -u

STATE_DIR="$1"
REPO=/home/ubuntu/ai-assistant-workspace/onnx_image_processing
cd "$REPO" || exit 1
PY="$REPO/.venv/bin/python"

echo $$ > "$STATE_DIR/pid"
: > "$STATE_DIR/rcs"
: > "$STATE_DIR/task.log"

for seq in desk2 room; do
  echo "[$(date +%H:%M:%S)] START kfmotion $seq" >> "$STATE_DIR/task.log"
  "$PY" eval/eval_tum_vo.py vo \
    --model eval/pyramid_k512_l2_wd.onnx \
    --seq "$seq" --stride 2 --method magsac --threshold 1.4 --dbin 0.1 \
    --loop-closure --loop-min-gap 30 --loop-window 40 --loop-temporal-k 1 \
    --edge-scale --scale-prior-sigma 0.5 --kf-mode motion \
    --kf-rot-thresh 30 --kf-trans-thresh 8 \
    --out "eval/results/kfmotion_${seq}.json" >> "$STATE_DIR/task.log" 2>&1
  echo "kfmotion $seq rc=$?" >> "$STATE_DIR/rcs"
  echo "[$(date +%H:%M:%S)] END   kfmotion $seq" >> "$STATE_DIR/task.log"
done

cat > "$STATE_DIR/compare.py" << 'EOF_CMP'
import json
def load(p):
    try: return json.load(open(p))[0]
    except Exception: return None
base = {r["seq"]: r for r in json.load(open("eval/results/loop_baseline_candidateA.json"))}
print(f"{'seq':7} {'cfg':14} {'ATEmed':>8} {'nKF':>5} {'nloop':>6}")
for seq in ["desk", "desk2", "room"]:
    for name, path in [("baseline", None),
                       ("decim_scale", f"eval/results/scale_all_{seq}.json"),
                       ("motion_scale", f"eval/results/kfmotion_{seq}.json")]:
        r = base.get(seq) if path is None else load(path)
        if not r:
            print(f"{seq:7} {name:14} {'-':>8}")
            continue
        print(f"{seq:7} {name:14} {r['ATE_median']:8.3f} {r.get('n_keyframes',0) or 0:5d} "
              f"{r.get('n_loop',0):6d}")
EOF_CMP

"$PY" "$STATE_DIR/compare.py" > "$STATE_DIR/compare.txt" 2>&1
echo 0 > "$STATE_DIR/exit"
