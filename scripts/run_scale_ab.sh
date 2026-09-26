#!/bin/bash
# Per-edge scale optimization A/B: baseline / rotation-only (from earlier runs)
# vs edge-scale (all edges) vs edge-scale (loop edges only).
# Usage: run_scale_ab.sh <state_dir>
set -u

STATE_DIR="$1"
REPO=/home/ubuntu/ai-assistant-workspace/onnx_image_processing
CHANNEL=1548834903150567605
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
    --loop-closure --keyframe-decim 8 --loop-min-gap 30 --loop-window 40 \
    --loop-temporal-k 1 --out "eval/results/scale_${cfg}_${seq}.json" "$@" \
    >> "$STATE_DIR/task.log" 2>&1
  echo "$cfg $seq rc=$?" >> "$STATE_DIR/rcs"
  echo "[$(date +%H:%M:%S)] END   $cfg $seq" >> "$STATE_DIR/task.log"
}

for seq in desk desk2 room; do
  run all "$seq" --edge-scale --scale-prior-sigma 0.5
done
# Quick ablation: free only loop edges on desk.
run looponly desk --edge-scale --scale-prior-sigma 0.5 --scale-loop-only

cat > "$STATE_DIR/compare.py" << 'EOF_CMP'
import json

def load(path):
    try:
        return json.load(open(path))[0]
    except Exception:
        return None

base = {r["seq"]: r for r in json.load(open("eval/results/loop_baseline_candidateA.json"))}
rows = []
for seq in ["desk", "desk2", "room"]:
    rows.append((seq, "baseline", base.get(seq)))
    rows.append((seq, "rot_only", load(f"eval/results/rotonly_{seq}.json")))
    rows.append((seq, "scale_all", load(f"eval/results/scale_all_{seq}.json")))
rows.append(("desk", "scale_loop", load("eval/results/scale_looponly_desk.json")))

print(f"{'seq':7} {'cfg':11} {'ATEmed':>8} {'RMSE':>7} {'ok':>6} {'inl':>6} {'nloop':>6}")
for seq, name, r in rows:
    if not r:
        print(f"{seq:7} {name:11} {'-':>8}")
        continue
    print(f"{seq:7} {name:11} {r['ATE_median']:8.3f} {r['ATE_RMSE']:7.3f} "
          f"{r.get('ok_rate',0):6.3f} {r.get('mean_inlier_ratio',0):6.3f} "
          f"{r.get('n_loop',0):6d}")
EOF_CMP

"$PY" "$STATE_DIR/compare.py" > "$STATE_DIR/compare.txt" 2>&1
echo 0 > "$STATE_DIR/exit"
if command -v xangi >/dev/null 2>&1; then
  xangi tool trigger --channel "$CHANNEL" \
    --message "辺ごとの倍率最適化の比較が完了しました。compare.txt を確認してください" \
    --source scale-ab
fi
