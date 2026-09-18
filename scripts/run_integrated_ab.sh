#!/bin/bash
# Integrated keyframe-matching + loop-closure A/B evaluation.
# Compares the baseline (candidate A, no loop closure) against the integrated
# single-pass keyframe matching / loop closure with two gate settings and a
# local-refinement ablation.
#
# Usage: run_integrated_ab.sh <state_dir>
# Writes eval/results/integr_<cfg>_<seq>.json and <state_dir>/{rcs,task.log,compare.txt,exit}.
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
    --out "eval/results/integr_${cfg}_${seq}.json" "$@" \
    >> "$STATE_DIR/task.log" 2>&1
  echo "$cfg $seq rc=$?" >> "$STATE_DIR/rcs"
  echo "[$(date +%H:%M:%S)] END   $cfg $seq" >> "$STATE_DIR/task.log"
}

for seq in desk desk2 room; do
  # Integrated, temporal gate off (new defaults, local refine on).
  run tk1 "$seq" --loop-closure --keyframe-decim 8 --loop-min-gap 30 \
      --loop-window 40 --loop-temporal-k 1
  # Integrated, temporal-k=3 + weak loop edges (loopv2-style gates).
  run tk3 "$seq" --loop-closure --keyframe-decim 8 --loop-min-gap 30 \
      --loop-window 40 --loop-temporal-k 3 --loop-sigma-scale 2.0
  # Ablation: local refinement disabled.
  run nolocal "$seq" --loop-closure --keyframe-decim 8 --loop-min-gap 30 \
      --loop-window 40 --loop-temporal-k 1 --no-local-refine
done

cat > "$STATE_DIR/compare.py" << 'EOF_CMP'
import json

base = {r["seq"]: r for r in json.load(open("eval/results/loop_baseline_candidateA.json"))}
names = ["baseline", "tk1", "tk3", "nolocal"]
rows = []
for seq in ["desk", "desk2", "room"]:
    for name in names:
        if name == "baseline":
            r = base.get(seq)
        else:
            try:
                r = json.load(open(f"eval/results/integr_{name}_{seq}.json"))[0]
            except Exception:
                r = None
        if not r:
            continue
        rows.append((seq, name, r))

print(f"{'seq':7} {'cfg':9} {'ATEmed':>8} {'RMSE':>7} {'ok':>6} {'inl':>6} "
      f"{'nloop':>6} {'nloc':>5} {'badR>20':>9}")
for seq, name, r in rows:
    le = r.get("loop_edges", [])
    bad = sum(1 for e in le if e.get("R_err_deg", 0.0) > 20.0)
    print(f"{seq:7} {name:9} {r['ATE_median']:8.3f} {r['ATE_RMSE']:7.3f} "
          f"{r['ok_rate']:6.3f} {r['mean_inlier_ratio']:6.3f} "
          f"{r.get('n_loop', 0):6d} {r.get('n_local', 0):5d} "
          f"{bad:4d}/{len(le):<4d}")
EOF_CMP

"$PY" "$STATE_DIR/compare.py" > "$STATE_DIR/compare.txt" 2>&1
echo 0 > "$STATE_DIR/exit"
if command -v xangi >/dev/null 2>&1; then
  xangi tool trigger --channel "$CHANNEL" \
    --message "統合パス A/B 評価が完了しました。state の compare.txt と eval/results/integr_*.json を確認してください" \
    --source integr-ab
fi
