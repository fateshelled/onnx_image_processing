#!/bin/bash
# Re-measure rotation-only loop closure with the fixed Levenberg-Marquardt
# solver, so it is comparable to the edge-scale runs (which used the fix).
# Usage: run_lm_recompare.sh <state_dir>
set -u

STATE_DIR="$1"
REPO=/home/ubuntu/ai-assistant-workspace/onnx_image_processing
CHANNEL=1548834903150567605
cd "$REPO" || exit 1
PY="$REPO/.venv/bin/python"

echo $$ > "$STATE_DIR/pid"
: > "$STATE_DIR/rcs"
: > "$STATE_DIR/task.log"

for seq in desk desk2 room; do
  echo "[$(date +%H:%M:%S)] START lmrot $seq" >> "$STATE_DIR/task.log"
  "$PY" eval/eval_tum_vo.py vo \
    --model eval/pyramid_k512_l2_wd.onnx \
    --seq "$seq" --stride 2 --method magsac --threshold 1.4 --dbin 0.1 \
    --loop-closure --keyframe-decim 8 --loop-min-gap 30 --loop-window 40 \
    --loop-temporal-k 1 --loop-rot-only \
    --out "eval/results/lmrot_${seq}.json" >> "$STATE_DIR/task.log" 2>&1
  echo "lmrot $seq rc=$?" >> "$STATE_DIR/rcs"
  echo "[$(date +%H:%M:%S)] END   lmrot $seq" >> "$STATE_DIR/task.log"
done

cat > "$STATE_DIR/compare.py" << 'EOF_CMP'
import json

def load(path):
    try:
        return json.load(open(path))[0]
    except Exception:
        return None

base = {r["seq"]: r for r in json.load(open("eval/results/loop_baseline_candidateA.json"))}
print(f"{'seq':7} {'cfg':13} {'ATEmed':>8} {'RMSE':>7} {'ok':>6} {'nloop':>6}")
for seq in ["desk", "desk2", "room"]:
    rows = [
        ("baseline", base.get(seq)),
        ("rot_oldLM", load(f"eval/results/rotonly_{seq}.json")),
        ("rot_newLM", load(f"eval/results/lmrot_{seq}.json")),
        ("scale_all", load(f"eval/results/scale_all_{seq}.json")),
    ]
    for name, r in rows:
        if not r:
            print(f"{seq:7} {name:13} {'-':>8}")
            continue
        print(f"{seq:7} {name:13} {r['ATE_median']:8.3f} {r['ATE_RMSE']:7.3f} "
              f"{r.get('ok_rate',0):6.3f} {r.get('n_loop',0):6d}")
EOF_CMP

"$PY" "$STATE_DIR/compare.py" > "$STATE_DIR/compare.txt" 2>&1
echo 0 > "$STATE_DIR/exit"
if command -v xangi >/dev/null 2>&1; then
  xangi tool trigger --channel "$CHANNEL" \
    --message "LM修正後の回転のみ再測定が完了しました。compare.txt を確認してください" \
    --source lm-recompare
fi
