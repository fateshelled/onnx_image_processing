#!/bin/bash
# Full-scale loop-closure eval: wd model (descriptor cache path), candidate A,
# production defaults (stride 2, decim 8, min_gap 30, min_inlier 0.4).
state_dir="$1"
cat > "$state_dir/task.sh" << 'EOF_TASK'
set -x
REPO=/home/ubuntu/ai-assistant-workspace/onnx_image_processing
cd "$REPO"
for seq in desk desk2 room; do
  .venv/bin/python eval/eval_tum_vo.py vo \
    --model eval/pyramid_k512_l2_wd.onnx \
    --seq $seq --stride 2 --method magsac --threshold 1.4 --dbin 0.1 \
    --loop-closure \
    --out eval/results/loop_wd_$seq.json
  echo "$?" >> "$state_dir/rcs"
done
echo 0 > "$state_dir/exit"
cat > "$state_dir/compare.py" << 'EOF_CMP'
import json
base = json.load(open("eval/results/loop_baseline_candidateA.json"))
rows = {}
for r in base:
    rows.setdefault(r["seq"], {})["baseline"] = r
for seq in ["desk", "desk2", "room"]:
    try:
        r = json.load(open(f"eval/results/loop_wd_{seq}.json"))[0]
        rows.setdefault(seq, {})["loop_wd"] = r
    except Exception as e:
        print(seq, "missing:", e)
print(f"{'seq':8} {'config':16} {'ATE_med':>8} {'inl':>6} {'ok':>9} {'n_loop':>6}")
for seq, ds in rows.items():
    for name, r in ds.items():
        print(f"{seq:8} {name:16} {r['ATE_median']:8.3f} {r['mean_inlier_ratio']:6.3f} "
              f"{r['ok_rate']:9.3f} {r.get('n_loop', 0):6d}")
EOF_CMP
cd "$REPO" && .venv/bin/python "$state_dir/compare.py" > "$state_dir/compare.txt" 2>&1
EOF_TASK

echo $$ > "$state_dir/pid"
bash "$state_dir/task.sh" > "$state_dir/task.log" 2>&1
