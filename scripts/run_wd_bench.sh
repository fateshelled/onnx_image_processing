#!/bin/bash
# Chain: wait for the reference descloop job to finish, then run the
# --with-descriptors pair-model cache path and compare.
state_dir="$1"
REPO=/home/ubuntu/ai-assistant-workspace/onnx_image_processing
while [ ! -f "$state_dir/exit" ]; do sleep 30; done
cd "$REPO"
.venv/bin/python eval/eval_tum_vo.py vo --model eval/pyramid_k512_l2_wd.onnx \
  --seq room --stride 2 --method magsac --threshold 1.4 --dbin 0.1 \
  --loop-closure --out eval/results/loop_wd_model_room.json
echo "$?" > "$state_dir/exit2"
 cat > "$state_dir/compare.py" << 'EOF_CMP'
import json
rows = {}
for name in ["loop_pair_model_room", "loop_desc_model_room", "loop_wd_model_room"]:
    try:
        d = json.load(open(f"eval/results/{name}.json"))
        r = d[0] if isinstance(d, list) else d
        rows[name] = (r.get("n_loop"), round(r.get("mean_inlier_ratio",0),4),
                      round(r.get("ATE_median",0),4), r.get("n_ok"))
    except Exception as e:
        rows[name] = f"missing ({e})"
for k, v in rows.items():
    print(k, "n_loop/inl/ATE_med/ok =", v)
EOF_CMP
cd "$REPO" && .venv/bin/python "$state_dir/compare.py" > "$state_dir/compare.txt" 2>&1
echo 9 > "$state_dir/exit2"
