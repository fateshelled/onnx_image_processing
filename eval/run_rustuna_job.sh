#!/bin/bash
# [9] Rustuna matching/pose hyperparameter tuning job.
# 1) build match cache (ONNX once per pair)
# 2) equivalence check of the cache evaluator on known configs [1]/[8]
# 3) Rustuna TPE study (100 trials)
# 4) verify best params through the real eval_tum_vo.py pipeline
set -u
cd /home/ubuntu/ai-assistant-workspace/onnx_image_processing
PY=.venv/bin/python
MODEL=eval/pyramid_k512_l2.onnx
OUT=eval/results

echo "=== [9.1] cache build $(date) ==="
if [ -f $OUT/tune_cache/desk.npz ] && [ -f $OUT/tune_cache/desk2.npz ] && [ -f $OUT/tune_cache/room.npz ]; then
    echo "cache exists, skip"
else
    $PY eval/rustuna_tune.py --model $MODEL --seq all --stride 2 --build-cache || exit 1
fi

echo "=== [9.2] cache equivalence check $(date) ==="
$PY eval/rustuna_tune.py --model $MODEL --seq all --stride 2 --check || exit 1

echo "=== [9.3] Rustuna study $(date) ==="
$PY eval/rustuna_tune.py --model $MODEL --seq all --stride 2 \
    --n-trials 100 --seed 42 --out $OUT/rustuna_tune.json || exit 1

echo "=== [9.4] verify best params in real pipeline $(date) ==="
$PY - <<'EOF' > eval/results/rustuna_best_flags.txt
import json
p = json.load(open("eval/results/rustuna_tune.json"))["best_params"]
flags = ["--method", p["method"], "--threshold", str(p["threshold"]),
         "--dbin", str(p["dbin"])]
if p["guided_inlier_thresh"] > 0:
    flags += ["--guided", "--guided-inlier-thresh", str(p["guided_inlier_thresh"]),
              "--guided-sampson", str(p["guided_sampson"])]
print(" ".join(flags))
EOF
$PY eval/eval_tum_vo.py vo --model $MODEL --seq all --stride 2 \
    $(cat $OUT/rustuna_best_flags.txt) --out $OUT/rustuna_best_verify.json || exit 1

echo "=== done $(date) ==="
