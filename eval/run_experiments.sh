#!/bin/bash
set -u
cd /home/ubuntu/ai-assistant-workspace/onnx_image_processing
PY=.venv/bin/python
MODEL=eval/pyramid_k512_l2.onnx
OUT=eval/results
echo "=== start $(date) ==="
echo "--- [1] RANSAC baseline (thr=1.0, stride2) ---"
$PY eval/eval_tum_vo.py vo --model $MODEL --seq all --stride 2 --method ransac --threshold 1.0 --out $OUT/ransac_baseline.json
echo "--- [2] MAGSAC thr=2.0 (stride2) ---"
$PY eval/eval_tum_vo.py vo --model $MODEL --seq all --stride 2 --method magsac --threshold 2.0 --out $OUT/magsac_2p0.json
echo "--- [3] MAGSAC thr=1.0 (stride2) ---"
$PY eval/eval_tum_vo.py vo --model $MODEL --seq all --stride 2 --method magsac --threshold 1.0 --out $OUT/magsac_1p0.json
echo "--- [4] Guided (correct Sampson retry, R1=ransac1.0) ---"
$PY eval/eval_tum_vo.py vo --model $MODEL --seq all --stride 2 --method ransac --threshold 1.0 --guided --out $OUT/guided.json
echo "--- [5] GT-check stride sweep (2/4/8) ---"
$PY eval/eval_tum_vo.py gtcheck --model $MODEL --seq all --stride 2 --out $OUT/gtcheck_s2.json
$PY eval/eval_tum_vo.py gtcheck --model $MODEL --seq all --stride 4 --out $OUT/gtcheck_s4.json
$PY eval/eval_tum_vo.py gtcheck --model $MODEL --seq all --stride 8 --out $OUT/gtcheck_s8.json
echo "=== done $(date) ==="
