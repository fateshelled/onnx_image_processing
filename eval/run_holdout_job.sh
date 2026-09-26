#!/bin/bash
# [10] Out-of-sample validation of the tuned config on held-out TUM fr1 sequences
# (360, xyz). Downloads the sequences if missing, builds match caches, then runs
# the same sensitivity grid as the in-sample one.
set -u
cd /home/ubuntu/ai-assistant-workspace/onnx_image_processing
PY=.venv/bin/python
MODEL=eval/pyramid_k512_l2.onnx
DS=/home/ubuntu/datasets/tum_rgbd
OUT=eval/results

for SEQ in 360 xyz; do
  DIR=$DS/rgbd_dataset_freiburg1_${SEQ}
  if [ ! -f "$DIR/rgb.txt" ]; then
    echo "=== download fr1_${SEQ} $(date) ==="
    curl -L --fail --retry 2 -o /tmp/opencode/rgbd_dataset_freiburg1_${SEQ}.tgz \
      "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_${SEQ}.tgz" || exit 1
    tar -xzf /tmp/opencode/rgbd_dataset_freiburg1_${SEQ}.tgz -C $DS || exit 1
    rm -f /tmp/opencode/rgbd_dataset_freiburg1_${SEQ}.tgz
    echo "=== extracted $(date) ==="
  fi
  if [ ! -f $OUT/tune_cache/${SEQ}.npz ]; then
    echo "=== cache build fr1_${SEQ} $(date) ==="
    $PY eval/rustuna_tune.py --model $MODEL --seq $SEQ --stride 2 --build-cache || exit 1
  fi
  echo "=== sensitivity on fr1_${SEQ} $(date) ==="
  $PY eval/param_sensitivity.py --model $MODEL --seq $SEQ \
    --out $OUT/sensitivity_holdout_${SEQ}.json || exit 1
done
echo "=== done $(date) ==="
