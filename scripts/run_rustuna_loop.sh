#!/bin/bash
# Rustuna tuning of the loop-closure / scale / keyframe stage.
# Tunes on desk+desk2 (fast) and verifies the best config on desk/desk2/room.
# Usage: run_rustuna_loop.sh <state_dir>
set -u

STATE_DIR="$1"
REPO=/home/ubuntu/ai-assistant-workspace/onnx_image_processing
cd "$REPO" || exit 1
PY="$REPO/.venv/bin/python"
CACHE=eval/results/tune_cache_loop

echo $$ > "$STATE_DIR/pid"
: > "$STATE_DIR/task.log"

build() {  # build <seq>
  if [ ! -f "$CACHE/$1.npz" ]; then
    echo "[$(date +%H:%M:%S)] build cache $1" >> "$STATE_DIR/task.log"
    "$PY" eval/rustuna_tune_loop.py --build-cache --seq "$1" \
      >> "$STATE_DIR/task.log" 2>&1
  fi
}
build desk
build desk2
build room

echo "[$(date +%H:%M:%S)] START study" >> "$STATE_DIR/task.log"
"$PY" eval/rustuna_tune_loop.py --seq desk,desk2 --n-trials 24 --tune-iterations 20 \
  --out eval/results/rustuna_tune_loop.json >> "$STATE_DIR/task.log" 2>&1
echo "study rc=$?" >> "$STATE_DIR/rcs"

echo "[$(date +%H:%M:%S)] START verify" >> "$STATE_DIR/task.log"
"$PY" - <<'PY' >> "$STATE_DIR/task.log" 2>&1
import importlib.util, sys, json
sys.path.insert(0, "eval")
s = importlib.util.spec_from_file_location("rtl", "eval/rustuna_tune_loop.py")
m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
from vo.sinkhorn_numpy import NumpySinkhornMatcher
cam = m.CameraIntrinsics(fx=525, fy=525, cx=320, cy=240, width=640, height=480)
dm = NumpySinkhornMatcher(iterations=20, epsilon=0.05, unused_score=1.0, distance_type="l2")
res = json.load(open("eval/results/rustuna_tune_loop.json"))
p = dict(res["best_params"]); p["loop_iterations"] = 60
p.setdefault("loop_rot_only", False); p.setdefault("cycle_threshold_deg", 0.0)
out = {}
for seq in ["desk", "desk2", "room"]:
    c = m.load_cache("eval/results/tune_cache_loop", seq)
    r = m.eval_seq(c, p, cam, dm)
    out[seq] = {"ATE_median": r["ATE_median"], "n_loop": r["n_loop"], "n_kf": r["n_kf"]}
    print(seq, round(r["ATE_median"], 4), "nloop", r["n_loop"], "nKF", r["n_kf"], flush=True)
json.dump({"best_params": res["best_params"], "tune_best_value": res["best_value"],
           "verify": out}, open("eval/results/rustuna_loop_best.json", "w"), indent=2)
print("wrote eval/results/rustuna_loop_best.json")
PY

echo "[$(date +%H:%M:%S)] END verify" >> "$STATE_DIR/task.log"
echo 0 > "$STATE_DIR/exit"
