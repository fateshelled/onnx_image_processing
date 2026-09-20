#!/bin/bash
# Re-tune the room objective WITH the linear-KF scale pre-pass enabled
# (the earlier room-best was tuned without KF). Adaptive gate on, loop_min_*
# fixed (proven ineffective for room) to reduce the search space.
# Usage: run_rustuna_room_kf_tune.sh <state_dir> [n_trials]
set -u

STATE_DIR="$1"
TRIALS="${2:-100}"
REPO=/home/ubuntu/ai-assistant-workspace/onnx_image_processing
cd "$REPO" || exit 1
PY="$REPO/.venv/bin/python"

echo $$ > "$STATE_DIR/pid"
: > "$STATE_DIR/rcs"
: > "$STATE_DIR/log"

export TORCH_THREADS=4
echo "[$(date +%H:%M:%S)] START room KF tuning (${TRIALS} trials)" >> "$STATE_DIR/log"
"$PY" eval/rustuna_tune_loop.py --seq room --fix-odom-ref kf --fix-kf-mode motion \
  --scale-kf --scale-kf-adapt-loop-ratio 0.15 \
  --scale-kf-q 1e-3 --scale-kf-r 0.1 --scale-kf-sigma 0.5 \
  --fix-loop-min-gap 30 --fix-loop-min-inlier 0.4 \
  --n-trials "$TRIALS" --tune-iterations 10 --matcher torch \
  --storage eval/results/room_kf_tuned.db --study-name room_kf_tuned \
  --out eval/results/rustuna_tune_loop_room_kf_tuned.json >> "$STATE_DIR/log" 2>&1
echo "study rc=$?" >> "$STATE_DIR/rcs"

echo "[$(date +%H:%M:%S)] START verify (numpy, iterations=60)" >> "$STATE_DIR/log"
"$PY" - <<'PY' >> "$STATE_DIR/log" 2>&1
import importlib.util, sys, json
sys.path.insert(0, "eval")
s = importlib.util.spec_from_file_location("rtl", "eval/rustuna_tune_loop.py")
m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
from eval_tum_vo import intrinsics_for
from vo.pose_estimation import CameraIntrinsics
from vo.sinkhorn_numpy import NumpySinkhornMatcher
res = json.load(open("eval/results/rustuna_tune_loop_room_kf_tuned.json"))
p = dict(res["best_params"]); p["loop_iterations"] = 60
matcher = NumpySinkhornMatcher(iterations=20, epsilon=0.05, unused_score=1.0,
                               distance_type="l2")
verify = {}
for seq in ["desk", "desk2", "room"]:
    fx, fy, cx, cy = intrinsics_for("/home/ubuntu/datasets/tum_rgbd", seq,
                                    (525., 525., 320., 240.))
    cam = CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=640, height=480)
    c = m.load_cache("eval/results/tune_cache_loop", seq)
    r = m.eval_seq(c, p, cam, matcher)
    verify[seq] = {"ATE_median": r["ATE_median"], "n_loop": r["n_loop"],
                   "n_kf": r["n_kf"]}
    print(seq, round(r["ATE_median"], 4), "nloop", r["n_loop"],
          "nKF", r["n_kf"], flush=True)
out = {"best_params": res["best_params"], "tune_best_value": res["best_value"],
       "loop_iterations": 60, "verify": verify}
json.dump(out, open("eval/results/room_kf_tuned_best.json", "w"), indent=2)
print("wrote eval/results/room_kf_tuned_best.json")
PY

echo "[$(date +%H:%M:%S)] END verify" >> "$STATE_DIR/log"
echo 0 > "$STATE_DIR/exit"
