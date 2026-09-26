#!/bin/bash
# Room-only Rustuna tuning (objective = room ATE_sim3_med), verify on
# desk/desk2. Waits for the desk+desk2 study to finish first.
# Usage: run_rustuna_room.sh <state_dir>
set -u

STATE_DIR="$1"
REPO=/home/ubuntu/ai-assistant-workspace/onnx_image_processing
cd "$REPO" || exit 1
PY="$REPO/.venv/bin/python"

echo $$ > "$STATE_DIR/pid"
: > "$STATE_DIR/task.log"

# Wait for the desk+desk2 study (state rustuna_loop) to finish.
echo "[$(date +%H:%M:%S)] waiting for rustuna_loop to finish" >> "$STATE_DIR/task.log"
for _ in $(seq 1 360); do
  [ -f eval/results/rustuna_loop/exit ] && break
  sleep 30
done
echo "[$(date +%H:%M:%S)] rustuna_loop done (or timeout)" >> "$STATE_DIR/task.log"

echo "[$(date +%H:%M:%S)] START room study" >> "$STATE_DIR/task.log"
"$PY" eval/rustuna_tune_loop.py --seq room --n-trials 12 --tune-iterations 10 \
  --out eval/results/rustuna_tune_loop_room.json >> "$STATE_DIR/task.log" 2>&1
echo "room study rc=$?" >> "$STATE_DIR/rcs"

echo "[$(date +%H:%M:%S)] START verify" >> "$STATE_DIR/task.log"
"$PY" - <<'PY' >> "$STATE_DIR/task.log" 2>&1
import importlib.util, sys, json
sys.path.insert(0, "eval")
s = importlib.util.spec_from_file_location("rtl", "eval/rustuna_tune_loop.py")
m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
from vo.sinkhorn_numpy import NumpySinkhornMatcher
cam = m.CameraIntrinsics(fx=525, fy=525, cx=320, cy=240, width=640, height=480)
dm = NumpySinkhornMatcher(iterations=20, epsilon=0.05, unused_score=1.0, distance_type="l2")
res = json.load(open("eval/results/rustuna_tune_loop_room.json"))
p = dict(res["best_params"]); p["loop_iterations"] = 60
p.setdefault("loop_rot_only", False); p.setdefault("cycle_threshold_deg", 0.0)
verify = {}
for seq in ["desk", "desk2", "room"]:
    c = m.load_cache("eval/results/tune_cache_loop", seq)
    r = m.eval_seq(c, p, cam, dm)
    verify[seq] = {"ATE_median": r["ATE_median"], "n_loop": r["n_loop"], "n_kf": r["n_kf"]}
    print(seq, round(r["ATE_median"], 4), "nloop", r["n_loop"], "nKF", r["n_kf"], flush=True)
out = {"best_params": res["best_params"], "tune_best_value": res["best_value"], "verify": verify}
try:
    dd = json.load(open("eval/results/rustuna_tune_loop.json"))
    out["desk2_objective_best_params"] = dd["best_params"]
except Exception:
    pass
json.dump(out, open("eval/results/rustuna_room_best.json", "w"), indent=2)
print("wrote eval/results/rustuna_room_best.json")
PY

echo "[$(date +%H:%M:%S)] END verify" >> "$STATE_DIR/task.log"
echo 0 > "$STATE_DIR/exit"
