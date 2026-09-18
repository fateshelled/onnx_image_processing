#!/bin/bash
# Room-objective tuning with odom_ref=kf and kf_mode=motion fixed, to see
# whether the long-baseline additive edges can be rescued for room.
# Usage: run_rustuna_room_kf.sh <state_dir>
set -u

STATE_DIR="$1"
REPO=/home/ubuntu/ai-assistant-workspace/onnx_image_processing
cd "$REPO" || exit 1
PY="$REPO/.venv/bin/python"

echo $$ > "$STATE_DIR/pid"
: > "$STATE_DIR/rcs"
: > "$STATE_DIR/task.log"

if [ ! -f eval/results/tune_cache_loop/room.npz ]; then
  echo "[$(date +%H:%M:%S)] build room cache" >> "$STATE_DIR/task.log"
  "$PY" eval/rustuna_tune_loop.py --build-cache --seq room >> "$STATE_DIR/task.log" 2>&1
fi

echo "[$(date +%H:%M:%S)] START room-kf study" >> "$STATE_DIR/task.log"
# Resume: a killed RUNNING trial is stale; count only COMPLETE trials toward
# the 36-trial budget so the study finishes at 36 valid trials.
REMAINING=$("$PY" - <<'PY'
import sqlite3
c = sqlite3.connect("eval/results/rustuna_room_kf.db")
c.execute("update trials set state='FAIL' where state='RUNNING'")
c.commit()
done = c.execute("select count(*) from trials where state='COMPLETE'").fetchone()[0]
print(max(0, 36 - done))
PY
)
echo "[$(date +%H:%M:%S)] resume: ${REMAINING} trials remaining" >> "$STATE_DIR/task.log"
export TORCH_THREADS=4
if [ "$REMAINING" -gt 0 ]; then
  TRIAL_ARGS="--n-trials $REMAINING"
else
  TRIAL_ARGS="--report-only"
fi
"$PY" eval/rustuna_tune_loop.py --seq room --fix-odom-ref kf --fix-kf-mode motion \
  $TRIAL_ARGS --tune-iterations 10 --matcher torch \
  --storage eval/results/rustuna_room_kf.db --study-name room_kf \
  --out eval/results/rustuna_tune_loop_room_kf.json >> "$STATE_DIR/task.log" 2>&1
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
res = json.load(open("eval/results/rustuna_tune_loop_room_kf.json"))
p = dict(res["best_params"]); p["loop_iterations"] = 60
p["odom_ref"] = "kf"; p["kf_mode"] = "motion"
p.setdefault("loop_rot_only", False); p.setdefault("cycle_threshold_deg", 0.0)
verify = {}
for seq in ["desk", "desk2", "room"]:
    c = m.load_cache("eval/results/tune_cache_loop", seq)
    r = m.eval_seq(c, p, cam, dm)
    verify[seq] = {"ATE_median": r["ATE_median"], "n_loop": r["n_loop"], "n_kf": r["n_kf"]}
    print(seq, round(r["ATE_median"], 4), "nloop", r["n_loop"], "nKF", r["n_kf"], flush=True)
out = {"best_params": res["best_params"], "tune_best_value": res["best_value"], "verify": verify}
try:
    out["prev_room_best"] = json.load(open("eval/results/rustuna_room_best.json"))["verify"]
except Exception:
    pass
json.dump(out, open("eval/results/rustuna_room_kf_best.json", "w"), indent=2)
print("wrote eval/results/rustuna_room_kf_best.json")
PY

echo "[$(date +%H:%M:%S)] END verify" >> "$STATE_DIR/task.log"
echo 0 > "$STATE_DIR/exit"
