#!/usr/bin/env python3
"""Room error decomposition: is rotation drift the dominant ATE cause?

Runs the room odometry (stride 2, candidate A) and records each accepted
pair's estimated (R, t) plus the GT relative rotation. Then integrates three
trajectories:

  est          : estimated R and unit t  (the baseline)
  rot_oracle   : GT relative rotation, estimated unit t   (rotation error removed)
  trans_oracle : estimated R, GT translation direction     (translation error removed)

Aligning each with a similarity transform to the GT positions and comparing
the median ATE shows which error source drives the position drift.
"""

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path("/home/ubuntu/ai-assistant-workspace/onnx_image_processing")
sys.path.insert(0, str(REPO))
import onnxruntime as ort  # noqa: E402

spec = importlib.util.spec_from_file_location("et", REPO / "eval/eval_tum_vo.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

DATASET = Path("/home/ubuntu/datasets/tum_rgbd")
SEQ = sys.argv[1] if len(sys.argv) > 1 else "room"
STRIDE = 2

sess = ort.InferenceSession(str(REPO / "eval/pyramid_k512_l2_wd.onnx"),
                            providers=["CPUExecutionProvider"])
cam = m.CameraIntrinsics(fx=525, fy=525, cx=320, cy=240, width=640, height=480)
base = DATASET / f"rgbd_dataset_freiburg1_{SEQ}"
gt_poses = [(ts, m.quat_to_se3(v[:3], v[3:7]))
            for ts, v in m.read_tum_file(base / "groundtruth.txt")]
frames = []
for ts, rel in m.read_path_file(base / "rgb.txt"):
    gm = m.nearest_timestamp(gt_poses, ts, 0.05)
    if gm is not None:
        frames.append((ts, base / rel, gm[1]))

args = m.argparse.Namespace(
    pose_source="essential", method="magsac", threshold=1.4, max_matches=1024,
    dbin=0.1, match_threshold=0.1, guided=False, guided_inlier_thresh=0.35,
    guided_sampson=2.0, depth_scale=5000.0, min_depth=0.1, max_depth=10.0)

pairs = {}
n_pairs = 0
t0 = time.time()
for i in range(0, len(frames) - STRIDE, STRIDE):
    n_pairs += 1
    _, pa, ga = frames[i]
    _, pb, gb = frames[i + STRIDE]
    res = m.estimate_pair(sess, str(pa), str(pb), cam, args, depth_path1=None)
    if res and res.get("ok"):
        pairs[(i, i + STRIDE)] = (np.asarray(res["R"], float),
                                  np.asarray(res["t"], float).reshape(3), ga, gb)
    if n_pairs % 100 == 0:
        print(f"  {n_pairs} pairs, ok={len(pairs)}, {time.time()-t0:.0f}s",
              flush=True)
print(f"pairs ok {len(pairs)}/{n_pairs} in {time.time()-t0:.0f}s", flush=True)


def integrate(rot_mode, trans_mode):
    traj = m.Trajectory()
    positions = [traj.get_current_position().copy()]
    gt_pos = [frames[0][2][:3, 3].copy()]
    R_est, R_gt = [], []
    for i in range(0, len(frames) - STRIDE, STRIDE):
        j = i + STRIDE
        if (i, j) in pairs:
            R, t, ga, gb = pairs[(i, j)]
            R_g = gb[:3, :3].T @ ga[:3, :3]
            t_g = gb[:3, :3].T @ (ga[:3, 3] - gb[:3, 3])
            n = np.linalg.norm(t_g)
            if n > 0:
                t_g = t_g / n
            R_use = R_g if rot_mode == "gt" else R
            t_use = t_g if trans_mode == "gt" else t
            traj.add_relative_pose(R_use, t_use)
            R_est.append(traj.get_current_pose()[:3, :3])
            R_gt.append(gb[:3, :3])
        positions.append(traj.get_current_position().copy())
        gt_pos.append(frames[j][2][:3, 3].copy())
    return np.array(positions), np.array(gt_pos), R_est, R_gt


def ate_med(est, gt):
    s, R, t = m.umeyama(est, gt, with_scale=True)
    aligned = s * (est @ R.T) + t
    err = np.linalg.norm(aligned - gt, axis=1)
    return (float(np.median(err)), float(np.sqrt(np.mean(err ** 2))), R)


def rot_deg(R1, R2):
    c = (np.trace(np.asarray(R1, float) @ np.asarray(R2, float).T) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


out = {"seq": SEQ, "stride": STRIDE, "n_pairs": n_pairs, "n_ok": len(pairs)}
for name, (rm, tm) in [("est", ("est", "est")),
                       ("rot_oracle", ("gt", "est")),
                       ("trans_oracle", ("est", "gt"))]:
    est, gt, Re, Rg = integrate(rm, tm)
    med, rmse, R_align = ate_med(est, gt)
    # Orientation drift after applying the alignment rotation (frame-matched).
    ori = [rot_deg(R_align @ a, b) for a, b in zip(Re, Rg)]
    out[name] = {"ATE_med": round(med, 4), "ATE_rmse": round(rmse, 4),
                 "orient_err_med_deg": round(float(np.median(ori)), 2),
                 "orient_err_max_deg": round(float(np.max(ori)), 2)}
    print(name, json.dumps(out[name]), flush=True)

# per-pair relative rotation error of the estimated relative rotations
rel = [rot_deg(R, gb[:3, :3].T @ ga[:3, :3])
       for (R, t, ga, gb) in pairs.values()]
out["rel_rot_err_deg"] = {"med": round(float(np.median(rel)), 2),
                          "max": round(float(np.max(rel)), 2),
                          "p90": round(float(np.percentile(rel, 90)), 2)}
print("rel_rot_err", json.dumps(out["rel_rot_err_deg"]), flush=True)

# per-pair translation direction error (estimated unit t vs GT unit t)
tdir = []
for (R, t, ga, gb) in pairs.values():
    t_g = gb[:3, :3].T @ (ga[:3, 3] - gb[:3, 3])
    n = np.linalg.norm(t_g)
    if n < 1e-9:
        continue
    t_g = t_g / n
    c = float(np.clip(np.dot(t, t_g) / max(np.linalg.norm(t), 1e-12), -1.0, 1.0))
    tdir.append(float(np.degrees(np.arccos(c))))
out["trans_dir_err_deg"] = {"med": round(float(np.median(tdir)), 2),
                            "max": round(float(np.max(tdir)), 2),
                            "p90": round(float(np.percentile(tdir, 90)), 2)}
print("trans_dir_err", json.dumps(out["trans_dir_err_deg"]), flush=True)

outpath = REPO / f"eval/results/diag_{SEQ}_error.json"
json.dump(out, open(outpath, "w"), indent=2)
print("wrote", outpath)
