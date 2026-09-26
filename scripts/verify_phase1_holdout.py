"""Holdout validation of the Phase-1 (odometry / RPE) tuning.

Phase 1 tuned the loop-disabled odometry/graph parameters on the dev sequences
``desk, desk2, room`` (study ``odom_v1``). This script evaluates the selected
parameter sets on sequences that were NEVER used for selection, with loop
closure disabled (so only the odometry is under test):

* fr2: freiburg2_desk / freiburg2_rpy / freiburg2_xyz
* fr3: freiburg3_long_office_household / freiburg3_sitting_xyz
* fr1 extras: 360 / xyz

Configs compared:

* ``default``     : shipped ``DEFAULT_PARAMS`` (loop disabled)
* ``odom_fixed``  : tuner's starting point (``ODOMETRY_FIXED``) + defaults
* ``phase1_mean`` : best by mean RPE (trial 205)
* ``phase1_robust``: best by mean+lambda*worst (trial 134)

RPE is gauge-free (global Umeyama scale fixed once). Results are written
incrementally so the job can be resumed. Read-only w.r.t. the study.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "eval"))

spec = importlib.util.spec_from_file_location(
    "rtl", REPO / "eval/rustuna_tune_loop.py")
rtl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rtl)

from eval.torch_sinkhorn import TorchSinkhornMatcher  # noqa: E402
from vo.online_graph import DEFAULT_PARAMS  # noqa: E402
from vo.pose_estimation import CameraIntrinsics  # noqa: E402

DATASET_ROOT = "/home/ubuntu/datasets/tum_rgbd"
CACHE = REPO / "eval/results/tune_cache_loop"
PHASE1_JSON = REPO / "eval/results/tune_odom.json"
SUMMARY_JSON = REPO / "eval/results/tune_odom_summary.json"
OUT = REPO / "eval/results/phase1_holdout.json"

HOLDOUT = [
    "freiburg2_desk",
    "freiburg2_rpy",
    "freiburg2_xyz",
    "freiburg3_long_office_household",
    "freiburg3_sitting_xyz",
    "360",
    "xyz",
]


ODOM_FIXED = {
    "odom_ref": "kf", "kf_mode": "motion", "keyframe_decim": 15,
    "graph_mode": "kf_prior", "tsvd_ratio": 0.0,
    "nl_reg": True, "nl_reg_c": 10.0, "nl_reg_tau": 10.0, "nl_reg_length": 1.0,
    "kf_local_map_k": 1, "kf_edge_min_inlier": 0.0, "trans_gate_deg": 0.0,
    "loop_iterations": 10, "seq_tsvd_ratio": 0.0, "scale_kf": False,
    "loop_enable": False, "loop_verifier": "none", "cycle_threshold_deg": 0.0,
}


def load_params():
    configs = {}
    base = {**DEFAULT_PARAMS,
            "loop_enable": False, "loop_verifier": "none",
            "cycle_threshold_deg": 0.0}
    configs["default"] = base
    configs["odom_fixed"] = {**ODOM_FIXED}
    phase1 = json.loads(PHASE1_JSON.read_text())
    configs["phase1_mean"] = phase1["best_params"]
    summary = json.loads(SUMMARY_JSON.read_text())
    configs["phase1_robust"] = summary["by_lambda"]["1.0"]["params"]
    return configs


def params_sha(params):
    blob = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def _atomic_pickle(path, obj):
    tmp = Path(str(path) + ".tmp")
    with tmp.open("wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def _atomic_dump(path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", default=",".join(HOLDOUT))
    ap.add_argument("--configs",
                    default="default,odom_fixed,phase1_mean,phase1_robust")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    out_path = Path(args.out)
    configs = load_params()
    shas = {name: params_sha({**p, "loop_enable": False,
                              "loop_verifier": "none"})
            for name, p in configs.items()}
    matcher = TorchSinkhornMatcher(iterations=20, epsilon=0.05,
                                   unused_score=1.0, distance_type="l2")
    result = json.loads(out_path.read_text()) if out_path.exists() else {}

    seqs = [s for s in args.seqs.split(",") if s]
    for seq in seqs:
        npz = CACHE / f"{seq}.npz"
        if not npz.exists():
            print(f"{seq:36} SKIP (no npz cache)", flush=True)
            continue
        c = rtl.load_cache(CACHE, seq)
        frames = rtl.load_frames(DATASET_ROOT, seq)
        if len(frames) != int(c.get("n_frames", len(frames))):
            raise ValueError(f"{seq}: frame count mismatch")
        gt_pose = np.array(
            [frames[i][1] for i in range(0, len(frames), int(c["stride"]))],
            dtype=np.float64)
        if len(gt_pose) != len(c["gt_pos"]):
            raise ValueError(f"{seq}: gt_pose length mismatch")
        c["gt_pose"] = gt_pose
        pkl = CACHE / f"match_cache_{seq}_torch.pkl"
        if not pkl.exists():
            pkl = CACHE / f"match_cache_{seq}_numpy.pkl"
        with pkl.open("rb") as handle:
            mc = pickle.load(handle)
        fx, fy, cx, cy = rtl.intrinsics_for(DATASET_ROOT, seq,
                                            (525., 525., 320., 240.))
        cam = CameraIntrinsics(fx, fy, cx, cy, 640, 480)

        for name in [x for x in args.configs.split(",") if x]:
            key = f"{name}:{seq}"
            prev = result.get(key)
            if prev is not None and prev.get("params_sha") == shas[name]:
                print(f"{key:40} SKIP (done)", flush=True)
                continue
            params = {**configs[name], "loop_enable": False,
                      "loop_verifier": "none"}
            t = time.perf_counter()
            r = rtl.eval_seq(c, params, cam, matcher, mc)
            _atomic_pickle(pkl, mc)  # persist newly matched pairs
            rpe2m = r.get("RPE_trans_2m_median")
            if rpe2m is None:
                print(f"{key:40} WARNING: RPE is None (degenerate align?)",
                      flush=True)
            result[key] = {
                "config": name,
                "seq": seq,
                "params_sha": shas[name],
                "RPE_trans_2m_median": rpe2m,
                "RPE_trans_1m_median": r.get("RPE_trans_1m_median"),
                "RPE_trans_5m_median": r.get("RPE_trans_5m_median"),
                "RPE_rot_2m_median_deg": r.get("RPE_rot_2m_median_deg"),
                "ATE_median": r.get("ATE_median"),
                "n_loop": r.get("n_loop"),
                "n_kf": r.get("n_kf"),
                "sec": round(time.perf_counter() - t, 1),
            }
            _atomic_dump(out_path, result)
            v = result[key]
            print(f"{key:40} RPE2m={v['RPE_trans_2m_median']} "
                  f"ATE={v['ATE_median']} n_kf={v['n_kf']} ({v['sec']}s)",
                  flush=True)

    print("wrote", out_path, flush=True)


if __name__ == "__main__":
    main()
