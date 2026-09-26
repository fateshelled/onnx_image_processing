"""Targeted A/B: can ATE and RPE both improve over the shipped default?

No tuning: a small hand-picked grid combining
  * odometry: shipped default / Phase1 t205 / t205 with NL-Reg length 0.5
  * loop: off / shipped default / loop-best from tune_rpe study (t358, t283, t153)

All configs evaluated on the dev sequences (desk, desk2, room) with the same
evaluator as the tuner. Results written incrementally (atomic) and resumable.
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
OUT = REPO / "eval/results/ab_ate_rpe_joint.json"
DEV = ["desk", "desk2", "room"]

# --- odometry bases -------------------------------------------------------
O_DEFAULT = dict(DEFAULT_PARAMS)
O_T205 = json.loads((REPO / "eval/results/tune_odom.json").read_text())["best_params"]
O_T205_L05 = {**O_T205, "nl_reg_length": 0.5}

# --- loop parameter sets --------------------------------------------------
L_OFF = {"loop_enable": False, "loop_verifier": "none"}
L_DEFAULT = {
    "loop_enable": True, "loop_verifier": "none", "cycle_threshold_deg": 0.0,
    "loop_window": 60, "loop_min_gap": 20, "loop_min_inlier": 0.5,
    "loop_temporal_k": 2, "loop_sigma_scale": 1.0, "loop_robust": "none",
}
L_T358 = {
    "loop_enable": True, "loop_verifier": "sim3", "loop_verifier_gate": 4,
    "loop_window": 80, "loop_min_gap": 20, "loop_min_inlier": 0.4,
    "loop_temporal_k": 2, "loop_sigma_scale": 3.0, "loop_robust": "none",
    "cycle_threshold_deg": 0.0,
}
L_T283 = {
    "loop_enable": True, "loop_verifier": "sim3", "loop_verifier_gate": 4,
    "loop_window": 40, "loop_min_gap": 40, "loop_min_inlier": 0.4,
    "loop_temporal_k": 2, "loop_sigma_scale": 2.0, "loop_robust": "none",
    "cycle_threshold_deg": 0.0,
}
L_T153 = {
    "loop_enable": True, "loop_verifier": "none",
    "loop_window": 60, "loop_min_gap": 60, "loop_min_inlier": 0.4,
    "loop_temporal_k": 4, "loop_sigma_scale": 1.0, "loop_robust": "none",
    "cycle_threshold_deg": 0.0,
}

CONFIGS = {
    "base":          (O_DEFAULT,  L_DEFAULT),
    "base_t358":     (O_DEFAULT,  L_T358),
    "base_t283":     (O_DEFAULT,  L_T283),
    "base_t153":     (O_DEFAULT,  L_T153),
    "t205_off":      (O_T205,     L_OFF),
    "t205_default":  (O_T205,     L_DEFAULT),
    "t205_t358":     (O_T205,     L_T358),
    "t205_t283":     (O_T205,     L_T283),
    "t205L05_t358":  (O_T205_L05, L_T358),
}


def build_params(om, lp):
    return {**om, **lp}


def params_sha(params):
    return hashlib.sha1(json.dumps(params, sort_keys=True, default=str)
                        .encode()).hexdigest()[:12]


def _atomic_dump(path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    os.replace(tmp, path)


def _atomic_pickle(path, obj):
    tmp = Path(str(path) + ".tmp")
    with tmp.open("wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", default=",".join(DEV))
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    out_path = Path(args.out)
    result = json.loads(out_path.read_text()) if out_path.exists() else {}
    matcher = TorchSinkhornMatcher(iterations=20, epsilon=0.05,
                                   unused_score=1.0, distance_type="l2")

    for seq in [s for s in args.seqs.split(",") if s]:
        c = rtl.load_cache(CACHE, seq)
        frames = rtl.load_frames(DATASET_ROOT, seq)
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
            om, lp = CONFIGS[name]
            params = build_params(om, lp)
            key = f"{name}:{seq}"
            sha = params_sha(params)
            prev = result.get(key)
            if prev is not None and prev.get("params_sha") == sha:
                print(f"{key:24} SKIP (done)", flush=True)
                continue
            t = time.perf_counter()
            r = rtl.eval_seq(c, params, cam, matcher, mc)
            _atomic_pickle(pkl, mc)
            result[key] = {
                "config": name, "seq": seq, "params_sha": sha,
                "loop": bool(params.get("loop_enable")),
                "RPE_trans_2m_median": r.get("RPE_trans_2m_median"),
                "RPE_trans_1m_median": r.get("RPE_trans_1m_median"),
                "RPE_trans_5m_median": r.get("RPE_trans_5m_median"),
                "RPE_rot_2m_median_deg": r.get("RPE_rot_2m_median_deg"),
                "ATE_median": r.get("ATE_median"),
                "n_loop": r.get("n_loop"), "n_kf": r.get("n_kf"),
                "sec": round(time.perf_counter() - t, 1),
            }
            _atomic_dump(out_path, result)
            v = result[key]
            print(f"{key:24} RPE2m={v['RPE_trans_2m_median']} "
                  f"ATE={v['ATE_median']} n_loop={v['n_loop']} "
                  f"n_kf={v['n_kf']} ({v['sec']}s)", flush=True)

    print("wrote", out_path, flush=True)


if __name__ == "__main__":
    main()
