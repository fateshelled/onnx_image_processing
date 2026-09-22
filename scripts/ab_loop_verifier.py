"""A/B the online Sim(3) loop verifier against the baseline (fixed cache).

Runs the kf_prior evaluator with several ``loop_verifier`` operating points
(gate and abstain policy) on cached sequences and prints the ATE plus the
verifier counters, so the gate can be re-calibrated on the ATE criterion
rather than on the offline false-accept margin alone.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "eval"))

from eval.rustuna_tune_loop import SEQ_OPT1_DEFAULTS, eval_seq, load_cache  # noqa: E402
from eval.eval_tum_vo import intrinsics_for  # noqa: E402
from eval.torch_sinkhorn import TorchSinkhornMatcher  # noqa: E402
from vo.pose_estimation import CameraIntrinsics  # noqa: E402

CACHE = REPO / "eval/results/tune_cache_loop"
FIELDS = ("ATE_median", "n_loop", "n_kf", "n_verifier_rejected",
          "n_verifier_abstained", "verifier_accept", "verifier_reject",
          "verifier_abstain", "verifier_abstain_no_cloud",
          "verifier_abstain_no_match", "verifier_abstain_few_tracks")


def _finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _configs(gates, policies, keyframes=False):
    configs = [("baseline", {"loop_verifier": "none"})]
    prefix = "sim3k" if keyframes else "sim3"
    for gate in gates:
        for policy in policies:
            overrides = {"loop_verifier": "sim3",
                          "loop_verifier_gate": gate,
                         "loop_verifier_abstain": policy,
                         "loop_verifier_keyframes": bool(keyframes)}
            configs.append((f"{prefix}_g{gate}_{policy}", overrides))
    return configs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq", default="desk,desk2,room")
    parser.add_argument("--gates", default="6,8,10")
    parser.add_argument("--policies", default="reject",
                        help="comma-separated abstain policies; use 'accept' "
                             "with --keyframes for the deployment profile")
    parser.add_argument("--dataset-root", default="/home/ubuntu/datasets/tum_rgbd")
    parser.add_argument("--keyframes", action="store_true",
                        help="include keyframe-baseline clouds (S1); combine "
                             "with --policies accept for the deployment "
                             "profile")
    parser.add_argument("--output", type=Path)
    opts = parser.parse_args()

    matcher = TorchSinkhornMatcher(iterations=20, epsilon=0.05,
                                   unused_score=1.0, distance_type="l2")
    reports = {}
    for seq in (name.strip() for name in opts.seq.split(",") if name):
        c = load_cache(CACHE, seq)
        stride = int(c["stride"])
        fx, fy, cx, cy = intrinsics_for(opts.dataset_root, seq,
                                        (525., 525., 320., 240.))
        cam = CameraIntrinsics(fx, fy, cx, cy, 640, 480)
        match_path = CACHE / f"match_cache_{seq}_torch.pkl"
        if not match_path.exists():
            match_path = CACHE / f"match_cache_{seq}_numpy.pkl"
        with match_path.open("rb") as handle:
            match_cache = pickle.load(handle)
        for tag, overrides in _configs(
                [int(g) for g in opts.gates.split(",") if g],
                [p.strip() for p in opts.policies.split(",") if p],
                keyframes=opts.keyframes):
            params = {**SEQ_OPT1_DEFAULTS, **overrides}
            try:
                result = eval_seq(c, params, cam, matcher, match_cache)
                report = {name: _finite(result.get(name))
                          for name in FIELDS}
            except Exception as exc:  # noqa: BLE001 - keep other configs going
                report = {"error": f"{type(exc).__name__}: {exc}"}
            reports[f"{seq}:{tag}"] = report
            print(f"{seq}:{tag} {report}", flush=True)
    text = json.dumps(reports, indent=2, ensure_ascii=False, allow_nan=False)
    if opts.output:
        opts.output.parent.mkdir(parents=True, exist_ok=True)
        opts.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
