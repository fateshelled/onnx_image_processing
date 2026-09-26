"""One-off probe: why do freiburg2_xyz local clouds have so few tracks?

Mirrors diag_loop_sim3._local_cloud for a handful of candidate endpoints and
prints the per-window triangulation validity, parallax and cloud sizes.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "eval"))

from eval.rustuna_tune_loop import DEFAULT_ARGS, load_cache  # noqa: E402
from eval.eval_tum_vo import intrinsics_for  # noqa: E402
from eval.torch_sinkhorn import TorchSinkhornMatcher  # noqa: E402
from vo.pose_estimation import CameraIntrinsics  # noqa: E402
from vo.sim3_verification import triangulate_local  # noqa: E402

import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "diag", REPO / "scripts" / "diag_loop_sim3.py")
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)


def main():
    seq = sys.argv[1] if len(sys.argv) > 1 else "freiburg2_xyz"
    c = load_cache("eval/results/tune_cache_loop", seq)
    stride = int(c["stride"])
    fx, fy, cx, cy = intrinsics_for("/home/ubuntu/datasets/tum_rgbd", seq,
                                    (525.0, 525.0, 320.0, 240.0))
    cam = CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=640, height=480)
    matcher = TorchSinkhornMatcher(iterations=20, epsilon=0.05,
                                   unused_score=1.0, distance_type="l2")
    args = SimpleNamespace(**vars(DEFAULT_ARGS))

    for endpoint in [4, 484]:
        print(f"== endpoint {endpoint}")
        for first, last, R, t, forward in diag._local_windows(
                c, endpoint, stride, 2):
            pair = diag._details(c, matcher, cam, first, last, args,
                                 require_pose=False)
            if pair is None:
                print(f"  window ({first},{last}) forward={forward}: no pair")
                continue
            p_first, p_last, _ = pair
            tri = triangulate_local(p_first, p_last, R, t, cam.K,
                                    min_parallax_deg=1.0)
            baseline = float(np.linalg.norm(t))
            print(f"  window ({first},{last}) forward={forward}: "
                  f"matches={len(p_first)} valid={int(tri.valid.sum())} "
                  f"baseline={baseline:.4g} "
                  f"parallax_med={np.median(tri.parallax_deg):.3f} "
                  f"parallax_p90={np.percentile(tri.parallax_deg, 90):.3f} "
                  f"depth_med={np.nanmedian(tri.points[:, 2]):.4g} "
                  f"depth_limit={100.0 * baseline:.4g}")


if __name__ == "__main__":
    main()
