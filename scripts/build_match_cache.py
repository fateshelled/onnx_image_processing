"""Build a torch match cache for one sequence in the diag/tuner schema.

The tuner memoizes match results only for pairs it happens to visit, so
sequences that were never tuned (e.g. ``freiburg2_xyz``) have no cache.  This
script enumerates endpoint pairs at a fixed set of raw-frame gaps and stores
the same ``{ok, R, t, inlier_ratio, n_matches}`` records the diagnostic
expects, checkpointing as it goes.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "eval"))

from eval.rustuna_tune_loop import DEFAULT_ARGS, load_cache  # noqa: E402
from eval.eval_tum_vo import estimate_pose_from_matches, intrinsics_for  # noqa: E402
from eval.torch_sinkhorn import TorchSinkhornMatcher  # noqa: E402
from vo.onnx_matcher import extract_matches  # noqa: E402
from vo.pose_estimation import CameraIntrinsics  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq", required=True)
    parser.add_argument("--cache-dir", default="eval/results/tune_cache_loop")
    parser.add_argument("--dataset-root", default="/home/ubuntu/datasets/tum_rgbd")
    parser.add_argument("--gaps", default="30,60,120,240,480,960,1920,2880",
                        help="raw-frame gaps between endpoints")
    parser.add_argument("--frame-step", type=int, default=2,
                        help="subsample stride frames per gap")
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--fx", type=float, default=525.0)
    parser.add_argument("--fy", type=float, default=525.0)
    parser.add_argument("--cx", type=float, default=320.0)
    parser.add_argument("--cy", type=float, default=240.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    opts = parser.parse_args()

    cache_dir = Path(opts.cache_dir)
    cache = load_cache(cache_dir, opts.seq)
    stride = int(cache["stride"])
    n_frames = int(cache["n_frames"])
    fx, fy, cx, cy = intrinsics_for(opts.dataset_root, opts.seq,
                                    (opts.fx, opts.fy, opts.cx, opts.cy))
    cam = CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy,
                           width=opts.width, height=opts.height)
    matcher = TorchSinkhornMatcher(iterations=20, epsilon=0.05,
                                   unused_score=1.0, distance_type="l2")
    output = cache_dir / f"match_cache_{opts.seq}_torch.pkl"
    if output.exists():
        with output.open("rb") as handle:
            matches = pickle.load(handle)
        print(f"[build] resuming with {len(matches)} cached pairs", flush=True)
    else:
        matches = {}

    pairs = set()
    for gap in (int(g) for g in opts.gaps.split(",") if g):
        step = stride * opts.frame_step
        pairs.update((start, start + gap)
                     for start in range(0, n_frames - gap, step)
                     if start % stride == 0 and (start + gap) % stride == 0)
    todo = sorted(pair for pair in pairs if pair not in matches)
    print(f"[build] {len(pairs)} pairs total, {len(todo)} to compute",
          flush=True)

    def save():
        tmp = output.with_name(output.name + ".tmp")
        with tmp.open("wb") as handle:
            pickle.dump(matches, handle, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(output)

    for index, (a, b) in enumerate(todo, start=1):
        ka, da = cache["feat"].get(a, (None, None))
        kb, db = cache["feat"].get(b, (None, None))
        if ka is None or kb is None:
            matches[(a, b)] = {"ok": False, "inlier_ratio": 0.0,
                               "n_matches": 0}
            continue
        P = matcher.match_probs(da[0], db[0])
        pa, pb, _ = extract_matches(ka, kb, P[None],
                                    DEFAULT_ARGS.match_threshold,
                                    DEFAULT_ARGS.max_matches,
                                    DEFAULT_ARGS.dbin)
        result = estimate_pose_from_matches(pa, pb, cam, DEFAULT_ARGS)
        matches[(a, b)] = {
            "ok": bool(result.get("ok", False)),
            "R": result.get("R"), "t": result.get("t"),
            "inlier_ratio": float(result.get("inlier_ratio", 0.0)),
            "n_matches": int(len(pa)),
        }
        if index % 500 == 0:
            print(f"[build] {index}/{len(todo)}", flush=True)
        if index % opts.checkpoint_every == 0:
            save()
            print(f"[build] checkpoint at {index}", flush=True)
    save()
    ok = sum(1 for value in matches.values() if value.get("ok"))
    print(f"[build] saved {len(matches)} pairs ({ok} ok) -> {output}",
          flush=True)


if __name__ == "__main__":
    main()
