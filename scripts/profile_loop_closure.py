"""Small profiler: measure where loop-closure time is spent.

Runs desk with and without --loop-closure, monkeypatching estimate_pair and
SlidingWindowOptimizer.optimize to accumulate wall-clock time and call counts.
loop re-match cost = (with_loop - without_loop) - optimize_time.
"""
import importlib.util
import sys
import time

REPO = "/home/ubuntu/ai-assistant-workspace/onnx_image_processing"
MODULE_PATH = f"{REPO}/eval/eval_tum_vo.py"

spec = importlib.util.spec_from_file_location("eval_tum_vo_prof", MODULE_PATH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

timing = {"est": 0.0, "est_n": 0, "opt": 0.0, "opt_n": 0}

orig_est = m.estimate_pair


def timed_est(*a, **k):
    t0 = time.perf_counter()
    r = orig_est(*a, **k)
    timing["est"] += time.perf_counter() - t0
    timing["est_n"] += 1
    return r


m.estimate_pair = timed_est

orig_opt = m.SlidingWindowOptimizer.optimize


def timed_opt(self, verbose=False):
    t0 = time.perf_counter()
    c = orig_opt(self, verbose)
    timing["opt"] += time.perf_counter() - t0
    timing["opt_n"] += 1
    return c


m.SlidingWindowOptimizer.optimize = timed_opt


def run(loop: bool):
    timing["est"] = timing["est_n"] = timing["opt"] = timing["opt_n"] = 0
    argv = [
        "eval_tum_vo.py", "vo",
        "--model", f"{REPO}/eval/pyramid_k512_l2.onnx",
        "--seq", "desk",
        "--stride", "2",
        "--method", "magsac",
        "--threshold", "1.4",
        "--dbin", "0.1",
        "--out", f"/tmp/prof_desk_{'loop' if loop else 'noloop'}.json",
    ]
    if loop:
        argv += [
            "--loop-closure",
            "--keyframe-decim", "8",
            "--loop-min-gap", "30",
            "--loop-min-inlier", "0.4",
            "--loop-iterations", "60",
        ]
    sys.argv = argv
    wall0 = time.perf_counter()
    m.main()
    wall = time.perf_counter() - wall0
    return {
        "wall": wall,
        "est_time": timing["est"],
        "est_n": timing["est_n"],
        "opt_time": timing["opt"],
        "opt_n": timing["opt_n"],
    }


print("=== desk WITHOUT loop-closure ===")
a = run(False)
print(a)
print("=== desk WITH loop-closure ===")
b = run(True)
print(b)

loop_rematch = (b["est_time"] - a["est_time"])
print("\n--- breakdown (desk) ---")
print(f"odometry estimate_pair : {a['est_time']:.1f}s over {a['est_n']} calls "
      f"(~{a['est_time']/max(a['est_n'],1):.3f}s/call)")
print(f"loop re-match estimate_pair: {loop_rematch:.1f}s over "
      f"{b['est_n']-a['est_n']} calls "
      f"(~{loop_rematch/max(b['est_n']-a['est_n'],1):.3f}s/call)")
print(f"graph optimize (GN)   : {b['opt_time']:.1f}s over {b['opt_n']} calls")
print(f"total wall (no loop)  : {a['wall']:.1f}s")
print(f"total wall (loop)     : {b['wall']:.1f}s")
