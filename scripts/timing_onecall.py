"""Quick timing: one estimate_pair call + frame counts per sequence.

Shows per-call cost of estimate_pair and how many loop-candidate pairs the
O(K^2) re-match would create for desk/desk2/room at keyframe_decim=8,
loop_min_gap=30.
"""
import importlib.util
import time

REPO = "/home/ubuntu/ai-assistant-workspace/onnx_image_processing"
MODULE_PATH = f"{REPO}/eval/eval_tum_vo.py"

spec = importlib.util.spec_from_file_location("eval_tum_vo_q", MODULE_PATH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

# camera + session
cam = m.CameraIntrinsics(fx=525.0, fy=525.0, cx=320.0, cy=240.0,
                         width=640, height=480)
session = m.ort.InferenceSession(
    f"{REPO}/eval/pyramid_k512_l2.onnx", providers=["CPUExecutionProvider"])

# build a fake args namespace with the fields estimate_pair needs
class A:
    pass
args = A()
args.pose_source = "essential"
args.method = "magsac"
args.threshold = 1.4
args.depth_scale = 5000.0
args.min_depth = 0.1
args.max_depth = 10.0
args.guided = False
args.guided_inlier_thresh = 0.35
args.guided_sampson = 2.0
args.match_threshold = 0.1
args.max_matches = 1024
args.dbin = 0.1

# one timed call on desk frames 0 and 2
base = f"{REPO}/../datasets/tum_rgbd/rgbd_dataset_freiburg1_desk"
# dataset_root default is /home/ubuntu/datasets/tum_rgbd
import os
dsroot = "/home/ubuntu/datasets/tum_rgbd"
frames = m.read_path_file(os.path.join(dsroot, "rgbd_dataset_freiburg1_desk", "rgb.txt"))
pa = os.path.join(dsroot, "rgbd_dataset_freiburg1_desk", frames[0][1])
pb = os.path.join(dsroot, "rgbd_dataset_freiburg1_desk", frames[2][1])

ts = []
for _ in range(3):
    t0 = time.perf_counter()
    res = m.estimate_pair(session=session, img_path1=pa, img_path2=pb,
                          cam=cam, args=args, depth_path1=None)
    ts.append(time.perf_counter() - t0)
print(f"[estimate_pair x3] {[f'{x:.2f}s' for x in ts]} -> per call ~{sum(ts)/3:.2f}s")

# frame counts + loop-candidate pair counts
kd, gap = 8, 30
for seq in ["desk", "desk2", "room"]:
    fl = m.read_path_file(os.path.join(dsroot, f"rgbd_dataset_freiburg1_{seq}", "rgb.txt"))
    n = len(fl)
    nodes = (n - 2) // 2  # approximate pair count (stride 2)
    kf = nodes // kd
    pairs = kf * (kf - 1) // 2  # upper bound; minus gap filtering
    # crude gap filter: fraction of pairs with index gap>=gap
    keep = sum(1 for ai in range(kf) for bi in range(ai + 1, kf)
               if (bi - ai) * kd >= gap)
    est_loop_s = keep * (sum(ts) / 3)
    print(f"[{seq}] frames={n} nodes~={nodes} keyframes~={kf} "
          f"loop_pairs(gap>= {gap})={keep} -> est loop re-match ~{est_loop_s:.0f}s "
          f"(+ odom {nodes * (sum(ts)/3):.0f}s)")
