import importlib.util, sys, time
from pathlib import Path
REPO = Path("/home/ubuntu/ai-assistant-workspace/onnx_image_processing")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO/"eval"))
spec = importlib.util.spec_from_file_location("et", f"{REPO}/eval/eval_tum_vo.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
import numpy as np, onnxruntime as ort, cv2

sess = ort.InferenceSession(f"{REPO}/eval/pyramid_k512_l2_wd.onnx", providers=["CPUExecutionProvider"])
cam = m.CameraIntrinsics(fx=525, fy=525, cx=320, cy=240, width=640, height=480)
base = Path("/home/ubuntu/datasets/tum_rgbd/rgbd_dataset_freiburg1_room")
frames = [l.split()[1] for l in open(base/"rgb.txt") if l.strip() and not l.startswith("#")]
t0 = time.perf_counter(); cache = {}
args = m.argparse.Namespace(pose_source="essential", method="magsac", threshold=1.4, max_matches=1024, dbin=0.1, match_threshold=0.1, guided=False, guided_inlier_thresh=0.35, guided_sampson=2.0, depth_scale=5000, min_depth=0.1, max_depth=10.0)
for i in range(0, 40, 8):
    a = cv2.resize(cv2.imread(str(base/frames[i]), 0), (640,480)).astype(np.float32)[None,None]
    b = cv2.resize(cv2.imread(str(base/frames[i+8]), 0), (640,480)).astype(np.float32)[None,None]
    outs = sess.run(None, {"image1": a, "image2": b})
    k1,k2,d1,d2,P = outs
    cache[i] = (k1,d1); cache[i+8]=(k2,d2)
    mk1, mk2, sc = m.extract_matches(k1,k2,P,0.1,1024,0.1)
    r = m.estimate_pose_from_matches(mk1, mk2, cam, args)
    print(f"pair {i}-{i+8}: n={len(mk1)} ok={r['ok']} t={time.perf_counter()-t0:.1f}s", flush=True)
