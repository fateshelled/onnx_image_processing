# onnx_image_processing
ONNX exportable pytorch model for image processing.

- Feature Detection
  - Shi-Tomasi corner detection
  - AKAZE feature detection
  - DoG (Difference of Gaussians) feature detection
  - BAD (Box Average Difference) descriptor
  - Shi-Tomasi + BAD unified feature detection
  - Shi-Tomasi + Angle + Sparse BAD (rotation-invariant descriptors)
  - AKAZE + Sparse BAD (rotation-invariant descriptors)
- Feature Matching
  - Sinkhorn matcher
  - Shi-Tomasi + BAD + Sinkhorn unified feature matcher (end-to-end, dense BAD)
  - Shi-Tomasi + Sparse BAD + Sinkhorn unified feature matcher (end-to-end, sparse BAD at keypoints only)
  - Shi-Tomasi + Angle + Sparse BAD + Sinkhorn (rotation-invariant matching)
  - AKAZE + Sparse BAD + Sinkhorn (rotation-invariant matching)
- Threshold
  - Otsu threshold
  - Multi-Otsu threshold
- Depth
  - Depth image to Point cloud
  - Depth image to Point cloud with Normal
  - Align depth image to rgb image
- Point Cloud
  - Voxel Downsampling
- Visual Odometry
  - Camera pose estimation
  - Trajectory management and visualization
  - ONNX-based feature matching for VO

## Sample: Feature Detection (Shi-Tomasi + BAD)

Detect feature points from an image using the Shi-Tomasi + BAD ONNX model and visualize the results.

### Requirements

```bash
pip install torch onnx onnxruntime onnxscript numpy Pillow
```

### 1. Export ONNX model

```bash
python onnx_export/export_shi_tomasi_bad.py -o shi_tomasi_bad.onnx -H 480 -W 640
```

### 2. Run feature detection sample

```bash
python sample/feature_detection.py --model shi_tomasi_bad.onnx --input image.png --output result.png
```

#### Options

| Option | Default | Description |
|---|---|---|
| `--model`, `-m` | (required) | Path to the exported ONNX model file |
| `--input`, `-i` | (required) | Input image path |
| `--output`, `-o` | `feature_detection_result.png` | Output visualization image path |
| `--threshold`, `-t` | `0.01` | Score threshold for keypoint selection |
| `--max-keypoints`, `-k` | `1000` | Maximum number of keypoints to detect |
| `--nms-radius` | `3` | Non-maximum suppression radius |
| `--circle-radius` | `3` | Radius of keypoint circles in visualization |
| `--colorize` | (flag) | Colorize keypoint circles by score (blue=low, red=high) |

---

## Sample: Image Matching (Shi-Tomasi + BAD + Sinkhorn)

Match feature points between two images using the end-to-end Shi-Tomasi + BAD + Sinkhorn ONNX model and visualize the matched pairs.

### 1. Export ONNX model

```bash
# Dense model
python onnx_export/export_shi_tomasi_bad_sinkhorn.py -o shi_tomasi_bad_sinkhorn.onnx -H 480 -W 640 --max-keypoints 512

# Sparse Model
python onnx_export/export_shi_tomasi_sparse_bad_sinkhorn.py -o shi_tomasi_sparse_bad_sinkhorn.onnx -H 480 -W 640 --max-keypoints 512
```

### 2. Run image matching sample

```bash
# Dense model
python sample/image_matching.py --model shi_tomasi_bad_sinkhorn.onnx --input1 image1.png --input2 image2.png --output result.png

# Sparse Model
python sample/image_matching.py --model shi_tomasi_sparse_bad_sinkhorn.onnx --input1 image1.png --input2 image2.png --output result.png
```

#### Options

| Option | Default | Description |
|---|---|---|
| `--model`, `-m` | (required) | Path to the exported ONNX model file |
| `--input1` | (required) | First input image path |
| `--input2` | (required) | Second input image path |
| `--output`, `-o` | `image_matching_result.png` | Output visualization image path |
| `--threshold`, `-t` | `0.1` | Minimum match probability threshold |
| `--max-matches` | `100` | Maximum number of matches to visualize |
| `--colorize` | (flag) | Colorize match lines by confidence (blue=low, red=high) |

---

## Sample: Visual Odometry

Estimate camera trajectory from video or image sequence using the ONNX feature matching model.

### 1. Export ONNX model

```bash
python onnx_export/export_shi_tomasi_angle_sparse_bad_sinkhorn.py -o matcher.onnx -H 480 -W 640 --max-keypoints 512
```

### 2. Run visual odometry

```bash
# From video file
python sample/visual_odometry.py --model matcher.onnx --video video.mp4 --fx 525 --fy 525 --cx 320 --cy 240 --save-trajectory trajectory.npz --save-plot trajectory.png

# From image sequence
python sample/visual_odometry.py --model matcher.onnx --image-dir frames/ --fx 525 --fy 525 --cx 320 --cy 240 --save-plot trajectory_3d.png --plot-3d

# From webcam (real-time)
python sample/visual_odometry.py --model matcher.onnx --camera 0 --fx 525 --fy 525 --cx 320 --cy 240 --display

# From RealSense camera (auto-detects intrinsics)
python sample/visual_odometry.py --model matcher.onnx --camera 0 --camera-backend realsense --display --camera-width 640 --camera-height 480

# From RealSense camera (manual intrinsics)
python sample/visual_odometry.py --model matcher.onnx --camera 0 --camera-backend realsense --fx 525 --fy 525 --cx 320 --cy 240 --display --camera-width 640 --camera-height 480

# From Orbbec camera (auto-detects intrinsics)
python sample/visual_odometry.py --model matcher.onnx --camera 0 --camera-backend orbbec --display --camera-width 640 --camera-height 480

# From OAK-D camera (auto-detects intrinsics)
python sample/visual_odometry.py --model matcher.onnx --camera 0 --camera-backend oak --display --camera-width 640 --camera-height 480
```

### Requirements

```bash
pip install torch onnx onnxruntime onnxscript numpy opencv-python matplotlib
```

#### Options

| Option | Default | Description |
|---|---|---|
| `--model`, `-m` | (required) | Path to the exported ONNX model file |
| `--video`, `-v` | (mutually exclusive) | Input video file path |
| `--image-dir`, `-d` | (mutually exclusive) | Input image directory path |
| `--camera`, `-c` | (mutually exclusive) | Webcam device ID (e.g., 0) |
| `--fx` | Required for OpenCV/video, auto-detected for RealSense/Orbbec/OAK | Focal length in x direction (pixels) |
| `--fy` | Required for OpenCV/video, auto-detected for RealSense/Orbbec/OAK | Focal length in y direction (pixels) |
| `--cx` | Required for OpenCV/video, auto-detected for RealSense/Orbbec/OAK | Principal point x coordinate (pixels) |
| `--cy` | Required for OpenCV/video, auto-detected for RealSense/Orbbec/OAK | Principal point y coordinate (pixels) |
| `--camera-backend` | `opencv` | Camera backend (opencv, realsense, orbbec, or oak) |
| `--camera-width` | `640` | Camera resolution width |
| `--camera-height` | `480` | Camera resolution height |
| `--camera-fps` | `30` | Camera framerate |
| `--match-threshold`, `-t` | `0.1` | Match probability threshold |
| `--ransac-threshold` | `1.0` | RANSAC reprojection threshold (pixels) |
| `--min-matches` | `20` | Minimum number of matches required |
| `--min-inlier-ratio` | `0.5` | Minimum RANSAC inlier ratio (0-1) to accept pose |
| `--min-motion-pixels` | `1.0` | Minimum RMS pixel motion to attempt pose estimation |
| `--max-reference-age` | `30` | Maximum frames before forced reference update |
| `--skip-frames` | `0` | Process every N-th frame (0=all frames) |
| `--max-frames` | `None` | Maximum number of frames to process |
| `--display` | (flag) | Display frames and trajectory in real-time |
| `--save-trajectory` | `None` | Save trajectory to file (*.npz) |
| `--save-plot` | `None` | Save trajectory plot to file (*.png) |
| `--plot-3d` | (flag) | Plot 3D trajectory instead of 2D |
| `--quiet`, `-q` | (flag) | Suppress progress output |

For detailed documentation, see [docs/VISUAL_ODOMETRY.md](docs/VISUAL_ODOMETRY.md).
