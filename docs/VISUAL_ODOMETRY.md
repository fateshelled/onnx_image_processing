# Visual Odometry with ONNX Models

This guide explains how to use the Visual Odometry (VO) implementation with ONNX feature matching models.

## Overview

Visual Odometry estimates camera motion (trajectory) from a sequence of images or video. The implementation uses:

1. **Feature Detection & Matching**: ONNX model (Shi-Tomasi + Angle + Sparse BAD + Sinkhorn)
2. **Pose Estimation**: Essential Matrix estimation via RANSAC
3. **Trajectory Management**: Accumulates relative poses to build absolute camera trajectory

## Architecture

```
pytorch_model/vo/
├── __init__.py           # Module exports
├── pose_estimation.py    # Camera pose estimation utilities
└── trajectory.py         # Trajectory management and visualization

sample/
└── visual_odometry.py    # Main VO sample script
```

## Quick Start

### 1. Export ONNX Model

First, export the feature matching model to ONNX format:

```bash
python onnx_export/export_shi_tomasi_angle_sparse_bad_sinkhorn.py \
    -o matcher.onnx \
    -H 480 \
    -W 640 \
    --max-keypoints 512
```

You can also use the model with filters for better robustness:

```bash
python onnx_export/export_shi_tomasi_angle_sparse_bad_sinkhorn_with_filters.py \
    -o matcher_filtered.onnx \
    -H 480 \
    -W 640 \
    --max-keypoints 512 \
    --ratio-threshold 2.0 \
    --dustbin-margin 0.3
```

### 2. Prepare Camera Intrinsics

You need to know your camera's intrinsic parameters:
- `fx`, `fy`: Focal lengths (in pixels)
- `cx`, `cy`: Principal point coordinates (in pixels)

These can be obtained through camera calibration (e.g., using OpenCV's calibration tools).

### 3. Run Visual Odometry

#### From Video File

```bash
python sample/visual_odometry.py \
    --model matcher.onnx \
    --video video.mp4 \
    --fx 525.0 \
    --fy 525.0 \
    --cx 319.5 \
    --cy 239.5 \
    --save-trajectory trajectory.npz \
    --save-plot trajectory.png
```

#### From Image Sequence

```bash
python sample/visual_odometry.py \
    --model matcher.onnx \
    --image-dir frames/ \
    --fx 525.0 \
    --fy 525.0 \
    --cx 319.5 \
    --cy 239.5 \
    --save-trajectory trajectory.npz \
    --save-plot trajectory.png
```

#### From Webcam (Real-time)

```bash
python sample/visual_odometry.py \
    --model matcher.onnx \
    --camera 0 \
    --fx 525.0 \
    --fy 525.0 \
    --cx 319.5 \
    --cy 239.5 \
    --display
```

Press `q` to quit or `s` to save trajectory during real-time processing.

#### From RealSense Camera

```bash
# Auto-detect camera intrinsics (recommended)
python sample/visual_odometry.py \
    --model matcher.onnx \
    --camera 0 \
    --camera-backend realsense \
    --camera-width 640 \
    --camera-height 480 \
    --display

# Or specify intrinsics manually
python sample/visual_odometry.py \
    --model matcher.onnx \
    --camera 0 \
    --camera-backend realsense \
    --fx 525.0 \
    --fy 525.0 \
    --cx 319.5 \
    --cy 239.5 \
    --camera-width 640 \
    --camera-height 480 \
    --display
```

**Note:**
- Requires `pyrealsense2` package. Install with: `pip install pyrealsense2`
- Camera intrinsics are automatically detected from RealSense cameras
- Manual specification overrides auto-detection

#### 3D Trajectory Visualization

```bash
python sample/visual_odometry.py \
    --model matcher.onnx \
    --video video.mp4 \
    --fx 525.0 \
    --fy 525.0 \
    --cx 319.5 \
    --cy 239.5 \
    --save-plot trajectory_3d.png \
    --plot-3d
```

## Command-Line Options

### Required Arguments

| Option | Description |
|--------|-------------|
| `--model`, `-m` | Path to ONNX model file |
| `--video`, `-v` OR `--image-dir`, `-d` OR `--camera`, `-c` | Input source: video file, image directory, or camera device ID |
| `--fx` | Focal length in x direction (pixels). **Optional for RealSense** (auto-detected). |
| `--fy` | Focal length in y direction (pixels). **Optional for RealSense** (auto-detected). |
| `--cx` | Principal point x coordinate (pixels). **Optional for RealSense** (auto-detected). |
| `--cy` | Principal point y coordinate (pixels). **Optional for RealSense** (auto-detected). |

### Camera Options (for --camera mode)

| Option | Default | Description |
|--------|---------|-------------|
| `--camera-backend` | opencv | Camera backend (opencv or realsense) |
| `--camera-width` | 640 | Camera resolution width |
| `--camera-height` | 480 | Camera resolution height |
| `--camera-fps` | 30 | Camera framerate |

### Optional Arguments

| Option | Default | Description |
|--------|---------|-------------|
| `--match-threshold`, `-t` | 0.1 | Match probability threshold |
| `--ransac-threshold` | 1.0 | RANSAC reprojection threshold (pixels) |
| `--min-matches` | 20 | Minimum number of matches required |
| `--skip-frames` | 0 | Process every N-th frame (0=all frames) |
| `--max-frames` | None | Maximum number of frames to process |
| `--display` | False | Display frames and trajectory in real-time |
| `--save-trajectory` | None | Save trajectory to file (*.npz) |
| `--save-plot` | None | Save trajectory plot (*.png) |
| `--plot-3d` | False | Plot 3D trajectory instead of 2D |
| `--quiet`, `-q` | False | Suppress progress output |

## Camera Backends

The Visual Odometry implementation supports multiple camera backends through a unified interface.

### OpenCV Backend (Default)

Standard USB/webcam cameras using OpenCV's `cv2.VideoCapture`:

```bash
python sample/visual_odometry.py \
    --model matcher.onnx \
    --camera 0 \
    --camera-backend opencv \
    --fx 525 --fy 525 --cx 320 --cy 240 \
    --display
```

**Features:**
- Works with any USB webcam
- Simple setup, no additional dependencies
- Resolution and FPS configurable (subject to hardware support)

### RealSense Backend

Intel RealSense D400/D500 series cameras using `pyrealsense2`:

```bash
# Install RealSense SDK
pip install pyrealsense2

# Run VO with RealSense (auto-detect intrinsics)
python sample/visual_odometry.py \
    --model matcher.onnx \
    --camera 0 \
    --camera-backend realsense \
    --camera-width 640 \
    --camera-height 480 \
    --camera-fps 30 \
    --display
```

**Features:**
- High-quality RGB streams
- Built-in depth sensor (for future RGBD-SLAM)
- Hardware-synchronized RGB-D frames
- Auto-exposure and white balance
- **Automatic intrinsics detection** - no need to specify fx, fy, cx, cy

**Note:** The current implementation uses RGB only. Depth support will be added for RGBD-SLAM.

### Camera Wrapper Architecture

The implementation uses an abstract `BaseCamera` class with concrete implementations:

```python
from pytorch_model.vo import create_camera

# Create OpenCV camera
camera = create_camera(backend="opencv", device_id=0, width=640, height=480)

# Create RealSense camera
camera = create_camera(backend="realsense", device_id=0, width=640, height=480, fps=30)

# Unified interface
ret, frame = camera.read()
camera.release()
```

This architecture makes it easy to add support for other camera types in the future.

## Algorithm Details

### Feature Matching Pipeline

1. **Feature Detection**: Shi-Tomasi corner detection with angle estimation
2. **Descriptor Extraction**: Rotation-invariant BAD descriptors at keypoints
3. **Matching**: Sinkhorn optimal transport matching
4. **Mutual Nearest Neighbor**: Extract consistent matches

### Pose Estimation Pipeline

1. **Essential Matrix**: Estimated from matched points using 5-point RANSAC
2. **Pose Recovery**: Decompose Essential Matrix to R (rotation) and t (translation)
3. **Inlier Filtering**: Only use geometrically consistent matches
4. **Trajectory Update**: Compose current pose with previous poses

### Coordinate Systems

- **Image coordinates**: (y, x) format, origin at top-left
- **Camera coordinates**: Right-handed, Z-axis forward
- **World coordinates**: Accumulated from identity at start

## Performance Tips

### Frame Skipping

For long videos or real-time processing, use `--skip-frames`:

```bash
# Process every 5th frame
python sample/visual_odometry.py \
    --model matcher.onnx \
    --video video.mp4 \
    --fx 525 --fy 525 --cx 320 --cy 240 \
    --skip-frames 4
```

### Match Quality Tuning

Adjust matching thresholds for better accuracy vs. robustness:

```bash
# Stricter matching (fewer but more reliable matches)
python sample/visual_odometry.py \
    --model matcher.onnx \
    --video video.mp4 \
    --fx 525 --fy 525 --cx 320 --cy 240 \
    --match-threshold 0.2 \
    --ransac-threshold 0.5 \
    --min-matches 30
```

### Model Selection

For better robustness, use the filtered model:

```bash
# Export model with built-in filters
python onnx_export/export_shi_tomasi_angle_sparse_bad_sinkhorn_with_filters.py \
    -o matcher_filtered.onnx \
    -H 480 -W 640 \
    --max-keypoints 512 \
    --ratio-threshold 2.0 \
    --dustbin-margin 0.3

# Run VO with filtered model
python sample/visual_odometry.py \
    --model matcher_filtered.onnx \
    --video video.mp4 \
    --fx 525 --fy 525 --cx 320 --cy 240
```

## Python API Usage

You can also use the VO modules programmatically:

```python
import numpy as np
from pytorch_model.vo import (
    CameraIntrinsics,
    estimate_pose_ransac,
    Trajectory,
)

# Create camera intrinsics
camera = CameraIntrinsics(
    fx=525.0, fy=525.0,
    cx=319.5, cy=239.5,
    width=640, height=480
)

# Estimate pose from matched keypoints
R, t, inlier_mask = estimate_pose_ransac(
    keypoints1,  # (N, 2) array
    keypoints2,  # (N, 2) array
    camera,
    ransac_threshold=1.0
)

# Build trajectory
trajectory = Trajectory()
trajectory.add_relative_pose(R, t)

# Visualize
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
trajectory.plot_2d(ax, show_orientation=True)
plt.show()
```

## Limitations and Future Work

### Current Limitations

1. **Scale Ambiguity**: Monocular VO has inherent scale ambiguity
   - Solution: Use known object sizes or stereo/depth cameras
2. **Drift**: Pose errors accumulate over time
   - Solution: Loop closure detection (VSLAM)
3. **Rotation-only Motion**: Cannot estimate translation during pure rotation
   - Solution: Require sufficient parallax

### Roadmap to VSLAM

The current implementation provides the foundation for Visual SLAM:

- [x] Feature detection and matching
- [x] Pose estimation
- [x] Trajectory management
- [ ] Map management (3D point cloud)
- [ ] Loop closure detection
- [ ] Bundle adjustment
- [ ] Relocalization

## Troubleshooting

### "Insufficient matches" Error

- Lower `--match-threshold` (e.g., from 0.1 to 0.05)
- Increase `--max-keypoints` in ONNX export
- Ensure sufficient texture in scenes

### "Pose estimation failed" Error

- Lower `--ransac-threshold` (e.g., from 1.0 to 0.5)
- Lower `--min-matches` (e.g., from 20 to 10)
- Check camera intrinsics are correct

### Poor Trajectory Quality

- Use filtered model with `ratio_threshold` and `dustbin_margin`
- Increase `--min-matches` for more reliable estimates
- Ensure sufficient camera motion (avoid pure rotation)

## References

- [ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2) - Classic Visual SLAM
- [OpenCV Visual Odometry Tutorial](https://docs.opencv.org/master/d0/d13/classcv_1_1stereoVO.html)
- [Multiple View Geometry in Computer Vision](https://www.robots.ox.ac.uk/~vgg/hzbook/) - Theory reference
