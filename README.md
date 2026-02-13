# onnx_image_processing
ONNX exportable pytorch model for image processing.

- Feature Detection
  - Shi-Tomasi corner detection
  - AKAZE feature detection
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
