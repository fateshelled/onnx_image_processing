# onnx_image_processing
ONNX exportable pytorch model for image processing.

- Feature Detection
  - Shi-Tomasi corner detection
  - BAD (Box Average Difference) descriptor
  - Shi-Tomasi + BAD unified feature detection
- Feature Matching
  - Sinkhorn matcher
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
