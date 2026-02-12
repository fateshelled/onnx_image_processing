# AKAZE Feature Detector

PyTorch implementation of AKAZE (Accelerated-KAZE) feature detector with full ONNX export compatibility.

## Overview

This implementation provides an ONNX-exportable AKAZE feature detector that includes:

- **Non-linear diffusion** using Fast Explicit Diffusion (FED) scheme for scale-space construction
- **Hessian-based feature detection** with non-maximum suppression
- **Orientation estimation** using intensity centroid method
- **Pure tensor operations** - no dynamic control flow (compatible with ONNX export)

## Key Features

### 1. Non-Linear Diffusion (FED Scheme)

The implementation uses Perona-Malik anisotropic diffusion with the conduction function:

```
c(|∇L|) = 1 / (1 + (|∇L|/κ)²)
```

where κ (kappa) is the contrast parameter controlling edge preservation.

The diffusion equation:
```
∂L/∂t = div(c(|∇L|) · ∇L)
```

is solved using a fixed number of iterations (unrolled for ONNX export).

### 2. Hessian-Based Feature Detection

Features are detected using the determinant of the Hessian matrix:

```
H = [Lxx  Lxy]
    [Lxy  Lyy]

Response = det(H) = Lxx * Lyy - Lxy²
```

### 3. Non-Maximum Suppression

NMS is implemented using `MaxPool2d` with a fixed window size, ensuring ONNX compatibility.

### 4. Orientation Estimation

Orientation is computed using Gaussian-weighted intensity centroid:

```
θ = atan2(m01, m10)
```

where m10 and m01 are the first-order moments computed via convolution.

**Important:** Orientations are computed at each scale level, and the orientation from the scale with the maximum feature response is selected for each pixel. This ensures that the orientation corresponds to the scale where the feature was actually detected, following the AKAZE specification.

## Usage

### Basic Usage

```python
import torch
from pytorch_model.feature_detection.akaze import AKAZE

# Create model
model = AKAZE(
    num_scales=3,
    diffusion_iterations=3,
    kappa=0.05,
    threshold=0.001,
    nms_size=5,
    orientation_patch_size=15,
    orientation_sigma=2.5,
)
model.eval()

# Input: grayscale image (N, 1, H, W)
image = torch.randn(1, 1, 480, 640)

# Detect features
with torch.no_grad():
    scores, orientations = model(image)

# scores: (N, 1, H, W) - feature point strength map
# orientations: (N, 1, H, W) - orientation map in radians [-π, π]
```

### Export to ONNX

#### Using the export script:

```bash
python onnx_export/export_akaze.py \
    --output akaze.onnx \
    --height 480 \
    --width 640 \
    --num-scales 3 \
    --diffusion-iterations 3 \
    --kappa 0.05 \
    --threshold 0.001 \
    --nms-size 5
```

#### Programmatically:

```python
torch.onnx.export(
    model,
    dummy_input,
    "akaze.onnx",
    export_params=True,
    opset_version=18,
    input_names=["input"],
    output_names=["scores", "orientations"],
)
```

## Parameters

### AKAZE Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_scales` | int | 3 | Number of scale levels in the scale-space pyramid |
| `diffusion_iterations` | int | 3 | Number of FED iterations per scale |
| `kappa` | float | 0.05 | Contrast parameter for edge preservation (lower = more preservation) |
| `threshold` | float | 0.001 | Feature detection threshold |
| `nms_size` | int | 5 | Non-maximum suppression window size (must be odd) |
| `orientation_patch_size` | int | 15 | Patch size for orientation computation (must be odd) |
| `orientation_sigma` | float | 2.5 | Gaussian sigma for orientation weighting |

## Implementation Details

### ONNX Compatibility Constraints

All technical constraints for ONNX export are strictly followed:

1. **No dynamic control flow**: All loops are unrolled using `nn.ModuleList` and `range()`
2. **Pure tensor operations**: Only `torch.nn.functional` operations (conv2d, max_pool2d, etc.)
3. **Fixed shape operations**: Non-maximum suppression uses fixed-size `MaxPool2d`
4. **No dynamic indexing**: Features are output as score maps, not coordinate lists

### Architecture

```
AKAZE Module
├── NonLinearDiffusion (per scale)
│   ├── Sobel gradient computation
│   ├── Perona-Malik conduction function
│   └── Fixed FED iterations
├── HessianDetector
│   ├── Hessian matrix computation
│   ├── Determinant calculation
│   └── MaxPool2d-based NMS
└── OrientationEstimator
    ├── Gaussian-weighted coordinate grids
    └── Convolution-based moment computation
```

### Multi-Scale Processing

The forward pass processes multiple scales sequentially:

1. **For each scale**:
   - Apply non-linear diffusion to build the scale space
   - Compute feature scores using Hessian detector
   - Compute orientations using intensity centroid method
   - Store both scores and orientations

2. **Scale selection**:
   - Stack all scale scores and orientations
   - Find the scale with maximum response at each pixel using `torch.max()`
   - Select the corresponding orientation from that scale using one-hot encoding and tensor multiplication

This ensures that the orientation at each pixel corresponds to the scale where the feature was most strongly detected, which is crucial for accurate feature description and matching.

## References

1. **AKAZE Features**: Pablo F. Alcantarilla, Jesús Nuevo, Adrien Bartoli.
   *"Fast Explicit Diffusion for Accelerated Features in Nonlinear Scale Spaces"*
   BMVC 2013

2. **FED Scheme**: S. Grewenig, J. Weickert, C. Schroers, A. Bruhn.
   *"Cyclic Schemes for PDE-Based Image Analysis"*

## Example Output

After running the model on an input image (1, 1, 480, 640):

- **Scores**: (1, 1, 480, 640) - Higher values indicate stronger features
- **Orientations**: (1, 1, 480, 640) - Orientation in radians at each pixel

Post-processing (not included in ONNX model):
- Threshold scores to find keypoints
- Extract keypoint locations from score map
- Use orientation map to get keypoint orientations

## Notes

- Input images should be grayscale (single channel)
- Input values can be in range [0, 1] or [0, 255]
- The model outputs dense maps, not sparse keypoint lists
- For keypoint extraction, apply thresholding and coordinate extraction in post-processing
- All operations are differentiable (can be used in training pipelines)
