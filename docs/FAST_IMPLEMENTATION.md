# FAST Corner Detection - PyTorch Implementation

## Overview

This is an ONNX-friendly PyTorch implementation of FAST (Features from Accelerated Segment Test) corner detection, optimized using the "Faster than FAST" (2025) binary encoding strategy.

## Features

- **ONNX-Compatible**: No dynamic loops or conditional branches
- **GPU-Optimized**: Eliminates warp divergence through binary encoding
- **Batched Processing**: Supports batch processing for multiple images
- **Non-Maximum Suppression**: Optional NMS for cleaner results
- **Efficient**: Uses bitwise operations instead of if-else branches

## Algorithm Overview

### Traditional FAST

FAST detects corners by checking if 9 or more contiguous pixels in a circle of 16 pixels around a candidate point are all brighter or all darker than the center pixel by more than a threshold.

Traditional implementations use:
1. Pre-check: Test 4 diagonal pairs
2. Loop-based counting: Increment counter for consecutive dark/bright pixels
3. Early exit: Break when 9 consecutive found

**Problem on GPU**: If-else branches (~6% of instructions) and early-exit loops cause warp divergence and idle threads.

### Optimized Strategy: Binary Encoding

Our implementation uses a branch-free approach:

#### Step 1: Encode Differences (32-bit buffer)
```
For each of 16 circle pixels:
  Dt = I_circle - I_center

  If Dt >= threshold:
    dark_bit = 1, bright_bit = 0
  Else if Dt <= -threshold:
    dark_bit = 0, bright_bit = 1
  Else:
    dark_bit = 0, bright_bit = 0

Encode into 32-bit integer:
  buffer_32b = (dark_bits << 16) | bright_bits
```

Uses conditional operators (`torch.where`) instead of if-else, compiling to branch-free select instructions.

#### Step 2: Create Circular Buffer (24-bit)
```
To handle wraparound (e.g., bits 14,15,16,1,2,...):
  buffer_24b = bits_16 | (bits_16[0:8] << 16)
```

This converts circular pattern detection to linear pattern detection.

#### Step 3: Detect 9 Consecutive Bits
```
For i in [0..15]:
  segment = (buffer_24b >> i) & 0x1FF  # Extract 9 bits
  if segment == 0x1FF:  # All 9 bits set?
    corner detected
```

All 16 checks are done in parallel using tensor operations.

## Usage

### PyTorch Model

```python
from pytorch_model.detector.fast import FASTScore

# Create detector
fast = FASTScore(threshold=20, use_nms=True, nms_radius=3)
fast.eval()

# Detect corners
image = torch.randn(1, 1, 480, 640) * 255  # (N, 1, H, W)
score_map = fast(image)  # (N, 1, H, W) - binary 0/1

# Extract coordinates
corners = (score_map > 0.5).nonzero()[:, 2:]  # (num_corners, 2)
```

### Export to ONNX

```bash
python onnx_export/export_fast.py \
  --output fast.onnx \
  --height 480 \
  --width 640 \
  --threshold 20 \
  --use-nms
```

### Run Inference

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("fast.onnx")
image = np.random.randn(1, 1, 480, 640).astype(np.float32) * 255
score_map = session.run(["output"], {"input": image})[0]
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | int | 20 | Intensity difference threshold |
| `use_nms` | bool | False | Enable non-maximum suppression |
| `nms_radius` | int | 3 | NMS neighborhood radius |

## Implementation Details

### Circle Pattern (Bresenham, radius=3)

16 pixels in clockwise order from top:
```
        12 11 10  9
      13           8
    14               7
   15                 6
   16                 5
    1                4
      2            3
```

Offsets: `[(0,-3), (1,-3), (2,-2), (3,-1), (3,0), (3,1), ...]`

### Border Handling

Images are padded with 3 pixels on all sides using replication mode to handle border pixels correctly.

### Output

- Binary score map: `1.0` for detected corners, `0.0` otherwise
- Same spatial dimensions as input
- Can be converted to keypoint list via `torch.nonzero()`

## Performance Characteristics

- **No branches**: All `if-else` replaced with `torch.where`
- **No loops**: 16 shift operations done in parallel via `torch.stack`
- **Warp-friendly**: All threads in a warp execute same instructions
- **Memory-efficient**: Only stores 32-bit buffer per pixel temporarily

## Comparison with OpenCV FAST

| Aspect | OpenCV FAST | This Implementation |
|--------|-------------|---------------------|
| Language | C++ | PyTorch |
| ONNX Export | ❌ | ✅ |
| GPU Batching | ❌ | ✅ |
| Dynamic Loops | ✅ | ❌ |
| Branch Instructions | ~6% | ~0% |
| Sub-pixel Refinement | ✅ | Future work |

## Future Enhancements

- [ ] Sub-pixel refinement (parabola fitting)
- [ ] Score computation (not just binary)
- [ ] Multi-scale detection
- [ ] Adaptive thresholding

## References

1. **Faster than FAST** (2025) - Binary encoding optimization strategy
2. Rosten, E. & Drummond, T. (2006). "Machine learning for high-speed corner detection." ECCV 2006.
3. CUDA_ORB implementation - Baseline for optimization

## License

Same as parent project.
