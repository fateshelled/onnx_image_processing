#!/usr/bin/env python3
"""
Feature detection sample using Shi-Tomasi + BAD.

Detects feature points using the unified ShiTomasiBADDetector model
and visualizes the results. Keypoints are selected by thresholding and
non-maximum suppression on Shi-Tomasi corner scores.

Usage:
    python sample/feature_detection.py --input image.png --output result.png
    python sample/feature_detection.py --input image.png --output result.png --max-keypoints 500
    python sample/feature_detection.py --input image.png --output result.png --threshold 0.01
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Add parent directory to path for importing pytorch_model
sys.path.insert(0, str(Path(__file__).parent.parent))

from pytorch_model.feature_detection.shi_tomasi_bad import ShiTomasiBADDetector


def load_image_as_tensor(image_path: str) -> torch.Tensor:
    """
    Load an image file as a grayscale float tensor.

    Args:
        image_path: Path to the input image.

    Returns:
        Grayscale image tensor of shape (1, 1, H, W) with values in [0, 255].
    """
    from PIL import Image
    import numpy as np

    img = Image.open(image_path).convert("L")
    arr = np.array(img, dtype=np.float32)
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    return tensor


def nms_keypoints(
    scores: torch.Tensor,
    nms_radius: int = 3,
) -> torch.Tensor:
    """
    Apply non-maximum suppression to corner scores.

    Keeps only pixels whose score is the local maximum within a
    (2*nms_radius+1) x (2*nms_radius+1) neighborhood.

    Args:
        scores: Corner score map of shape (1, 1, H, W).
        nms_radius: Radius of the NMS window. Default is 3.

    Returns:
        Score map with non-maxima suppressed to zero.
    """
    kernel_size = 2 * nms_radius + 1
    padding = nms_radius
    local_max = F.max_pool2d(
        scores,
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
    )
    is_max = (scores == local_max)
    return scores * is_max.float()


def select_keypoints(
    scores: torch.Tensor,
    threshold: float = 0.01,
    max_keypoints: int = 1000,
    nms_radius: int = 3,
) -> torch.Tensor:
    """
    Select keypoints from corner score map.

    Applies NMS and threshold filtering, then returns the top-k keypoints
    sorted by score in descending order.

    Args:
        scores: Corner score map of shape (1, 1, H, W).
        threshold: Minimum score threshold. Default is 0.01.
        max_keypoints: Maximum number of keypoints to return. Default is 1000.
        nms_radius: Radius of the NMS window. Default is 3.

    Returns:
        Keypoint coordinates of shape (N, 3) where each row is (y, x, score).
    """
    # Apply NMS
    nms_scores = nms_keypoints(scores, nms_radius=nms_radius)
    score_map = nms_scores.squeeze(0).squeeze(0)  # (H, W)

    # Threshold
    mask = score_map > threshold
    if mask.sum() == 0:
        return torch.zeros(0, 3)

    ys, xs = torch.where(mask)
    vals = score_map[mask]

    # Sort by score descending and take top-k
    top_k = min(max_keypoints, vals.shape[0])
    _, indices = torch.topk(vals, top_k)
    ys = ys[indices]
    xs = xs[indices]
    vals = vals[indices]

    keypoints = torch.stack([ys.float(), xs.float(), vals], dim=-1)  # (N, 3)
    return keypoints


def visualize_keypoints(
    image_path: str,
    keypoints: torch.Tensor,
    output_path: str,
    circle_radius: int = 3,
    color: tuple = (0, 255, 0),
) -> None:
    """
    Draw detected keypoints on the image and save the result.

    Args:
        image_path: Path to the original input image.
        keypoints: Keypoint coordinates of shape (N, 3) where each row
                   is (y, x, score).
        output_path: Path to save the output visualization image.
        circle_radius: Radius of the keypoint circles. Default is 3.
        color: BGR color tuple for drawing keypoints. Default is green.
    """
    from PIL import Image, ImageDraw

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    for i in range(keypoints.shape[0]):
        y = int(keypoints[i, 0].item())
        x = int(keypoints[i, 1].item())
        r = circle_radius
        draw.ellipse(
            [x - r, y - r, x + r, y + r],
            outline=color,
            width=1,
        )

    img.save(output_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Feature detection sample using Shi-Tomasi + BAD"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input image path"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="feature_detection_result.png",
        help="Output visualization image path (default: feature_detection_result.png)"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.01,
        help="Score threshold for keypoint selection (default: 0.01)"
    )
    parser.add_argument(
        "--max-keypoints", "-k",
        type=int,
        default=1000,
        help="Maximum number of keypoints to detect (default: 1000)"
    )
    parser.add_argument(
        "--nms-radius",
        type=int,
        default=3,
        help="Non-maximum suppression radius (default: 3)"
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=3,
        help="Block size for Shi-Tomasi (default: 3)"
    )
    parser.add_argument(
        "--num-pairs",
        type=int,
        default=256,
        help="Number of BAD descriptor pairs (default: 256)"
    )
    parser.add_argument(
        "--box-size",
        type=int,
        default=5,
        help="Box size for BAD averaging (default: 5)"
    )
    parser.add_argument(
        "--pattern-scale",
        type=float,
        default=16.0,
        help="Pattern scale for BAD sampling offsets (default: 16.0)"
    )
    parser.add_argument(
        "--circle-radius",
        type=int,
        default=3,
        help="Radius of keypoint circles in visualization (default: 3)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Build unified model
    model = ShiTomasiBADDetector(
        block_size=args.block_size,
        num_pairs=args.num_pairs,
        box_size=args.box_size,
        pattern_scale=args.pattern_scale,
    )
    model.eval()

    # Load image
    image = load_image_as_tensor(args.input)
    print(f"Input image: {args.input} ({image.shape[2]}x{image.shape[3]})")

    # Run feature detection
    with torch.no_grad():
        scores, descriptors = model(image)

    print(f"Score map range: [{scores.min().item():.4f}, {scores.max().item():.4f}]")
    print(f"Descriptor map shape: {list(descriptors.shape)}")

    # Select keypoints
    keypoints = select_keypoints(
        scores,
        threshold=args.threshold,
        max_keypoints=args.max_keypoints,
        nms_radius=args.nms_radius,
    )
    print(f"Detected keypoints: {keypoints.shape[0]}")

    if keypoints.shape[0] == 0:
        print("No keypoints detected. Try lowering --threshold.")
        return

    # Visualize
    visualize_keypoints(
        args.input,
        keypoints,
        args.output,
        circle_radius=args.circle_radius,
    )
    print(f"Saved visualization to: {args.output}")


if __name__ == "__main__":
    main()
