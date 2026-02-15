#!/usr/bin/env python3
"""
FAST corner detection usage example.

Demonstrates how to use the FASTScore module to detect corners in an image
and visualize the results.

Usage:
    python examples/fast_usage_example.py
"""

import sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image, ImageDraw

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pytorch_model.detector.fast import FASTScore


def load_image(image_path: str) -> tuple[torch.Tensor, Image.Image]:
    """
    Load an image file as a grayscale tensor.

    Args:
        image_path: Path to the input image.

    Returns:
        Tuple of:
            - Grayscale image tensor of shape (1, 1, H, W) with values in [0, 255].
            - Original PIL Image in RGB mode (for visualization).
    """
    img = Image.open(image_path).convert("L")
    arr = np.array(img, dtype=np.float32)
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    rgb = img.convert("RGB")
    return tensor, rgb


def extract_keypoints(score_map: torch.Tensor) -> np.ndarray:
    """
    Extract keypoint coordinates from FAST score map.

    Args:
        score_map: FAST score map of shape (1, 1, H, W) with binary values.

    Returns:
        Keypoint array of shape (N, 2) where each row is (y, x).
    """
    # Convert to numpy and squeeze
    scores = score_map.squeeze().cpu().numpy()  # (H, W)

    # Find non-zero positions
    ys, xs = np.where(scores > 0.5)

    if len(ys) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    keypoints = np.stack([ys, xs], axis=-1).astype(np.float32)  # (N, 2)
    return keypoints


def visualize_keypoints(
    image: Image.Image,
    keypoints: np.ndarray,
    output_path: str,
    circle_radius: int = 3,
    color: tuple = (0, 255, 0),
) -> None:
    """
    Draw detected keypoints on the image and save the result.

    Args:
        image: RGB PIL Image to draw on.
        keypoints: Keypoint array of shape (N, 2) where each row is (y, x).
        output_path: Path to save the output visualization image.
        circle_radius: Radius of the keypoint circles. Default is 3.
        color: RGB color tuple for drawing keypoints. Default is green.
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)

    for i in range(keypoints.shape[0]):
        y = int(keypoints[i, 0])
        x = int(keypoints[i, 1])
        r = circle_radius
        draw.ellipse(
            [x - r, y - r, x + r, y + r],
            outline=color,
            width=2,
        )

    img.save(output_path)
    print(f"Saved visualization to: {output_path}")


def main():
    # Example parameters
    image_path = "test_image.jpg"  # Replace with your image path
    output_path = "fast_result.png"
    threshold = 20
    use_nms = True

    # Create FAST detector
    fast_detector = FASTScore(threshold=threshold, use_nms=use_nms, nms_radius=3)
    fast_detector.eval()

    # Load image
    print(f"Loading image: {image_path}")
    image_tensor, image_rgb = load_image(image_path)
    print(f"  Image shape: {image_tensor.shape}")

    # Detect corners
    print(f"Detecting FAST corners (threshold={threshold}, NMS={use_nms})...")
    with torch.no_grad():
        score_map = fast_detector(image_tensor)

    print(f"  Score map shape: {score_map.shape}")
    print(f"  Detected corners: {(score_map > 0.5).sum().item()}")

    # Extract keypoints
    keypoints = extract_keypoints(score_map)
    print(f"  Keypoint coordinates: {keypoints.shape[0]} points")

    if keypoints.shape[0] == 0:
        print("No keypoints detected. Try lowering the threshold.")
        return

    # Visualize
    visualize_keypoints(image_rgb, keypoints, output_path, circle_radius=2)


if __name__ == "__main__":
    main()
