#!/usr/bin/env python3
"""
Feature detection sample using Shi-Tomasi + BAD ONNX model.

Runs feature point detection on an exported ONNX model using onnxruntime
and visualizes the results. Keypoints are selected by thresholding and
non-maximum suppression on Shi-Tomasi corner scores.

Usage:
    # First, export the ONNX model:
    python onnx_export/export_shi_tomasi_bad.py -o shi_tomasi_bad.onnx -H 480 -W 640

    # Then, run the sample:
    python sample/feature_detection.py --model shi_tomasi_bad.onnx --input image.png --output result.png
    python sample/feature_detection.py --model shi_tomasi_bad.onnx --input image.png --max-keypoints 500
    python sample/feature_detection.py --model shi_tomasi_bad.onnx --input image.png --threshold 0.01
"""

import argparse

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image file as a grayscale float32 array.

    Args:
        image_path: Path to the input image.

    Returns:
        Grayscale image array of shape (1, 1, H, W) with values in [0, 255].
    """
    img = Image.open(image_path).convert("L")
    arr = np.array(img, dtype=np.float32)
    return arr[np.newaxis, np.newaxis, :, :]  # (1, 1, H, W)


def nms_keypoints(scores: np.ndarray, nms_radius: int = 3) -> np.ndarray:
    """
    Apply non-maximum suppression to corner scores using a sliding window.

    Keeps only pixels whose score is the local maximum within a
    (2*nms_radius+1) x (2*nms_radius+1) neighborhood.

    Args:
        scores: Corner score map of shape (H, W).
        nms_radius: Radius of the NMS window. Default is 3.

    Returns:
        Score map with non-maxima suppressed to zero, shape (H, W).
    """
    H, W = scores.shape
    suppressed = np.zeros_like(scores)

    for y in range(H):
        for x in range(W):
            if scores[y, x] == 0:
                continue
            y_min = max(0, y - nms_radius)
            y_max = min(H, y + nms_radius + 1)
            x_min = max(0, x - nms_radius)
            x_max = min(W, x + nms_radius + 1)
            local_patch = scores[y_min:y_max, x_min:x_max]
            if scores[y, x] >= local_patch.max():
                suppressed[y, x] = scores[y, x]

    return suppressed


def select_keypoints(
    scores: np.ndarray,
    threshold: float = 0.01,
    max_keypoints: int = 1000,
    nms_radius: int = 3,
) -> np.ndarray:
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
        Keypoint array of shape (N, 3) where each row is (y, x, score).
        Returns empty array of shape (0, 3) if no keypoints are found.
    """
    score_map = scores[0, 0]  # (H, W)

    # Apply NMS
    score_map = nms_keypoints(score_map, nms_radius=nms_radius)

    # Threshold
    ys, xs = np.where(score_map > threshold)
    if len(ys) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    vals = score_map[ys, xs]

    # Sort by score descending and take top-k
    top_k = min(max_keypoints, len(vals))
    indices = np.argsort(vals)[::-1][:top_k]

    keypoints = np.stack([
        ys[indices].astype(np.float32),
        xs[indices].astype(np.float32),
        vals[indices],
    ], axis=-1)  # (N, 3)
    return keypoints


def visualize_keypoints(
    image_path: str,
    keypoints: np.ndarray,
    output_path: str,
    circle_radius: int = 3,
    color: tuple = (0, 255, 0),
) -> None:
    """
    Draw detected keypoints on the image and save the result.

    Args:
        image_path: Path to the original input image.
        keypoints: Keypoint array of shape (N, 3) where each row
                   is (y, x, score).
        output_path: Path to save the output visualization image.
        circle_radius: Radius of the keypoint circles. Default is 3.
        color: RGB color tuple for drawing keypoints. Default is green.
    """
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    for i in range(keypoints.shape[0]):
        y = int(keypoints[i, 0])
        x = int(keypoints[i, 1])
        r = circle_radius
        draw.ellipse(
            [x - r, y - r, x + r, y + r],
            outline=color,
            width=1,
        )

    img.save(output_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Feature detection sample using Shi-Tomasi + BAD ONNX model"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Path to the exported ONNX model file"
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
        "--circle-radius",
        type=int,
        default=3,
        help="Radius of keypoint circles in visualization (default: 3)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Create ONNX Runtime session
    session = ort.InferenceSession(args.model)
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]
    print(f"ONNX model: {args.model}")
    print(f"  Input:   {input_name} {session.get_inputs()[0].shape}")
    print(f"  Outputs: {output_names}")

    # Load image
    image = load_image(args.input)
    print(f"Input image: {args.input} ({image.shape[2]}x{image.shape[3]})")

    # Run inference
    results = session.run(output_names, {input_name: image})
    scores = results[0]       # (1, 1, H, W)
    descriptors = results[1]  # (1, num_pairs, H, W)

    print(f"Score map range: [{scores.min():.4f}, {scores.max():.4f}]")
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
