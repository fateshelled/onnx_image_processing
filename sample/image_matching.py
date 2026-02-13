#!/usr/bin/env python3
"""
Image matching sample using Shi-Tomasi + BAD + Sinkhorn ONNX model.

Runs end-to-end feature matching between two images using the exported
ONNX model and visualizes the matched keypoint pairs.

Usage:
    # First, export the ONNX model:
    python onnx_export/export_shi_tomasi_bad_sinkhorn.py -o shi_tomasi_bad_sinkhorn.onnx -H 480 -W 640

    # Then, run the sample:
    python sample/image_matching.py --model shi_tomasi_bad_sinkhorn.onnx --input1 image1.png --input2 image2.png
    python sample/image_matching.py --model shi_tomasi_bad_sinkhorn.onnx --input1 image1.png --input2 image2.png --threshold 0.1
    python sample/image_matching.py --model shi_tomasi_bad_sinkhorn.onnx --input1 image1.png --input2 image2.png --max-matches 50
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pytorch_model.matching.outlier_filters import (
    probability_ratio_filter,
    dustbin_margin_filter,
)


def load_image(image_path: str, height: int, width: int) -> tuple[np.ndarray, Image.Image]:
    """
    Load an image file as a grayscale float32 array, resized to the model input size.

    Args:
        image_path: Path to the input image.
        height: Target height to resize.
        width: Target width to resize.

    Returns:
        Tuple of:
            - Grayscale image array of shape (1, 1, H, W) with values in [0, 255].
            - Resized PIL Image in RGB mode (for visualization).
    """
    img = Image.open(image_path).convert("L")
    img_resized = img.resize((width, height), Image.BILINEAR)
    arr = np.array(img_resized, dtype=np.float32)
    rgb = img_resized.convert("RGB")
    return arr[np.newaxis, np.newaxis, :, :], rgb


def extract_matches(
    matching_probs: np.ndarray,
    keypoints1: np.ndarray,
    keypoints2: np.ndarray,
    threshold: float = 0.1,
    max_matches: int = 100,
    ratio_threshold: float = None,
    dustbin_margin: float = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract mutual nearest-neighbor matches from the Sinkhorn probability matrix.

    A match between keypoint i in image1 and keypoint j in image2 is accepted
    when both of the following hold:
    - j is the argmax of P[i, :M] (best match for i)
    - i is the argmax of P[:N, j] (best match for j)
    and the match probability exceeds the threshold.

    Optionally, additional outlier filters can be applied:
    - Probability ratio filter: Rejects ambiguous matches where the best match
      probability is not significantly higher than the second-best.
    - Dustbin margin filter: Rejects matches where the dustbin (unmatched)
      probability is too high relative to the best match probability.

    Args:
        matching_probs: Sinkhorn probability matrix of shape (1, K+1, K+1).
        keypoints1: Keypoints in image1 of shape (1, K, 2) as (y, x).
        keypoints2: Keypoints in image2 of shape (1, K, 2) as (y, x).
        threshold: Minimum match probability. Default is 0.1.
        max_matches: Maximum number of matches to return. Default is 100.
        ratio_threshold: Minimum ratio between best and second-best match probabilities.
                        If None, ratio filtering is disabled. Default is None.
        dustbin_margin: Minimum margin between best match and dustbin probabilities.
                       If None, dustbin margin filtering is disabled. Default is None.

    Returns:
        Tuple of:
            - matched_kpts1: (N, 2) matched keypoint coordinates in image1.
            - matched_kpts2: (N, 2) matched keypoint coordinates in image2.
            - match_scores: (N,) match probability scores.
    """
    P = matching_probs[0]  # (K+1, K+1)
    kpts1 = keypoints1[0]  # (K, 2)
    kpts2 = keypoints2[0]  # (K, 2)

    K = kpts1.shape[0]

    # Core probability matrix excluding dustbin
    P_core = P[:K, :K]  # (K, K)

    # Mutual nearest neighbors
    max_j_for_i = np.argmax(P_core, axis=1)  # (K,) best match in image2 for each i
    max_i_for_j = np.argmax(P_core, axis=0)  # (K,) best match in image1 for each j

    # Check mutual consistency: i matches j AND j matches i
    mutual_mask = np.zeros(K, dtype=bool)
    for i in range(K):
        j = max_j_for_i[i]
        if max_i_for_j[j] == i:
            mutual_mask[i] = True

    # Apply probability ratio filter if enabled
    if ratio_threshold is not None:
        ratio_mask = probability_ratio_filter(P_core, ratio_threshold)
        mutual_mask = mutual_mask & ratio_mask

    # Apply dustbin margin filter if enabled
    if dustbin_margin is not None:
        dustbin_mask = dustbin_margin_filter(P, dustbin_margin)
        mutual_mask = mutual_mask & dustbin_mask

    # Get match probabilities
    match_indices_i = np.where(mutual_mask)[0]
    match_indices_j = max_j_for_i[match_indices_i]
    scores = P_core[match_indices_i, match_indices_j]

    # Apply threshold
    above_threshold = scores >= threshold
    match_indices_i = match_indices_i[above_threshold]
    match_indices_j = match_indices_j[above_threshold]
    scores = scores[above_threshold]

    # Sort by score descending and take top matches
    sort_order = np.argsort(scores)[::-1][:max_matches]
    match_indices_i = match_indices_i[sort_order]
    match_indices_j = match_indices_j[sort_order]
    scores = scores[sort_order]

    matched_kpts1 = kpts1[match_indices_i]  # (N, 2)
    matched_kpts2 = kpts2[match_indices_j]  # (N, 2)

    return matched_kpts1, matched_kpts2, scores


def score_to_color(normalized_score: float) -> tuple[int, int, int]:
    """
    Map a normalized score in [0, 1] to an RGB color using a jet-like colormap.

    Args:
        normalized_score: Score value in [0, 1].

    Returns:
        RGB tuple with values in [0, 255].
    """
    t = max(0.0, min(1.0, normalized_score))
    if t < 0.25:
        r = 0
        g = int(255 * (t / 0.25))
        b = 255
    elif t < 0.5:
        r = 0
        g = 255
        b = int(255 * (1.0 - (t - 0.25) / 0.25))
    elif t < 0.75:
        r = int(255 * ((t - 0.5) / 0.25))
        g = 255
        b = 0
    else:
        r = 255
        g = int(255 * (1.0 - (t - 0.75) / 0.25))
        b = 0
    return (r, g, b)


def visualize_matches(
    image1: Image.Image,
    image2: Image.Image,
    matched_kpts1: np.ndarray,
    matched_kpts2: np.ndarray,
    scores: np.ndarray,
    output_path: str,
    circle_radius: int = 3,
    line_width: int = 1,
    colorize_by_score: bool = False,
) -> None:
    """
    Create a side-by-side visualization of matched keypoints.

    Draws both images side by side with lines connecting matched keypoints.

    Args:
        image1: First RGB PIL Image.
        image2: Second RGB PIL Image.
        matched_kpts1: (N, 2) matched keypoint coordinates in image1 as (y, x).
        matched_kpts2: (N, 2) matched keypoint coordinates in image2 as (y, x).
        scores: (N,) match scores.
        output_path: Path to save the output visualization image.
        circle_radius: Radius of keypoint circles. Default is 3.
        line_width: Width of match lines. Default is 1.
        colorize_by_score: Color matches by score. Default is False.
    """
    w1, h1 = image1.size
    w2, h2 = image2.size

    # Create side-by-side canvas
    canvas_width = w1 + w2
    canvas_height = max(h1, h2)
    canvas = Image.new("RGB", (canvas_width, canvas_height), (0, 0, 0))
    canvas.paste(image1, (0, 0))
    canvas.paste(image2, (w1, 0))

    draw = ImageDraw.Draw(canvas)

    # Precompute normalized scores
    if colorize_by_score and len(scores) > 0:
        score_min = scores.min()
        score_max = scores.max()
        if score_max > score_min:
            normalized = (scores - score_min) / (score_max - score_min)
        else:
            normalized = np.ones_like(scores)

    for i in range(len(scores)):
        y1 = int(matched_kpts1[i, 0])
        x1 = int(matched_kpts1[i, 1])
        y2 = int(matched_kpts2[i, 0])
        x2 = int(matched_kpts2[i, 1]) + w1  # offset for side-by-side

        if colorize_by_score:
            color = score_to_color(normalized[i])
        else:
            color = (0, 255, 0)

        # Draw keypoint circles
        r = circle_radius
        draw.ellipse([x1 - r, y1 - r, x1 + r, y1 + r], outline=color, width=1)
        draw.ellipse([x2 - r, y2 - r, x2 + r, y2 + r], outline=color, width=1)

        # Draw match line
        draw.line([(x1, y1), (x2, y2)], fill=color, width=line_width)

    canvas.save(output_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Image matching sample using Shi-Tomasi + BAD + Sinkhorn ONNX model"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Path to the exported ONNX model file"
    )
    parser.add_argument(
        "--input1", "-i1",
        type=str,
        required=True,
        help="First input image path"
    )
    parser.add_argument(
        "--input2", "-i2",
        type=str,
        required=True,
        help="Second input image path"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="image_matching_result.png",
        help="Output visualization image path (default: image_matching_result.png)"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.1,
        help="Match probability threshold (default: 0.1)"
    )
    parser.add_argument(
        "--ratio-threshold",
        type=float,
        default=None,
        help="Probability ratio threshold for outlier filtering. "
             "Minimum ratio between best and second-best match probabilities. "
             "Higher values are more strict (e.g., 2.0 means best must be 2x better). "
             "If not specified, ratio filtering is disabled."
    )
    parser.add_argument(
        "--dustbin-margin",
        type=float,
        default=None,
        help="Dustbin margin threshold for outlier filtering. "
             "Minimum margin between best match probability and dustbin probability. "
             "Higher values are more strict (e.g., 0.3 means best match must be 0.3 higher). "
             "If not specified, dustbin margin filtering is disabled."
    )
    parser.add_argument(
        "--max-matches",
        type=int,
        default=100,
        help="Maximum number of matches to visualize (default: 100)"
    )
    parser.add_argument(
        "--circle-radius",
        type=int,
        default=3,
        help="Radius of keypoint circles in visualization (default: 3)"
    )
    parser.add_argument(
        "--line-width",
        type=int,
        default=1,
        help="Width of match lines in visualization (default: 1)"
    )
    parser.add_argument(
        "--colorize",
        action="store_true",
        help="Colorize matches by score (blue=low, red=high)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Create ONNX Runtime session
    session = ort.InferenceSession(args.model)
    inputs = session.get_inputs()
    outputs = session.get_outputs()

    input_names = [inp.name for inp in inputs]
    output_names = [out.name for out in outputs]

    # Get model input dimensions
    input_shape1 = inputs[0].shape  # [B, 1, H, W]
    input_shape2 = inputs[1].shape  # [B, 1, H, W]
    model_height1 = input_shape1[2]
    model_width1 = input_shape1[3]
    model_height2 = input_shape2[2]
    model_width2 = input_shape2[3]

    print(f"ONNX model: {args.model}")
    for inp in inputs:
        print(f"  Input:  {inp.name} {inp.shape}")
    for out in outputs:
        print(f"  Output: {out.name} {out.shape}")

    # Load images
    image1, image1_rgb = load_image(args.input1, model_height1, model_width1)
    image2, image2_rgb = load_image(args.input2, model_height2, model_width2)
    print(f"Image 1: {args.input1} (resized to {model_height1}x{model_width1})")
    print(f"Image 2: {args.input2} (resized to {model_height2}x{model_width2})")

    # Run inference
    results = session.run(
        output_names,
        {input_names[0]: image1, input_names[1]: image2},
    )
    keypoints1 = results[0]       # (1, K, 2)
    keypoints2 = results[1]       # (1, K, 2)
    matching_probs = results[2]   # (1, K+1, K+1)

    print(f"Keypoints in image 1: {keypoints1.shape[1]}")
    print(f"Keypoints in image 2: {keypoints2.shape[1]}")
    print(f"Matching probability matrix shape: {list(matching_probs.shape)}")

    # Extract matches
    matched_kpts1, matched_kpts2, scores = extract_matches(
        matching_probs,
        keypoints1,
        keypoints2,
        threshold=args.threshold,
        max_matches=args.max_matches,
        ratio_threshold=args.ratio_threshold,
        dustbin_margin=args.dustbin_margin,
    )

    filter_info = f"threshold={args.threshold}"
    if args.ratio_threshold is not None:
        filter_info += f", ratio_threshold={args.ratio_threshold}"
    if args.dustbin_margin is not None:
        filter_info += f", dustbin_margin={args.dustbin_margin}"
    print(f"Matches found: {len(scores)} ({filter_info})")

    if len(scores) == 0:
        print("No matches found. Try lowering --threshold.")
        return

    print(f"Match score range: [{scores.min():.4f}, {scores.max():.4f}]")

    # Visualize
    visualize_matches(
        image1_rgb,
        image2_rgb,
        matched_kpts1,
        matched_kpts2,
        scores,
        args.output,
        circle_radius=args.circle_radius,
        line_width=args.line_width,
        colorize_by_score=args.colorize,
    )
    print(f"Saved visualization to: {args.output}")


if __name__ == "__main__":
    main()
