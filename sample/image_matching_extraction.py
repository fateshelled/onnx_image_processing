#!/usr/bin/env python3
"""
Image matching sample using ONNX models with built-in match extraction.

This script works with ONNX models exported with the --with-extraction flag,
which directly output matched keypoint pairs instead of probability matrices.

Usage:
    # First, export an ONNX model with match extraction:
    python onnx_export/export_shi_tomasi_bad_sinkhorn.py \
        -o model_with_extraction.onnx -H 480 -W 640 \
        --with-extraction --max-matches 100 --match-threshold 0.1

    # Then, run this sample:
    python sample/image_matching_extraction.py --model model_with_extraction.onnx --input1 image1.png --input2 image2.png
    python sample/image_matching_extraction.py --model model_with_extraction.onnx --input1 image1.png --input2 image2.png --colorize
"""

import argparse
import time

import numpy as np
from PIL import Image, ImageDraw

from provider_utils import create_session


def load_image(
    image_path: str, height: int, width: int
) -> tuple[np.ndarray, Image.Image]:
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
        description="Image matching sample using ONNX models with built-in match extraction"
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        help="Path to the ONNX model file (exported with --with-extraction)",
    )
    parser.add_argument(
        "--input1", "-i1", type=str, required=True, help="First input image path"
    )
    parser.add_argument(
        "--input2", "-i2", type=str, required=True, help="Second input image path"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="image_matching_extraction_result.png",
        help="Output visualization image path (default: image_matching_extraction_result.png)",
    )
    parser.add_argument(
        "--circle-radius",
        type=int,
        default=3,
        help="Radius of keypoint circles in visualization (default: 3)",
    )
    parser.add_argument(
        "--line-width",
        type=int,
        default=1,
        help="Width of match lines in visualization (default: 1)",
    )
    parser.add_argument(
        "--colorize",
        action="store_true",
        help="Colorize matches by score (blue=low, red=high)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Create ONNX Runtime session
    session = create_session(args.model)
    inputs = session.get_inputs()
    outputs = session.get_outputs()

    input_names = [inp.name for inp in inputs]
    output_names = [out.name for out in outputs]

    # Verify this is a model with match extraction
    expected_outputs = ["matched_kpts1", "matched_kpts2", "scores", "valid_mask"]
    if not all(name in output_names for name in expected_outputs):
        print(f"Error: This model does not have match extraction outputs.")
        print(f"Expected outputs: {expected_outputs}")
        print(f"Found outputs: {output_names}")
        print(f"\nPlease export your model with the --with-extraction flag.")
        return

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

    # Run inference and Time measurement
    WARMUP_ITER = 5
    MEASURE_ITER = 10
    for _ in range(WARMUP_ITER):
        results = session.run(
            output_names,
            {input_names[0]: image1, input_names[1]: image2},
        )

    elapsed = 0.0
    for _ in range(MEASURE_ITER):  # warmup
        start_time = time.time()
        results = session.run(
            output_names,
            {input_names[0]: image1, input_names[1]: image2},
        )
        elapsed += time.time() - start_time

    # Extract results
    matched_kpts1 = results[0]  # (1, max_matches, 2)
    matched_kpts2 = results[1]  # (1, max_matches, 2)
    scores = results[2]  # (1, max_matches)
    valid_mask = results[3]  # (1, max_matches) - float type for TensorRT compatibility

    # Filter valid matches (convert float mask to bool)
    valid_mask_bool = valid_mask[0] > 0.5
    matched_kpts1 = matched_kpts1[0][valid_mask_bool]  # (N, 2)
    matched_kpts2 = matched_kpts2[0][valid_mask_bool]  # (N, 2)
    scores = scores[0][valid_mask_bool]  # (N,)

    print(f"\nMatches found: {len(scores)}")

    if len(scores) == 0:
        print("No matches found. The model may need different parameters:")
        print("  - Try exporting with a lower --match-threshold")
        print("  - Try exporting with a higher --max-matches")
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
    print(f"Elapsed: {elapsed * 1000 / MEASURE_ITER} ms/frame")


if __name__ == "__main__":
    main()
