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
    python sample/feature_detection.py --model shi_tomasi_bad.onnx --input image.png --colorize
"""

import argparse

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw


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
    return arr[np.newaxis, np.newaxis, :, :], rgb  # (1, 1, H, W)


def refine_keypoints_subpixel(
    score_map: np.ndarray,
    keypoints: np.ndarray,
) -> np.ndarray:
    """
    Refine keypoint positions to sub-pixel accuracy using parabola fitting.

    For each keypoint, fits a 1D parabola independently along the y and x
    axes using the 3 neighboring score values, then computes the sub-pixel
    offset of the peak.

    Given three consecutive values f(-1), f(0), f(1) along one axis, the
    parabola peak offset is:
        delta = (f(-1) - f(1)) / (2 * (f(-1) - 2*f(0) + f(1)))

    The offset is applied only when the fitted parabola is concave
    (negative second derivative) and the shift is less than 1 pixel, i.e.
    the refinement is numerically stable. Keypoints on image borders where
    a full 3-point neighborhood is unavailable are left unchanged.

    Args:
        score_map: Corner score map of shape (H, W).
        keypoints: Keypoint array of shape (N, 3) where each row is
                   (y, x, score) with integer pixel coordinates.

    Returns:
        Refined keypoint array of shape (N, 3) with sub-pixel (y, x) and
        interpolated score. A copy is returned; the input is not modified.
    """
    if keypoints.shape[0] == 0:
        return keypoints.copy()

    H, W = score_map.shape
    refined = keypoints.copy()

    ys = keypoints[:, 0].astype(np.intp)
    xs = keypoints[:, 1].astype(np.intp)

    # Mask for interior keypoints where a 3-point neighborhood is available
    interior = (ys >= 1) & (ys < H - 1) & (xs >= 1) & (xs < W - 1)
    if not np.any(interior):
        return refined

    yi = ys[interior]
    xi = xs[interior]

    # Gather 3-point neighborhoods along y-axis and x-axis
    fy_neg = score_map[yi - 1, xi].astype(np.float64)
    fy_ctr = score_map[yi, xi].astype(np.float64)
    fy_pos = score_map[yi + 1, xi].astype(np.float64)

    fx_neg = score_map[yi, xi - 1].astype(np.float64)
    fx_ctr = fy_ctr  # same center value
    fx_pos = score_map[yi, xi + 1].astype(np.float64)

    # Parabola fitting along y-axis
    denom_y = 2.0 * (fy_neg - 2.0 * fy_ctr + fy_pos)
    with np.errstate(divide="ignore", invalid="ignore"):
        dy = np.where(denom_y < -1e-6, (fy_neg - fy_pos) / denom_y, 0.0)
    dy = np.where(np.abs(dy) < 1.0, dy, 0.0)

    # Parabola fitting along x-axis
    denom_x = 2.0 * (fx_neg - 2.0 * fx_ctr + fx_pos)
    with np.errstate(divide="ignore", invalid="ignore"):
        dx = np.where(denom_x < -1e-6, (fx_neg - fx_pos) / denom_x, 0.0)
    dx = np.where(np.abs(dx) < 1.0, dx, 0.0)

    # Update coordinates
    refined[interior, 0] = yi + dy
    refined[interior, 1] = xi + dx

    # Interpolate score at refined position using quadratic peak value:
    #   f(delta) = f(0) + (b/2) * delta  where b = (f(1) - f(-1)) / 2
    # which simplifies to: f(0) + 0.25 * delta * (f(1) - f(-1))
    score_y = fy_ctr + 0.25 * dy * (fy_pos - fy_neg)
    score_x = fx_ctr + 0.25 * dx * (fx_pos - fx_neg)
    refined[interior, 2] = ((score_y + score_x) / 2.0).astype(np.float32)

    return refined


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
    subpixel: bool = True,
) -> np.ndarray:
    """
    Select keypoints from corner score map.

    Applies NMS and threshold filtering, then returns the top-k keypoints
    sorted by score in descending order. When ``subpixel`` is True, keypoint
    positions are refined to sub-pixel accuracy via parabola fitting on the
    original (pre-NMS) score map.

    Args:
        scores: Corner score map of shape (1, 1, H, W).
        threshold: Minimum score threshold. Default is 0.01.
        max_keypoints: Maximum number of keypoints to return. Default is 1000.
        nms_radius: Radius of the NMS window. Default is 3.
        subpixel: If True, refine keypoint positions to sub-pixel accuracy
                  using parabola fitting. Default is True.

    Returns:
        Keypoint array of shape (N, 3) where each row is (y, x, score).
        When subpixel is True, y and x are floating-point sub-pixel coordinates.
        Returns empty array of shape (0, 3) if no keypoints are found.
    """
    raw_score_map = scores[0, 0]  # (H, W) — keep original for sub-pixel fitting
    score_map = nms_keypoints(raw_score_map, nms_radius=nms_radius)

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

    # Sub-pixel refinement on the original (pre-NMS) score map
    if subpixel:
        keypoints = refine_keypoints_subpixel(raw_score_map, keypoints)

    return keypoints


def score_to_color(normalized_score: float) -> tuple[int, int, int]:
    """
    Map a normalized score in [0, 1] to an RGB color using a jet-like colormap.

    The mapping goes: blue (0.0) -> cyan (0.25) -> green (0.5) ->
    yellow (0.75) -> red (1.0).

    Args:
        normalized_score: Score value in [0, 1].

    Returns:
        RGB tuple with values in [0, 255].
    """
    t = max(0.0, min(1.0, normalized_score))
    # Jet-like colormap: blue -> cyan -> green -> yellow -> red
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


def visualize_keypoints(
    image: Image.Image,
    keypoints: np.ndarray,
    output_path: str,
    circle_radius: int = 3,
    color: tuple = (0, 255, 0),
    colorize_by_score: bool = False,
) -> None:
    """
    Draw detected keypoints on the image and save the result.

    Args:
        image: RGB PIL Image to draw on.
        keypoints: Keypoint array of shape (N, 3) where each row
                   is (y, x, score).
        output_path: Path to save the output visualization image.
        circle_radius: Radius of the keypoint circles. Default is 3.
        color: RGB color tuple for drawing keypoints. Default is green.
        colorize_by_score: If True, color each keypoint circle according
                          to its score using a jet-like colormap (blue=low,
                          red=high). Overrides the ``color`` parameter.
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)

    # Precompute normalized scores for colorization
    if colorize_by_score and keypoints.shape[0] > 0:
        scores = keypoints[:, 2]
        score_min = scores.min()
        score_max = scores.max()
        if score_max > score_min:
            normalized = (scores - score_min) / (score_max - score_min)
        else:
            normalized = np.ones_like(scores)

    for i in range(keypoints.shape[0]):
        y = int(keypoints[i, 0])
        x = int(keypoints[i, 1])
        r = circle_radius
        if colorize_by_score:
            c = score_to_color(normalized[i])
        else:
            c = color
        draw.ellipse(
            [x - r, y - r, x + r, y + r],
            outline=c,
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
    parser.add_argument(
        "--colorize",
        action="store_true",
        help="Colorize keypoint circles by score (blue=low, red=high)"
    )
    parser.add_argument(
        "--no-subpixel",
        action="store_true",
        help="Disable sub-pixel refinement of keypoint positions"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Create ONNX Runtime session
    session = ort.InferenceSession(args.model)
    input_info = session.get_inputs()[0]
    input_name = input_info.name
    input_shape = input_info.shape  # [N, C, H, W]
    output_names = [o.name for o in session.get_outputs()]
    model_height = input_shape[2]
    model_width = input_shape[3]
    print(f"ONNX model: {args.model}")
    print(f"  Input:   {input_name} {input_shape}")
    print(f"  Outputs: {output_names}")

    # Load and resize image to model input size
    image, image_rgb = load_image(args.input, model_height, model_width)
    print(f"Input image: {args.input} (resized to {model_height}x{model_width})")

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
        subpixel=not args.no_subpixel,
    )
    print(f"Detected keypoints: {keypoints.shape[0]}")

    if keypoints.shape[0] == 0:
        print("No keypoints detected. Try lowering --threshold.")
        return

    # Visualize
    visualize_keypoints(
        image_rgb,
        keypoints,
        args.output,
        circle_radius=args.circle_radius,
        colorize_by_score=args.colorize,
    )
    print(f"Saved visualization to: {args.output}")


if __name__ == "__main__":
    main()
