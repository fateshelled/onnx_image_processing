#!/usr/bin/env python3
"""
ONNX export script for Shi-Tomasi + Angle + Sparse BAD + Sinkhorn matching with outlier filters.

Exports a complete feature matching pipeline with integrated outlier filtering:
- Probability ratio filter: Rejects ambiguous matches
- Dustbin margin filter: Rejects matches with high dustbin probability

Usage:
    python export_shi_tomasi_angle_sparse_bad_sinkhorn_with_filters.py --output model.onnx
    python export_shi_tomasi_angle_sparse_bad_sinkhorn_with_filters.py --output model.onnx --ratio-threshold 2.0
    python export_shi_tomasi_angle_sparse_bad_sinkhorn_with_filters.py --output model.onnx --dustbin-margin 0.3
    python export_shi_tomasi_angle_sparse_bad_sinkhorn_with_filters.py --output model.onnx --ratio-threshold 2.0 --dustbin-margin 0.3
"""

import argparse
import sys
from pathlib import Path

import torch

# Add parent directory to path for importing pytorch_model
sys.path.insert(0, str(Path(__file__).parent.parent))

from pytorch_model.feature_detection.shi_tomasi_angle_sparse_bad_sinkhorn import (
    ShiTomasiAngleSparseBADSinkhornMatcherWithFilters,
)
from onnx_export.optimize import optimize_onnx_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export Shi-Tomasi + Angle + Sparse BAD + Sinkhorn with outlier filters to ONNX"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="shi_tomasi_angle_sparse_bad_sinkhorn_with_filters.onnx",
        help="Output ONNX file path (default: shi_tomasi_angle_sparse_bad_sinkhorn_with_filters.onnx)"
    )
    parser.add_argument(
        "--height", "-H",
        type=int,
        default=480,
        help="Input image height (default: 480)"
    )
    parser.add_argument(
        "--width", "-W",
        type=int,
        default=640,
        help="Input image width (default: 640)"
    )
    parser.add_argument(
        "--max-keypoints", "-k",
        type=int,
        default=1024,
        help="Maximum number of keypoints per image (default: 1024)"
    )
    # --- Outlier filter parameters ---
    parser.add_argument(
        "--ratio-threshold",
        type=float,
        default=10.0,
        help="Probability ratio threshold for outlier filtering. "
             "Minimum ratio between best and second-best match probabilities. "
             "Higher values are more strict (e.g., 10.0). "
             "If not specified, ratio filtering is disabled."
    )
    parser.add_argument(
        "--dustbin-margin",
        type=float,
        default=0.3,
        help="Dustbin margin threshold for outlier filtering. "
             "Minimum margin between best match and dustbin probabilities. "
             "Higher values are more strict (e.g., 0.3). "
             "If not specified, dustbin margin filtering is disabled."
    )
    # --- Shi-Tomasi detector parameters ---
    parser.add_argument(
        "--block-size",
        type=int,
        default=5,
        help="Shi-Tomasi block size (default: 5)"
    )
    # --- Angle estimation parameters ---
    parser.add_argument(
        "--patch-size",
        type=int,
        default=15,
        help="Patch size for angle estimation (default: 15)"
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=2.5,
        help="Gaussian sigma for angle estimation (default: 2.5)"
    )
    # --- BAD descriptor parameters ---
    parser.add_argument(
        "--num-pairs", "-n",
        type=int,
        choices=[256, 512],
        default=512,
        help="Number of BAD descriptor pairs (choices: 256 or 512, default: 512)"
    )
    parser.add_argument(
        "--binarization",
        type=str,
        choices=["none", "soft", "hard"],
        default="hard",
        help="BAD binarization mode: none, soft (sigmoid), hard (binary) (default: hard)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=10.0,
        help="Temperature for soft sigmoid binarization (default: 10.0)"
    )
    # --- Sinkhorn parameters ---
    parser.add_argument(
        "--sinkhorn-iterations", "-i",
        type=int,
        default=20,
        help="Number of Sinkhorn iterations (default: 20)"
    )
    parser.add_argument(
        "--epsilon", "-e",
        type=float,
        default=0.05,
        help="Entropy regularization parameter for Sinkhorn (default: 0.05)"
    )
    parser.add_argument(
        "--unused-score",
        type=float,
        default=1.0,
        help="Score for dustbin entries in Sinkhorn (default: 1.0)"
    )
    parser.add_argument(
        "--normalize-descriptors",
        action="store_true",
        default=True,
        help="L2-normalize descriptors before matching (default: True)"
    )
    parser.add_argument(
        "--no-normalize-descriptors",
        dest="normalize_descriptors",
        action="store_false",
        help="Disable descriptor normalization"
    )
    parser.add_argument(
        "--distance-type",
        type=str,
        choices=["l1", "l2"],
        default="l2",
        help="Distance metric for Sinkhorn cost matrix (default: l2)"
    )
    # --- Pipeline parameters ---
    parser.add_argument(
        "--nms-radius",
        type=int,
        default=5,
        help="Radius for non-maximum suppression (default: 5)"
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.0,
        help="Minimum score threshold for keypoint selection (default: 0.0)"
    )
    parser.add_argument(
        "--sampling-mode",
        type=str,
        choices=["nearest", "bilinear"],
        default="nearest",
        help="Sampling mode for sparse BAD descriptor extraction (default: nearest)"
    )
    # --- ONNX export options ---
    parser.add_argument(
        "--opset-version",
        type=int,
        default=18,
        help="ONNX opset version (default: 18)"
    )
    parser.add_argument(
        "--dynamic-axes",
        action="store_true",
        help="Enable dynamic input shape (batch, height, width)"
    )
    parser.add_argument(
        "--disable-dynamo",
        action="store_true",
        help="Disable dynamo"
    )
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Disable ONNX model optimization (onnxsim/onnxoptimizer)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Create model
    binarize = args.binarization != "none"
    soft_binarize = args.binarization == "soft"

    model = ShiTomasiAngleSparseBADSinkhornMatcherWithFilters(
        max_keypoints=args.max_keypoints,
        block_size=args.block_size,
        patch_size=args.patch_size,
        sigma=args.sigma,
        num_pairs=args.num_pairs,
        binarize=binarize,
        soft_binarize=soft_binarize,
        temperature=args.temperature,
        sinkhorn_iterations=args.sinkhorn_iterations,
        epsilon=args.epsilon,
        unused_score=args.unused_score,
        distance_type=args.distance_type,
        ratio_threshold=args.ratio_threshold,
        dustbin_margin=args.dustbin_margin,
        nms_radius=args.nms_radius,
        score_threshold=args.score_threshold,
        normalize_descriptors=args.normalize_descriptors,
        sampling_mode=args.sampling_mode,
    )
    model.eval()

    # Create dummy inputs (B, 1, H, W)
    dummy_image1 = torch.randn(1, 1, args.height, args.width)
    dummy_image2 = torch.randn(1, 1, args.height, args.width)

    # Configure dynamic axes if requested
    dynamic_axes = None
    if args.dynamic_axes:
        dynamic_axes = {
            "image1": {0: "batch", 2: "height", 3: "width"},
            "image2": {0: "batch", 2: "height", 3: "width"},
            "keypoints1": {0: "batch"},
            "keypoints2": {0: "batch"},
            "matching_probs": {0: "batch"},
            "valid_mask": {0: "batch"},
        }

    # Export to ONNX
    print(f"Exporting Shi-Tomasi + Angle + Sparse BAD + Sinkhorn + Filters to ONNX...")
    torch.onnx.export(
        model,
        (dummy_image1, dummy_image2),
        args.output,
        export_params=True,
        opset_version=args.opset_version,
        do_constant_folding=True,
        input_names=["image1", "image2"],
        output_names=["keypoints1", "keypoints2", "matching_probs", "valid_mask"],
        dynamic_axes=dynamic_axes,
        dynamo=not args.disable_dynamo,
    )

    # Optimize ONNX model
    optimization = "skipped"
    if not args.no_optimize:
        print("Optimizing ONNX model...")
        optimization = optimize_onnx_model(args.output)

    K = args.max_keypoints
    print(f"\nExported ONNX model to: {args.output}")
    print(f"  Model type: Shi-Tomasi + Angle + Sparse BAD + Sinkhorn + Filters")
    print(f"  Input image1 shape: (B, 1, {args.height}, {args.width})")
    print(f"  Input image2 shape: (B, 1, {args.height}, {args.width})")
    print(f"  Output keypoints1 shape: (B, {K}, 2)")
    print(f"  Output keypoints2 shape: (B, {K}, 2)")
    print(f"  Output matching_probs shape: (B, {K + 1}, {K + 1})")
    print(f"  Output valid_mask shape: (B, {K})")
    print(f"  Max keypoints: {K}")
    print(f"  Outlier filters:")
    if args.ratio_threshold is not None:
        print(f"    - Probability ratio threshold: {args.ratio_threshold}")
    else:
        print(f"    - Probability ratio filter: DISABLED")
    if args.dustbin_margin is not None:
        print(f"    - Dustbin margin: {args.dustbin_margin}")
    else:
        print(f"    - Dustbin margin filter: DISABLED")
    print(f"  Block size: {args.block_size}")
    print(f"  Patch size: {args.patch_size}")
    print(f"  Sigma: {args.sigma}")
    print(f"  Number of pairs: {args.num_pairs}")
    print(f"  Binarization: {args.binarization}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Sinkhorn iterations: {args.sinkhorn_iterations}")
    print(f"  Epsilon: {args.epsilon}")
    print(f"  Unused score: {args.unused_score}")
    print(f"  Distance type: {args.distance_type}")
    print(f"  Sampling mode: {args.sampling_mode}")
    print(f"  Normalize descriptors: {args.normalize_descriptors}")
    print(f"  NMS radius: {args.nms_radius}")
    print(f"  Score threshold: {args.score_threshold}")
    print(f"  Opset version: {args.opset_version}")
    print(f"  Dynamic axes: {args.dynamic_axes}")
    print(f"  Optimization: {optimization}")


if __name__ == "__main__":
    main()
