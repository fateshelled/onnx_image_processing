#!/usr/bin/env python3
"""
ONNX export script for Shi-Tomasi + Angle + Sparse BAD descriptor model.

Exports the ShiTomasiAngleSparseBAD model which combines Shi-Tomasi corner
detection with angle estimation and rotation-aware sparse BAD descriptors.

This exports the descriptor computation component. For a complete matching
pipeline with Sinkhorn, see export_shi_tomasi_angle_sparse_bad_sinkhorn.py.

Usage:
    python export_shi_tomasi_angle_sparse_bad.py --output shi_tomasi_angle_sparse_bad.onnx
    python export_shi_tomasi_angle_sparse_bad.py --output model.onnx --num-pairs 512
    python export_shi_tomasi_angle_sparse_bad.py --output model.onnx --max-keypoints 256 --binarization soft
"""

import argparse
import sys
from pathlib import Path

import torch

# Add parent directory to path for importing pytorch_model
sys.path.insert(0, str(Path(__file__).parent.parent))

from pytorch_model.feature_detection.shi_tomasi_angle import (
    ShiTomasiAngleSparseBADDetector,
)
from onnx_export.optimize import optimize_onnx_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export Shi-Tomasi + Angle + Sparse BAD descriptor model to ONNX format"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="shi_tomasi_angle_sparse_bad.onnx",
        help="Output ONNX file path (default: shi_tomasi_angle_sparse_bad.onnx)"
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
        help="BAD binarization mode: none (threshold-centered response), soft (sigmoid), hard (binary) (default: hard)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=10.0,
        help="Temperature for soft sigmoid binarization (default: 10.0)"
    )
    parser.add_argument(
        "--normalize-descriptors",
        action="store_true",
        default=True,
        help="L2-normalize descriptors (default: True)"
    )
    parser.add_argument(
        "--no-normalize-descriptors",
        dest="normalize_descriptors",
        action="store_false",
        help="Disable descriptor normalization"
    )
    parser.add_argument(
        "--sampling-mode",
        type=str,
        choices=["nearest", "bilinear"],
        default="nearest",
        help="Sampling mode for sparse BAD descriptor extraction (default: nearest)"
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

    model = ShiTomasiAngleSparseBADDetector(
        max_keypoints=args.max_keypoints,
        block_size=args.block_size,
        patch_size=args.patch_size,
        sigma=args.sigma,
        num_pairs=args.num_pairs,
        binarize=binarize,
        soft_binarize=soft_binarize,
        temperature=args.temperature,
        normalize_descriptors=args.normalize_descriptors,
        sampling_mode=args.sampling_mode,
        nms_radius=args.nms_radius,
        score_threshold=args.score_threshold,
    )
    model.eval()

    # Create dummy input (B, 1, H, W)
    dummy_image = torch.randn(1, 1, args.height, args.width)

    # Configure dynamic axes if requested
    dynamic_axes = None
    if args.dynamic_axes:
        dynamic_axes = {
            "image": {0: "batch", 2: "height", 3: "width"},
            "keypoints": {0: "batch"},
            "scores": {0: "batch"},
            "descriptors": {0: "batch"},
        }

    # Export to ONNX
    print(f"Exporting Shi-Tomasi + Angle + Sparse BAD model to ONNX format...")
    torch.onnx.export(
        model,
        dummy_image,
        args.output,
        export_params=True,
        opset_version=args.opset_version,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["keypoints", "scores", "descriptors"],
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
    print(f"  Model type: Shi-Tomasi + Angle + Sparse BAD (rotation-aware)")
    print(f"  Input image shape: (B, 1, {args.height}, {args.width})")
    print(f"  Output keypoints shape: (B, {K}, 2)")
    print(f"  Output scores shape: (B, {K})")
    print(f"  Output descriptors shape: (B, {K}, {args.num_pairs})")
    print(f"  Max keypoints: {K}")
    print(f"  Block size: {args.block_size}")
    print(f"  Patch size: {args.patch_size}")
    print(f"  Sigma: {args.sigma}")
    print(f"  Number of pairs: {args.num_pairs}")
    print(f"  Binarization: {args.binarization}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Sampling mode: {args.sampling_mode}")
    print(f"  Normalize descriptors: {args.normalize_descriptors}")
    print(f"  NMS radius: {args.nms_radius}")
    print(f"  Score threshold: {args.score_threshold}")
    print(f"  Opset version: {args.opset_version}")
    print(f"  Dynamic axes: {args.dynamic_axes}")
    print(f"  Optimization: {optimization}")


if __name__ == "__main__":
    main()
