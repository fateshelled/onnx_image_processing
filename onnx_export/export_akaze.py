#!/usr/bin/env python3
"""
ONNX export script for AKAZE feature detection model.

Exports the AKAZE model that computes feature point scores and orientations
using non-linear diffusion scale space and Hessian-based detection.

Usage:
    python export_akaze.py --output akaze.onnx --height 480 --width 640
"""

import argparse
import sys
from pathlib import Path

import torch

# Add parent directory to path for importing pytorch_model
sys.path.insert(0, str(Path(__file__).parent.parent))

from pytorch_model.feature_detection.akaze import AKAZE
from onnx_export.optimize import optimize_onnx_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export AKAZE feature detection model to ONNX format"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="akaze.onnx",
        help="Output ONNX file path (default: akaze.onnx)"
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
        "--num-scales",
        type=int,
        default=3,
        help="Number of scale levels (default: 3)"
    )
    parser.add_argument(
        "--diffusion-iterations",
        type=int,
        default=3,
        help="Number of FED iterations per scale (default: 3)"
    )
    parser.add_argument(
        "--kappa",
        type=float,
        default=0.05,
        help="Contrast parameter for diffusion (default: 0.05)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.001,
        help="Feature detection threshold (default: 0.001)"
    )
    parser.add_argument(
        "--nms-size",
        type=int,
        default=5,
        help="Non-maximum suppression window size (default: 5)"
    )
    parser.add_argument(
        "--orientation-patch-size",
        type=int,
        default=15,
        help="Patch size for orientation computation (default: 15)"
    )
    parser.add_argument(
        "--orientation-sigma",
        type=float,
        default=2.5,
        help="Gaussian sigma for orientation weighting (default: 2.5)"
    )
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
        "--disable_dynamo",
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
    model = AKAZE(
        num_scales=args.num_scales,
        diffusion_iterations=args.diffusion_iterations,
        kappa=args.kappa,
        threshold=args.threshold,
        nms_size=args.nms_size,
        orientation_patch_size=args.orientation_patch_size,
        orientation_sigma=args.orientation_sigma,
    )
    model.eval()

    # Create dummy input (N, 1, H, W)
    dummy_input = torch.randn(1, 1, args.height, args.width)

    # Configure dynamic axes if requested
    dynamic_axes = None
    if args.dynamic_axes:
        dynamic_axes = {
            "input": {0: "batch", 2: "height", 3: "width"},
            "scores": {0: "batch", 2: "height", 3: "width"},
            "orientations": {0: "batch", 2: "height", 3: "width"},
        }

    # Export to ONNX
    print(f"Exporting AKAZE model to ONNX format...")
    print(f"  Input shape: (N, 1, {args.height}, {args.width})")
    print(f"  Number of scales: {args.num_scales}")
    print(f"  Diffusion iterations per scale: {args.diffusion_iterations}")
    print(f"  This may take a moment...")

    torch.onnx.export(
        model,
        dummy_input,
        args.output,
        export_params=True,
        opset_version=args.opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["scores", "orientations"],
        dynamic_axes=dynamic_axes,
        dynamo=not args.disable_dynamo,
    )

    # Optimize ONNX model
    optimization = "skipped"
    if not args.no_optimize:
        print("Optimizing ONNX model...")
        optimization = optimize_onnx_model(args.output)

    print(f"\nExported ONNX model to: {args.output}")
    print(f"  Input shape:  (N, 1, {args.height}, {args.width})")
    print(f"  Scores shape: (N, 1, {args.height}, {args.width})")
    print(f"  Orientations shape: (N, 1, {args.height}, {args.width})")
    print(f"  Number of scales: {args.num_scales}")
    print(f"  Diffusion iterations: {args.diffusion_iterations}")
    print(f"  Kappa: {args.kappa}")
    print(f"  Threshold: {args.threshold}")
    print(f"  NMS size: {args.nms_size}")
    print(f"  Orientation patch size: {args.orientation_patch_size}")
    print(f"  Orientation sigma: {args.orientation_sigma}")
    print(f"  Opset version: {args.opset_version}")
    print(f"  Dynamic axes: {args.dynamic_axes}")
    print(f"  Optimization: {optimization}")


if __name__ == "__main__":
    main()
