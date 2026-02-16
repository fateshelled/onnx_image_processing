#!/usr/bin/env python3
"""
ONNX export script for DoG (Difference of Gaussians) feature detector.

Usage:
    python export_dog.py --output dog.onnx --height 480 --width 640
    python export_dog.py --output dog_with_score.onnx --height 480 --width 640 --with-score
"""

import argparse
import math
import sys
from pathlib import Path

import torch

# Add parent directory to path for importing pytorch_model
sys.path.insert(0, str(Path(__file__).parent.parent))

from pytorch_model.detector.dog import DoGDetector, DoGDetectorWithScore
from onnx_export.optimize import optimize_onnx_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export DoG feature detector model to ONNX format"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="dog.onnx",
        help="Output ONNX file path (default: dog.onnx)"
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
        default=5,
        help="Number of scale levels in the pyramid (default: 5)"
    )
    parser.add_argument(
        "--sigma-base",
        type=float,
        default=1.6,
        help="Base sigma value for the first scale (default: 1.6)"
    )
    parser.add_argument(
        "--sigma-ratio",
        type=float,
        default=math.sqrt(2),
        help="Ratio between consecutive sigma values (default: sqrt(2))"
    )
    parser.add_argument(
        "--kernel-size",
        type=int,
        default=None,
        help="Size of Gaussian kernel (default: auto-computed as 6*sigma_max+1)"
    )
    parser.add_argument(
        "--with-score",
        action="store_true",
        help="Use DoGDetectorWithScore to output single score map instead of multi-scale responses"
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

    # Validate output path to prevent path traversal attacks
    output_path = Path(args.output).resolve()

    # Ensure parent directory exists
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError) as e:
        print(f"Error: Cannot create output directory: {e}", file=sys.stderr)
        sys.exit(1)

    # Use the validated path
    args.output = str(output_path)

    # Create model
    if args.with_score:
        model = DoGDetectorWithScore(
            num_scales=args.num_scales,
            sigma_base=args.sigma_base,
            sigma_ratio=args.sigma_ratio,
            kernel_size=args.kernel_size,
        )
    else:
        model = DoGDetector(
            num_scales=args.num_scales,
            sigma_base=args.sigma_base,
            sigma_ratio=args.sigma_ratio,
            kernel_size=args.kernel_size,
        )
    model.eval()

    # Create dummy input (N, 1, H, W)
    dummy_input = torch.randn(1, 1, args.height, args.width)

    # Configure dynamic axes if requested
    dynamic_axes = None
    if args.dynamic_axes:
        dynamic_axes = {
            "input": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height", 3: "width"}
        }

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        args.output,
        export_params=True,
        opset_version=args.opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        dynamo=not args.disable_dynamo,
    )

    # Optimize ONNX model
    optimization = "skipped"
    if not args.no_optimize:
        optimization = optimize_onnx_model(args.output)

    print(f"Exported ONNX model to: {args.output}")
    print(f"  Model type: {'DoGDetectorWithScore' if args.with_score else 'DoGDetector'}")
    print(f"  Input shape: (N, 1, {args.height}, {args.width})")
    if args.with_score:
        print(f"  Output shape: (N, 1, {args.height}, {args.width})")
    else:
        print(f"  Output shape: (N, {args.num_scales - 1}, {args.height}, {args.width})")
    print(f"  Number of scales: {args.num_scales}")
    print(f"  Sigma base: {args.sigma_base}")
    print(f"  Sigma ratio: {args.sigma_ratio}")
    print(f"  Kernel size: {args.kernel_size if args.kernel_size else 'auto'}")
    print(f"  Opset version: {args.opset_version}")
    print(f"  Dynamic axes: {args.dynamic_axes}")
    print(f"  Optimization: {optimization}")


if __name__ == "__main__":
    main()
