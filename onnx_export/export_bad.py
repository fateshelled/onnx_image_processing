#!/usr/bin/env python3
"""
ONNX export script for BAD (Box Average Difference) descriptor.

Usage:
    python export_bad.py --output bad.onnx --height 480 --width 640
"""

import argparse
import sys
from pathlib import Path

import torch

# Add parent directory to path for importing pytorch_model
sys.path.insert(0, str(Path(__file__).parent.parent))

from pytorch_model.descriptor.bad import BADDescriptor
from onnx_export.optimize import optimize_onnx_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export BAD descriptor model to ONNX format"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="bad.onnx",
        help="Output ONNX file path (default: bad.onnx)"
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
        "--num-pairs", "-n",
        type=int,
        choices=[256, 512],
        default=256,
        help="Number of BAD descriptor bits (choices: 256 or 512, default: 256)"
    )
    parser.add_argument(
        "--box-size", "-b",
        type=int,
        default=5,
        help="Box size for averaging (default: 5)"
    )
    parser.add_argument(
        "--pattern-scale", "-s",
        type=float,
        default=16.0,
        help="Pattern scale for sampling offsets (default: 16.0)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling pattern (default: 42)"
    )
    parser.add_argument(
        "--binarization",
        type=str,
        choices=["none", "soft", "hard"],
        default="none",
        help="BAD binarization mode: none (threshold-centered response), soft (sigmoid), hard (binary) (default: none)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=10.0,
        help="Temperature for soft sigmoid binarization (default: 10.0)"
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

    # NOTE: Learned BAD patterns are fixed for 256/512 bits.
    # pattern_scale/seed are accepted for backward-compatible CLI only.

    # Create model
    binarize = args.binarization != "none"
    soft_binarize = args.binarization == "soft"
    model = BADDescriptor(
        num_pairs=args.num_pairs,
        box_size=args.box_size,
        pattern_scale=args.pattern_scale,
        seed=args.seed,
        binarize=binarize,
        soft_binarize=soft_binarize,
        temperature=args.temperature,
    )
    model.eval()

    # Create dummy input (N, 1, H, W)
    dummy_input = torch.randn(1, 1, args.height, args.width)

    # Configure dynamic axes if requested
    dynamic_axes = None
    if args.dynamic_axes:
        dynamic_axes = {
            "input": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 1: "num_pairs", 2: "height", 3: "width"}
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
    print(f"  Input shape: (N, 1, {args.height}, {args.width})")
    print(f"  Output shape: (N, {args.num_pairs}, {args.height}, {args.width})")
    print(f"  Number of pairs: {args.num_pairs}")
    print(f"  Box size: {args.box_size}")
    print(f"  Pattern scale (unused with learned pattern): {args.pattern_scale}")
    print(f"  Seed (unused with learned pattern): {args.seed}")
    print(f"  Binarization: {args.binarization}")
    print(f"  Opset version: {args.opset_version}")
    print(f"  Dynamic axes: {args.dynamic_axes}")
    print(f"  Optimization: {optimization}")


if __name__ == "__main__":
    main()
