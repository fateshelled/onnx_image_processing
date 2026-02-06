#!/usr/bin/env python3
"""
ONNX export script for Shi-Tomasi corner detection score calculator.

Usage:
    python export_shi_tomasi.py --output shi_tomasi.onnx --height 480 --width 640
"""

import argparse
import sys
from pathlib import Path

import torch

# Add parent directory to path for importing pytorch_model
sys.path.insert(0, str(Path(__file__).parent.parent))

from pytorch_model.corner.shi_tomasi import ShiTomasiScore


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export Shi-Tomasi corner detection model to ONNX format"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="shi_tomasi.onnx",
        help="Output ONNX file path (default: shi_tomasi.onnx)"
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
        "--block-size", "-b",
        type=int,
        default=3,
        help="Block size for structure tensor computation (default: 3)"
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
    return parser.parse_args()


def main():
    args = parse_args()

    # Create model
    model = ShiTomasiScore(block_size=args.block_size, sobel_size=3)
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

    print(f"Exported ONNX model to: {args.output}")
    print(f"  Input shape: (N, 1, {args.height}, {args.width})")
    print(f"  Block size: {args.block_size}")
    print(f"  Opset version: {args.opset_version}")
    print(f"  Dynamic axes: {args.dynamic_axes}")


if __name__ == "__main__":
    main()
