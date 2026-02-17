#!/usr/bin/env python3
"""
ONNX export script for FAST corner detection score calculator.

Usage:
    python export_fast.py --output fast.onnx --height 480 --width 640 --threshold 20
"""

import argparse
import sys
from pathlib import Path

import torch

# Add parent directory to path for importing pytorch_model
sys.path.insert(0, str(Path(__file__).parent.parent))

from pytorch_model.detector.fast import FASTScore
from onnx_export.optimize import optimize_onnx_model, remove_external_data


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export FAST corner detection model to ONNX format"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="fast.onnx",
        help="Output ONNX file path (default: fast.onnx)"
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
        "--threshold", "-t",
        type=int,
        default=20,
        help="FAST intensity difference threshold (default: 20)"
    )
    parser.add_argument(
        "--use-nms",
        action="store_true",
        help="Enable non-maximum suppression"
    )
    parser.add_argument(
        "--nms-radius",
        type=int,
        default=3,
        help="NMS radius (default: 3)"
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
    model = FASTScore(
        threshold=args.threshold,
        use_nms=args.use_nms,
        nms_radius=args.nms_radius
    )
    model.eval()

    # Create dummy input (N, 1, H, W)
    dummy_input = torch.randn(1, 1, args.height, args.width) * 255

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
    else:
        remove_external_data(args.output)

    print(f"Exported ONNX model to: {args.output}")
    print(f"  Input shape: (N, 1, {args.height}, {args.width})")
    print(f"  Threshold: {args.threshold}")
    print(f"  Use NMS: {args.use_nms}")
    if args.use_nms:
        print(f"  NMS radius: {args.nms_radius}")
    print(f"  Opset version: {args.opset_version}")
    print(f"  Dynamic axes: {args.dynamic_axes}")
    print(f"  Optimization: {optimization}")


if __name__ == "__main__":
    main()
