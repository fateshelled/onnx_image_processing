#!/usr/bin/env python3
"""
ONNX export script for Voxel Downsampling.

Usage:
    python export_voxel_downsampling.py --output voxel_downsampling.onnx --num-points 1000
"""

import argparse
import sys
from pathlib import Path

import torch

# Add parent directory to path for importing pytorch_model
sys.path.insert(0, str(Path(__file__).parent.parent))

from pytorch_model.pointcloud.voxel_downsampling import VoxelDownsampling
from onnx_export.optimize import optimize_onnx_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export Voxel Downsampling model to ONNX format"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="voxel_downsampling.onnx",
        help="Output ONNX file path (default: voxel_downsampling.onnx)"
    )
    parser.add_argument(
        "--num-points", "-n",
        type=int,
        default=1000,
        help="Number of points for dummy input (default: 1000)"
    )
    parser.add_argument(
        "--leaf-size", "-l",
        type=float,
        default=0.05,
        help="Voxel leaf size for dummy input (default: 0.05)"
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
        help="Enable dynamic input shape (num_points)"
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
    model = VoxelDownsampling()
    model.eval()

    # Create dummy input
    dummy_points = torch.randn(args.num_points, 3, dtype=torch.float32)
    dummy_leaf_size = torch.tensor(args.leaf_size, dtype=torch.float32)

    # Configure dynamic axes if requested
    dynamic_axes = None
    if args.dynamic_axes:
        dynamic_axes = {
            "points": {0: "N"},
            "output_points": {0: "N"},
            "mask": {0: "N"}
        }

    # Export to ONNX
    torch.onnx.export(
        model,
        (dummy_points, dummy_leaf_size),
        args.output,
        export_params=True,
        opset_version=args.opset_version,
        do_constant_folding=True,
        input_names=["points", "leaf_size"],
        output_names=["output_points", "mask"],
        dynamic_axes=dynamic_axes
    )

    # Optimize ONNX model
    optimization = "skipped"
    if not args.no_optimize:
        optimization = optimize_onnx_model(args.output)

    print(f"Exported ONNX model to: {args.output}")
    print(f"  Input points shape: (N, 3), dummy N={args.num_points}")
    print(f"  Leaf size: {args.leaf_size}")
    print(f"  Opset version: {args.opset_version}")
    print(f"  Dynamic axes: {args.dynamic_axes}")
    print(f"  Optimization: {optimization}")


if __name__ == "__main__":
    main()
