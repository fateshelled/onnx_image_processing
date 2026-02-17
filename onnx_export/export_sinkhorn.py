#!/usr/bin/env python3
"""
ONNX export script for Sinkhorn feature matcher.

Usage:
    python export_sinkhorn.py --output sinkhorn_matcher.onnx --num-points1 100 --num-points2 80
    python export_sinkhorn.py --output sinkhorn_matcher.onnx --dynamic-axes
"""

import argparse
import sys
from pathlib import Path

import torch

# Add parent directory to path for importing pytorch_model
sys.path.insert(0, str(Path(__file__).parent.parent))

from pytorch_model.matching.sinkhorn import SinkhornMatcher
from onnx_export.optimize import optimize_onnx_model, remove_external_data


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export Sinkhorn feature matcher model to ONNX format"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="sinkhorn_matcher.onnx",
        help="Output ONNX file path (default: sinkhorn_matcher.onnx)"
    )
    parser.add_argument(
        "--num-points1", "-n1",
        type=int,
        default=100,
        help="Number of points in first descriptor set (default: 100)"
    )
    parser.add_argument(
        "--num-points2", "-n2",
        type=int,
        default=100,
        help="Number of points in second descriptor set (default: 100)"
    )
    parser.add_argument(
        "--desc-dim", "-d",
        type=int,
        default=256,
        help="Descriptor dimension (default: 256)"
    )
    parser.add_argument(
        "--iterations", "-i",
        type=int,
        default=20,
        help="Number of Sinkhorn iterations (default: 20)"
    )
    parser.add_argument(
        "--epsilon", "-e",
        type=float,
        default=1.0,
        help="Entropy regularization parameter (default: 1.0)"
    )
    parser.add_argument(
        "--unused-score",
        type=float,
        default=1.0,
        help="Score for dustbin entries (default: 1.0)"
    )
    parser.add_argument(
        "--distance-type",
        type=str,
        choices=["l1", "l2"],
        default="l2",
        help="Distance metric for cost matrix (default: l2)"
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
        help="Enable dynamic input shape (batch, num_points)"
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
    model = SinkhornMatcher(
        iterations=args.iterations,
        epsilon=args.epsilon,
        unused_score=args.unused_score,
        distance_type=args.distance_type,
    )
    model.eval()

    # Create dummy inputs [B, N, D] and [B, M, D]
    dummy_desc1 = torch.randn(1, args.num_points1, args.desc_dim)
    dummy_desc2 = torch.randn(1, args.num_points2, args.desc_dim)

    # Configure dynamic axes if requested
    dynamic_axes = None
    if args.dynamic_axes:
        dynamic_axes = {
            "desc1": {0: "batch", 1: "num_points1"},
            "desc2": {0: "batch", 1: "num_points2"},
            "matching_probs": {0: "batch", 1: "num_points1_plus1", 2: "num_points2_plus1"}
        }

    # Export to ONNX
    torch.onnx.export(
        model,
        (dummy_desc1, dummy_desc2),
        args.output,
        export_params=True,
        opset_version=args.opset_version,
        do_constant_folding=True,
        input_names=["desc1", "desc2"],
        output_names=["matching_probs"],
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
    print(f"  Input desc1 shape: (B, {args.num_points1}, {args.desc_dim})")
    print(f"  Input desc2 shape: (B, {args.num_points2}, {args.desc_dim})")
    print(f"  Output shape: (B, {args.num_points1 + 1}, {args.num_points2 + 1})")
    print(f"  Iterations: {args.iterations}")
    print(f"  Epsilon: {args.epsilon}")
    print(f"  Unused score: {args.unused_score}")
    print(f"  Distance type: {args.distance_type}")
    print(f"  Opset version: {args.opset_version}")
    print(f"  Dynamic axes: {args.dynamic_axes}")
    print(f"  Optimization: {optimization}")


if __name__ == "__main__":
    main()
