#!/usr/bin/env python3
"""
ONNX export script for Shi-Tomasi + BAD + Sinkhorn image matching model.

Exports the unified ShiTomasiBADSinkhornMatcher model that detects keypoints,
computes descriptors, and performs Sinkhorn matching in a single forward pass.

Usage:
    python export_shi_tomasi_bad_sinkhorn.py --output shi_tomasi_bad_sinkhorn.onnx --height 480 --width 640
    python export_shi_tomasi_bad_sinkhorn.py --output shi_tomasi_bad_sinkhorn.onnx --max-keypoints 256
"""

import argparse
import sys
from pathlib import Path

import torch

# Add parent directory to path for importing pytorch_model
sys.path.insert(0, str(Path(__file__).parent.parent))

from pytorch_model.feature_detection.shi_tomasi_bad_sinkhorn import ShiTomasiBADSinkhornMatcher


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export Shi-Tomasi + BAD + Sinkhorn image matching model to ONNX format"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="shi_tomasi_bad_sinkhorn.onnx",
        help="Output ONNX file path (default: shi_tomasi_bad_sinkhorn.onnx)"
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
        default=512,
        help="Maximum number of keypoints per image (default: 512)"
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=3,
        help="Block size for Shi-Tomasi structure tensor computation (default: 3)"
    )
    parser.add_argument(
        "--num-pairs", "-n",
        type=int,
        default=256,
        help="Number of BAD descriptor pairs/bits (default: 256)"
    )
    parser.add_argument(
        "--box-size", "-b",
        type=int,
        default=5,
        help="Box size for BAD averaging (default: 5)"
    )
    parser.add_argument(
        "--pattern-scale", "-s",
        type=float,
        default=16.0,
        help="Pattern scale for BAD sampling offsets (default: 16.0)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for BAD sampling pattern (default: 42)"
    )
    parser.add_argument(
        "--binarization",
        type=str,
        choices=["none", "soft", "hard"],
        default="none",
        help="BAD binarization mode: none (raw diff), soft (sigmoid), hard (sign) (default: none)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=10.0,
        help="Temperature for soft sigmoid binarization (default: 10.0)"
    )
    parser.add_argument(
        "--sinkhorn-iterations", "-i",
        type=int,
        default=20,
        help="Number of Sinkhorn iterations (default: 20)"
    )
    parser.add_argument(
        "--epsilon", "-e",
        type=float,
        default=1.0,
        help="Entropy regularization parameter for Sinkhorn (default: 1.0)"
    )
    parser.add_argument(
        "--unused-score",
        type=float,
        default=1.0,
        help="Score for dustbin entries in Sinkhorn (default: 1.0)"
    )
    parser.add_argument(
        "--distance-type",
        type=str,
        choices=["l1", "l2"],
        default="l2",
        help="Distance metric for Sinkhorn cost matrix (default: l2)"
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
    binarize = args.binarization != "none"
    soft_binarize = args.binarization == "soft"
    model = ShiTomasiBADSinkhornMatcher(
        max_keypoints=args.max_keypoints,
        block_size=args.block_size,
        sobel_size=3,
        num_pairs=args.num_pairs,
        box_size=args.box_size,
        pattern_scale=args.pattern_scale,
        seed=args.seed,
        binarize=binarize,
        soft_binarize=soft_binarize,
        temperature=args.temperature,
        sinkhorn_iterations=args.sinkhorn_iterations,
        epsilon=args.epsilon,
        unused_score=args.unused_score,
        distance_type=args.distance_type,
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
        }

    # Export to ONNX
    torch.onnx.export(
        model,
        (dummy_image1, dummy_image2),
        args.output,
        export_params=True,
        opset_version=args.opset_version,
        do_constant_folding=True,
        input_names=["image1", "image2"],
        output_names=["keypoints1", "keypoints2", "matching_probs"],
        dynamic_axes=dynamic_axes,
        dynamo=not args.disable_dynamo,
    )

    K = args.max_keypoints
    print(f"Exported ONNX model to: {args.output}")
    print(f"  Input image1 shape: (B, 1, {args.height}, {args.width})")
    print(f"  Input image2 shape: (B, 1, {args.height}, {args.width})")
    print(f"  Output keypoints1 shape: (B, {K}, 2)")
    print(f"  Output keypoints2 shape: (B, {K}, 2)")
    print(f"  Output matching_probs shape: (B, {K + 1}, {K + 1})")
    print(f"  Max keypoints: {K}")
    print(f"  Block size: {args.block_size}")
    print(f"  Number of pairs: {args.num_pairs}")
    print(f"  Box size: {args.box_size}")
    print(f"  Pattern scale: {args.pattern_scale}")
    print(f"  Binarization: {args.binarization}")
    print(f"  Sinkhorn iterations: {args.sinkhorn_iterations}")
    print(f"  Epsilon: {args.epsilon}")
    print(f"  Distance type: {args.distance_type}")
    print(f"  Opset version: {args.opset_version}")
    print(f"  Dynamic axes: {args.dynamic_axes}")


if __name__ == "__main__":
    main()
