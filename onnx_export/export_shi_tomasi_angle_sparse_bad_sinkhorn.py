#!/usr/bin/env python3
"""
ONNX export script for Shi-Tomasi + Angle + Sparse BAD + Sinkhorn matching model.

Exports a complete feature matching pipeline that combines Shi-Tomasi corner
detection, angle estimation, rotation-aware sparse BAD descriptors, and
Sinkhorn matching. This is analogous to AKAZESparseBADSinkhornMatcher but
uses Shi-Tomasi instead of AKAZE for feature detection.

Usage:
    python export_shi_tomasi_angle_sparse_bad_sinkhorn.py --output shi_tomasi_angle_sparse_bad_sinkhorn.onnx
    python export_shi_tomasi_angle_sparse_bad_sinkhorn.py --output model.onnx --max-keypoints 512
    python export_shi_tomasi_angle_sparse_bad_sinkhorn.py --output model.onnx --num-pairs 512 --binarization soft
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Add parent directory to path for importing pytorch_model
sys.path.insert(0, str(Path(__file__).parent.parent))

from pytorch_model.feature_detection.shi_tomasi_angle import ShiTomasiAngleSparseBAD
from pytorch_model.matching.sinkhorn import SinkhornMatcher
from pytorch_model.utils import apply_nms_maxpool, select_topk_keypoints
from onnx_export.optimize import optimize_onnx_model


class ShiTomasiAngleSparseBADSinkhornMatcher(torch.nn.Module):
    """
    Complete feature matching pipeline with Shi-Tomasi + Angle + Sparse BAD + Sinkhorn.

    Pipeline:
        1. Shi-Tomasi corner detection -> score maps
        2. Angle estimation -> orientation maps
        3. NMS + top-k -> keypoint selection
        4. Rotation-aware BAD descriptor computation (sparse)
        5. Sinkhorn matching

    Inputs:
        image1: (B, 1, H, W) first grayscale image
        image2: (B, 1, H, W) second grayscale image

    Outputs:
        keypoints1: (B, K, 2) keypoints in first image (y, x)
        keypoints2: (B, K, 2) keypoints in second image (y, x)
        matching_probs: (B, K+1, K+1) matching probability matrix
    """

    def __init__(
        self,
        max_keypoints: int,
        block_size: int = 5,
        patch_size: int = 15,
        sigma: float = 2.5,
        num_pairs: int = 256,
        binarize: bool = False,
        soft_binarize: bool = True,
        temperature: float = 10.0,
        sinkhorn_iterations: int = 20,
        epsilon: float = 1.0,
        unused_score: float = 1.0,
        distance_type: str = "l2",
        nms_radius: int = 3,
        score_threshold: float = 0.0,
        normalize_descriptors: bool = True,
        sampling_mode: str = "nearest",
    ):
        super().__init__()

        self.max_keypoints = max_keypoints
        self.nms_radius = nms_radius
        self.score_threshold = score_threshold
        self.normalize_descriptors = normalize_descriptors

        # Feature detector + descriptor
        self.descriptor = ShiTomasiAngleSparseBAD(
            block_size=block_size,
            patch_size=patch_size,
            sigma=sigma,
            num_pairs=num_pairs,
            binarize=binarize,
            soft_binarize=soft_binarize,
            temperature=temperature,
            normalize_descriptors=normalize_descriptors,
            sampling_mode=sampling_mode,
        )

        # Feature matcher
        self.matcher = SinkhornMatcher(
            iterations=sinkhorn_iterations,
            epsilon=epsilon,
            unused_score=unused_score,
            distance_type=distance_type,
        )

    def forward(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Detect keypoints and compute matches between two images.

        Args:
            image1: First grayscale image (B, 1, H, W)
            image2: Second grayscale image (B, 1, H, W)

        Returns:
            keypoints1: Detected keypoints in first image (B, K, 2) in (y, x)
            keypoints2: Detected keypoints in second image (B, K, 2) in (y, x)
            matching_probs: Matching probability matrix (B, K+1, K+1)
        """
        # 1. Detect features and compute orientations
        scores1, angles1 = self.descriptor.detect_and_orient(image1)
        scores2, angles2 = self.descriptor.detect_and_orient(image2)
        scores1 = scores1.squeeze(1)  # (B, H, W)
        scores2 = scores2.squeeze(1)

        # 2. Apply NMS
        nms_mask1 = apply_nms_maxpool(scores1, self.nms_radius)
        nms_mask2 = apply_nms_maxpool(scores2, self.nms_radius)

        # 3. Select top-k keypoints
        keypoints1, _ = select_topk_keypoints(
            scores1, nms_mask1, self.max_keypoints, self.score_threshold
        )
        keypoints2, _ = select_topk_keypoints(
            scores2, nms_mask2, self.max_keypoints, self.score_threshold
        )

        # 4. Compute rotation-aware descriptors at keypoints
        desc1 = self.descriptor.describe(image1, keypoints1, angles1)
        desc2 = self.descriptor.describe(image2, keypoints2, angles2)

        # 5. Perform Sinkhorn matching
        matching_probs = self.matcher(desc1, desc2)  # (B, K+1, K+1)

        return keypoints1, keypoints2, matching_probs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export Shi-Tomasi + Angle + Sparse BAD + Sinkhorn matching model to ONNX format"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="shi_tomasi_angle_sparse_bad_sinkhorn.onnx",
        help="Output ONNX file path (default: shi_tomasi_angle_sparse_bad_sinkhorn.onnx)"
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
        default=256,
        help="Number of BAD descriptor pairs (choices: 256 or 512, default: 256)"
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
        default=3,
        help="Radius for non-maximum suppression (default: 3)"
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

    model = ShiTomasiAngleSparseBADSinkhornMatcher(
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
        }

    # Export to ONNX
    print(f"Exporting Shi-Tomasi + Angle + Sparse BAD + Sinkhorn model to ONNX format...")
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

    # Optimize ONNX model
    optimization = "skipped"
    if not args.no_optimize:
        print("Optimizing ONNX model...")
        optimization = optimize_onnx_model(args.output)

    K = args.max_keypoints
    print(f"\nExported ONNX model to: {args.output}")
    print(f"  Model type: Shi-Tomasi + Angle + Sparse BAD + Sinkhorn")
    print(f"  Input image1 shape: (B, 1, {args.height}, {args.width})")
    print(f"  Input image2 shape: (B, 1, {args.height}, {args.width})")
    print(f"  Output keypoints1 shape: (B, {K}, 2)")
    print(f"  Output keypoints2 shape: (B, {K}, 2)")
    print(f"  Output matching_probs shape: (B, {K + 1}, {K + 1})")
    print(f"  Max keypoints: {K}")
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
