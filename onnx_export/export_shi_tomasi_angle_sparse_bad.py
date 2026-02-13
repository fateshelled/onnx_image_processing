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

from pytorch_model.feature_detection.shi_tomasi_angle import ShiTomasiAngleSparseBAD
from onnx_export.optimize import optimize_onnx_model


class ShiTomasiAngleSparseBADWrapper(torch.nn.Module):
    """
    Wrapper for ONNX export that combines detection and description.

    This wrapper provides a single forward pass that:
    1. Detects features and computes orientations
    2. Selects top-k keypoints
    3. Computes rotation-aware descriptors

    Inputs:
        image: (B, 1, H, W) grayscale image

    Outputs:
        keypoints: (B, K, 2) keypoint coordinates in (y, x) format
        scores: (B, K) keypoint scores
        descriptors: (B, K, num_pairs) rotation-aware descriptors
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
        normalize_descriptors: bool = True,
        sampling_mode: str = "nearest",
        nms_radius: int = 3,
        score_threshold: float = 0.0,
    ):
        super().__init__()

        self.max_keypoints = max_keypoints
        self.nms_radius = nms_radius
        self.score_threshold = score_threshold

        self.model = ShiTomasiAngleSparseBAD(
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

    def _apply_nms_maxpool(self, scores: torch.Tensor) -> torch.Tensor:
        """Apply non-maximum suppression using max pooling."""
        import torch.nn.functional as F

        kernel_size = 2 * self.nms_radius + 1
        padding = self.nms_radius

        scores_padded = F.pad(
            scores.unsqueeze(1),
            (padding, padding, padding, padding),
            mode="constant",
            value=float("-inf"),
        )

        local_max = F.max_pool2d(
            scores_padded,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
        ).squeeze(1)

        nms_mask = (scores >= (local_max - 1e-7)).float()
        return nms_mask

    def _select_topk_keypoints(
        self,
        scores: torch.Tensor,
        nms_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select top-k keypoints from score map after NMS."""
        B, H, W = scores.shape
        K = self.max_keypoints

        scores_masked = scores * nms_mask
        scores_masked = torch.where(
            scores_masked > self.score_threshold,
            scores_masked,
            torch.zeros_like(scores_masked),
        )

        scores_flat = scores_masked.reshape(B, -1)

        topk_scores, topk_indices = torch.topk(
            scores_flat,
            k=K,
            dim=1,
            largest=True,
            sorted=True,
        )

        y_coords = (topk_indices // W).float()
        x_coords = (topk_indices % W).float()
        keypoints = torch.stack([y_coords, x_coords], dim=-1)

        valid_mask = (topk_scores > 0).float()
        invalid_keypoints = torch.full_like(keypoints, -1.0)
        keypoints = torch.where(
            valid_mask.unsqueeze(-1) > 0.5,
            keypoints,
            invalid_keypoints,
        )
        topk_scores = topk_scores * valid_mask

        return keypoints, topk_scores

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass: detect, select, and describe keypoints.

        Args:
            image: Input grayscale image (B, 1, H, W)

        Returns:
            keypoints: (B, K, 2) in (y, x) format, invalid keypoints at (-1, -1)
            scores: (B, K) keypoint scores
            descriptors: (B, K, num_pairs) rotation-aware descriptors
        """
        # 1. Detect and orient
        score_map, angles = self.model.detect_and_orient(image)
        score_map = score_map.squeeze(1)  # (B, H, W)

        # 2. Apply NMS
        nms_mask = self._apply_nms_maxpool(score_map)

        # 3. Select top-k keypoints
        keypoints, scores = self._select_topk_keypoints(score_map, nms_mask)

        # 4. Compute rotation-aware descriptors
        descriptors = self.model.describe(image, keypoints, angles)

        return keypoints, scores, descriptors


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
        default=3,
        help="Radius for non-maximum suppression (default: 3)"
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

    model = ShiTomasiAngleSparseBADWrapper(
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
