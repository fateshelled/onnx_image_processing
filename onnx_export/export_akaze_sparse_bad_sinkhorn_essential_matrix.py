#!/usr/bin/env python3
"""
ONNX export script for AKAZE + Sparse BAD + Sinkhorn + Essential Matrix.

Exports a combined model that detects keypoints with AKAZE, matches them with
Sinkhorn, and estimates the Essential Matrix from the actual detected keypoint
positions using the weighted 8-point algorithm with Hartley normalisation.

Unlike the standalone EssentialMatrixEstimator (which assumes keypoints lie on
a regular pixel grid), this model uses the actual pixel coordinates of the
AKAZE keypoints for geometrically accurate Essential Matrix estimation.

Note on ONNX exporter
---------------------
The Essential Matrix estimation step uses fixed-iteration power iteration
(to avoid torch.linalg.svd which is unsupported at opset 14). The dynamo
exporter is therefore disabled; use the TorchScript-based exporter
(dynamo=False) for reliable export.

Usage:
    python export_akaze_sparse_bad_sinkhorn_essential_matrix.py \\
        --output model.onnx \\
        --fx 525 --fy 525 --cx 320 --cy 240 \\
        --height 480 --width 640 \\
        --max-keypoints 512
"""

import argparse
import sys
from pathlib import Path

import torch

# Add parent directory to path for importing pytorch_model
sys.path.insert(0, str(Path(__file__).parent.parent))

from pytorch_model.feature_detection.akaze_sparse_bad_sinkhorn_essential_matrix import (
    AKAZESparseBADSinkhornWithEssentialMatrix,
)
from onnx_export.optimize import optimize_onnx_model, remove_external_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export AKAZE + Sparse BAD + Sinkhorn + "
            "Essential Matrix model to ONNX format"
        )
    )

    # ── Output ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="akaze_sparse_bad_sinkhorn_essential_matrix.onnx",
        help=(
            "Output ONNX file path "
            "(default: akaze_sparse_bad_sinkhorn_essential_matrix.onnx)"
        ),
    )

    # ── Input image dimensions ───────────────────────────────────────────
    parser.add_argument(
        "--height", "-H",
        type=int,
        default=480,
        help="Input image height (default: 480)",
    )
    parser.add_argument(
        "--width", "-W",
        type=int,
        default=640,
        help="Input image width (default: 640)",
    )

    # ── Camera intrinsics ────────────────────────────────────────────────
    parser.add_argument(
        "--fx",
        type=float,
        default=525.0,
        help="Focal length in x direction [px] (default: 525.0)",
    )
    parser.add_argument(
        "--fy",
        type=float,
        default=525.0,
        help="Focal length in y direction [px] (default: 525.0)",
    )
    parser.add_argument(
        "--cx",
        type=float,
        default=320.0,
        help="Principal point x [px] (default: 320.0)",
    )
    parser.add_argument(
        "--cy",
        type=float,
        default=240.0,
        help="Principal point y [px] (default: 240.0)",
    )

    # ── AKAZE detector parameters ────────────────────────────────────────
    parser.add_argument(
        "--max-keypoints", "-k",
        type=int,
        default=1024,
        help="Maximum number of keypoints per image (default: 1024)",
    )
    parser.add_argument(
        "--num-scales",
        type=int,
        default=3,
        help="Number of AKAZE scale levels (default: 3)",
    )
    parser.add_argument(
        "--diffusion-iterations",
        type=int,
        default=3,
        help="Number of FED iterations per scale (default: 3)",
    )
    parser.add_argument(
        "--kappa",
        type=float,
        default=0.05,
        help="Contrast parameter for AKAZE diffusion (default: 0.05)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.001,
        help="AKAZE feature detection threshold (default: 0.001)",
    )
    parser.add_argument(
        "--akaze-nms-size",
        type=int,
        default=5,
        help="NMS window size inside AKAZE detector (must be odd, default: 5)",
    )
    parser.add_argument(
        "--orientation-patch-size",
        type=int,
        default=15,
        help="Patch size for AKAZE orientation estimation (must be odd, default: 15)",
    )
    parser.add_argument(
        "--orientation-sigma",
        type=float,
        default=2.5,
        help="Gaussian sigma for AKAZE orientation weighting (default: 2.5)",
    )

    # ── BAD descriptor ───────────────────────────────────────────────────
    parser.add_argument(
        "--num-pairs", "-n",
        type=int,
        choices=[256, 512],
        default=512,
        help="Number of BAD descriptor pairs (256 or 512, default: 512)",
    )
    parser.add_argument(
        "--binarization",
        type=str,
        choices=["none", "soft", "hard"],
        default="hard",
        help=(
            "BAD binarization mode: none (threshold-centred response), "
            "soft (sigmoid), hard (binary). Default: hard"
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=10.0,
        help="Temperature for soft sigmoid binarization (default: 10.0)",
    )
    parser.add_argument(
        "--sampling-mode",
        type=str,
        choices=["nearest", "bilinear"],
        default="nearest",
        help="Sampling mode for sparse BAD descriptor extraction (default: nearest)",
    )
    parser.add_argument(
        "--normalize-descriptors",
        action="store_true",
        default=True,
        help="L2-normalize descriptors before matching (default: True)",
    )
    parser.add_argument(
        "--no-normalize-descriptors",
        dest="normalize_descriptors",
        action="store_false",
        help="Disable descriptor normalization",
    )

    # ── Sinkhorn matching ────────────────────────────────────────────────
    parser.add_argument(
        "--sinkhorn-iterations", "-i",
        type=int,
        default=20,
        help="Number of Sinkhorn iterations (default: 20)",
    )
    parser.add_argument(
        "--epsilon", "-e",
        type=float,
        default=0.05,
        help="Entropy regularization parameter for Sinkhorn (default: 0.05)",
    )
    parser.add_argument(
        "--unused-score",
        type=float,
        default=1.0,
        help="Score for dustbin entries in Sinkhorn (default: 1.0)",
    )
    parser.add_argument(
        "--distance-type",
        type=str,
        choices=["l1", "l2"],
        default="l2",
        help="Distance metric for Sinkhorn cost matrix (default: l2)",
    )

    # ── Pipeline parameters ──────────────────────────────────────────────
    parser.add_argument(
        "--nms-radius",
        type=int,
        default=5,
        help="Radius for pipeline-level non-maximum suppression (default: 5)",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.0,
        help="Minimum score threshold for keypoint selection (default: 0.0)",
    )

    # ── Essential Matrix estimation ──────────────────────────────────────
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help=(
            "Bidirectional top-K entries kept per row/column of the "
            "probability matrix for Essential Matrix estimation (default: 3)"
        ),
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=30,
        help=(
            "Power-iteration steps for the 9x9 eigenvector solve "
            "in Essential Matrix estimation (default: 30). "
            "More iterations -> better accuracy but larger ONNX graph."
        ),
    )
    parser.add_argument(
        "--n-iter-manifold",
        type=int,
        default=10,
        help=(
            "Power-iteration steps for each 3x3 eigenvector solve inside "
            "the Essential Matrix manifold projection (default: 10)."
        ),
    )

    # ── ONNX export options ──────────────────────────────────────────────
    parser.add_argument(
        "--opset-version",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)",
    )
    parser.add_argument(
        "--dynamic-axes",
        action="store_true",
        help="Enable dynamic height and width axes on image inputs.",
    )
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Disable ONNX model optimization (onnxsim / onnxoptimizer)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── Build camera intrinsic matrix ───────────────────────────────────
    K = torch.tensor(
        [[args.fx,   0.0,   args.cx],
         [0.0,    args.fy,  args.cy],
         [0.0,      0.0,      1.0  ]],
        dtype=torch.float32,
    )

    # ── Instantiate combined model ───────────────────────────────────────
    binarize      = args.binarization != "none"
    soft_binarize = args.binarization == "soft"

    model = AKAZESparseBADSinkhornWithEssentialMatrix(
        K=K,
        max_keypoints=args.max_keypoints,
        num_scales=args.num_scales,
        diffusion_iterations=args.diffusion_iterations,
        kappa=args.kappa,
        threshold=args.threshold,
        akaze_nms_size=args.akaze_nms_size,
        orientation_patch_size=args.orientation_patch_size,
        orientation_sigma=args.orientation_sigma,
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
        top_k=args.top_k,
        n_iter=args.n_iter,
        n_iter_manifold=args.n_iter_manifold,
    )
    model.eval()

    # ── Dummy inputs (batch_size=1 required for E estimation) ───────────
    dummy_image1 = torch.randn(1, 1, args.height, args.width)
    dummy_image2 = torch.randn(1, 1, args.height, args.width)

    # ── Dynamic axes ────────────────────────────────────────────────────
    dynamic_axes = None
    if args.dynamic_axes:
        dynamic_axes = {
            "image1": {2: "height", 3: "width"},
            "image2": {2: "height", 3: "width"},
        }

    # ── Export to ONNX (TorchScript exporter; dynamo not used) ──────────
    # The Essential Matrix estimation step uses fixed-iteration power
    # iteration which is ONNX-compatible but safer with the TorchScript
    # exporter at the required opset level.
    print(
        "Exporting AKAZE + Sparse BAD + Sinkhorn + "
        "Essential Matrix to ONNX..."
    )
    print(
        f"  This may take a moment "
        f"(AKAZE has {args.num_scales} scales x {args.diffusion_iterations} iterations)..."
    )
    torch.onnx.export(
        model,
        (dummy_image1, dummy_image2),
        args.output,
        export_params=True,
        opset_version=args.opset_version,
        do_constant_folding=True,
        input_names=["image1", "image2"],
        output_names=["keypoints1", "keypoints2", "matching_probs", "E"],
        dynamic_axes=dynamic_axes,
    )

    # ── Optimize ONNX model ──────────────────────────────────────────────
    optimization = "skipped"
    if not args.no_optimize:
        print("Optimizing ONNX model...")
        optimization = optimize_onnx_model(args.output)
    else:
        remove_external_data(args.output)

    # ── Summary ──────────────────────────────────────────────────────────
    mk = args.max_keypoints
    print(f"\nExported ONNX model to: {args.output}")
    print(f"  Model type: AKAZE + Sparse BAD + Sinkhorn + Essential Matrix")
    print(f"  Input image1 shape: (1, 1, {args.height}, {args.width})")
    print(f"  Input image2 shape: (1, 1, {args.height}, {args.width})")
    print(f"  Output keypoints1 shape:     (1, {mk}, 2)")
    print(f"  Output keypoints2 shape:     (1, {mk}, 2)")
    print(f"  Output matching_probs shape: (1, {mk + 1}, {mk + 1})")
    print(f"  Output E shape:              (3, 3)")
    print(f"  Camera intrinsics: fx={args.fx}, fy={args.fy}, cx={args.cx}, cy={args.cy}")
    print(f"  Max keypoints: {mk}")
    print(f"  AKAZE scales: {args.num_scales}")
    print(f"  Diffusion iterations: {args.diffusion_iterations}")
    print(f"  Kappa: {args.kappa}")
    print(f"  Threshold: {args.threshold}")
    print(f"  AKAZE NMS size: {args.akaze_nms_size}")
    print(f"  Orientation patch size: {args.orientation_patch_size}")
    print(f"  Orientation sigma: {args.orientation_sigma}")
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
    print(f"  Top-K (E estimation): {args.top_k}")
    print(f"  Power-iteration steps (9x9): {args.n_iter}")
    print(f"  Power-iteration steps (3x3 manifold): {args.n_iter_manifold}")
    print(f"  Opset version: {args.opset_version}")
    print(f"  Dynamic axes: {args.dynamic_axes}")
    print(f"  Optimization: {optimization}")


if __name__ == "__main__":
    main()
