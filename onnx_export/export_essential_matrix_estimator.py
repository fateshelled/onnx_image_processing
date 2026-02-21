#!/usr/bin/env python3
"""
ONNX export script for EssentialMatrixEstimator.

Estimates the Essential Matrix from a Sinkhorn probability matrix using the
weighted 8-point algorithm with Hartley normalisation.

Note on ONNX exporter
---------------------
This model uses fixed-iteration power iteration to replace ``torch.linalg.svd``
and related ops that are unsupported by the TorchScript-based ONNX exporter at
any opset version.  The dynamo exporter is therefore **not** used here; the
script always calls ``torch.onnx.export`` without ``dynamo=True``.
The required minimum opset version is 14.

Usage:
    python export_essential_matrix_estimator.py --output essential_matrix_estimator.onnx
    python export_essential_matrix_estimator.py --fx 525 --fy 525 --cx 320 --cy 240 \\
        --image-height 480 --image-width 640 --output essential_matrix_estimator.onnx
"""

import argparse
import sys
from pathlib import Path

import torch

# Add parent directory to path for importing pytorch_model
sys.path.insert(0, str(Path(__file__).parent.parent))

from pytorch_model.geometry.essential_matrix_estimator import EssentialMatrixEstimator
from onnx_export.optimize import optimize_onnx_model, remove_external_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export EssentialMatrixEstimator model to ONNX format"
    )

    # ── Output ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="essential_matrix_estimator.onnx",
        help="Output ONNX file path (default: essential_matrix_estimator.onnx)",
    )

    # ── Camera intrinsics ────────────────────────────────────────────────
    parser.add_argument(
        "--fx",
        type=float,
        default=16.0,
        help="Focal length in x direction [px] (default: 16.0)",
    )
    parser.add_argument(
        "--fy",
        type=float,
        default=16.0,
        help="Focal length in y direction [px] (default: 16.0)",
    )
    parser.add_argument(
        "--cx",
        type=float,
        default=16.0,
        help="Principal point x [px] (default: 16.0)",
    )
    parser.add_argument(
        "--cy",
        type=float,
        default=16.0,
        help="Principal point y [px] (default: 16.0)",
    )

    # ── Feature-point grid ───────────────────────────────────────────────
    parser.add_argument(
        "--image-height", "-H",
        type=int,
        default=32,
        help=(
            "Height of the feature-point grid (default: 32). "
            "Feature point index i maps to pixel y = i // image-width."
        ),
    )
    parser.add_argument(
        "--image-width", "-W",
        type=int,
        default=32,
        help=(
            "Width of the feature-point grid (default: 32). "
            "Feature point index i maps to pixel x = i %% image-width."
        ),
    )

    # ── Algorithm hyper-parameters ───────────────────────────────────────
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Bidirectional top-K entries kept per row/column of P (default: 3)",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=30,
        help=(
            "Power-iteration steps for the 9x9 eigenvector solve (default: 30). "
            "More iterations → better accuracy but larger ONNX graph."
        ),
    )
    parser.add_argument(
        "--n-iter-manifold",
        type=int,
        default=10,
        help=(
            "Power-iteration steps for each 3x3 eigenvector solve inside the "
            "E-manifold projection (default: 10). "
            "3x3 matrices converge much faster than the 9x9 case."
        ),
    )

    # ── ONNX export options ──────────────────────────────────────────────
    parser.add_argument(
        "--opset-version",
        type=int,
        default=14,
        help=(
            "ONNX opset version (default: 14). "
            "Must be >= 14; dynamo exporter is not supported for this model."
        ),
    )
    parser.add_argument(
        "--dynamic-axes",
        action="store_true",
        help=(
            "Enable dynamic N and M axes on input P "
            "(N+1 = rows, M+1 = columns). "
            "The pixel-grid buffer size in the model still constrains "
            "max(N, M) <= image-height * image-width."
        ),
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
        [[args.fx,    0.0,   args.cx],
         [   0.0,  args.fy,  args.cy],
         [   0.0,    0.0,      1.0  ]],
        dtype=torch.float32,
    )

    # ── Instantiate model ────────────────────────────────────────────────
    model = EssentialMatrixEstimator(
        K=K,
        image_shape=(args.image_height, args.image_width),
        top_k=args.top_k,
        n_iter=args.n_iter,
        n_iter_manifold=args.n_iter_manifold,
    )
    model.eval()

    # ── Dummy input: P of shape (N+1, M+1) ──────────────────────────────
    # N = M = image_height * image_width (square grid, same for both images)
    n_points = args.image_height * args.image_width
    dummy_P = torch.rand(n_points + 1, n_points + 1)

    # ── Dynamic axes ────────────────────────────────────────────────────
    dynamic_axes = None
    if args.dynamic_axes:
        dynamic_axes = {
            "P": {0: "N_plus_1", 1: "M_plus_1"},
        }

    # ── Export to ONNX (TorchScript exporter; dynamo not supported) ─────
    torch.onnx.export(
        model,
        (dummy_P,),
        args.output,
        export_params=True,
        opset_version=args.opset_version,
        do_constant_folding=True,
        input_names=["P"],
        output_names=["E"],
        dynamic_axes=dynamic_axes,
    )

    # ── Optimize ONNX model ──────────────────────────────────────────────
    optimization = "skipped"
    if not args.no_optimize:
        optimization = optimize_onnx_model(args.output)
    else:
        remove_external_data(args.output)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"Exported ONNX model to: {args.output}")
    print(f"  K (intrinsics): fx={args.fx}, fy={args.fy}, cx={args.cx}, cy={args.cy}")
    print(f"  Image grid: {args.image_height}x{args.image_width} "
          f"({n_points} feature points)")
    print(f"  Input P shape: ({n_points + 1}, {n_points + 1})")
    print(f"  Output E shape: (3, 3)")
    print(f"  Top-K: {args.top_k}")
    print(f"  Power-iteration steps (9x9): {args.n_iter}")
    print(f"  Power-iteration steps (3x3 manifold): {args.n_iter_manifold}")
    print(f"  Opset version: {args.opset_version}")
    print(f"  Dynamic axes: {args.dynamic_axes}")
    print(f"  Optimization: {optimization}")


if __name__ == "__main__":
    main()
