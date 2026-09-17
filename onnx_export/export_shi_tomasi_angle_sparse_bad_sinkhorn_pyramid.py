#!/usr/bin/env python3
"""
ONNX export script for the pyramid variant of the Shi-Tomasi + Angle + Sparse
BAD + Sinkhorn matching model.

Mirrors ``export_shi_tomasi_angle_sparse_bad_sinkhorn.py`` but instantiates
``ShiTomasiAngleSparseBADSinkhornMatcherPyramid`` (multi-scale detection) and
adds ``--num-levels`` / ``--level-weights``.

Usage:
    python export_shi_tomasi_angle_sparse_bad_sinkhorn_pyramid.py \
        -o pyramid.onnx --max-keypoints 512 --num-levels 2
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from pytorch_model.feature_detection.shi_tomasi_angle_sparse_bad_sinkhorn_pyramid import (
    ShiTomasiAngleSparseBADSinkhornMatcherPyramid,
)
from pytorch_model.feature_detection.single_image_pyramid_features import (
    SingleImagePyramidFeatures,
    PairWithDescriptors,
)
from pytorch_model.feature_detection.match_extraction_wrapper import MatchExtractionWrapper
from onnx_export.optimize import optimize_onnx_model, remove_external_data


def parse_args():
    p = argparse.ArgumentParser(
        description="Export pyramid Shi-Tomasi+Angle+SparseBAD+Sinkhorn matcher to ONNX"
    )
    p.add_argument("--output", "-o", default="pyramid.onnx")
    p.add_argument("--height", "-H", type=int, default=480)
    p.add_argument("--width", "-W", type=int, default=640)
    p.add_argument("--max-keypoints", "-k", type=int, default=1024)
    p.add_argument("--num-levels", "-L", type=int, default=2)
    p.add_argument("--level-weights", type=float, nargs="+", default=None,
                   help="Per-level keypoint budget weights (length == num-levels, sum 1).")
    # Shi-Tomasi / angle / BAD / sinkhorn params (same as base export)
    p.add_argument("--block-size", type=int, default=5)
    p.add_argument("--patch-size", type=int, default=15)
    p.add_argument("--sigma", type=float, default=2.5)
    p.add_argument("--num-pairs", "-n", type=int, choices=[256, 512], default=512)
    p.add_argument("--binarization", type=str, choices=["none", "soft", "hard"], default="hard")
    p.add_argument("--temperature", type=float, default=10.0)
    p.add_argument("--sinkhorn-iterations", "-i", type=int, default=20)
    p.add_argument("--epsilon", "-e", type=float, default=0.05)
    p.add_argument("--unused-score", type=float, default=1.0)
    p.add_argument("--normalize-descriptors", action="store_true", default=True)
    p.add_argument("--no-normalize-descriptors", dest="normalize_descriptors",
                   action="store_false")
    p.add_argument("--distance-type", type=str, choices=["l1", "l2"], default="l2")
    p.add_argument("--nms-radius", type=int, default=5)
    p.add_argument("--score-threshold", type=float, default=0.0)
    p.add_argument("--sampling-mode", type=str, choices=["nearest", "bilinear"], default="nearest")
    p.add_argument("--cas-sharpness", type=float, default=0.0)
    p.add_argument("--with-extraction", action="store_true")
    p.add_argument("--single-image", action="store_true",
                   help="Export a single-image wrapper that returns "
                        "(keypoints, descriptors) for feature caching.")
    p.add_argument("--with-descriptors", action="store_true",
                   help="Export the standard pair model with the internally "
                        "computed descriptors added as outputs "
                        "(keypoints1, keypoints2, descriptors1, "
                        "descriptors2, matching_probs).")
    p.add_argument("--max-matches", type=int, default=100)
    p.add_argument("--match-threshold", type=float, default=0.1)
    p.add_argument("--opset-version", type=int, default=18)
    p.add_argument("--dynamic-axes", action="store_true")
    p.add_argument("--disable-dynamo", action="store_true")
    p.add_argument("--no-optimize", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    binarize = args.binarization != "none"
    soft_binarize = args.binarization == "soft"

    pyramid_kwargs = dict(
        max_keypoints=args.max_keypoints,
        num_levels=args.num_levels,
        level_weights=args.level_weights,
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
        cas_sharpness=args.cas_sharpness,
    )

    base_model = ShiTomasiAngleSparseBADSinkhornMatcherPyramid(**pyramid_kwargs)

    if args.single_image:
        model = SingleImagePyramidFeatures(base_model)
    elif args.with_descriptors:
        model = PairWithDescriptors(base_model)
    elif args.with_extraction:
        model = MatchExtractionWrapper(
            feature_matcher=base_model,
            max_matches=args.max_matches,
            match_threshold=args.match_threshold,
        )
    else:
        model = base_model

    model.eval()

    single_image = args.single_image
    dummy1 = torch.randn(1, 1, args.height, args.width)
    dummy2 = torch.randn(1, 1, args.height, args.width)

    if single_image:
        output_names = ["keypoints", "descriptors"]
        sample_inputs = (dummy1,)
        input_names = ["image"]
        dynamic_axes = None
        if args.dynamic_axes:
            dynamic_axes = {
                "image": {0: "batch", 2: "height", 3: "width"},
            }
    else:
        if args.with_descriptors:
            output_names = ["keypoints1", "keypoints2", "descriptors1",
                            "descriptors2", "matching_probs"]
        elif args.with_extraction:
            output_names = ["matched_kpts1", "matched_kpts2", "scores", "valid_mask"]
        else:
            output_names = ["keypoints1", "keypoints2", "matching_probs"]
        sample_inputs = (dummy1, dummy2)
        input_names = ["image1", "image2"]
        dynamic_axes = None
        if args.dynamic_axes:
            dynamic_axes = {
                "image1": {0: "batch", 2: "height", 3: "width"},
                "image2": {0: "batch", 2: "height", 3: "width"},
            }

    print(f"Exporting pyramid matcher to ONNX (levels={args.num_levels}, "
          f"single_image={single_image}, with_descriptors={args.with_descriptors})...")
    torch.onnx.export(
        model,
        sample_inputs,
        args.output,
        export_params=True,
        opset_version=args.opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        dynamo=not args.disable_dynamo,
    )

    optimization = "skipped"
    if not args.no_optimize:
        print("Optimizing ONNX model...")
        optimization = optimize_onnx_model(args.output)
    else:
        remove_external_data(args.output)

    print(f"\nExported: {args.output}")
    print(f"  num_levels={args.num_levels} level_weights={args.level_weights}")
    print(f"  max_keypoints={args.max_keypoints} binarization={args.binarization}")
    print(f"  epsilon={args.epsilon} distance_type={args.distance_type}")
    print(f"  optimization={optimization}")


if __name__ == "__main__":
    main()
