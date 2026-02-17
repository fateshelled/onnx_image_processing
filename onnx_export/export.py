#!/usr/bin/env python3
"""
Export all ONNX models.

This script serves as a central registry for all ONNX export configurations.
Add new models to the EXPORT_CONFIGS list below.

Usage:
    python export.py --output-dir ./models
    python export.py --output-dir ./models --dynamic-axes
"""

import argparse
import subprocess
import sys
from pathlib import Path


# Registry of export configurations
# Each entry: (script_name, output_name, extra_args)
EXPORT_CONFIGS = [
    (
        "export_shi_tomasi_sparse_bad_sinkhorn.py",
        "shi_tomasi_sparse_bad_sinkhorn.onnx",
        ["--height", "480", "--width", "640", "--num-pairs", "512", "--max-keypoints", "1024"]
    ),
    (
        "export_shi_tomasi_sparse_bad_sinkhorn.py",
        "shi_tomasi_sparse_bad_sinkhorn_extraction.onnx",
        ["--height", "480", "--width", "640", "--num-pairs", "512", "--max-keypoints", "1024", "--with-extraction"]
    ),
    (
        "export_shi_tomasi_angle_sparse_bad_sinkhorn.py",
        "shi_tomasi_angle_sparse_bad_sinkhorn.onnx",
        ["--height", "480", "--width", "640", "--num-pairs", "512", "--max-keypoints", "1024"]
    ),
    (
        "export_shi_tomasi_angle_sparse_bad_sinkhorn.py",
        "shi_tomasi_angle_sparse_bad_sinkhorn_extraction.onnx",
        ["--height", "480", "--width", "640", "--num-pairs", "512", "--max-keypoints", "1024", "--with-extraction"]
    ),
    (
        "export_shi_tomasi_angle_sparse_bad_sinkhorn_with_filters.py",
        "shi_tomasi_angle_sparse_bad_sinkhorn_with_filters.onnx",
        ["--height", "480", "--width", "640", "--num-pairs", "512", "--max-keypoints", "1024"]
    ),
    (
        "export_shi_tomasi_angle_sparse_bad_sinkhorn_with_filters.py",
        "shi_tomasi_angle_sparse_bad_sinkhorn_with_filters_extraction.onnx",
        ["--height", "480", "--width", "640", "--num-pairs", "512", "--max-keypoints", "1024", "--with-extraction"]
    ),
    (
        "export_akaze_sparse_bad_sinkhorn.py",
        "akaze_sparse_bad_sinkhorn.onnx",
        ["--height", "480", "--width", "640", "--num-pairs", "512", "--max-keypoints", "1024"]
    ),
    (
        "export_akaze_sparse_bad_sinkhorn.py",
        "akaze_sparse_bad_sinkhorn_extraction.onnx",
        ["--height", "480", "--width", "640", "--num-pairs", "512", "--max-keypoints", "1024", "--with-extraction"]
    ),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export all ONNX models"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=".",
        help="Output directory for ONNX files (default: current directory)"
    )
    parser.add_argument(
        "--dynamic-axes",
        action="store_true",
        help="Enable dynamic input shape for all models that support it"
    )
    parser.add_argument(
        "--disable-dynamo",
        action="store_true",
        help="Disable dynamo for all exports"
    )
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Disable ONNX model optimization (onnxsim/onnxoptimizer) for all exports"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    script_dir = Path(__file__).parent
    exported_models = []
    failed_models = []

    for script_name, output_name, extra_args in EXPORT_CONFIGS:
        script_path = script_dir / script_name

        if not script_path.exists():
            print(f"Warning: {script_name} not found, skipping")
            failed_models.append(output_name)
            continue

        # Build base output path
        output_path = output_dir / output_name

        # Export with static axes
        cmd = [
            sys.executable,
            str(script_path),
            "--output", str(output_path),
            *extra_args
        ]
        if args.disable_dynamo:
            cmd.append("--disable-dynamo")
        if args.no_optimize:
            cmd.append("--no-optimize")

        print(f"Exporting: {output_name}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"  Error: {result.stderr}")
            failed_models.append(output_name)
        else:
            exported_models.append(str(output_path))
            print(f"  OK: {output_path}")

        # Export with dynamic axes if requested
        if args.dynamic_axes:
            dynamic_output_name = output_name.replace(".onnx", "_dynamic.onnx")
            dynamic_output_path = output_dir / dynamic_output_name

            cmd_dynamic = [
                sys.executable,
                str(script_path),
                "--output", str(dynamic_output_path),
                *extra_args,
                "--dynamic-axes",
                "--disable-dynamo",
            ]
            if args.no_optimize:
                cmd_dynamic.append("--no-optimize")

            print(f"Exporting: {dynamic_output_name}")
            result = subprocess.run(cmd_dynamic, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"  Error: {result.stderr}")
                failed_models.append(dynamic_output_name)
            else:
                exported_models.append(str(dynamic_output_path))
                print(f"  OK: {dynamic_output_path}")

    # Summary
    print("\n" + "=" * 50)
    print(f"Exported: {len(exported_models)} models")
    if failed_models:
        print(f"Failed: {len(failed_models)} models")
        for name in failed_models:
            print(f"  - {name}")
        sys.exit(1)

    # Print exported model paths for CI
    print("\nExported models:")
    for model_path in exported_models:
        print(f"  {model_path}")


if __name__ == "__main__":
    main()
