"""
Example usage of AKAZE feature detector.

This script demonstrates how to use the AKAZE model for feature detection
and how to export it to ONNX format.
"""

import torch
from pytorch_model.feature_detection.akaze import AKAZE


def main():
    # Create AKAZE model with custom parameters
    model = AKAZE(
        num_scales=3,               # Number of scale levels
        diffusion_iterations=3,     # FED iterations per scale
        kappa=0.05,                 # Edge preservation parameter
        threshold=0.001,            # Feature detection threshold
        nms_size=5,                 # NMS window size
        orientation_patch_size=15,  # Patch size for orientation
        orientation_sigma=2.5,      # Gaussian sigma for orientation
    )
    model.eval()

    # Create dummy input image (batch=1, channels=1, height=480, width=640)
    # In practice, this would be your actual grayscale image
    # normalized to [0, 1] or [0, 255]
    dummy_image = torch.randn(1, 1, 480, 640)

    # Run feature detection
    with torch.no_grad():
        scores, orientations = model(dummy_image)

    print(f"Input shape:        {dummy_image.shape}")
    print(f"Scores shape:       {scores.shape}")
    print(f"Orientations shape: {orientations.shape}")
    print(f"\nScore statistics:")
    print(f"  Min:  {scores.min().item():.6f}")
    print(f"  Max:  {scores.max().item():.6f}")
    print(f"  Mean: {scores.mean().item():.6f}")
    print(f"\nOrientation statistics (radians):")
    print(f"  Min:  {orientations.min().item():.6f}")
    print(f"  Max:  {orientations.max().item():.6f}")
    print(f"  Mean: {orientations.mean().item():.6f}")

    # Export to ONNX
    onnx_path = "akaze_example.onnx"
    torch.onnx.export(
        model,
        dummy_image,
        onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["scores", "orientations"],
    )
    print(f"\nModel exported to: {onnx_path}")


if __name__ == "__main__":
    main()
