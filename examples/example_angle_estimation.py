"""
Example: Using AngleEstimator with Shi-Tomasi feature detection.

This example demonstrates how to use the angle estimation module
in the complete pipeline: Shi-Tomasi → Angle Estimation → Sparse BAD → Sinkhorn

Run this script after installing dependencies:
    pip install -r requirements.txt
    python examples/example_angle_estimation.py
"""

import torch
import sys
import os
import tempfile

# Add pytorch_model to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pytorch_model'))

from corner.shi_tomasi import ShiTomasiScore
from orientation.angle_estimation import AngleEstimator
from feature_detection.shi_tomasi_angle import ShiTomasiWithAngle, ShiTomasiAngleSparseBAD


def example_basic_usage():
    """Example 1: Basic angle estimation."""
    print("=" * 60)
    print("Example 1: Basic Angle Estimation")
    print("=" * 60)

    # Create a sample image
    image = torch.randn(1, 1, 480, 640)

    # Create angle estimator
    estimator = AngleEstimator(patch_size=15, sigma=2.5)

    # Compute angles
    angles = estimator(image)

    print(f"Input shape:  {image.shape}")
    print(f"Output shape: {angles.shape}")
    print(f"Angle range:  [{angles.min():.4f}, {angles.max():.4f}] radians")
    print(f"Mean angle:   {angles.mean():.4f} radians")
    print()


def example_with_shi_tomasi():
    """Example 2: Integration with Shi-Tomasi."""
    print("=" * 60)
    print("Example 2: Shi-Tomasi + Angle Estimation")
    print("=" * 60)

    # Create a sample image
    image = torch.randn(1, 1, 480, 640)

    # Detect features using Shi-Tomasi
    detector = ShiTomasiScore(block_size=5)
    scores = detector(image)

    # Compute angles
    estimator = AngleEstimator(patch_size=15, sigma=2.5)
    angles = estimator(image)

    # Select top-k feature points
    k = 100
    scores_flat = scores.view(1, -1)
    values, indices = torch.topk(scores_flat, k, dim=1)

    # Get corresponding angles
    angles_flat = angles.view(1, -1)
    selected_angles = angles_flat.gather(1, indices)

    print(f"Detected {k} feature points")
    print(f"Score range:  [{values.min():.6f}, {values.max():.6f}]")
    print(f"Angle range:  [{selected_angles.min():.4f}, {selected_angles.max():.4f}]")
    print(f"Mean angle:   {selected_angles.mean():.4f} radians")
    print()


def example_unified_module():
    """Example 3: Using the unified ShiTomasiWithAngle module."""
    print("=" * 60)
    print("Example 3: Unified Module (ShiTomasiWithAngle)")
    print("=" * 60)

    # Create a sample image
    image = torch.randn(1, 1, 480, 640)

    # Use unified module
    detector = ShiTomasiWithAngle(
        block_size=5,
        patch_size=15,
        sigma=2.5
    )

    # Get both scores and angles in one call
    scores, angles = detector(image)

    print(f"Scores shape: {scores.shape}")
    print(f"Angles shape: {angles.shape}")
    print(f"Score range:  [{scores.min():.6f}, {scores.max():.6f}]")
    print(f"Angle range:  [{angles.min():.4f}, {angles.max():.4f}]")
    print()


def example_feature_matching_pipeline():
    """Example 4: Feature matching pipeline demonstration."""
    print("=" * 60)
    print("Example 4: Feature Matching Pipeline")
    print("=" * 60)

    # Create two sample images
    image1 = torch.randn(1, 1, 480, 640)
    image2 = torch.randn(1, 1, 480, 640)

    # Detector with angle estimation
    detector = ShiTomasiWithAngle(block_size=5, patch_size=15)

    # Detect features in both images
    scores1, angles1 = detector(image1)
    scores2, angles2 = detector(image2)

    print("Image 1:")
    print(f"  Scores shape: {scores1.shape}")
    print(f"  Angles shape: {angles1.shape}")

    print("\nImage 2:")
    print(f"  Scores shape: {scores2.shape}")
    print(f"  Angles shape: {angles2.shape}")

    # Select top-k keypoints
    k = 100
    scores1_flat = scores1.view(1, -1)
    scores2_flat = scores2.view(1, -1)

    _, indices1 = torch.topk(scores1_flat, k, dim=1)
    _, indices2 = torch.topk(scores2_flat, k, dim=1)

    # Get keypoint coordinates
    h, w = scores1.shape[2], scores1.shape[3]
    y1 = (indices1 // w).float()
    x1 = (indices1 % w).float()
    y2 = (indices2 // w).float()
    x2 = (indices2 % w).float()

    keypoints1 = torch.stack([x1, y1], dim=-1)  # (1, k, 2)
    keypoints2 = torch.stack([x2, y2], dim=-1)  # (1, k, 2)

    # Get angles for keypoints
    angles1_flat = angles1.view(1, -1)
    angles2_flat = angles2.view(1, -1)
    kp_angles1 = angles1_flat.gather(1, indices1)
    kp_angles2 = angles2_flat.gather(1, indices2)

    print(f"\nSelected {k} keypoints from each image")
    print(f"Keypoints1 shape: {keypoints1.shape}")
    print(f"Keypoints2 shape: {keypoints2.shape}")
    print(f"Angles1 shape: {kp_angles1.shape}")
    print(f"Angles2 shape: {kp_angles2.shape}")

    print("\nNext steps in the pipeline:")
    print("  1. ✓ Shi-Tomasi feature detection")
    print("  2. ✓ Angle estimation")
    print("  3. → Sparse BAD descriptor computation (rotation-aware)")
    print("  4. → Sinkhorn matching algorithm")
    print()


def example_onnx_export():
    """Example 5: ONNX export."""
    print("=" * 60)
    print("Example 5: ONNX Export")
    print("=" * 60)

    # Create model
    model = AngleEstimator(patch_size=15, sigma=2.5)
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 1, 256, 256)

    # Export to ONNX
    output_path = os.path.join(tempfile.gettempdir(), "angle_estimator_example.onnx")

    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            input_names=['image'],
            output_names=['angles'],
            dynamic_axes={
                'image': {0: 'batch', 2: 'height', 3: 'width'},
                'angles': {0: 'batch', 2: 'height', 3: 'width'}
            },
            opset_version=17
        )

        print(f"✓ Model exported to: {output_path}")

        # Verify with onnx
        try:
            import onnx
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            print("✓ ONNX model validation passed")
        except ImportError:
            print("  (onnx package not available for validation)")

        # Clean up
        os.remove(output_path)

    except Exception as e:
        print(f"✗ Export failed: {e}")

    print()


def example_sparse_bad_descriptors():
    """Example 6: Complete pipeline with Sparse BAD descriptors."""
    print("=" * 60)
    print("Example 6: Sparse BAD Descriptor Computation")
    print("=" * 60)

    # Create two sample images
    image1 = torch.randn(1, 1, 480, 640)
    image2 = torch.randn(1, 1, 480, 640)

    # Create model with Sparse BAD descriptors
    model = ShiTomasiAngleSparseBAD(
        block_size=5,
        patch_size=15,
        num_pairs=256,
        binarize=True,
        soft_binarize=True,
        temperature=10.0,
        normalize_descriptors=True
    )

    print("Processing first image:")
    # 1. Detect and orient
    scores1, angles1 = model.detect_and_orient(image1)
    print(f"  Scores shape: {scores1.shape}")
    print(f"  Angles shape: {angles1.shape}")

    # 2. Select keypoints
    k = 100
    scores1_flat = scores1.view(1, -1)
    _, indices1 = torch.topk(scores1_flat, k, dim=1)
    h, w = scores1.shape[2], scores1.shape[3]
    y1 = (indices1 // w).float()
    x1 = (indices1 % w).float()
    keypoints1 = torch.stack([y1, x1], dim=-1)  # (1, k, 2) in (y, x)

    # 3. Compute rotation-aware descriptors
    descriptors1 = model.describe(image1, keypoints1, angles1)
    print(f"  Keypoints shape: {keypoints1.shape}")
    print(f"  Descriptors shape: {descriptors1.shape}")
    print(f"  Descriptor range: [{descriptors1.min():.4f}, {descriptors1.max():.4f}]")

    print("\nProcessing second image:")
    scores2, angles2 = model.detect_and_orient(image2)
    scores2_flat = scores2.view(1, -1)
    _, indices2 = torch.topk(scores2_flat, k, dim=1)
    y2 = (indices2 // w).float()
    x2 = (indices2 % w).float()
    keypoints2 = torch.stack([y2, x2], dim=-1)
    descriptors2 = model.describe(image2, keypoints2, angles2)
    print(f"  Descriptors shape: {descriptors2.shape}")

    # Compute descriptor distances (for matching)
    # L2 distance between all pairs
    desc1_expanded = descriptors1.unsqueeze(2)  # (1, k, 1, 256)
    desc2_expanded = descriptors2.unsqueeze(1)  # (1, 1, k, 256)
    distances = torch.norm(desc1_expanded - desc2_expanded, dim=-1)  # (1, k, k)

    print(f"\n✓ Distance matrix shape: {distances.shape}")
    print(f"  Distance range: [{distances.min():.4f}, {distances.max():.4f}]")
    print(f"  Mean distance: {distances.mean():.4f}")

    # Find nearest neighbors
    min_distances, nearest_neighbors = distances.min(dim=2)
    print(f"\n✓ Nearest neighbor distances:")
    print(f"  Mean: {min_distances.mean():.4f}")
    print(f"  Min: {min_distances.min():.4f}")
    print(f"  Max: {min_distances.max():.4f}")

    print("\nPipeline summary:")
    print("  1. ✓ Shi-Tomasi feature detection")
    print("  2. ✓ Angle estimation (rotation awareness)")
    print("  3. ✓ Sparse BAD descriptor computation")
    print("  4. → Ready for Sinkhorn matching")
    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Angle Estimation Module - Usage Examples")
    print("=" * 60)
    print()

    # Set random seed for reproducibility
    torch.manual_seed(42)

    try:
        example_basic_usage()
        example_with_shi_tomasi()
        example_unified_module()
        example_feature_matching_pipeline()
        example_onnx_export()
        example_sparse_bad_descriptors()

        print("=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
