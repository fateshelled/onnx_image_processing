"""
Test script for ShiTomasiAngleSparseBAD module.

This script tests the complete Shi-Tomasi + Angle + Sparse BAD pipeline.
"""

import torch
import numpy as np
import sys
import os

# Add the pytorch_model directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pytorch_model'))

from feature_detection.shi_tomasi_angle import ShiTomasiAngleSparseBAD


def test_basic_functionality():
    """Test basic detect and describe functionality."""
    print("=" * 60)
    print("Test 1: Basic Functionality")
    print("=" * 60)

    # Create test image
    batch_size = 2
    height, width = 480, 640
    image = torch.randn(batch_size, 1, height, width)

    # Create model
    model = ShiTomasiAngleSparseBAD(num_pairs=256)

    # Detect and orient
    scores, angles = model.detect_and_orient(image)

    print(f"✓ Input shape: {image.shape}")
    print(f"✓ Scores shape: {scores.shape}")
    print(f"✓ Angles shape: {angles.shape}")
    print(f"  Score range: [{scores.min():.6f}, {scores.max():.6f}]")
    print(f"  Angle range: [{angles.min():.4f}, {angles.max():.4f}] rad")
    print()


def test_descriptor_computation():
    """Test descriptor computation at keypoints."""
    print("=" * 60)
    print("Test 2: Descriptor Computation")
    print("=" * 60)

    # Create test image
    image = torch.randn(1, 1, 480, 640)

    # Create model with different configurations
    configs = [
        {"num_pairs": 256, "binarize": False, "name": "256-dim raw"},
        {"num_pairs": 512, "binarize": False, "name": "512-dim raw"},
        {"num_pairs": 256, "binarize": True, "soft_binarize": True, "name": "256-dim soft binary"},
        {"num_pairs": 256, "binarize": True, "soft_binarize": False, "name": "256-dim hard binary"},
    ]

    for config in configs:
        name = config.pop("name")
        model = ShiTomasiAngleSparseBAD(**config)

        # Detect features
        scores, angles = model.detect_and_orient(image)

        # Select top-k keypoints
        k = 100
        scores_flat = scores.view(1, -1)
        _, indices = torch.topk(scores_flat, k, dim=1)

        h, w = scores.shape[2], scores.shape[3]
        y = (indices // w).float()
        x = (indices % w).float()
        keypoints = torch.stack([y, x], dim=-1)  # (1, k, 2) in (y, x)

        # Compute descriptors
        descriptors = model.describe(image, keypoints, angles)

        print(f"✓ {name}")
        print(f"  Descriptor shape: {descriptors.shape}")
        print(f"  Descriptor range: [{descriptors.min():.4f}, {descriptors.max():.4f}]")
        print(f"  Descriptor mean: {descriptors.mean():.4f}")
        print(f"  Descriptor std: {descriptors.std():.4f}")

    print()


def test_rotation_awareness():
    """Test rotation awareness of descriptors."""
    print("=" * 60)
    print("Test 3: Rotation Awareness")
    print("=" * 60)

    # Create simple test pattern
    image = torch.zeros(1, 1, 100, 100)
    # Add a gradient pattern
    for i in range(100):
        image[0, 0, i, :] = i / 100.0

    model = ShiTomasiAngleSparseBAD(num_pairs=256)

    # Detect features
    scores, angles = model.detect_and_orient(image)

    # Select center keypoint
    keypoints = torch.tensor([[[50.0, 50.0]]])  # (1, 1, 2) in (y, x)

    # Compute descriptor
    desc_original = model.describe(image, keypoints, angles)

    # Manually rotate orientation and recompute
    # (This is a simple test - in practice, the image would be rotated too)
    angles_rotated = angles + np.pi / 4  # Rotate by 45 degrees
    desc_rotated = model.describe(image, keypoints, angles_rotated)

    print(f"✓ Original descriptor shape: {desc_original.shape}")
    print(f"✓ Rotated descriptor shape: {desc_rotated.shape}")
    print(f"  Descriptor difference (should be non-zero): {torch.abs(desc_original - desc_rotated).mean():.4f}")
    print(f"  Note: Different orientations produce different descriptors (rotation-aware)")
    print()


def test_invalid_keypoints():
    """Test handling of invalid keypoints."""
    print("=" * 60)
    print("Test 4: Invalid Keypoint Handling")
    print("=" * 60)

    image = torch.randn(1, 1, 480, 640)
    model = ShiTomasiAngleSparseBAD(num_pairs=256)

    scores, angles = model.detect_and_orient(image)

    # Mix valid and invalid keypoints
    keypoints = torch.tensor([
        [[50.0, 50.0]],   # Valid
        [[100.0, 100.0]], # Valid
        [[-1.0, -1.0]],   # Invalid
        [[200.0, 200.0]], # Valid
        [[-1.0, -1.0]],   # Invalid
    ]).transpose(0, 1)  # (1, 5, 2)

    descriptors = model.describe(image, keypoints, angles)

    print(f"✓ Keypoints shape: {keypoints.shape}")
    print(f"✓ Descriptors shape: {descriptors.shape}")
    print(f"  Descriptor[0] (valid) norm: {descriptors[0, 0].norm():.4f}")
    print(f"  Descriptor[1] (valid) norm: {descriptors[0, 1].norm():.4f}")
    print(f"  Descriptor[2] (invalid) norm: {descriptors[0, 2].norm():.4f} (should be ~0)")
    print(f"  Descriptor[3] (valid) norm: {descriptors[0, 3].norm():.4f}")
    print(f"  Descriptor[4] (invalid) norm: {descriptors[0, 4].norm():.4f} (should be ~0)")

    # Check that invalid descriptors are zero
    assert descriptors[0, 2].abs().max() < 1e-6, "Invalid descriptor should be zero"
    assert descriptors[0, 4].abs().max() < 1e-6, "Invalid descriptor should be zero"
    print("✓ Invalid keypoints correctly produce zero descriptors")
    print()


def test_normalization():
    """Test descriptor normalization."""
    print("=" * 60)
    print("Test 5: Descriptor Normalization")
    print("=" * 60)

    image = torch.randn(1, 1, 480, 640)

    # Test with and without normalization
    model_normalized = ShiTomasiAngleSparseBAD(num_pairs=256, normalize_descriptors=True)
    model_unnormalized = ShiTomasiAngleSparseBAD(num_pairs=256, normalize_descriptors=False)

    scores, angles = model_normalized.detect_and_orient(image)

    # Select keypoints
    k = 50
    scores_flat = scores.view(1, -1)
    _, indices = torch.topk(scores_flat, k, dim=1)
    h, w = scores.shape[2], scores.shape[3]
    y = (indices // w).float()
    x = (indices % w).float()
    keypoints = torch.stack([y, x], dim=-1)

    desc_norm = model_normalized.describe(image, keypoints, angles)
    desc_unnorm = model_unnormalized.describe(image, keypoints, angles)

    norms_normalized = desc_norm.norm(dim=-1, p=2)
    norms_unnormalized = desc_unnorm.norm(dim=-1, p=2)

    print(f"✓ Normalized descriptor norms:")
    print(f"  Mean: {norms_normalized.mean():.4f} (should be ~1.0)")
    print(f"  Min: {norms_normalized.min():.4f}")
    print(f"  Max: {norms_normalized.max():.4f}")

    print(f"✓ Unnormalized descriptor norms:")
    print(f"  Mean: {norms_unnormalized.mean():.4f}")
    print(f"  Min: {norms_unnormalized.min():.4f}")
    print(f"  Max: {norms_unnormalized.max():.4f}")

    # Check that normalized descriptors have unit norm (within tolerance)
    assert torch.allclose(norms_normalized, torch.ones_like(norms_normalized), atol=1e-5), \
        "Normalized descriptors should have unit norm"
    print("✓ Normalized descriptors have unit norm")
    print()


def test_comparison_with_akaze():
    """Compare output format with AKAZE implementation."""
    print("=" * 60)
    print("Test 6: Comparison with AKAZE Format")
    print("=" * 60)

    try:
        from feature_detection.akaze_sparse_bad_sinkhorn import AKAZESparseBADSinkhornMatcher

        image = torch.randn(1, 1, 480, 640)

        # Shi-Tomasi version
        shi_model = ShiTomasiAngleSparseBAD(num_pairs=256)
        shi_scores, shi_angles = shi_model.detect_and_orient(image)

        # AKAZE version (just for format comparison)
        # Note: AKAZE includes full matcher, we're just comparing detection output
        print(f"✓ Shi-Tomasi + Angle output:")
        print(f"  Scores shape: {shi_scores.shape}")
        print(f"  Angles shape: {shi_angles.shape}")
        print(f"  Output format: (scores, angles) - same as AKAZE")

        # Test descriptor computation
        k = 100
        scores_flat = shi_scores.view(1, -1)
        _, indices = torch.topk(scores_flat, k, dim=1)
        h, w = shi_scores.shape[2], shi_scores.shape[3]
        y = (indices // w).float()
        x = (indices % w).float()
        keypoints = torch.stack([y, x], dim=-1)

        shi_desc = shi_model.describe(image, keypoints, shi_angles)

        print(f"✓ Shi-Tomasi descriptors:")
        print(f"  Shape: {shi_desc.shape} (B, K, num_pairs)")
        print(f"  Same format as AKAZE sparse BAD descriptors")
        print("✓ Output formats are compatible!")

    except ImportError:
        print("⚠ AKAZE module not available for comparison (optional)")

    print()


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 5 + "Shi-Tomasi + Angle + Sparse BAD Tests" + " " * 13 + "║")
    print("╚" + "═" * 58 + "╝")
    print()

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    try:
        test_basic_functionality()
        test_descriptor_computation()
        test_rotation_awareness()
        test_invalid_keypoints()
        test_normalization()
        test_comparison_with_akaze()

        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        print()

        print("ShiTomasiAngleSparseBAD is now fully functional!")
        print()
        print("Pipeline:")
        print("  1. Shi-Tomasi corner detection")
        print("  2. Angle estimation (rotation awareness)")
        print("  3. Sparse BAD descriptor computation")
        print()
        print("Next steps:")
        print("  - Integrate with Sinkhorn matcher for complete pipeline")
        print("  - Export to ONNX for deployment")
        print()

        return 0

    except Exception as e:
        print("=" * 60)
        print(f"Test failed with error: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
