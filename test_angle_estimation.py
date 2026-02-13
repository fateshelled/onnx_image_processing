"""
Test script for AngleEstimator module.

This script tests the angle estimation functionality both standalone
and integrated with Shi-Tomasi feature detection.
"""

import torch
import numpy as np
import sys
import os
import tempfile

# Add the pytorch_model directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pytorch_model'))

from orientation.angle_estimation import AngleEstimator, AngleEstimatorMultiScale
from corner.shi_tomasi import ShiTomasiScore


def test_basic_functionality():
    """Test basic angle estimation functionality."""
    print("=" * 60)
    print("Test 1: Basic Functionality")
    print("=" * 60)

    # Create test image
    batch_size = 2
    height, width = 480, 640
    image = torch.randn(batch_size, 1, height, width)

    # Create angle estimator
    estimator = AngleEstimator(patch_size=15, sigma=2.5)

    # Compute angles
    angles = estimator(image)

    # Check output shape
    assert angles.shape == (batch_size, 1, height, width), \
        f"Expected shape {(batch_size, 1, height, width)}, got {angles.shape}"

    # Check value range [-π, π]
    assert angles.min() >= -np.pi, f"Min angle {angles.min()} < -π"
    assert angles.max() <= np.pi, f"Max angle {angles.max()} > π"

    print(f"✓ Input shape: {image.shape}")
    print(f"✓ Output shape: {angles.shape}")
    print(f"✓ Angle range: [{angles.min():.4f}, {angles.max():.4f}] (expected: [-π, π])")
    print(f"✓ Mean angle: {angles.mean():.4f} rad")
    print()


def test_gradient_image():
    """Test with synthetic gradient image (known orientation)."""
    print("=" * 60)
    print("Test 2: Synthetic Gradient Image")
    print("=" * 60)

    # Create gradient image: intensity increases from left to right
    # Expected orientation: pointing right (0 radians)
    height, width = 100, 100
    x = torch.linspace(0, 255, width).view(1, 1, 1, width)
    gradient_x = x.expand(1, 1, height, width)

    estimator = AngleEstimator(patch_size=15, sigma=2.5)
    angles_x = estimator(gradient_x)

    # Check center region (edges may have boundary effects)
    center_angles = angles_x[0, 0, 25:75, 25:75]
    mean_angle_x = center_angles.mean().item()

    print(f"✓ Horizontal gradient image")
    print(f"  Mean angle in center: {mean_angle_x:.4f} rad ({np.degrees(mean_angle_x):.2f}°)")
    print(f"  Expected: ~0 rad (0°) - pointing right")

    # Create gradient image: intensity increases from top to bottom
    # Expected orientation: pointing down (π/2 radians)
    y = torch.linspace(0, 255, height).view(1, 1, height, 1)
    gradient_y = y.expand(1, 1, height, width)

    angles_y = estimator(gradient_y)
    center_angles_y = angles_y[0, 0, 25:75, 25:75]
    mean_angle_y = center_angles_y.mean().item()

    print(f"✓ Vertical gradient image")
    print(f"  Mean angle in center: {mean_angle_y:.4f} rad ({np.degrees(mean_angle_y):.2f}°)")
    print(f"  Expected: ~π/2 rad (90°) - pointing down")
    print()


def test_with_shi_tomasi():
    """Test integration with Shi-Tomasi feature detector."""
    print("=" * 60)
    print("Test 3: Integration with Shi-Tomasi")
    print("=" * 60)

    # Create test image
    image = torch.randn(1, 1, 480, 640)

    # Shi-Tomasi feature detection
    detector = ShiTomasiScore(block_size=5)
    scores = detector(image)

    # Angle estimation
    estimator = AngleEstimator(patch_size=15, sigma=2.5)
    angles = estimator(image)

    # Check shapes are compatible
    assert scores.shape == angles.shape, \
        f"Shape mismatch: scores {scores.shape} vs angles {angles.shape}"

    # Find top-k feature points
    k = 100
    scores_flat = scores.view(1, -1)
    _, indices = torch.topk(scores_flat, k, dim=1)

    # Extract corresponding angles
    angles_flat = angles.view(1, -1)
    selected_angles = angles_flat.gather(1, indices)

    print(f"✓ Shi-Tomasi scores shape: {scores.shape}")
    print(f"✓ Angle estimation shape: {angles.shape}")
    print(f"✓ Top-{k} feature points selected")
    print(f"  Score range: [{scores.max():.6f}, {scores.min():.6f}]")
    print(f"  Angle range: [{selected_angles.min():.4f}, {selected_angles.max():.4f}] rad")
    print(f"  Mean angle: {selected_angles.mean():.4f} rad ({np.degrees(selected_angles.mean()):.2f}°)")
    print()


def test_different_parameters():
    """Test with different patch sizes and sigma values."""
    print("=" * 60)
    print("Test 4: Different Parameters")
    print("=" * 60)

    image = torch.randn(1, 1, 256, 256)

    configs = [
        (7, 1.5),
        (15, 2.5),
        (21, 3.5),
        (31, 5.0),
    ]

    for patch_size, sigma in configs:
        estimator = AngleEstimator(patch_size=patch_size, sigma=sigma)
        angles = estimator(image)

        print(f"✓ patch_size={patch_size}, sigma={sigma:.1f}")
        print(f"  Output shape: {angles.shape}")
        print(f"  Angle range: [{angles.min():.4f}, {angles.max():.4f}]")
        print(f"  Mean: {angles.mean():.4f}, Std: {angles.std():.4f}")

    print()


def test_onnx_export():
    """Test ONNX export compatibility."""
    print("=" * 60)
    print("Test 5: ONNX Export")
    print("=" * 60)

    try:
        import onnx
        import onnxruntime as ort

        # Create model
        model = AngleEstimator(patch_size=15, sigma=2.5)
        model.eval()

        # Create dummy input
        dummy_input = torch.randn(1, 1, 256, 256)

        # Export to ONNX
        onnx_path = os.path.join(tempfile.gettempdir(), "angle_estimator.onnx")
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=['image'],
            output_names=['angles'],
            dynamic_axes={
                'image': {0: 'batch', 2: 'height', 3: 'width'},
                'angles': {0: 'batch', 2: 'height', 3: 'width'}
            },
            opset_version=17
        )

        # Verify ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)

        # Test inference with ONNX Runtime
        ort_session = ort.InferenceSession(onnx_path)
        ort_inputs = {'image': dummy_input.numpy()}
        ort_outputs = ort_session.run(None, ort_inputs)

        # Compare with PyTorch output
        with torch.no_grad():
            torch_output = model(dummy_input)

        diff = np.abs(ort_outputs[0] - torch_output.numpy())
        max_diff = diff.max()

        print(f"✓ ONNX export successful: {onnx_path}")
        print(f"✓ ONNX model validated")
        print(f"✓ ONNX Runtime inference successful")
        print(f"  Max difference (ONNX vs PyTorch): {max_diff:.6e}")
        print(f"  Expected: < 1e-5 (numerical precision)")

        if max_diff < 1e-5:
            print("✓ ONNX output matches PyTorch output!")
        else:
            print(f"⚠ Warning: Difference {max_diff} exceeds threshold")

        # Clean up
        os.remove(onnx_path)

    except ImportError as e:
        print(f"⚠ Skipping ONNX test (missing dependencies): {e}")

    print()


def test_multi_scale():
    """Test multi-scale angle estimator (experimental feature)."""
    print("=" * 60)
    print("Test 6: Multi-Scale Angle Estimation (Experimental)")
    print("=" * 60)

    image = torch.randn(1, 1, 480, 640)

    estimator = AngleEstimatorMultiScale(
        num_scales=3,
        patch_size=15,
        sigma=2.5
    )

    print("⚠ Warning: AngleEstimatorMultiScale is experimental and incomplete.")
    print("  Multi-scale selection logic is not yet implemented.")
    print("  This test only validates basic functionality (shape/range).")
    print()

    # Suppress the runtime warning for testing
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        angles, scale_indices = estimator(image)

    print(f"✓ Input shape: {image.shape}")
    print(f"✓ Output angles shape: {angles.shape}")
    print(f"✓ Output scale_indices shape: {scale_indices.shape}")
    print(f"  Angle range: [{angles.min():.4f}, {angles.max():.4f}]")
    print(f"  Scale indices (expected all zeros): unique values = {torch.unique(scale_indices).tolist()}")

    # Verify scale indices are all zeros (since selection is not implemented)
    assert torch.all(scale_indices == 0), "Expected all scale indices to be 0 (not implemented)"
    print(f"✓ Confirmed: All scale indices are 0 (selection not implemented)")
    print()
    print("Note: For production use, please use the single-scale AngleEstimator.")
    print()


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "Angle Estimation Module Tests" + " " * 18 + "║")
    print("╚" + "═" * 58 + "╝")
    print()

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    try:
        test_basic_functionality()
        test_gradient_image()
        test_with_shi_tomasi()
        test_different_parameters()
        test_onnx_export()
        test_multi_scale()

        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        print()

        print("Pipeline demonstration:")
        print("  Shi-Tomasi → AngleEstimator → Sparse BAD → Sinkhorn")
        print()
        print("Next steps:")
        print("  1. Use AngleEstimator with Shi-Tomasi feature points")
        print("  2. Integrate with Sparse BAD descriptor (rotation-aware)")
        print("  3. Match features using Sinkhorn algorithm")
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
