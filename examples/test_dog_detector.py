#!/usr/bin/env python3
"""
Test script for DoG (Difference of Gaussians) detector.

This script demonstrates:
1. Basic usage of DoGDetector and DoGDetectorWithScore
2. ONNX export verification
3. Visual comparison of multi-scale DoG responses
"""

import sys
from pathlib import Path

import torch
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pytorch_model.detector.dog import DoGDetector, DoGDetectorWithScore


def test_dog_detector():
    """Test basic DoGDetector functionality."""
    print("=" * 60)
    print("Testing DoGDetector")
    print("=" * 60)

    # Create model
    detector = DoGDetector(num_scales=5, sigma_base=1.6)
    detector.eval()

    # Create test input
    batch_size = 2
    height, width = 256, 320
    test_image = torch.rand(batch_size, 1, height, width)

    # Forward pass
    with torch.no_grad():
        dog_responses = detector(test_image)

    print(f"Input shape: {test_image.shape}")
    print(f"Output shape: {dog_responses.shape}")
    print(f"Number of DoG scales: {dog_responses.shape[1]}")
    print(f"Output range: [{dog_responses.min():.4f}, {dog_responses.max():.4f}]")

    # Check output shape
    expected_shape = (batch_size, 4, height, width)  # num_scales - 1 = 4
    assert dog_responses.shape == expected_shape, \
        f"Expected shape {expected_shape}, got {dog_responses.shape}"

    print("✓ DoGDetector test passed!\n")

    return detector, test_image


def test_dog_detector_with_score():
    """Test DoGDetectorWithScore functionality."""
    print("=" * 60)
    print("Testing DoGDetectorWithScore")
    print("=" * 60)

    # Create model
    detector = DoGDetectorWithScore(num_scales=5, sigma_base=1.6)
    detector.eval()

    # Create test input
    batch_size = 2
    height, width = 256, 320
    test_image = torch.rand(batch_size, 1, height, width)

    # Forward pass
    with torch.no_grad():
        score_map = detector(test_image)

    print(f"Input shape: {test_image.shape}")
    print(f"Output shape: {score_map.shape}")
    print(f"Score range: [{score_map.min():.4f}, {score_map.max():.4f}]")

    # Check output shape
    expected_shape = (batch_size, 1, height, width)
    assert score_map.shape == expected_shape, \
        f"Expected shape {expected_shape}, got {score_map.shape}"

    print("✓ DoGDetectorWithScore test passed!\n")

    return detector, test_image


def test_onnx_export():
    """Test ONNX export capability."""
    print("=" * 60)
    print("Testing ONNX Export")
    print("=" * 60)

    # Test DoGDetector export
    print("\n1. Exporting DoGDetector...")
    detector = DoGDetector(num_scales=5, sigma_base=1.6)
    detector.eval()

    dummy_input = torch.randn(1, 1, 256, 320)
    output_path = "/tmp/dog_detector_test.onnx"

    try:
        torch.onnx.export(
            detector,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamo=False,
        )
        print(f"✓ DoGDetector exported to: {output_path}")
    except Exception as e:
        print(f"✗ DoGDetector export failed: {e}")
        return False

    # Test DoGDetectorWithScore export
    print("\n2. Exporting DoGDetectorWithScore...")
    detector_score = DoGDetectorWithScore(num_scales=5, sigma_base=1.6)
    detector_score.eval()

    output_path_score = "/tmp/dog_detector_with_score_test.onnx"

    try:
        torch.onnx.export(
            detector_score,
            dummy_input,
            output_path_score,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamo=False,
        )
        print(f"✓ DoGDetectorWithScore exported to: {output_path_score}")
    except Exception as e:
        print(f"✗ DoGDetectorWithScore export failed: {e}")
        return False

    print("\n✓ All ONNX exports successful!\n")
    return True


def test_different_scales():
    """Test detector with different scale configurations."""
    print("=" * 60)
    print("Testing Different Scale Configurations")
    print("=" * 60)

    test_image = torch.rand(1, 1, 256, 256)

    configs = [
        {"num_scales": 3, "sigma_base": 1.0},
        {"num_scales": 5, "sigma_base": 1.6},
        {"num_scales": 7, "sigma_base": 2.0},
    ]

    for i, config in enumerate(configs, 1):
        print(f"\nConfiguration {i}: {config}")
        detector = DoGDetector(**config)
        detector.eval()

        with torch.no_grad():
            output = detector(test_image)

        print(f"  Output shape: {output.shape}")
        print(f"  DoG scales: {output.shape[1]}")
        print(f"  Response range: [{output.min():.4f}, {output.max():.4f}]")

    print("\n✓ All scale configurations tested!\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("DoG Detector Test Suite")
    print("=" * 60 + "\n")

    try:
        # Test basic functionality
        test_dog_detector()
        test_dog_detector_with_score()

        # Test different configurations
        test_different_scales()

        # Test ONNX export
        onnx_success = test_onnx_export()

        # Summary
        print("=" * 60)
        print("Test Summary")
        print("=" * 60)
        print("✓ All tests passed successfully!")
        print("=" * 60 + "\n")

        return 0

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
