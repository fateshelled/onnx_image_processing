#!/usr/bin/env python3
"""
Unit tests for FAST corner detection.

To run:
    pytest tests/test_fast.py -v
"""

import sys
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from pytorch_model.detector.fast import FASTScore


class TestFASTScore:
    """Test suite for FASTScore module."""

    def test_initialization(self):
        """Test model initialization with different parameters."""
        # Default parameters
        model1 = FASTScore()
        assert model1.threshold == 20
        assert model1.use_nms is False

        # Custom parameters
        model2 = FASTScore(threshold=30, use_nms=True, nms_radius=5)
        assert model2.threshold == 30
        assert model2.use_nms is True
        assert model2.nms_radius == 5

    def test_circle_offsets(self):
        """Test that circle offsets form a valid Bresenham circle."""
        model = FASTScore()
        offsets = model.circle_offsets.cpu().numpy()

        # Should have 16 points
        assert offsets.shape == (16, 2)

        # Check that all points are on a radius-3 circle (approximately)
        for dy, dx in offsets:
            # Bresenham circle: max(|dx|, |dy|) == 3
            assert max(abs(dy), abs(dx)) == 3

    def test_forward_shape(self):
        """Test that forward pass produces correct output shape."""
        model = FASTScore(threshold=20)
        model.eval()

        # Test different input sizes
        for H, W in [(64, 64), (128, 256), (480, 640)]:
            x = torch.randn(1, 1, H, W) * 255
            with torch.no_grad():
                output = model(x)

            assert output.shape == (1, 1, H, W)
            assert output.dtype == torch.float32

    def test_output_range(self):
        """Test that output values are in valid range."""
        model = FASTScore(threshold=20)
        model.eval()

        x = torch.randn(2, 1, 64, 64) * 255
        with torch.no_grad():
            output = model(x)

        # Output should be binary (0 or 1)
        assert torch.all((output == 0.0) | (output == 1.0))

    def test_corner_detection(self):
        """Test that FAST detects corners in a synthetic image."""
        model = FASTScore(threshold=20, use_nms=False)
        model.eval()

        # Create a synthetic corner pattern
        # A bright square on dark background should have corners
        H, W = 128, 128
        image = torch.zeros(1, 1, H, W)
        # Draw a bright square
        image[:, :, 40:80, 40:80] = 255.0

        with torch.no_grad():
            output = model(image)

        # Should detect some corners
        num_corners = (output > 0.5).sum().item()
        assert num_corners > 0, "Should detect corners in synthetic image"

    def test_nms(self):
        """Test that NMS reduces the number of detected corners."""
        # Create image with many close corners
        H, W = 128, 128
        image = torch.randn(1, 1, H, W) * 50 + 128

        # Without NMS
        model_no_nms = FASTScore(threshold=10, use_nms=False)
        model_no_nms.eval()
        with torch.no_grad():
            output_no_nms = model_no_nms(image)
        count_no_nms = (output_no_nms > 0.5).sum().item()

        # With NMS
        model_with_nms = FASTScore(threshold=10, use_nms=True, nms_radius=3)
        model_with_nms.eval()
        with torch.no_grad():
            output_with_nms = model_with_nms(image)
        count_with_nms = (output_with_nms > 0.5).sum().item()

        # NMS should reduce or maintain the number of corners
        assert count_with_nms <= count_no_nms

    def test_batch_processing(self):
        """Test that batch processing works correctly."""
        model = FASTScore(threshold=20)
        model.eval()

        # Batch of 4 images
        batch_size = 4
        x = torch.randn(batch_size, 1, 64, 64) * 255

        with torch.no_grad():
            output = model(x)

        assert output.shape == (batch_size, 1, 64, 64)

    def test_binary_encoding(self):
        """Test binary encoding function."""
        model = FASTScore(threshold=20)
        model.eval()

        # Create simple test case
        center = torch.tensor([[[[100.0]]]])  # (1, 1, 1, 1)
        # Create circle pixels: 16 values
        circle = torch.tensor([
            [[[150.0]]], [[[70.0]]], [[[90.0]]], [[[130.0]]],
            [[[80.0]]], [[[120.0]]], [[[85.0]]], [[[140.0]]],
            [[[75.0]]], [[[155.0]]], [[[95.0]]], [[[125.0]]],
            [[[65.0]]], [[[135.0]]], [[[60.0]]], [[[145.0]]],
        ])  # 16 x (1, 1, 1, 1)
        circle = torch.cat(circle, dim=-1).squeeze(0).squeeze(0)  # (1, 1, 16)

        center_val = center.squeeze()  # scalar
        dark_bits, bright_bits = model._binary_encoding(center_val, circle)

        # Check that bits are integers
        assert dark_bits.dtype == torch.int32
        assert bright_bits.dtype == torch.int32

    def test_consecutive_detection(self):
        """Test 9-consecutive bit detection."""
        model = FASTScore(threshold=20)
        model.eval()

        # Test case 1: All bits set -> should detect
        bits_all = torch.tensor([0xFFFF], dtype=torch.int32)  # 16 bits all 1
        result = model._detect_9_consecutive(bits_all)
        assert result.item() is True

        # Test case 2: No bits set -> should not detect
        bits_none = torch.tensor([0x0000], dtype=torch.int32)
        result = model._detect_9_consecutive(bits_none)
        assert result.item() is False

        # Test case 3: Exactly 9 consecutive bits starting at position 0
        bits_9 = torch.tensor([0b111111111], dtype=torch.int32)  # 9 bits set
        result = model._detect_9_consecutive(bits_9)
        assert result.item() is True

        # Test case 4: Only 8 consecutive bits -> should not detect
        bits_8 = torch.tensor([0b11111111], dtype=torch.int32)  # 8 bits set
        result = model._detect_9_consecutive(bits_8)
        assert result.item() is False

    def test_deterministic(self):
        """Test that results are deterministic."""
        model = FASTScore(threshold=20)
        model.eval()

        x = torch.randn(1, 1, 64, 64) * 255

        with torch.no_grad():
            output1 = model(x)
            output2 = model(x)

        assert torch.all(output1 == output2)


if __name__ == "__main__":
    # Run a quick sanity check
    print("Running quick sanity checks...")

    test = TestFASTScore()

    print("✓ Testing initialization...")
    test.test_initialization()

    print("✓ Testing circle offsets...")
    test.test_circle_offsets()

    print("✓ Testing forward shape...")
    test.test_forward_shape()

    print("✓ Testing output range...")
    test.test_output_range()

    print("✓ Testing corner detection...")
    test.test_corner_detection()

    print("✓ Testing batch processing...")
    test.test_batch_processing()

    print("\n✅ All sanity checks passed!")
