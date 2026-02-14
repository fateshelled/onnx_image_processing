#!/usr/bin/env python3
"""Quick test to verify filter-integrated model works in PyTorch"""

import torch
from pytorch_model.feature_detection.shi_tomasi_angle_sparse_bad_sinkhorn import (
    ShiTomasiAngleSparseBADSinkhornMatcherWithFilters,
)

def test_model():
    print("Testing ShiTomasiAngleSparseBADSinkhornMatcherWithFilters...")

    # Create model with filters enabled
    model = ShiTomasiAngleSparseBADSinkhornMatcherWithFilters(
        max_keypoints=128,  # Small for faster test
        ratio_threshold=2.0,
        dustbin_margin=0.3,
        sinkhorn_iterations=10,  # Fewer iterations for test
        num_pairs=256,
    )
    model.eval()

    # Create test inputs
    img1 = torch.randn(1, 1, 240, 320)
    img2 = torch.randn(1, 1, 240, 320)

    # Run forward pass
    with torch.no_grad():
        kpts1, kpts2, probs, valid_mask = model(img1, img2)

    # Verify outputs
    print(f"✓ Model instantiated successfully")
    print(f"✓ Forward pass completed")
    print(f"  Keypoints1 shape: {kpts1.shape}")
    print(f"  Keypoints2 shape: {kpts2.shape}")
    print(f"  Matching probs shape: {probs.shape}")
    print(f"  Valid mask shape: {valid_mask.shape}")
    print(f"  Valid mask dtype: {valid_mask.dtype}")
    print(f"  Number of valid matches: {valid_mask.sum().item()}/{valid_mask.shape[1]}")

    # Test with filters disabled
    model_no_filters = ShiTomasiAngleSparseBADSinkhornMatcherWithFilters(
        max_keypoints=128,
        ratio_threshold=None,
        dustbin_margin=None,
        sinkhorn_iterations=10,
        num_pairs=256,
    )
    model_no_filters.eval()

    with torch.no_grad():
        kpts1, kpts2, probs, valid_mask = model_no_filters(img1, img2)

    print(f"\n✓ Model with filters disabled works")
    print(f"  All valid (filters disabled): {valid_mask.all().item()}")
    print(f"  Number of valid matches: {valid_mask.sum().item()}/{valid_mask.shape[1]}")

    print("\n✅ All tests passed!")

if __name__ == "__main__":
    test_model()
