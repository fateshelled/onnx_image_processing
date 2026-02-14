#!/usr/bin/env python3
"""Test vectorized probability_ratio_filter implementation"""

import numpy as np
from pytorch_model.matching.outlier_filters import probability_ratio_filter

def test_basic_functionality():
    """Test basic functionality with the example from docstring"""
    P = np.array([[0.8, 0.1, 0.1],
                  [0.05, 0.9, 0.05],
                  [0.4, 0.35, 0.25]])

    mask = probability_ratio_filter(P, ratio_threshold=2.0)

    # Calculate expected ratios manually
    # Point 0: 0.8 / 0.1 = 8.0 (PASS)
    # Point 1: 0.9 / 0.05 = 18.0 (PASS)
    # Point 2: 0.4 / 0.35 = 1.14 (FAIL)

    expected = np.array([True, True, False])

    assert np.array_equal(mask, expected), f"Expected {expected}, got {mask}"
    print("✓ Basic functionality test passed")

def test_edge_case_single_point():
    """Test with single point (K=1)"""
    P = np.array([[1.0]])
    mask = probability_ratio_filter(P, ratio_threshold=2.0)

    # With K=1, should accept (no second-best to compare)
    expected = np.array([True])
    assert np.array_equal(mask, expected), f"Expected {expected}, got {mask}"
    print("✓ Single point edge case test passed")

def test_large_scale():
    """Test with larger matrix to verify vectorization works"""
    np.random.seed(42)
    K = 1000
    P = np.random.rand(K, K)

    # Normalize rows to sum to 1 (valid probability distribution)
    P = P / P.sum(axis=1, keepdims=True)

    mask = probability_ratio_filter(P, ratio_threshold=2.0)

    # Verify output shape and type
    assert mask.shape == (K,), f"Expected shape ({K},), got {mask.shape}"
    assert mask.dtype == bool, f"Expected dtype bool, got {mask.dtype}"

    # Verify at least some points pass and some fail
    num_passed = mask.sum()
    assert 0 < num_passed < K, f"Expected some passes and fails, got {num_passed}/{K}"

    print(f"✓ Large scale test passed: {num_passed}/{K} points passed filter")

def test_strict_threshold():
    """Test with very strict threshold"""
    P = np.array([[0.8, 0.1, 0.1],
                  [0.6, 0.4, 0.0]])

    # With threshold=3.0, need 3x difference
    # Point 0: 0.8/0.1 = 8.0 (PASS)
    # Point 1: 0.6/0.4 = 1.5 (FAIL)
    mask = probability_ratio_filter(P, ratio_threshold=3.0)
    expected = np.array([True, False])

    assert np.array_equal(mask, expected), f"Expected {expected}, got {mask}"
    print("✓ Strict threshold test passed")

def test_performance():
    """Compare performance of vectorized vs loop-based (if old implementation available)"""
    import time

    np.random.seed(42)
    K = 5000
    P = np.random.rand(K, K)
    P = P / P.sum(axis=1, keepdims=True)

    start = time.time()
    mask = probability_ratio_filter(P, ratio_threshold=2.0)
    elapsed = time.time() - start

    print(f"✓ Performance test: {K}x{K} matrix processed in {elapsed:.4f}s")
    assert elapsed < 1.0, f"Expected vectorized version to be fast, took {elapsed}s"

if __name__ == "__main__":
    test_basic_functionality()
    test_edge_case_single_point()
    test_large_scale()
    test_strict_threshold()
    test_performance()
    print("\n✅ All tests passed!")
