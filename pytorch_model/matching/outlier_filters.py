"""
Outlier filtering methods for Sinkhorn matching results.

This module provides various outlier removal techniques that can be applied
to Sinkhorn probability matrices to improve match quality.
"""

import numpy as np


def probability_ratio_filter(
    P: np.ndarray,
    ratio_threshold: float = 2.0,
) -> np.ndarray:
    """
    Filter matches based on the ratio between best and second-best match probabilities.

    A match is considered reliable if its probability is significantly higher than
    the second-best alternative. Low ratios indicate ambiguous matches that are
    likely to be outliers.

    Args:
        P: Matching probability matrix of shape (K, K) excluding dustbin.
           P[i, j] represents the probability of matching point i to point j.
        ratio_threshold: Minimum ratio between best and second-best probabilities.
                        Higher values are more strict. Default is 2.0.

    Returns:
        valid_mask: Boolean array of shape (K,) where True indicates the match
                   for point i passes the ratio test.

    Example:
        >>> P = np.array([[0.8, 0.1, 0.1],
        ...               [0.05, 0.9, 0.05],
        ...               [0.4, 0.35, 0.25]])
        >>> mask = probability_ratio_filter(P, ratio_threshold=2.0)
        >>> # Points 0 and 1 pass (8.0, 18.0), point 2 fails (1.14)
    """
    K = P.shape[0]
    valid_mask = np.zeros(K, dtype=bool)

    for i in range(K):
        # Get top 2 probabilities for point i
        row_probs = P[i, :]
        top2_indices = np.argpartition(row_probs, -2)[-2:]
        top2_probs = row_probs[top2_indices]
        top2_probs_sorted = np.sort(top2_probs)[::-1]  # Descending order

        best_prob = top2_probs_sorted[0]
        second_prob = top2_probs_sorted[1]

        # Calculate ratio (avoid division by zero)
        if second_prob > 1e-8:
            ratio = best_prob / second_prob
        else:
            # If second best is near zero, best match is very strong
            ratio = float('inf')

        # Accept if ratio exceeds threshold
        valid_mask[i] = ratio >= ratio_threshold

    return valid_mask
