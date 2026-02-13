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


def dustbin_margin_filter(
    P: np.ndarray,
    margin: float = 0.3,
) -> np.ndarray:
    """
    Filter matches based on the margin between best match and dustbin probabilities.

    The Sinkhorn algorithm with dustbin mechanism assigns probability mass to a
    "dustbin" (unmatched) state. This filter rejects matches where the dustbin
    probability is high relative to the best match probability, indicating the
    point is better left unmatched.

    Args:
        P: Full Sinkhorn probability matrix of shape (K+1, K+1) including dustbin.
           P[i, j] for i,j < K are real matches, P[i, K] and P[K, j] are dustbin.
        margin: Minimum margin that best match probability must exceed dustbin
               probability. Higher values are more strict. Default is 0.3.

    Returns:
        valid_mask: Boolean array of shape (K,) where True indicates the point i
                   has a sufficiently strong match compared to its dustbin probability.

    Example:
        >>> # Point 0: best_match=0.7, dustbin=0.2 -> margin=0.5 (PASS with margin=0.3)
        >>> # Point 1: best_match=0.4, dustbin=0.5 -> margin=-0.1 (FAIL)
        >>> P = np.array([[0.7, 0.1, 0.2],
        ...               [0.2, 0.3, 0.5],
        ...               [0.1, 0.6, 0.3]])
        >>> mask = dustbin_margin_filter(P, margin=0.3)
        >>> # Only point 0 passes
    """
    K = P.shape[0] - 1  # Exclude dustbin row/column

    # Extract dustbin probabilities for each point
    # P[i, K] is the probability that point i in image1 goes to dustbin
    dustbin_probs = P[:K, K]  # Shape: (K,)

    # Extract best match probabilities (excluding dustbin)
    # For each point i, find the maximum probability among real matches
    P_core = P[:K, :K]  # Shape: (K, K)
    best_match_probs = np.max(P_core, axis=1)  # Shape: (K,)

    # Calculate margin: best_match - dustbin
    # Positive margin means the point prefers a real match over dustbin
    margins = best_match_probs - dustbin_probs

    # Accept points where margin exceeds the threshold
    valid_mask = margins >= margin

    return valid_mask
