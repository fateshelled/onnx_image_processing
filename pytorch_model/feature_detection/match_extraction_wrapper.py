"""
Match extraction wrapper for feature matching pipelines.

This module provides a generic wrapper that adds match extraction
to any feature matching model, making it ONNX-exportable.
"""

import torch
from torch import nn

from pytorch_model.matching.match_extraction import MutualNearestNeighborMatcher


class MatchExtractionWrapper(nn.Module):
    """
    Generic wrapper that adds match extraction to any feature matcher.

    This wrapper takes any feature matching model that outputs:
    (keypoints1, keypoints2, matching_probs, ...)

    And adds mutual nearest neighbor match extraction, outputting:
    (matched_kpts1, matched_kpts2, scores, valid_mask)

    This makes it easy to add match extraction to any feature matching
    pipeline for ONNX export.

    Args:
        feature_matcher: The base feature matching model. Must output at least:
                        (keypoints1, keypoints2, matching_probs, ...)
                        where matching_probs is [B, N+1, M+1] Sinkhorn matrix.
        max_matches: Maximum number of matches to return. Default: 100.
        match_threshold: Minimum match probability threshold. Default: 0.1.

    Returns:
        Tuple of:
        - matched_kpts1: [B, max_matches, 2] matched keypoints from image 1 (y, x)
        - matched_kpts2: [B, max_matches, 2] matched keypoints from image 2 (y, x)
        - scores: [B, max_matches] match probability scores
        - valid_mask: [B, max_matches] boolean mask indicating valid matches

    Example:
        >>> from pytorch_model.feature_detection.shi_tomasi_angle_sparse_bad_sinkhorn import (
        ...     ShiTomasiAngleSparseBADSinkhornMatcherWithFilters
        ... )
        >>> base_matcher = ShiTomasiAngleSparseBADSinkhornMatcherWithFilters(
        ...     max_keypoints=1024,
        ...     ratio_threshold=10.0,
        ...     dustbin_margin=0.3
        ... )
        >>> model = MatchExtractionWrapper(
        ...     feature_matcher=base_matcher,
        ...     max_matches=100,
        ...     match_threshold=0.1
        ... )
        >>> img1 = torch.randn(1, 1, 480, 640)
        >>> img2 = torch.randn(1, 1, 480, 640)
        >>> mkpts1, mkpts2, scores, valid = model(img1, img2)
        >>> # Extract only valid matches
        >>> num_valid = valid[0].sum().item()
        >>> mkpts1_valid = mkpts1[0, :num_valid]  # [num_valid, 2]
        >>> mkpts2_valid = mkpts2[0, :num_valid]  # [num_valid, 2]
        >>> scores_valid = scores[0, :num_valid]  # [num_valid]
    """

    def __init__(
        self,
        feature_matcher: nn.Module,
        max_matches: int = 100,
        match_threshold: float = 0.1,
    ) -> None:
        super().__init__()

        # Store the base feature matcher
        self.feature_matcher = feature_matcher

        # Match extraction
        self.match_extractor = MutualNearestNeighborMatcher(
            max_matches=max_matches,
            threshold=match_threshold,
        )

    def forward(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform complete feature matching pipeline with match extraction.

        Args:
            image1: First input image [B, 1, H, W]
            image2: Second input image [B, 1, H, W]

        Returns:
            Tuple of:
            - matched_kpts1: [B, max_matches, 2] matched keypoints from image 1
            - matched_kpts2: [B, max_matches, 2] matched keypoints from image 2
            - scores: [B, max_matches] match probability scores
            - valid_mask: [B, max_matches] boolean mask indicating valid matches
        """
        # Step 1: Feature detection and matching
        # Assume feature_matcher returns at least (kp1, kp2, P, ...)
        outputs = self.feature_matcher(image1, image2)
        keypoints1 = outputs[0]
        keypoints2 = outputs[1]
        matching_probs = outputs[2]

        # Step 2: Extract matches
        matched_kpts1, matched_kpts2, scores, valid_mask = self.match_extractor(
            matching_probs, keypoints1, keypoints2
        )

        return matched_kpts1, matched_kpts2, scores, valid_mask
