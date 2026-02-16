"""
Extract matches from Sinkhorn probability matrix using mutual nearest neighbor criterion.

This module implements match extraction that can be exported to ONNX.
"""

import torch
from torch import nn


class MutualNearestNeighborMatcher(nn.Module):
    """
    Extract mutual nearest neighbor matches from probability matrix.

    Given a Sinkhorn probability matrix P, extracts matches where:
    - Point i in set 1 has point j as best match in set 2
    - Point j in set 2 has point i as best match in set 1
    - The match probability exceeds the threshold

    The output is fixed-size (max_matches) with padding for ONNX compatibility.

    Args:
        max_matches: Maximum number of matches to return. Default: 100.
        threshold: Minimum match probability to accept. Default: 0.1.

    Example:
        >>> matcher = MutualNearestNeighborMatcher(max_matches=100, threshold=0.1)
        >>> P = torch.randn(1, 101, 101).softmax(dim=-1)  # Probability matrix
        >>> kpts1 = torch.randn(1, 100, 2)  # Keypoints in image 1
        >>> kpts2 = torch.randn(1, 100, 2)  # Keypoints in image 2
        >>> mkpts1, mkpts2, scores, valid = matcher(P, kpts1, kpts2)
        >>> # mkpts1, mkpts2: [B, max_matches, 2]
        >>> # scores: [B, max_matches]
        >>> # valid: [B, max_matches] boolean mask indicating valid matches
    """

    def __init__(
        self,
        max_matches: int = 100,
        threshold: float = 0.1,
    ) -> None:
        super().__init__()
        self.max_matches = max_matches
        self.threshold = threshold

    def forward(
        self,
        P: torch.Tensor,
        keypoints1: torch.Tensor,
        keypoints2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract mutual nearest neighbor matches.

        Args:
            P: Probability matrix [B, N+1, M+1] including dustbin
            keypoints1: Keypoints in image 1 [B, N, 2] as (y, x)
            keypoints2: Keypoints in image 2 [B, M, 2] as (y, x)

        Returns:
            Tuple of:
            - matched_kpts1: [B, max_matches, 2] matched keypoints from image 1
            - matched_kpts2: [B, max_matches, 2] matched keypoints from image 2
            - scores: [B, max_matches] match probability scores
            - valid_mask: [B, max_matches] boolean mask (True = valid match)
        """
        B = P.shape[0]
        N = keypoints1.shape[1]
        M = keypoints2.shape[1]

        # Extract core probability matrix (excluding dustbin)
        P_core = P[:, :N, :M]  # [B, N, M]

        # Find best match for each keypoint
        # For each point in set 1, find best match in set 2
        max_j_for_i = torch.argmax(P_core, dim=2)  # [B, N]
        max_prob_i = torch.max(P_core, dim=2).values  # [B, N]

        # For each point in set 2, find best match in set 1
        max_i_for_j = torch.argmax(P_core, dim=1)  # [B, M]

        # Check mutual consistency: i matches j AND j matches i
        # For each i, check if max_i_for_j[max_j_for_i[i]] == i
        # This is done using gather operation for ONNX compatibility

        # Expand max_j_for_i to gather from max_i_for_j
        # max_j_for_i: [B, N] contains indices j
        # max_i_for_j: [B, M] contains indices i
        # We want: max_i_for_j[b, max_j_for_i[b, i]] for each b, i

        # Create batch indices for gather
        batch_idx = torch.arange(B, device=P.device).unsqueeze(1).expand(-1, N)  # [B, N]

        # Gather: for each (b, i), get max_i_for_j[b, max_j_for_i[b, i]]
        matched_i = torch.gather(
            max_i_for_j,  # [B, M]
            dim=1,
            index=max_j_for_i  # [B, N]
        )  # [B, N]

        # Check if matched_i == i (mutual consistency)
        i_indices = torch.arange(N, device=P.device).unsqueeze(0).expand(B, -1)  # [B, N]
        is_mutual = (matched_i == i_indices)  # [B, N]

        # Apply threshold
        above_threshold = (max_prob_i >= self.threshold)  # [B, N]

        # Valid matches: mutual AND above threshold
        valid_matches = is_mutual & above_threshold  # [B, N]

        # Get match indices and scores
        # For each batch, we need to extract valid matches
        # This is tricky in ONNX because we need dynamic indexing

        # Strategy: Use masked selection with fixed output size
        # 1. Create scores for sorting (invalid matches get -1)
        scores_for_sorting = torch.where(
            valid_matches,
            max_prob_i,
            torch.tensor(-1.0, device=P.device, dtype=P.dtype)
        )  # [B, N]

        # 2. Sort by score descending and take top max_matches
        sorted_scores, sorted_indices = torch.topk(
            scores_for_sorting,
            k=min(self.max_matches, N),
            dim=1,
            largest=True,
            sorted=True
        )  # [B, max_matches or N]

        # If N < max_matches, pad with zeros
        if N < self.max_matches:
            padding_size = self.max_matches - N
            sorted_scores = torch.cat([
                sorted_scores,
                torch.zeros(B, padding_size, device=P.device, dtype=P.dtype)
            ], dim=1)
            sorted_indices = torch.cat([
                sorted_indices,
                torch.zeros(B, padding_size, device=P.device, dtype=torch.long)
            ], dim=1)

        # 3. Extract keypoint coordinates using sorted indices
        # Gather keypoints from image 1
        # sorted_indices: [B, max_matches]
        # keypoints1: [B, N, 2]
        # We need to gather along dim=1

        # Expand sorted_indices to [B, max_matches, 2] for gathering
        sorted_indices_expanded = sorted_indices.unsqueeze(-1).expand(-1, -1, 2)  # [B, max_matches, 2]

        # Clamp indices to avoid out-of-bounds (for padded entries)
        sorted_indices_clamped = torch.clamp(sorted_indices_expanded, 0, N - 1)

        matched_kpts1 = torch.gather(
            keypoints1,  # [B, N, 2]
            dim=1,
            index=sorted_indices_clamped  # [B, max_matches, 2]
        )  # [B, max_matches, 2]

        # Gather corresponding j indices
        # max_j_for_i: [B, N]
        j_indices = torch.gather(
            max_j_for_i,  # [B, N]
            dim=1,
            index=sorted_indices.clamp(0, N - 1)  # [B, max_matches]
        )  # [B, max_matches]

        # Gather keypoints from image 2
        j_indices_expanded = j_indices.unsqueeze(-1).expand(-1, -1, 2)  # [B, max_matches, 2]
        j_indices_clamped = torch.clamp(j_indices_expanded, 0, M - 1)

        matched_kpts2 = torch.gather(
            keypoints2,  # [B, M, 2]
            dim=1,
            index=j_indices_clamped  # [B, max_matches, 2]
        )  # [B, max_matches, 2]

        # 4. Create valid mask (scores > 0 means valid)
        valid_mask = (sorted_scores > 0.0)  # [B, max_matches]

        # Return matched keypoints, scores, and valid mask
        return matched_kpts1, matched_kpts2, sorted_scores, valid_mask
