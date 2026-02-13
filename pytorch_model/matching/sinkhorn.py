"""
Hardware-Friendly Sinkhorn Algorithm for Feature Point Matching.

This module implements the Sinkhorn algorithm in log-space for matching
two sets of feature points. It is designed for ONNX export and efficient
execution on NPU/GPU hardware.

Key features:
- Fixed iterations (no dynamic branching for ONNX compatibility)
- Log-space computation (numerical stability)
- Dustbin mechanism (handles unmatched points/outliers)
- Supports L1 and L2 distance metrics

References:
- Sinkhorn, R. "A relationship between arbitrary positive matrices and
  doubly stochastic matrices." Ann. Math. Statist. 35.2 (1964): 876-879.
- Cuturi, M. "Sinkhorn distances: Lightspeed computation of optimal
  transport." NeurIPS 2013.
- Sarlin et al. "SuperGlue: Learning Feature Matching with Graph Neural
  Networks." CVPR 2020.
"""

import torch
from torch import nn
import torch.nn.functional as F


class SinkhornMatcher(nn.Module):
    """
    Sinkhorn-based feature matcher with dustbin support.

    Computes soft assignment (matching probability matrix) between two sets
    of feature descriptors using the Sinkhorn algorithm in log-space.
    The algorithm is designed for ONNX export with fixed iterations.

    The matching probability matrix P satisfies:
    - P[i, j] represents the probability that point i in set 1 matches point j in set 2
    - P[i, M] represents the probability that point i in set 1 is unmatched (dustbin)
    - P[N, j] represents the probability that point j in set 2 is unmatched (dustbin)

    Args:
        iterations: Number of Sinkhorn iterations. Higher values give more
                   accurate solutions but increase computation. Default: 20.
        epsilon: Entropy regularization parameter. Smaller values give sharper
                assignments but may cause numerical issues. Default: 1.0.
        unused_score: Score assigned to dustbin entries. Controls the threshold
                     for considering a point as unmatched. Default: 1.0.
        distance_type: Distance metric for cost matrix ('l1' or 'l2'). Default: 'l2'.

    Example:
        >>> matcher = SinkhornMatcher(iterations=20, epsilon=1.0)
        >>> desc1 = torch.randn(1, 100, 256)  # 100 points, 256-dim features
        >>> desc2 = torch.randn(1, 80, 256)   # 80 points, 256-dim features
        >>> P = matcher(desc1, desc2)         # [1, 101, 81] probability matrix
    """

    def __init__(
        self,
        iterations: int = 20,
        epsilon: float = 1.0,
        unused_score: float = 1.0,
        distance_type: str = "l2",
    ) -> None:
        super().__init__()

        if iterations <= 0:
            raise ValueError(f"iterations must be positive, got {iterations}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

        self.iterations = iterations
        self.epsilon = epsilon
        self.unused_score = unused_score
        self.distance_type = distance_type.lower()

        if self.distance_type not in ("l1", "l2"):
            raise ValueError(f"distance_type must be 'l1' or 'l2', got {distance_type}")

    def _compute_cost_matrix(
        self,
        desc1: torch.Tensor,
        desc2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute pairwise distance (cost) matrix between two descriptor sets.

        Args:
            desc1: First descriptor set [B, N, D]
            desc2: Second descriptor set [B, M, D]

        Returns:
            Cost matrix [B, N, M] where C[b, i, j] is distance between
            desc1[b, i] and desc2[b, j]
        """
        if self.distance_type == "l2":
            # Squared L2 distance: ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b
            # Using squared distance (no sqrt) for numerical stability
            norm1 = (desc1 ** 2).sum(dim=-1, keepdim=True)  # [B, N, 1]
            norm2 = (desc2 ** 2).sum(dim=-1, keepdim=True)  # [B, M, 1]
            # [B, N, M] = [B, N, 1] + [B, 1, M] - 2 * [B, N, D] @ [B, D, M]
            cost = norm1 + norm2.transpose(-2, -1) - 2.0 * torch.bmm(desc1, desc2.transpose(-2, -1))
            # Clamp to handle numerical errors
            cost = torch.clamp(cost, min=0.0)
        else:
            # L1 (Manhattan) distance
            # Expand dimensions for broadcasting: [B, N, 1, D] - [B, 1, M, D]
            diff = desc1.unsqueeze(2) - desc2.unsqueeze(1)  # [B, N, M, D]
            cost = torch.abs(diff).sum(dim=-1)  # [B, N, M]

        return cost

    def _log_sinkhorn_iterations(
        self,
        log_scores: torch.Tensor,
        log_mu: torch.Tensor,
        log_nu: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform Sinkhorn normalization in log-space with fixed iterations.

        This implements the log-space Sinkhorn algorithm which is numerically
        stable and ONNX-compatible (no dynamic loops).

        Args:
            log_scores: Log score matrix [B, N+1, M+1]
            log_mu: Log of row marginals [B, N+1]
            log_nu: Log of column marginals [B, M+1]

        Returns:
            Log assignment matrix [B, N+1, M+1]
        """
        # Initialize dual variables in log-space
        # u: row dual variables, v: column dual variables
        u = torch.zeros_like(log_mu)  # [B, N+1]
        v = torch.zeros_like(log_nu)  # [B, M+1]

        # Fixed iterations - unrolled loop for ONNX static graph
        for _ in range(self.iterations):
            # Row normalization: u = log_mu - logsumexp(log_scores + v, dim=-1)
            u = log_mu - torch.logsumexp(log_scores + v.unsqueeze(-2), dim=-1)
            # Column normalization: v = log_nu - logsumexp(log_scores + u, dim=-2)
            v = log_nu - torch.logsumexp(log_scores + u.unsqueeze(-1), dim=-2)

        # Compute final log assignment: log_P = log_scores + u + v
        log_P = log_scores + u.unsqueeze(-1) + v.unsqueeze(-2)

        return log_P

    def forward(
        self,
        desc1: torch.Tensor,
        desc2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute matching probability matrix between two descriptor sets.

        Args:
            desc1: First descriptor set of shape [B, N, D] where
                   B = batch size, N = number of points, D = descriptor dimension
            desc2: Second descriptor set of shape [B, M, D] where
                   M = number of points in second set

        Returns:
            Matching probability matrix P of shape [B, N+1, M+1] where:
            - P[b, i, j] (i < N, j < M): probability that point i matches point j
            - P[b, i, M] (i < N): probability that point i in set 1 is unmatched
            - P[b, N, j] (j < M): probability that point j in set 2 is unmatched
            - P[b, N, M]: probability mass for mutual dustbin (typically small)
        """
        B, N, D = desc1.shape
        M = desc2.shape[1]

        # Step 1: Compute cost matrix [B, N, M]
        cost = self._compute_cost_matrix(desc1, desc2)

        # Step 2: Convert cost to log scores
        # Higher cost = lower score, normalized by epsilon (temperature)
        log_scores_core = -cost / self.epsilon  # [B, N, M]

        # Step 3: Add dustbin row and column
        # Dustbin score: -unused_score / epsilon (converted to log space)
        dustbin_score = -self.unused_score / self.epsilon

        # Create augmented score matrix [B, N+1, M+1]
        # Pad core scores with dustbin_score on the right column and bottom row.
        # F.pad avoids slice assignment which produces ScatterND in ONNX.
        log_scores = F.pad(log_scores_core, (0, 1, 0, 1), value=dustbin_score)

        # Step 4: Define marginal distributions (row and column sums)
        # Each original point should have total probability 1
        # Dustbins get probability equal to the number of points they can absorb
        # Using uniform distribution: each row/col sums to 1, dustbin row/col sum to M/N

        # Row marginals: [1, 1, ..., 1, M] (N real points + dustbin absorbs M)
        # Column marginals: [1, 1, ..., 1, N] (M real points + dustbin absorbs N)
        # torch.cat avoids index assignment which produces ScatterND in ONNX.
        log_M = torch.log(torch.tensor(float(M), device=desc1.device, dtype=desc1.dtype))
        log_N = torch.log(torch.tensor(float(N), device=desc2.device, dtype=desc2.dtype))
        log_mu = torch.cat([desc1.new_zeros(B, N), log_M.reshape(1, 1).expand(B, -1)], dim=1)
        log_nu = torch.cat([desc2.new_zeros(B, M), log_N.reshape(1, 1).expand(B, -1)], dim=1)

        # Step 5: Sinkhorn iterations in log-space
        log_P = self._log_sinkhorn_iterations(log_scores, log_mu, log_nu)

        # Step 6: Convert from log space to probability
        P = torch.exp(log_P)

        return P


class SinkhornMatcherWithScores(SinkhornMatcher):
    """
    Extended Sinkhorn matcher that also returns match scores.

    This variant returns both the probability matrix and convenience outputs
    for extracting matches with confidence scores.

    Args:
        Same as SinkhornMatcher

    Returns:
        Tuple of:
        - P: Matching probability matrix [B, N+1, M+1]
        - scores0: Match confidence for set 1 points [B, N] (max prob excluding dustbin)
        - scores1: Match confidence for set 2 points [B, M] (max prob excluding dustbin)
    """

    def forward(
        self,
        desc1: torch.Tensor,
        desc2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute matching probabilities and confidence scores.

        Args:
            desc1: First descriptor set [B, N, D]
            desc2: Second descriptor set [B, M, D]

        Returns:
            Tuple of (P, scores0, scores1):
            - P: Matching probability matrix [B, N+1, M+1]
            - scores0: Confidence that each point in set 1 has a match [B, N]
            - scores1: Confidence that each point in set 2 has a match [B, M]
        """
        # Get base matching probability matrix
        P = super().forward(desc1, desc2)

        N = desc1.shape[1]
        M = desc2.shape[1]

        # Extract match scores (max probability excluding dustbin)
        # For each point in set 1: max probability of matching any point in set 2
        scores0 = P[:, :N, :M].max(dim=-1).values  # [B, N]

        # For each point in set 2: max probability of being matched by any point in set 1
        scores1 = P[:, :N, :M].max(dim=-2).values  # [B, M]

        return P, scores0, scores1


class SinkhornMatcherWithFilters(SinkhornMatcher):
    """
    Sinkhorn matcher with integrated outlier filtering.

    Extends the base SinkhornMatcher with two optional outlier removal filters:
    1. Probability ratio filter: Rejects ambiguous matches where the best match
       probability is not significantly higher than the second-best alternative.
    2. Dustbin margin filter: Rejects matches where the dustbin (unmatched)
       probability is too high relative to the best match probability.

    These filters are applied directly in the forward pass and can be exported to ONNX.

    Args:
        iterations: Number of Sinkhorn iterations. Default: 20.
        epsilon: Entropy regularization parameter. Default: 1.0.
        unused_score: Score assigned to dustbin entries. Default: 1.0.
        distance_type: Distance metric ('l1' or 'l2'). Default: 'l2'.
        ratio_threshold: Minimum ratio between best and second-best probabilities.
                        If None or <= 0, ratio filtering is disabled. Default: None.
        dustbin_margin: Minimum margin between best match and dustbin probabilities.
                       If None or < 0, dustbin filtering is disabled. Default: None.

    Returns:
        Tuple of:
        - P: Matching probability matrix [B, N+1, M+1]
        - valid_mask: Boolean mask [B, N] indicating which matches pass filters
                     (True = valid match, False = filtered outlier)

    Example:
        >>> matcher = SinkhornMatcherWithFilters(
        ...     iterations=20,
        ...     epsilon=1.0,
        ...     ratio_threshold=2.0,
        ...     dustbin_margin=0.3
        ... )
        >>> desc1 = torch.randn(1, 100, 256)
        >>> desc2 = torch.randn(1, 80, 256)
        >>> P, valid_mask = matcher(desc1, desc2)
        >>> # valid_mask[0] is shape [100], indicating which of 100 points passed filters
    """

    def __init__(
        self,
        iterations: int = 20,
        epsilon: float = 1.0,
        unused_score: float = 1.0,
        distance_type: str = "l2",
        ratio_threshold: float = None,
        dustbin_margin: float = None,
    ) -> None:
        super().__init__(iterations, epsilon, unused_score, distance_type)

        self.ratio_threshold = ratio_threshold if ratio_threshold is not None else -1.0
        self.dustbin_margin = dustbin_margin if dustbin_margin is not None else -1.0

    def _probability_ratio_filter(
        self,
        P_core: torch.Tensor,
        threshold: float,
    ) -> torch.Tensor:
        """
        Filter matches based on probability ratio between best and second-best.

        Args:
            P_core: Core probability matrix [B, N, M] excluding dustbin
            threshold: Minimum ratio value (e.g., 2.0)

        Returns:
            valid_mask: Boolean tensor [B, N] where True means the match passes
        """
        B, N, M = P_core.shape

        # Get top-2 probabilities along the M dimension for each point in N
        # topk returns (values, indices), we only need values
        # Shape: [B, N, 2] where [:, :, 0] is best, [:, :, 1] is second-best
        if M >= 2:
            top2_probs = torch.topk(P_core, k=2, dim=2, largest=True, sorted=True).values
            best_prob = top2_probs[:, :, 0]      # [B, N]
            second_prob = top2_probs[:, :, 1]    # [B, N]
        else:
            # Edge case: only 1 point in set 2
            best_prob = P_core[:, :, 0]          # [B, N]
            second_prob = torch.zeros_like(best_prob)

        # Calculate ratio with epsilon to avoid division by zero
        ratio = best_prob / (second_prob + 1e-8)

        # Accept points where ratio >= threshold
        # For ONNX compatibility, we use >= comparison (returns boolean tensor)
        valid_mask = ratio >= threshold

        return valid_mask

    def _dustbin_margin_filter(
        self,
        P: torch.Tensor,
        margin: float,
    ) -> torch.Tensor:
        """
        Filter matches based on margin between best match and dustbin probabilities.

        Args:
            P: Full probability matrix [B, N+1, M+1] including dustbin
            margin: Minimum margin value (e.g., 0.3)

        Returns:
            valid_mask: Boolean tensor [B, N] where True means the match passes
        """
        B = P.shape[0]
        N = P.shape[1] - 1  # Exclude dustbin row
        M = P.shape[2] - 1  # Exclude dustbin column

        # Extract core matches (excluding dustbin)
        P_core = P[:, :N, :M]  # [B, N, M]

        # Extract dustbin probabilities for each point in set 1
        dustbin_probs = P[:, :N, M]  # [B, N] - last column (dustbin column)

        # Get best match probability for each point (excluding dustbin)
        best_match_probs = P_core.max(dim=2).values  # [B, N]

        # Calculate margin: best_match - dustbin
        margins = best_match_probs - dustbin_probs  # [B, N]

        # Accept points where margin >= threshold
        valid_mask = margins >= margin

        return valid_mask

    def forward(
        self,
        desc1: torch.Tensor,
        desc2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute matching probability matrix with outlier filtering.

        Args:
            desc1: First descriptor set [B, N, D]
            desc2: Second descriptor set [B, M, D]

        Returns:
            Tuple of:
            - P: Matching probability matrix [B, N+1, M+1]
            - valid_mask: Boolean mask [B, N] indicating valid matches
        """
        # Get base matching probability matrix
        P = super().forward(desc1, desc2)

        B = P.shape[0]
        N = desc1.shape[1]
        M = desc2.shape[1]

        # Initialize valid mask as all True
        # Use torch.ones with dtype=torch.bool for ONNX compatibility
        valid_mask = torch.ones(B, N, dtype=torch.bool, device=desc1.device)

        # Extract core probability matrix (excluding dustbin)
        P_core = P[:, :N, :M]  # [B, N, M]

        # Apply probability ratio filter if enabled
        # Check if ratio_threshold > 0 (disabled if <= 0)
        if self.ratio_threshold > 0:
            ratio_mask = self._probability_ratio_filter(P_core, self.ratio_threshold)
            valid_mask = valid_mask & ratio_mask

        # Apply dustbin margin filter if enabled
        # Check if dustbin_margin >= 0 (disabled if < 0)
        if self.dustbin_margin >= 0:
            dustbin_mask = self._dustbin_margin_filter(P, self.dustbin_margin)
            valid_mask = valid_mask & dustbin_mask

        return P, valid_mask
