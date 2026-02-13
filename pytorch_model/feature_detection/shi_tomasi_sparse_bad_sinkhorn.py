"""
Shi-Tomasi + Sparse BAD + Sinkhorn Feature Matcher.

This module provides a feature matching model where BAD descriptors are
computed only at detected keypoint locations (sparse), rather than densely
for all pixels. This reduces computation when the number of keypoints is
much smaller than the total number of pixels.

Pipeline:
    1. Shi-Tomasi corner detection -> score map (full image)
    2. NMS + top-k -> keypoint selection
    3. BAD descriptor computation at keypoints only (sparse)
    4. Sinkhorn matching

Designed for ONNX export as a single integrated model.
"""

import torch
from torch import nn
import torch.nn.functional as F

from pytorch_model.detector.shi_tomasi import ShiTomasiScore
from pytorch_model.descriptor.bad import SparseBAD
from pytorch_model.matching.sinkhorn import SinkhornMatcher


class ShiTomasiSparseBADSinkhornMatcher(nn.Module):
    """
    Feature matching model with sparse BAD descriptor computation.

    Unlike ShiTomasiBADSinkhornMatcher which computes dense BAD descriptors
    for all pixels, this model first detects keypoints using Shi-Tomasi
    corner detection, then computes BAD descriptors only at those keypoint
    locations. This is more efficient when max_keypoints << H * W.

    Args:
        max_keypoints: Maximum number of keypoints to detect per image.
                       Output will be padded to this size.
        block_size: Block size for Shi-Tomasi structure tensor computation.
                    Must be a positive odd integer. Default is 3.
        sobel_size: Sobel kernel size for gradient computation.
                    Currently only supports 3. Default is 3.
        num_pairs: Number of BAD descriptor comparison pairs (descriptor
                   dimensionality). Default is 256.
        binarize: If True, output binarized BAD descriptors. Default is False.
        soft_binarize: If True and binarize=True, use sigmoid for soft
                       binarization. Default is True.
        temperature: Temperature for soft sigmoid binarization. Default is 10.0.
        sinkhorn_iterations: Number of Sinkhorn iterations for matching.
                            Default is 20.
        epsilon: Entropy regularization parameter for Sinkhorn. Default is 1.0.
        unused_score: Score for dustbin entries in Sinkhorn (controls match
                      threshold). Default is 1.0.
        distance_type: Distance metric for Sinkhorn cost matrix. Either 'l1'
                       or 'l2'. Default is 'l2'.
        nms_radius: Radius for non-maximum suppression. Default is 3.
        score_threshold: Minimum corner score threshold for keypoint selection.
                        Default is 0.0 (no filtering).
        normalize_descriptors: If True, L2-normalize descriptors before matching.
                              Default is True.
        sampling_mode: Sampling mode for descriptor grid sampling.
                       Choose 'nearest' for faster approximate sampling or
                       'bilinear' for smoother interpolation. Default is 'nearest'.
        border_margin: Margin from image border (in pixels) to exclude keypoints.
                      If None, uses descriptor's max_radius to ensure valid
                      descriptor computation. Set to 0 to disable border filtering.
                      Default is None (uses max_radius).

    Example:
        >>> model = ShiTomasiSparseBADSinkhornMatcher(max_keypoints=512)
        >>> img1 = torch.randn(1, 1, 480, 640)  # Grayscale image 1
        >>> img2 = torch.randn(1, 1, 480, 640)  # Grayscale image 2
        >>> kpts1, kpts2, probs = model(img1, img2)
        >>> print(kpts1.shape)  # [1, 512, 2]
        >>> print(kpts2.shape)  # [1, 512, 2]
        >>> print(probs.shape)  # [1, 513, 513]
    """

    def __init__(
        self,
        max_keypoints: int,
        block_size: int = 3,
        sobel_size: int = 3,
        num_pairs: int = 256,
        binarize: bool = False,
        soft_binarize: bool = True,
        temperature: float = 10.0,
        sinkhorn_iterations: int = 20,
        epsilon: float = 1.0,
        unused_score: float = 1.0,
        distance_type: str = "l2",
        nms_radius: int = 3,
        score_threshold: float = 0.0,
        normalize_descriptors: bool = True,
        sampling_mode: str = "nearest",
        border_margin: int | None = None,
    ) -> None:
        super().__init__()

        self.max_keypoints = max_keypoints
        self.nms_radius = nms_radius
        self.score_threshold = score_threshold

        # Corner detector only (no dense BAD computation)
        self.corner_detector = ShiTomasiScore(
            block_size=block_size,
            sobel_size=sobel_size,
        )

        # Sparse BAD descriptor computation
        self.descriptor = SparseBAD(
            num_pairs=num_pairs,
            binarize=binarize,
            soft_binarize=soft_binarize,
            temperature=temperature,
            normalize_descriptors=normalize_descriptors,
            sampling_mode=sampling_mode,
        )

        # Set border margin: if None, use descriptor's max_radius for safety
        if border_margin is None:
            self.border_margin = self.descriptor.max_radius
        else:
            self.border_margin = border_margin

        # Feature matcher: Sinkhorn
        self.matcher = SinkhornMatcher(
            iterations=sinkhorn_iterations,
            epsilon=epsilon,
            unused_score=unused_score,
            distance_type=distance_type,
        )

    def _apply_nms_maxpool(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Apply non-maximum suppression using max pooling.

        Args:
            scores: Corner score map of shape (B, H, W).

        Returns:
            NMS mask of shape (B, H, W) where 1.0 indicates local maximum.
        """
        kernel_size = 2 * self.nms_radius + 1
        padding = self.nms_radius

        scores_padded = F.pad(
            scores.unsqueeze(1),
            (padding, padding, padding, padding),
            mode="constant",
            value=float("-inf"),
        )

        local_max = F.max_pool2d(
            scores_padded,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
        ).squeeze(1)

        nms_mask = (scores >= (local_max - 1e-7)).float()
        return nms_mask

    def _select_topk_keypoints(
        self,
        scores: torch.Tensor,
        nms_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Select top-k keypoints from score map after NMS.

        Args:
            scores: Corner score map of shape (B, H, W).
            nms_mask: NMS mask of shape (B, H, W).

        Returns:
            Tuple of:
                - keypoints: Keypoint coordinates of shape (B, K, 2) in (y, x)
                  format, padded with (-1, -1) for invalid entries.
                - keypoint_scores: Scores for each keypoint of shape (B, K).
        """
        B, H, W = scores.shape
        K = self.max_keypoints

        # Create border mask to exclude keypoints near image boundaries.
        # Use comparison + broadcasting instead of slice assignment to avoid
        # ScatterND in ONNX (which causes warnings on CUDA with duplicate indices).
        if self.border_margin > 0:
            m = self.border_margin
            y_idx = torch.arange(H, device=scores.device)
            x_idx = torch.arange(W, device=scores.device)
            y_valid = ((y_idx >= m) & (y_idx < H - m)).float()
            x_valid = ((x_idx >= m) & (x_idx < W - m)).float()
            border_mask = y_valid.view(1, H, 1) * x_valid.view(1, 1, W)
        else:
            border_mask = torch.ones_like(scores)

        # Apply NMS mask, border mask, and score threshold
        scores_masked = scores * nms_mask * border_mask
        scores_masked = torch.where(
            scores_masked > self.score_threshold,
            scores_masked,
            torch.zeros_like(scores_masked),
        )

        scores_flat = scores_masked.reshape(B, -1)

        topk_scores, topk_indices = torch.topk(
            scores_flat,
            k=K,
            dim=1,
            largest=True,
            sorted=True,
        )

        y_coords = (topk_indices // W).float()
        x_coords = (topk_indices % W).float()
        keypoints = torch.stack([y_coords, x_coords], dim=-1)

        valid_mask = (topk_scores > 0).float()
        invalid_keypoints = torch.full_like(keypoints, -1.0)
        keypoints = torch.where(
            valid_mask.unsqueeze(-1) > 0.5,
            keypoints,
            invalid_keypoints,
        )
        topk_scores = topk_scores * valid_mask

        return keypoints, topk_scores

    def forward(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Detect keypoints and compute matches between two images.

        Args:
            image1: First grayscale image of shape (B, 1, H, W).
            image2: Second grayscale image of shape (B, 1, H, W).

        Returns:
            Tuple of:
                - keypoints1: Detected keypoints in first image of shape
                  (B, K, 2) in (y, x) format. Invalid keypoints are (-1, -1).
                - keypoints2: Detected keypoints in second image of shape
                  (B, K, 2) in (y, x) format. Invalid keypoints are (-1, -1).
                - matching_probs: Matching probability matrix of shape
                  (B, K+1, K+1). The last row/column is the dustbin.
        """
        # 1. Compute Shi-Tomasi corner scores only (no dense BAD)
        scores1 = self.corner_detector(image1)  # (B, 1, H, W)
        scores2 = self.corner_detector(image2)
        scores1 = scores1.squeeze(1)  # (B, H, W)
        scores2 = scores2.squeeze(1)

        # 2. Apply NMS
        nms_mask1 = self._apply_nms_maxpool(scores1)
        nms_mask2 = self._apply_nms_maxpool(scores2)

        # 3. Select top-k keypoints
        keypoints1, _ = self._select_topk_keypoints(scores1, nms_mask1)
        keypoints2, _ = self._select_topk_keypoints(scores2, nms_mask2)

        # 4. Compute BAD descriptors at keypoints only (sparse)
        desc1 = self.descriptor(image1, keypoints1)  # (B, K, num_pairs)
        desc2 = self.descriptor(image2, keypoints2)

        # 5. Perform Sinkhorn matching
        matching_probs = self.matcher(desc1, desc2)  # (B, K+1, K+1)

        return keypoints1, keypoints2, matching_probs
