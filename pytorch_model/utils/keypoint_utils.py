"""
Utility functions for keypoint detection and selection.

This module provides common utilities used across feature detection models,
including non-maximum suppression and top-k keypoint selection.
"""

import torch
import torch.nn.functional as F


def apply_nms_maxpool(
    scores: torch.Tensor,
    nms_radius: int
) -> torch.Tensor:
    """
    Apply non-maximum suppression using max pooling.

    Args:
        scores: Feature score map of shape (B, H, W).
        nms_radius: Radius for NMS window (kernel_size = 2 * radius + 1).

    Returns:
        NMS mask of shape (B, H, W) where 1.0 indicates local maximum.
    """
    kernel_size = 2 * nms_radius + 1
    padding = nms_radius

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


def select_topk_keypoints(
    scores: torch.Tensor,
    nms_mask: torch.Tensor,
    max_keypoints: int,
    score_threshold: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Select top-k keypoints from score map after NMS.

    Args:
        scores: Feature score map of shape (B, H, W).
        nms_mask: NMS mask of shape (B, H, W).
        max_keypoints: Maximum number of keypoints to select.
        score_threshold: Minimum score threshold for keypoint selection.

    Returns:
        Tuple of:
            - keypoints: Keypoint coordinates of shape (B, K, 2) in (y, x)
              format, padded with (-1, -1) for invalid entries.
            - keypoint_scores: Scores for each keypoint of shape (B, K).
    """
    B, H, W = scores.shape
    K = max_keypoints

    scores_masked = scores * nms_mask
    scores_masked = torch.where(
        scores_masked > score_threshold,
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
