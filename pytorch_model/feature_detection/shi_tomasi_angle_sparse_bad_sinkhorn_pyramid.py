"""
Shi-Tomasi + Angle + Sparse BAD + Sinkhorn Feature Matcher (Pyramid variant).

Extends the single-scale Shi-Tomasi matcher with a fixed shallow image
pyramid. Detection is performed independently at each pyramid level with
NMS and a per-level top-k keypoint budget (area-proportional, following
the ORB-SLAM3 convention). Keypoints are mapped back to full-resolution
coordinates, descriptors are computed at their own level, and the sets are
concatenated before a single Sinkhorn matching pass.

All operations are static-shape friendly for ONNX export: fixed number of
levels, fixed per-level keypoint budget, no dynamic branching, and no
ScatterND usage.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from pytorch_model.feature_detection.shi_tomasi_angle_sparse_bad_sinkhorn import (
    ShiTomasiAngleSparseBADSinkhornMatcher,
)
from pytorch_model.utils import apply_nms_maxpool, select_topk_keypoints


def _downsample(image: torch.Tensor) -> torch.Tensor:
    """Half-resolution image via bilinear resize (static graph friendly)."""
    return F.interpolate(image, scale_factor=0.5, mode="bilinear", align_corners=False)


class ShiTomasiAngleSparseBADSinkhornMatcherPyramid(
    ShiTomasiAngleSparseBADSinkhornMatcher
):
    """
    Multi-scale variant of ShiTomasiAngleSparseBADSinkhornMatcher.

    Detects keypoints at ``num_levels`` pyramid levels (level 0 is the
    original resolution, each further level is a 2x downsample). Keypoints
    are selected per level with a budget proportional to the level area:

        level l gets K * weights[l] keypoints

    (default weights are (1/4, 1/4, 1/2) reversed so that level 0 has the
    largest share). Keypoint coordinates are reported in full-resolution
    (y, x) coordinates so downstream VO code stays unchanged.

    Descriptors are computed on the level image the keypoint was detected
    at, which makes the descriptor scale-adaptive: coarse-level keypoints
    are described over a doubled physical patch extent.

    Args:
        Same as :class:`ShiTomasiAngleSparseBADSinkhornMatcher`, plus:

        num_levels: Number of pyramid levels (1 = same as the base class).
                    Must be >= 1. Default is 2.
        level_weights: Optional tuple of per-level keypoint budget weights.
                       Length must equal num_levels, entries positive and
                       must sum to 1.0. Default is None which uses an
                       area-proportional split.

    Example:
        >>> model = ShiTomasiAngleSparseBADSinkhornMatcherPyramid(
        ...     max_keypoints=512, num_levels=2
        ... )
        >>> img1 = torch.randn(1, 1, 480, 640)
        >>> img2 = torch.randn(1, 1, 480, 640)
        >>> kpts1, kpts2, probs = model(img1, img2)
        >>> print(kpts1.shape)  # [1, 512, 2]  (full-resolution coordinates)
        >>> print(probs.shape)  # [1, 513, 513]
    """

    def __init__(
        self,
        max_keypoints: int,
        num_levels: int = 2,
        level_weights: list[float] | None = None,
        **kwargs,
    ) -> None:
        if num_levels < 1:
            raise ValueError(f"num_levels must be >= 1, got {num_levels}")
        super().__init__(max_keypoints=max_keypoints, **kwargs)

        if num_levels > 3:
            raise ValueError(
                f"num_levels must be <= 3 (static per-level budget), got {num_levels}"
            )
        if level_weights is None:
            if num_levels == 1:
                weights = (1.0,)
            elif num_levels == 2:
                weights = (0.5, 0.5)
            else:
                weights = (0.5, 0.25, 0.25)
        else:
            weights = tuple(level_weights)
            if len(weights) != num_levels:
                raise ValueError(
                    f"level_weights length {len(weights)} != num_levels {num_levels}"
                )
            if any(w <= 0 for w in weights) or abs(sum(weights) - 1.0) > 1e-6:
                raise ValueError(f"level_weights must be positive and sum to 1, got {weights}")

        self.num_levels = num_levels
        # Per-level keypoint budgets (ceil/floor so the total is exact)
        wsum = sum(weights)
        budgets = [math.floor(self.max_keypoints * w / wsum) for w in weights]
        budgets[0] += self.max_keypoints - sum(budgets)
        self.level_keypoint_budgets = budgets

    def _level_keypoints(
        self,
        image: torch.Tensor,
        level: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Detect + select keypoints at one pyramid level.

        Returns (keypoints_fullres, descriptors, scores_l) where keypoints
        are already mapped back to full-resolution coordinates.

        """
        img_l = image
        for _ in range(level):
            img_l = _downsample(img_l)

        scores_l, angles_l = self.detector(img_l)
        scores_l = scores_l.squeeze(1)
        nms_l = apply_nms_maxpool(scores_l, self.nms_radius)
        kp_l, kp_scores_l = select_topk_keypoints(
            scores_l, nms_l,
            self.level_keypoint_budgets[level],
            self.score_threshold, self.border_margin,
        )

        # Descriptor uses level-local coordinates
        desc_l = self.descriptor(img_l, kp_l, angles_l)

        # Map valid coordinates back to full resolution ((y, x) * 2^level)
        if level > 0:
            invalid = kp_scores_l <= 0
            kp_l = kp_l * (2.0 ** level)
            kp_l = torch.where(invalid.unsqueeze(-1), torch.full_like(kp_l, -1.0), kp_l)

        return kp_l, desc_l, kp_scores_l

    def forward(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Detect multi-scale keypoints and compute matches between two images.

        Args:
            image1: First grayscale image of shape (B, 1, H, W). Expected
                range [0, 255] when ``cas_sharpness`` is enabled.
            image2: Second grayscale image of shape (B, 1, H, W).

        Returns:
            Tuple of:
                - keypoints1: Detected keypoints in first image of shape
                  (B, K, 2) in (y, x) format, full-resolution coordinates.
                  Invalid keypoints are (-1, -1).
                - keypoints2: Same as keypoints1, for the second image.
                - matching_probs: Matching probability matrix of shape
                  (B, K+1, K+1).
        """
        if self.sharpener is not None:
            image1 = self.sharpener(image1)
            image2 = self.sharpener(image2)

        kps1 = []
        ds1 = []
        kps2 = []
        ds2 = []
        for level in range(self.num_levels):
            kp1_l, d1_l, _ = self._level_keypoints(image1, level)
            kp2_l, d2_l, _ = self._level_keypoints(image2, level)
            kps1.append(kp1_l)
            ds1.append(d1_l)
            kps2.append(kp2_l)
            ds2.append(d2_l)

        keypoints1 = torch.cat(kps1, dim=1)
        keypoints2 = torch.cat(kps2, dim=1)
        desc1 = torch.cat(ds1, dim=1)
        desc2 = torch.cat(ds2, dim=1)

        matching_probs = self.matcher(desc1, desc2)
        return keypoints1, keypoints2, matching_probs
