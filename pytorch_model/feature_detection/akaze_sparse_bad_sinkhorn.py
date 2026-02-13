"""
AKAZE + Sparse BAD + Sinkhorn Feature Matcher.

This module provides a feature matching model that combines AKAZE feature
detection with orientation-aware sparse BAD descriptors and Sinkhorn matching.
AKAZE provides per-pixel orientation information which is used to rotate BAD
pair offsets at each keypoint, making the descriptor rotation-invariant.

Pipeline:
    1. AKAZE feature detection -> score map + orientation map (full image)
    2. NMS + top-k -> keypoint selection
    3. Orientation-aware BAD descriptor computation at keypoints (sparse)
    4. Sinkhorn matching

Designed for ONNX export as a single integrated model.
"""

import torch
from torch import nn
import torch.nn.functional as F

from pytorch_model.feature_detection.akaze import AKAZE
from pytorch_model.descriptor.bad import SparseBAD
from pytorch_model.matching.sinkhorn import SinkhornMatcher


class AKAZESparseBADSinkhornMatcher(nn.Module):
    """
    Feature matching model with AKAZE detection and orientation-aware BAD.

    Uses AKAZE for feature detection (providing both scores and orientations),
    then computes rotation-invariant BAD descriptors at keypoint locations by
    rotating pair offsets according to the local AKAZE orientation.

    Args:
        max_keypoints: Maximum number of keypoints to detect per image.
                       Output will be padded to this size.
        num_scales: Number of AKAZE scale levels. Default is 3.
        diffusion_iterations: Number of FED iterations per scale. Default is 3.
        kappa: Contrast parameter for AKAZE diffusion. Default is 0.05.
        threshold: AKAZE feature detection threshold. Default is 0.001.
        akaze_nms_size: NMS window size inside AKAZE detector (must be odd).
                        Default is 5.
        orientation_patch_size: Patch size for AKAZE orientation estimation
                                (must be odd). Default is 15.
        orientation_sigma: Gaussian sigma for AKAZE orientation weighting.
                           Default is 2.5.
        num_pairs: Number of BAD descriptor comparison pairs (descriptor
                   dimensionality). Must be 256 or 512. Default is 256.
        binarize: If True, output binarized BAD descriptors. Default is False.
        soft_binarize: If True and binarize=True, use sigmoid for soft
                       binarization. Default is True.
        temperature: Temperature for soft sigmoid binarization. Default is 10.0.
        sinkhorn_iterations: Number of Sinkhorn iterations for matching.
                            Default is 20.
        epsilon: Entropy regularization parameter for Sinkhorn. Default is 1.0.
        unused_score: Score for dustbin entries in Sinkhorn. Default is 1.0.
        distance_type: Distance metric for Sinkhorn cost matrix. Either 'l1'
                       or 'l2'. Default is 'l2'.
        nms_radius: Radius for pipeline-level non-maximum suppression on the
                    AKAZE score map. Default is 3.
        score_threshold: Minimum score threshold for keypoint selection.
                        Default is 0.0.
        normalize_descriptors: If True, L2-normalize descriptors before
                              matching. Default is True.
        sampling_mode: Sampling mode for descriptor grid sampling.
                       Choose 'nearest' or 'bilinear'. Default is 'nearest'.

    Example:
        >>> model = AKAZESparseBADSinkhornMatcher(max_keypoints=512)
        >>> img1 = torch.randn(1, 1, 480, 640)
        >>> img2 = torch.randn(1, 1, 480, 640)
        >>> kpts1, kpts2, probs = model(img1, img2)
        >>> print(kpts1.shape)  # [1, 512, 2]
        >>> print(kpts2.shape)  # [1, 512, 2]
        >>> print(probs.shape)  # [1, 513, 513]
    """

    def __init__(
        self,
        max_keypoints: int,
        num_scales: int = 3,
        diffusion_iterations: int = 3,
        kappa: float = 0.05,
        threshold: float = 0.001,
        akaze_nms_size: int = 5,
        orientation_patch_size: int = 15,
        orientation_sigma: float = 2.5,
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
    ) -> None:
        super().__init__()

        self.max_keypoints = max_keypoints
        self.nms_radius = nms_radius
        self.score_threshold = score_threshold

        # AKAZE feature detector (scores + orientations)
        self.detector = AKAZE(
            num_scales=num_scales,
            diffusion_iterations=diffusion_iterations,
            kappa=kappa,
            threshold=threshold,
            nms_size=akaze_nms_size,
            orientation_patch_size=orientation_patch_size,
            orientation_sigma=orientation_sigma,
        )

        # Sparse BAD descriptor computation with orientation support
        self.descriptor = SparseBAD(
            num_pairs=num_pairs,
            binarize=binarize,
            soft_binarize=soft_binarize,
            temperature=temperature,
            normalize_descriptors=normalize_descriptors,
            sampling_mode=sampling_mode,
        )

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
            scores: Feature score map of shape (B, H, W).

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
            scores: Feature score map of shape (B, H, W).
            nms_mask: NMS mask of shape (B, H, W).

        Returns:
            Tuple of:
                - keypoints: Keypoint coordinates of shape (B, K, 2) in (y, x)
                  format, padded with (-1, -1) for invalid entries.
                - keypoint_scores: Scores for each keypoint of shape (B, K).
        """
        B, H, W = scores.shape
        K = self.max_keypoints

        scores_masked = scores * nms_mask
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
        # 1. AKAZE feature detection (scores + orientations)
        scores1, orient1 = self.detector(image1)  # (B, 1, H, W) each
        scores2, orient2 = self.detector(image2)
        scores1 = scores1.squeeze(1)  # (B, H, W)
        scores2 = scores2.squeeze(1)

        # 2. Apply NMS
        nms_mask1 = self._apply_nms_maxpool(scores1)
        nms_mask2 = self._apply_nms_maxpool(scores2)

        # 3. Select top-k keypoints
        keypoints1, _ = self._select_topk_keypoints(scores1, nms_mask1)
        keypoints2, _ = self._select_topk_keypoints(scores2, nms_mask2)

        # 4. Compute orientation-aware BAD descriptors at keypoints
        desc1 = self.descriptor(image1, keypoints1, orient1)
        desc2 = self.descriptor(image2, keypoints2, orient2)

        # 5. Perform Sinkhorn matching
        matching_probs = self.matcher(desc1, desc2)  # (B, K+1, K+1)

        return keypoints1, keypoints2, matching_probs
