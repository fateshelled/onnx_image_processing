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
from pytorch_model.descriptor.bad import _get_bad_learned_params
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
        self.normalize_descriptors = normalize_descriptors
        self.sampling_mode = sampling_mode
        if num_pairs not in (256, 512):
            raise ValueError(
                f"num_pairs must be 256 or 512 to use learned BAD patterns, got {num_pairs}"
            )
        if self.sampling_mode not in ("nearest", "bilinear"):
            raise ValueError(
                f"sampling_mode must be 'nearest' or 'bilinear', got {sampling_mode}"
            )

        self.num_pairs = num_pairs
        self.binarize = binarize
        self.soft_binarize = soft_binarize
        self.temperature = temperature

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

        # BAD descriptor buffers (learned patterns)
        box_params, thresholds = _get_bad_learned_params(num_pairs)
        self.register_buffer("offset_x1", box_params[:, 0] - 16.0)
        self.register_buffer("offset_x2", box_params[:, 1] - 16.0)
        self.register_buffer("offset_y1", box_params[:, 2] - 16.0)
        self.register_buffer("offset_y2", box_params[:, 3] - 16.0)
        self.register_buffer("radii", box_params[:, 4].to(torch.int64))
        self.register_buffer("thresholds", thresholds)

        # Pre-reshape buffers for performance and ONNX graph clarity
        self.register_buffer("offset_y1_v", self.offset_y1.view(1, 1, -1))
        self.register_buffer("offset_x1_v", self.offset_x1.view(1, 1, -1))
        self.register_buffer("offset_y2_v", self.offset_y2.view(1, 1, -1))
        self.register_buffer("offset_x2_v", self.offset_x2.view(1, 1, -1))
        self.register_buffer("thresholds_v", self.thresholds.view(1, 1, -1))

        # One-hot selection matrix mapping each pair to its radius channel.
        radius_select = torch.zeros(int(torch.max(self.radii).item()) + 1, num_pairs)
        for i in range(num_pairs):
            radius_select[int(self.radii[i].item()), i] = 1.0
        self.register_buffer("radius_select", radius_select)

        # Bank of normalized box kernels for each radius.
        max_radius = int(torch.max(self.radii).item())
        self.max_radius = max_radius
        coords = torch.arange(-max_radius, max_radius + 1, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
        radius_values = torch.arange(max_radius + 1, dtype=torch.float32).view(-1, 1, 1)
        square_masks = (
            (grid_y.abs() <= radius_values) & (grid_x.abs() <= radius_values)
        ).to(torch.float32)
        denom = ((2.0 * radius_values + 1.0) ** 2).clamp_min(1.0)
        kernel_bank = (square_masks / denom).unsqueeze(1)
        self.register_buffer("box_kernel_bank", kernel_bank)

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

    def _compute_bad_at_keypoints(
        self,
        image: torch.Tensor,
        keypoints: torch.Tensor,
        orientation: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute orientation-aware BAD descriptors at keypoint locations.

        Each pair's sampling offsets are rotated by the local AKAZE orientation
        at the keypoint, making the descriptor rotation-invariant.

        Args:
            image: Input grayscale image of shape (B, 1, H, W).
            keypoints: Keypoint coordinates of shape (B, K, 2) in (y, x) format.
                       Invalid keypoints at (-1, -1) are handled by clamping.
            orientation: Orientation map of shape (B, 1, H, W) in radians.

        Returns:
            Descriptors of shape (B, K, num_pairs) at keypoint locations.
            Invalid keypoints get zero descriptors.
        """
        _B, _C, H, W = image.shape

        # Validity mask before clamping
        valid_mask = (keypoints[:, :, 0] >= 0).float()  # (B, K)

        # Clamp invalid keypoints to valid range for sampling
        y_clamped = torch.clamp(keypoints[:, :, 0], min=0.0, max=float(H - 1))
        x_clamped = torch.clamp(keypoints[:, :, 1], min=0.0, max=float(W - 1))
        kp_clamped = torch.stack([y_clamped, x_clamped], dim=-1)  # (B, K, 2)

        # Normalization scales for pixel -> [-1, 1] conversion
        norm_scale_y = 2.0 / (H - 1 + 1e-8)
        norm_scale_x = 2.0 / (W - 1 + 1e-8)

        # --- Sample orientation at keypoint locations ---
        ky_norm = y_clamped * norm_scale_y - 1.0  # (B, K)
        kx_norm = x_clamped * norm_scale_x - 1.0
        orient_grid = torch.stack([kx_norm, ky_norm], dim=-1).unsqueeze(2)  # (B, K, 1, 2)
        theta = F.grid_sample(
            orientation, orient_grid, mode="nearest",
            padding_mode="border", align_corners=True,
        ).squeeze(1).squeeze(-1)  # (B, K)

        cos_t = torch.cos(theta).unsqueeze(-1)  # (B, K, 1)
        sin_t = torch.sin(theta).unsqueeze(-1)

        # --- Rotate pair offsets per keypoint ---
        oy1 = self.offset_y1_v.to(dtype=image.dtype)  # (1, 1, P)
        ox1 = self.offset_x1_v.to(dtype=image.dtype)
        oy2 = self.offset_y2_v.to(dtype=image.dtype)
        ox2 = self.offset_x2_v.to(dtype=image.dtype)

        # 2-D rotation [cos -sin; sin cos] applied to (ox, oy) offsets
        rot_dy1 = ox1 * sin_t + oy1 * cos_t  # (B, K, P)
        rot_dx1 = ox1 * cos_t - oy1 * sin_t
        rot_dy2 = ox2 * sin_t + oy2 * cos_t
        rot_dx2 = ox2 * cos_t - oy2 * sin_t

        # --- Absolute sample positions ---
        kp_y = kp_clamped[:, :, 0:1]  # (B, K, 1)
        kp_x = kp_clamped[:, :, 1:2]

        pos1_y = kp_y + rot_dy1  # (B, K, P)
        pos1_x = kp_x + rot_dx1
        pos2_y = kp_y + rot_dy2
        pos2_x = kp_x + rot_dx2

        # --- Build normalized grids for grid_sample ---
        grid1 = torch.stack(
            [pos1_x * norm_scale_x - 1.0, pos1_y * norm_scale_y - 1.0],
            dim=-1,
        )  # (B, K, P, 2)
        grid2 = torch.stack(
            [pos2_x * norm_scale_x - 1.0, pos2_y * norm_scale_y - 1.0],
            dim=-1,
        )

        # --- Box averaging + sampling ---
        kernels = self.box_kernel_bank.to(device=image.device, dtype=image.dtype)
        padded = F.pad(
            image,
            (self.max_radius, self.max_radius, self.max_radius, self.max_radius),
            mode="replicate",
        )
        box_avg_bank = F.conv2d(padded, kernels, stride=1)  # (B, R+1, H, W)

        sampled1 = F.grid_sample(
            box_avg_bank, grid1, mode=self.sampling_mode,
            padding_mode="border", align_corners=True,
        )  # (B, R+1, K, P)
        sampled2 = F.grid_sample(
            box_avg_bank, grid2, mode=self.sampling_mode,
            padding_mode="border", align_corners=True,
        )

        # --- Select correct radius channel per pair via multiply+sum ---
        rs = self.radius_select.to(dtype=sampled1.dtype).view(1, -1, 1, self.num_pairs)
        sample1 = (sampled1 * rs).sum(dim=1)  # (B, K, P)
        sample2 = (sampled2 * rs).sum(dim=1)
        diff = sample1 - sample2

        centered = diff - self.thresholds_v.to(diff.dtype)

        if not self.binarize:
            desc = centered
        elif self.soft_binarize:
            desc = torch.sigmoid(-centered * self.temperature)
        else:
            desc = (centered <= 0).to(centered.dtype)

        # Zero out invalid keypoints' descriptors
        desc = desc * valid_mask.unsqueeze(-1)

        return desc

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
        desc1 = self._compute_bad_at_keypoints(image1, keypoints1, orient1)
        desc2 = self._compute_bad_at_keypoints(image2, keypoints2, orient2)

        # 5. Normalize descriptors if enabled
        if self.normalize_descriptors:
            desc1 = F.normalize(desc1, p=2, dim=-1)
            desc2 = F.normalize(desc2, p=2, dim=-1)

        # 6. Perform Sinkhorn matching
        matching_probs = self.matcher(desc1, desc2)  # (B, K+1, K+1)

        return keypoints1, keypoints2, matching_probs
