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

from pytorch_model.corner.shi_tomasi import ShiTomasiScore
from pytorch_model.descriptor.bad import _get_bad_learned_params
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

        # Corner detector only (no dense BAD computation)
        self.corner_detector = ShiTomasiScore(
            block_size=block_size,
            sobel_size=sobel_size,
        )

        # Use learned BAD pattern and learned thresholds.
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

        # Group pair indices by radius to avoid runtime gather on the
        # radius-channel dimension during descriptor sampling.
        unique_radii = torch.unique(self.radii, sorted=True).to(torch.int64)
        self.register_buffer("unique_radii", unique_radii)
        self.radius_group_index_names: list[str] = []
        for group_idx, radius in enumerate(unique_radii.tolist()):
            pair_indices = torch.nonzero(self.radii == int(radius), as_tuple=False).squeeze(1)
            name = f"radius_group_indices_{group_idx}"
            self.register_buffer(name, pair_indices.to(torch.int64))
            self.radius_group_index_names.append(name)

        # Precompute a bank of normalized box kernels for each radius so
        # descriptor computation can run without Python-side loops.
        max_radius = int(torch.max(self.radii).item())
        self.max_radius = max_radius
        coords = torch.arange(-max_radius, max_radius + 1, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
        radius_values = torch.arange(max_radius + 1, dtype=torch.float32).view(-1, 1, 1)
        square_masks = ((grid_y.abs() <= radius_values) & (grid_x.abs() <= radius_values)).to(
            torch.float32
        )
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
    ) -> torch.Tensor:
        """
        Compute BAD descriptors only at specified keypoint locations.

        Instead of computing dense descriptors for every pixel, this method
        computes box-averaged image once and then samples only at keypoint
        positions with pair offsets. This is more efficient when
        max_keypoints << H * W.

        Args:
            image: Input grayscale image of shape (B, 1, H, W).
            keypoints: Keypoint coordinates of shape (B, K, 2) in (y, x) format.
                       Invalid keypoints at (-1, -1) are handled by clamping
                       and subsequent masking.

        Returns:
            Descriptors of shape (B, K, num_pairs) at keypoint locations.
            Invalid keypoints get zero descriptors.
        """
        B, C, H, W = image.shape
        K = keypoints.shape[1]

        # Validity mask before clamping
        valid_mask = (keypoints[:, :, 0] >= 0).float()  # (B, K)

        # Clamp invalid keypoints to valid range for sampling
        y_clamped = torch.clamp(keypoints[:, :, 0], min=0.0, max=float(H - 1))
        x_clamped = torch.clamp(keypoints[:, :, 1], min=0.0, max=float(W - 1))
        kp_clamped = torch.stack([y_clamped, x_clamped], dim=-1)  # (B, K, 2)

        # Build sampling grids for keypoint-pair combinations.
        kp_expanded = kp_clamped.unsqueeze(2)  # (B, K, 1, 2)

        # Convert pixel coordinates to normalized [-1, 1] for grid_sample.
        norm_scale_y = 2.0 / (H - 1 + 1e-8)
        norm_scale_x = 2.0 / (W - 1 + 1e-8)

        kernels = self.box_kernel_bank.to(device=image.device, dtype=image.dtype)
        padded = F.pad(
            image,
            (self.max_radius, self.max_radius, self.max_radius, self.max_radius),
            mode="replicate",
        )
        box_avg_bank = F.conv2d(padded, kernels, stride=1)

        # Use pre-reshaped buffers, casting to the correct dtype
        offset_y1 = self.offset_y1_v.to(dtype=image.dtype)
        offset_x1 = self.offset_x1_v.to(dtype=image.dtype)
        offset_y2 = self.offset_y2_v.to(dtype=image.dtype)
        offset_x2 = self.offset_x2_v.to(dtype=image.dtype)

        pos1_y = kp_expanded[:, :, :, 0] + offset_y1
        pos1_x = kp_expanded[:, :, :, 1] + offset_x1
        pos2_y = kp_expanded[:, :, :, 0] + offset_y2
        pos2_x = kp_expanded[:, :, :, 1] + offset_x2

        grid1 = torch.stack(
            [pos1_x * norm_scale_x - 1.0, pos1_y * norm_scale_y - 1.0],
            dim=-1,
        )
        grid2 = torch.stack(
            [pos2_x * norm_scale_x - 1.0, pos2_y * norm_scale_y - 1.0],
            dim=-1,
        )

        sampled1 = F.grid_sample(
            box_avg_bank,
            grid1,
            mode=self.sampling_mode,
            padding_mode="border",
            align_corners=True,
        )
        sampled2 = F.grid_sample(
            box_avg_bank,
            grid2,
            mode=self.sampling_mode,
            padding_mode="border",
            align_corners=True,
        )

        # Reconstruct pair-aligned samples by selecting each radius channel
        # once and scattering grouped pair subsets back to original order.
        sample1 = sampled1.new_zeros(B, K, self.num_pairs)
        sample2 = sampled2.new_zeros(B, K, self.num_pairs)
        for group_idx, radius in enumerate(self.unique_radii.tolist()):
            pair_indices = getattr(self, self.radius_group_index_names[group_idx]).to(device=image.device)
            pair_indices_expanded = pair_indices.view(1, 1, -1).expand(B, K, -1)

            # sampled*: [B, R+1, K, num_pairs] -> [B, K, num_pairs] at radius channel
            sampled1_at_radius = sampled1[:, int(radius), :, :]
            sampled2_at_radius = sampled2[:, int(radius), :, :]

            selected1 = torch.index_select(sampled1_at_radius, dim=2, index=pair_indices)
            selected2 = torch.index_select(sampled2_at_radius, dim=2, index=pair_indices)

            sample1.scatter_(2, pair_indices_expanded, selected1)
            sample2.scatter_(2, pair_indices_expanded, selected2)
        diff = sample1 - sample2

        centered = diff - self.thresholds_v.to(diff.dtype)

        # BAD bit is 1 when response <= threshold.
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
        desc1 = self._compute_bad_at_keypoints(image1, keypoints1)  # (B, K, num_pairs)
        desc2 = self._compute_bad_at_keypoints(image2, keypoints2)

        # 5. Normalize descriptors if enabled
        if self.normalize_descriptors:
            desc1 = F.normalize(desc1, p=2, dim=-1)
            desc2 = F.normalize(desc2, p=2, dim=-1)

        # 6. Perform Sinkhorn matching
        matching_probs = self.matcher(desc1, desc2)  # (B, K+1, K+1)

        return keypoints1, keypoints2, matching_probs
