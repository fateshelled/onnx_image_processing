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
        box_size: Box size for BAD averaging window. Must be odd. Default is 5.
        pattern_scale: Scale factor for BAD sampling pattern spread in pixels.
                       Default is 16.0.
        seed: Random seed for reproducible BAD sampling pattern. Default is 42.
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
        box_size: int = 5,
        pattern_scale: float = 16.0,
        seed: int = 42,
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
    ) -> None:
        super().__init__()

        if box_size <= 0 or box_size % 2 == 0:
            raise ValueError(f"box_size must be a positive odd integer, got {box_size}")

        self.max_keypoints = max_keypoints
        self.nms_radius = nms_radius
        self.score_threshold = score_threshold
        self.normalize_descriptors = normalize_descriptors
        self.num_pairs = num_pairs
        self.box_size = box_size
        self.binarize = binarize
        self.soft_binarize = soft_binarize
        self.temperature = temperature

        # Corner detector only (no dense BAD computation)
        self.corner_detector = ShiTomasiScore(
            block_size=block_size,
            sobel_size=sobel_size,
        )

        # BAD sampling pattern (same generation as BADDescriptor for consistency)
        generator = torch.Generator()
        generator.manual_seed(seed)
        pair_offsets = (
            (torch.rand(num_pairs, 2, 2, generator=generator) - 0.5)
            * 2
            * pattern_scale
        )
        self.register_buffer("pair_offsets", pair_offsets)

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

        # Step 1: Compute box-averaged image
        pad = self.box_size // 2
        img_padded = F.pad(image, (pad, pad, pad, pad), mode="replicate")
        box_avg = F.avg_pool2d(img_padded, kernel_size=self.box_size, stride=1)
        # box_avg: (B, 1, H, W)

        # Step 2: Build sampling grids for all keypoint-pair combinations
        # pair_offsets: (num_pairs, 2, 2) where [:, 0, :] = (dy1, dx1), [:, 1, :] = (dy2, dx2)
        offsets1 = self.pair_offsets[:, 0, :]  # (num_pairs, 2)
        offsets2 = self.pair_offsets[:, 1, :]  # (num_pairs, 2)

        # Compute absolute sampling positions in pixel space
        # kp_clamped: (B, K, 1, 2) + offsets: (1, 1, num_pairs, 2) -> (B, K, num_pairs, 2)
        kp_expanded = kp_clamped.unsqueeze(2)
        pos1 = kp_expanded + offsets1.unsqueeze(0).unsqueeze(0)  # (B, K, num_pairs, 2)
        pos2 = kp_expanded + offsets2.unsqueeze(0).unsqueeze(0)

        # Convert pixel coordinates to normalized [-1, 1] for grid_sample (align_corners=True)
        # norm = pixel / (dim - 1) * 2 - 1
        norm_scale_y = 2.0 / (H - 1 + 1e-8)
        norm_scale_x = 2.0 / (W - 1 + 1e-8)

        grid1_y = pos1[:, :, :, 0] * norm_scale_y - 1.0
        grid1_x = pos1[:, :, :, 1] * norm_scale_x - 1.0
        grid2_y = pos2[:, :, :, 0] * norm_scale_y - 1.0
        grid2_x = pos2[:, :, :, 1] * norm_scale_x - 1.0

        # grid_sample expects (x, y) order: (B, K, num_pairs, 2)
        grid1 = torch.stack([grid1_x, grid1_y], dim=-1)
        grid2 = torch.stack([grid2_x, grid2_y], dim=-1)

        # Step 3: Sample box_avg at offset positions
        # input: (B, 1, H, W), grid: (B, K, num_pairs, 2)
        # output: (B, 1, K, num_pairs)
        sample1 = F.grid_sample(
            box_avg, grid1, mode="bilinear", padding_mode="border", align_corners=True,
        )
        sample2 = F.grid_sample(
            box_avg, grid2, mode="bilinear", padding_mode="border", align_corners=True,
        )

        # Step 4: Compute difference
        diff = (sample1 - sample2).squeeze(1)  # (B, K, num_pairs)

        # Step 5: Optional binarization
        if not self.binarize:
            desc = diff
        elif self.soft_binarize:
            desc = torch.sigmoid(diff * self.temperature)
        else:
            desc = (diff > 0).to(diff.dtype)

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
