"""
Shi-Tomasi Feature Detection with Angle Estimation.

This module combines Shi-Tomasi corner detection with angle estimation
to provide both feature point scores and orientations. This is the first
step in the pipeline:
    Shi-Tomasi + Angle Estimation → Sparse BAD → Sinkhorn

The angle information can be used to make descriptors rotation-invariant,
similar to how AKAZE uses orientations with BAD descriptors.
"""

import torch
from torch import nn
import torch.nn.functional as F

from ..corner.shi_tomasi import ShiTomasiScore
from ..orientation.angle_estimation import AngleEstimator
from ..descriptor.bad import _get_bad_learned_params


class ShiTomasiWithAngle(nn.Module):
    """
    Shi-Tomasi corner detection combined with angle estimation.

    This module provides a unified interface for detecting feature points
    using Shi-Tomasi and computing their orientations. It follows the same
    output format as AKAZE (scores + orientations) for compatibility.

    Args:
        block_size: Block size for Shi-Tomasi corner detection (must be odd).
        sobel_size: Sobel kernel size for gradient computation (default: 3).
        patch_size: Patch size for angle estimation (must be odd, default: 15).
        sigma: Gaussian sigma for angle estimation (default: 2.5).

    Example:
        >>> # Basic usage
        >>> detector = ShiTomasiWithAngle(block_size=5, patch_size=15)
        >>> img = torch.randn(1, 1, 480, 640)
        >>> scores, angles = detector(img)
        >>> print(scores.shape)  # [1, 1, 480, 640]
        >>> print(angles.shape)  # [1, 1, 480, 640]
        >>>
        >>> # Compatible with AKAZE interface
        >>> from pytorch_model.feature_detection.akaze import AKAZE
        >>> akaze = AKAZE()
        >>> akaze_scores, akaze_angles = akaze(img)
        >>> # Both have same output format!
    """

    def __init__(
        self,
        block_size: int = 5,
        sobel_size: int = 3,
        patch_size: int = 15,
        sigma: float = 2.5
    ):
        super().__init__()

        # Shi-Tomasi corner detector
        self.shi_tomasi = ShiTomasiScore(
            block_size=block_size,
            sobel_size=sobel_size
        )

        # Angle estimator
        self.angle_estimator = AngleEstimator(
            patch_size=patch_size,
            sigma=sigma
        )

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Detect feature points and compute their orientations.

        Args:
            image: Input grayscale image of shape (N, 1, H, W).

        Returns:
            Tuple of:
                - scores: Feature point scores (N, 1, H, W).
                  Higher values indicate stronger corners.
                - angles: Orientation map (N, 1, H, W) in radians [-π, π].
                  Provides dominant orientation at each pixel.

        Pipeline:
            1. Shi-Tomasi detects corner feature points
            2. AngleEstimator computes orientation at all pixels
            3. Results can be used for rotation-invariant matching
        """
        # Detect feature points using Shi-Tomasi
        scores = self.shi_tomasi(image)

        # Compute orientations using angle estimator
        angles = self.angle_estimator(image)

        return scores, angles


class ShiTomasiAngleSparseBAD(nn.Module):
    """
    Shi-Tomasi + Angle + Sparse BAD descriptor computation module.

    This module implements the complete feature detection and description pipeline:
        1. Shi-Tomasi corner detection
        2. Angle estimation for rotation invariance
        3. Sparse BAD descriptor computation with rotation compensation

    Similar to AKAZESparseBADSinkhornMatcher but uses Shi-Tomasi instead of
    AKAZE for feature detection. The angle information is used to rotate BAD
    pair offsets at each keypoint, making the descriptor rotation-invariant.

    Args:
        block_size: Shi-Tomasi block size (default: 5).
        patch_size: Angle estimation patch size (default: 15).
        sigma: Angle estimation sigma (default: 2.5).
        num_pairs: Number of BAD descriptor comparison pairs (descriptor
                   dimensionality). Must be 256 or 512 (default: 256).
        binarize: If True, output binarized BAD descriptors (default: False).
        soft_binarize: If True and binarize=True, use sigmoid for soft
                       binarization (default: True).
        temperature: Temperature for soft sigmoid binarization (default: 10.0).
        normalize_descriptors: If True, L2-normalize descriptors (default: True).
        sampling_mode: Sampling mode for descriptor grid sampling, 'nearest' or
                       'bilinear' (default: 'nearest').

    Example:
        >>> model = ShiTomasiAngleSparseBAD(num_pairs=256)
        >>> img = torch.randn(1, 1, 480, 640)
        >>>
        >>> # Detect features and compute orientations
        >>> scores, angles = model.detect_and_orient(img)
        >>>
        >>> # Select keypoints (example: top-100)
        >>> scores_flat = scores.view(1, -1)
        >>> _, indices = torch.topk(scores_flat, k=100, dim=1)
        >>> h, w = scores.shape[2], scores.shape[3]
        >>> y = (indices // w).float()
        >>> x = (indices % w).float()
        >>> keypoints = torch.stack([y, x], dim=-1)  # (1, 100, 2) in (y, x)
        >>>
        >>> # Compute rotation-aware descriptors
        >>> descriptors = model.describe(img, keypoints, angles)
        >>> print(descriptors.shape)  # (1, 100, 256)
    """

    def __init__(
        self,
        block_size: int = 5,
        patch_size: int = 15,
        sigma: float = 2.5,
        num_pairs: int = 256,
        binarize: bool = False,
        soft_binarize: bool = True,
        temperature: float = 10.0,
        normalize_descriptors: bool = True,
        sampling_mode: str = "nearest",
    ):
        super().__init__()

        if num_pairs not in (256, 512):
            raise ValueError(
                f"num_pairs must be 256 or 512 to use learned BAD patterns, got {num_pairs}"
            )
        if sampling_mode not in ("nearest", "bilinear"):
            raise ValueError(
                f"sampling_mode must be 'nearest' or 'bilinear', got {sampling_mode}"
            )

        self.num_pairs = num_pairs
        self.binarize = binarize
        self.soft_binarize = soft_binarize
        self.temperature = temperature
        self.normalize_descriptors = normalize_descriptors
        self.sampling_mode = sampling_mode

        # Feature detection + orientation
        self.detector = ShiTomasiWithAngle(
            block_size=block_size,
            patch_size=patch_size,
            sigma=sigma
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

        # One-hot selection matrix mapping each pair to its radius channel
        radius_select = torch.zeros(int(torch.max(self.radii).item()) + 1, num_pairs)
        for i in range(num_pairs):
            radius_select[int(self.radii[i].item()), i] = 1.0
        self.register_buffer("radius_select", radius_select)

        # Bank of normalized box kernels for each radius
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

    def detect_and_orient(
        self,
        image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Detect feature points and compute orientations.

        Args:
            image: Input image (N, 1, H, W).

        Returns:
            Tuple of (scores, angles) both with shape (N, 1, H, W).
        """
        return self.detector(image)

    def describe(
        self,
        image: torch.Tensor,
        keypoints: torch.Tensor,
        orientation: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute rotation-aware BAD descriptors at keypoint locations.

        Each pair's sampling offsets are rotated by the local orientation
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

        # Normalize descriptors if enabled
        if self.normalize_descriptors:
            desc = F.normalize(desc, p=2, dim=-1)

        return desc

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returns scores and angles.

        For full pipeline including descriptor computation, use:
        1. detect_and_orient() to get scores and angles
        2. Select keypoints from scores
        3. describe() to compute rotation-aware descriptors

        Returns:
            Tuple of (scores, angles) both with shape (N, 1, H, W).
        """
        return self.detect_and_orient(image)


# Example usage and integration guide
if __name__ == "__main__":
    print("Shi-Tomasi + Angle Estimation Pipeline")
    print("=" * 60)
    print()
    print("This module provides feature detection with orientation:")
    print("  1. Shi-Tomasi detects corner points")
    print("  2. AngleEstimator computes orientation angles")
    print("  3. Output can be used with rotation-aware descriptors")
    print()
    print("Usage example:")
    print("  >>> from pytorch_model.feature_detection.shi_tomasi_angle import ShiTomasiWithAngle")
    print("  >>> detector = ShiTomasiWithAngle(block_size=5, patch_size=15)")
    print("  >>> scores, angles = detector(image)")
    print()
    print("Integration with pipeline:")
    print("  Shi-Tomasi + Angle → Sparse BAD → Sinkhorn")
    print()
    print("Compare with AKAZE:")
    print("  - AKAZE: Multi-scale + Hessian detector + Orientation")
    print("  - Shi-Tomasi: Single scale + Minimum eigenvalue + Orientation")
    print("  - Both provide (scores, angles) output format")
    print()
