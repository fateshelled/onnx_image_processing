"""
BAD (Box Average Difference) Descriptor for Visual Odometry.

This module implements dense and sparse-compatible BAD descriptor operators
using the original learned measurement patterns from the public BAD release.
"""

import torch
from torch import nn
import torch.nn.functional as F

from .bad_params import _get_bad_learned_params

class BADDescriptor(nn.Module):
    """Dense BAD descriptor with learned box pattern and learned thresholds."""

    def __init__(
        self,
        num_pairs: int = 256,
        box_size: int = 5,
        pattern_scale: float = 16.0,
        seed: int = 42,
        binarize: bool = False,
        soft_binarize: bool = True,
        temperature: float = 10.0,
    ) -> None:
        super().__init__()

        if box_size <= 0 or box_size % 2 == 0:
            raise ValueError(f"box_size must be a positive odd integer, got {box_size}")

        # NOTE: pattern_scale/seed are kept for API compatibility but unused.
        self.num_pairs = num_pairs
        self.box_size = box_size
        self.pattern_scale = pattern_scale
        self.seed = seed
        self.binarize = binarize
        self.soft_binarize = soft_binarize
        self.temperature = temperature

        box_params, thresholds = _get_bad_learned_params(num_pairs)
        # BAD learned patch is 32x32 and rectified around patch center.
        self.register_buffer("offset_x1", box_params[:, 0] - 16.0)
        self.register_buffer("offset_x2", box_params[:, 1] - 16.0)
        self.register_buffer("offset_y1", box_params[:, 2] - 16.0)
        self.register_buffer("offset_y2", box_params[:, 3] - 16.0)
        self.register_buffer("radii", box_params[:, 4].to(torch.int64))
        self.register_buffer("thresholds", thresholds)
        # Pre-compute area for normalization
        area = (2.0 * self.radii.float() + 1.0) ** 2
        self.register_buffer("area", area.view(-1, 1, 1))
        # Pre-compute max_radius for padding
        self.max_radius = int(torch.max(self.radii).item())

    def _compute_diff_map(self, x: torch.Tensor) -> torch.Tensor:
        """Compute BAD average differences using learned box radii and offsets."""
        B, _, H, W = x.shape
        device = x.device
        dtype = x.dtype

        # Use pre-computed max_radius for ONNX compatibility
        max_radius = self.max_radius
        x_padded = F.pad(x, (max_radius, max_radius, max_radius, max_radius), mode="replicate")
        integral = torch.cumsum(torch.cumsum(x_padded, dim=2), dim=3)
        integral = F.pad(integral, (1, 0, 1, 0), mode="constant", value=0.0).squeeze(1)
        _, Hp1, Wp1 = integral.shape

        base_y = torch.arange(H, device=device, dtype=dtype).view(1, H, 1)
        base_x = torch.arange(W, device=device, dtype=dtype).view(1, 1, W)

        radii_i64 = self.radii.to(device=device).view(-1, 1, 1)

        def box_mean_for_offsets(offset_y: torch.Tensor, offset_x: torch.Tensor) -> torch.Tensor:
            center_y = torch.clamp(base_y + offset_y.view(-1, 1, 1), min=0.0, max=float(H - 1))
            center_x = torch.clamp(base_x + offset_x.view(-1, 1, 1), min=0.0, max=float(W - 1))

            center_y_i64 = center_y.to(torch.int64) + max_radius
            center_x_i64 = center_x.to(torch.int64) + max_radius

            y0 = center_y_i64 - radii_i64
            x0 = center_x_i64 - radii_i64
            y1 = center_y_i64 + radii_i64 + 1
            x1 = center_x_i64 + radii_i64 + 1

            flat = integral.reshape(B, -1)

            def gather(y_idx: torch.Tensor, x_idx: torch.Tensor) -> torch.Tensor:
                linear_idx = (y_idx * Wp1 + x_idx).reshape(-1)
                return flat[:, linear_idx].reshape(B, self.num_pairs, H, W)

            area_sum = gather(y1, x1) - gather(y0, x1) - gather(y1, x0) + gather(y0, x0)
            return area_sum / self.area.to(device=device, dtype=dtype)

        sample1 = box_mean_for_offsets(
            self.offset_y1.to(device=device, dtype=dtype),
            self.offset_x1.to(device=device, dtype=dtype),
        )
        sample2 = box_mean_for_offsets(
            self.offset_y2.to(device=device, dtype=dtype),
            self.offset_x2.to(device=device, dtype=dtype),
        )

        return sample1 - sample2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute dense BAD descriptor map using learned thresholds."""
        diff = self._compute_diff_map(x)
        centered = diff - self.thresholds.view(1, -1, 1, 1).to(diff.dtype)

        if not self.binarize:
            return centered
        if self.soft_binarize:
            # BAD bit is 1 when response <= threshold.
            return torch.sigmoid(-centered * self.temperature)
        return (centered <= 0).to(centered.dtype)


def extract_descriptors_at_keypoints(
    descriptor_map: torch.Tensor,
    keypoints: torch.Tensor,
) -> torch.Tensor:
    """
    Extract descriptor vectors at specific keypoint locations.

    This function demonstrates how to extract descriptors from the dense
    descriptor map at specific keypoint coordinates using gather operations.

    Args:
        descriptor_map: Dense descriptor map of shape (B, D, H, W)
                        where D is the descriptor dimensionality.
        keypoints: Keypoint coordinates of shape (B, N, 2) where N is the
                   number of keypoints and each keypoint is (y, x) in pixels.
                   Coordinates should be integers in valid range.

    Returns:
        Descriptor vectors of shape (B, N, D) for each keypoint.

    Example:
        >>> model = BADDescriptor(num_pairs=256)
        >>> img = torch.randn(1, 1, 480, 640)
        >>> desc_map = model(img)  # [1, 256, 480, 640]
        >>> # Define keypoints: 100 random points
        >>> kpts = torch.stack([
        ...     torch.randint(0, 480, (100,)),  # y coordinates
        ...     torch.randint(0, 640, (100,)),  # x coordinates
        ... ], dim=-1).unsqueeze(0)  # [1, 100, 2]
        >>> descriptors = extract_descriptors_at_keypoints(desc_map, kpts)
        >>> print(descriptors.shape)  # [1, 100, 256]
    """
    B, D, H, W = descriptor_map.shape
    _, N, _ = keypoints.shape

    # Convert 2D coordinates to 1D indices for gather
    # y * W + x gives the flattened index
    y_coords = keypoints[:, :, 0].long()  # [B, N]
    x_coords = keypoints[:, :, 1].long()  # [B, N]
    flat_indices = y_coords * W + x_coords  # [B, N]

    # Flatten spatial dimensions: [B, D, H, W] -> [B, D, H*W]
    desc_flat = descriptor_map.reshape(B, D, H * W)

    # Expand indices for all descriptor dimensions: [B, N] -> [B, D, N]
    indices_expanded = flat_indices.unsqueeze(1).expand(B, D, N)

    # Gather descriptors at keypoint locations: [B, D, N]
    descriptors = torch.gather(desc_flat, dim=2, index=indices_expanded)

    # Transpose to [B, N, D] for conventional descriptor format
    descriptors = descriptors.permute(0, 2, 1)

    return descriptors


def extract_descriptors_at_keypoints_subpixel(
    descriptor_map: torch.Tensor,
    keypoints: torch.Tensor,
) -> torch.Tensor:
    """
    Extract descriptor vectors at sub-pixel keypoint locations using bilinear interpolation.

    This version supports floating-point keypoint coordinates for sub-pixel
    accuracy, using grid_sample for bilinear interpolation.

    Args:
        descriptor_map: Dense descriptor map of shape (B, D, H, W)
                        where D is the descriptor dimensionality.
        keypoints: Keypoint coordinates of shape (B, N, 2) where N is the
                   number of keypoints and each keypoint is (y, x) in pixels.
                   Coordinates can be floating-point for sub-pixel precision.

    Returns:
        Descriptor vectors of shape (B, N, D) for each keypoint.

    Example:
        >>> model = BADDescriptor(num_pairs=256)
        >>> img = torch.randn(1, 1, 480, 640)
        >>> desc_map = model(img)  # [1, 256, 480, 640]
        >>> # Sub-pixel keypoints
        >>> kpts = torch.tensor([[[100.5, 200.3], [150.2, 300.7]]]).float()  # [1, 2, 2]
        >>> descriptors = extract_descriptors_at_keypoints_subpixel(desc_map, kpts)
        >>> print(descriptors.shape)  # [1, 2, 256]
    """
    B, D, H, W = descriptor_map.shape
    _, N, _ = keypoints.shape

    # Convert pixel coordinates to normalized coordinates [-1, 1]
    # For align_corners=True: pixel 0 -> -1, pixel (dim-1) -> 1
    y_norm = keypoints[:, :, 0] / (H - 1 + 1e-8) * 2.0 - 1.0  # [B, N]
    x_norm = keypoints[:, :, 1] / (W - 1 + 1e-8) * 2.0 - 1.0  # [B, N]

    # Create sampling grid: [B, N, 1, 2] for grid_sample
    # grid_sample expects (x, y) order and shape [B, H_out, W_out, 2]
    # Here we treat N keypoints as H_out=N, W_out=1
    grid = torch.stack([x_norm, y_norm], dim=-1)  # [B, N, 2]
    grid = grid.unsqueeze(2)  # [B, N, 1, 2]

    # Sample descriptors using bilinear interpolation
    # Output shape: [B, D, N, 1]
    sampled = F.grid_sample(
        descriptor_map,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )

    # Reshape to [B, N, D]
    descriptors = sampled.squeeze(-1).permute(0, 2, 1)

    return descriptors
