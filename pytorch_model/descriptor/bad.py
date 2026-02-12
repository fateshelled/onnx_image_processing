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
        binarize: bool = False,
        soft_binarize: bool = True,
        temperature: float = 10.0,
    ) -> None:
        super().__init__()

        self.num_pairs = num_pairs
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

        # --- Buffers for oriented mode (conv2d + grid_sample path) ---
        max_radius = self.max_radius
        radius_select = torch.zeros(max_radius + 1, num_pairs)
        for i in range(num_pairs):
            radius_select[int(self.radii[i].item()), i] = 1.0
        self.register_buffer("radius_select", radius_select)

        coords = torch.arange(-max_radius, max_radius + 1, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
        radius_values = torch.arange(max_radius + 1, dtype=torch.float32).view(-1, 1, 1)
        square_masks = (
            (grid_y.abs() <= radius_values) & (grid_x.abs() <= radius_values)
        ).to(torch.float32)
        denom = ((2.0 * radius_values + 1.0) ** 2).clamp_min(1.0)
        kernel_bank = (square_masks / denom).unsqueeze(1)
        self.register_buffer("box_kernel_bank", kernel_bank)

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

    def _compute_diff_map_oriented(
        self, x: torch.Tensor, orientation: torch.Tensor
    ) -> torch.Tensor:
        """Compute BAD average differences with per-pixel orientation rotation.

        Uses conv2d box averaging + grid_sample with rotation-aware sampling
        grids.  Each pair's offset is rotated by the local orientation angle
        so that the descriptor becomes rotation-invariant.

        Args:
            x: Input image of shape (B, 1, H, W).
            orientation: Per-pixel orientation map of shape (B, 1, H, W)
                         in radians [-pi, pi], e.g. from AKAZE.

        Returns:
            Difference map of shape (B, num_pairs, H, W).
        """
        B, _, H, W = x.shape
        P = self.num_pairs

        # 1. Box averaging via conv2d kernel bank
        mr = self.max_radius
        padded = F.pad(x, (mr, mr, mr, mr), mode="replicate")
        box_avg_bank = F.conv2d(
            padded, self.box_kernel_bank.to(dtype=x.dtype)
        )  # (B, R+1, H, W)

        # 2. Select per-pair channels: (B, R+1, H, W) x (R+1, P) -> (B, P, H, W)
        rs = self.radius_select.to(dtype=x.dtype)
        per_pair_avg = torch.einsum("brhw,rp->bphw", box_avg_bank, rs)

        # 3. Rotated offsets per pixel
        cos_t = torch.cos(orientation)  # (B, 1, H, W)
        sin_t = torch.sin(orientation)

        oy1 = self.offset_y1.to(dtype=x.dtype).view(1, -1, 1, 1)  # (1, P, 1, 1)
        ox1 = self.offset_x1.to(dtype=x.dtype).view(1, -1, 1, 1)
        oy2 = self.offset_y2.to(dtype=x.dtype).view(1, -1, 1, 1)
        ox2 = self.offset_x2.to(dtype=x.dtype).view(1, -1, 1, 1)

        # 2-D rotation [cos -sin; sin cos] applied to (ox, oy)
        rot_dy1 = ox1 * sin_t + oy1 * cos_t  # (B, P, H, W)
        rot_dx1 = ox1 * cos_t - oy1 * sin_t
        rot_dy2 = ox2 * sin_t + oy2 * cos_t
        rot_dx2 = ox2 * cos_t - oy2 * sin_t

        # 4. Absolute positions normalised to [-1, 1] for grid_sample
        base_y = torch.arange(H, device=x.device, dtype=x.dtype).view(1, 1, H, 1)
        base_x = torch.arange(W, device=x.device, dtype=x.dtype).view(1, 1, 1, W)
        norm_y = 2.0 / (H - 1 + 1e-8)
        norm_x = 2.0 / (W - 1 + 1e-8)

        gy1 = (base_y + rot_dy1) * norm_y - 1.0
        gx1 = (base_x + rot_dx1) * norm_x - 1.0
        gy2 = (base_y + rot_dy2) * norm_y - 1.0
        gx2 = (base_x + rot_dx2) * norm_x - 1.0

        # 5. Sample per-pair averages at rotated positions
        input_bp = per_pair_avg.reshape(B * P, 1, H, W)

        grid1 = torch.stack([gx1, gy1], dim=-1).reshape(B * P, H, W, 2)
        s1 = F.grid_sample(
            input_bp, grid1, mode="bilinear",
            padding_mode="border", align_corners=True,
        )

        grid2 = torch.stack([gx2, gy2], dim=-1).reshape(B * P, H, W, 2)
        s2 = F.grid_sample(
            input_bp, grid2, mode="bilinear",
            padding_mode="border", align_corners=True,
        )

        sample1 = s1.reshape(B, P, H, W)
        sample2 = s2.reshape(B, P, H, W)

        return sample1 - sample2

    def forward(
        self,
        x: torch.Tensor,
        orientation: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute dense BAD descriptor map.

        Args:
            x: Input image of shape (B, 1, H, W).
            orientation: Optional per-pixel orientation map (B, 1, H, W)
                         in radians from AKAZE. When provided, pair offsets
                         are rotated per-pixel for rotation-invariant
                         descriptors. When None, uses the fast integral-image
                         path (Shi-Tomasi compatible).

        Returns:
            Descriptor map of shape (B, num_pairs, H, W).
        """
        if orientation is not None:
            diff = self._compute_diff_map_oriented(x, orientation)
        else:
            diff = self._compute_diff_map(x)

        centered = diff - self.thresholds.view(1, -1, 1, 1).to(diff.dtype)

        if not self.binarize:
            return centered
        if self.soft_binarize:
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
