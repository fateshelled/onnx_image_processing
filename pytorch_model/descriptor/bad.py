"""
BAD (Box Average Difference) Descriptor for Visual Odometry.

This module implements a dense BAD descriptor that computes feature maps
for the entire image using matrix operations, suitable for ONNX export
and efficient NPU/GPU inference.

Reference:
    BAD: Bidimensional Anomaly Detection
    The descriptor compares average intensities within box regions
    at different offsets to create binary descriptors.
"""

import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Tuple


class BADDescriptor(nn.Module):
    """
    Dense BAD (Box Average Difference) Descriptor.

    Computes a dense descriptor map where each pixel gets a descriptor
    based on comparing box averages at different offset positions.
    All operations are matrix-based for efficient ONNX export.

    The descriptor works by:
    1. Computing box-averaged image using avg_pool2d
    2. Sampling the averaged image at offset positions using grid_sample
    3. Comparing pairs of sampled values to produce binary-like features

    Args:
        num_pairs: Number of comparison pairs (descriptor dimensionality).
                   Default is 256 for a 256-bit descriptor.
        box_size: Size of the averaging box window. Must be odd. Default is 5.
        pattern_scale: Scale factor for sampling pattern spread in pixels.
                       Default is 16.0.
        seed: Random seed for reproducible sampling pattern generation.
              Default is 42.
        binarize: If True, output binarized descriptors. If False, output
                  raw differences. Default is False for ONNX compatibility.
        soft_binarize: If True and binarize=True, use sigmoid for soft
                       binarization. If False, use sign for hard binarization.
                       Default is True for better gradient flow.
        temperature: Temperature for soft sigmoid binarization. Higher values
                     produce sharper transitions. Default is 10.0.

    Example:
        >>> model = BADDescriptor(num_pairs=256, box_size=5)
        >>> img = torch.randn(1, 1, 480, 640)  # Grayscale image
        >>> desc_map = model(img)  # [1, 256, 480, 640]
    """

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
        if num_pairs <= 0:
            raise ValueError(f"num_pairs must be positive, got {num_pairs}")
        if pattern_scale <= 0:
            raise ValueError(f"pattern_scale must be positive, got {pattern_scale}")

        self.num_pairs = num_pairs
        self.box_size = box_size
        self.pattern_scale = pattern_scale
        self.binarize = binarize
        self.soft_binarize = soft_binarize
        self.temperature = temperature

        # Generate random sampling pairs with fixed seed for reproducibility
        # Each pair has two box centers: (dy1, dx1) and (dy2, dx2)
        # These are pixel offsets from the center position
        # Shape: [num_pairs, 2, 2] where [:, 0, :] is (dy1, dx1), [:, 1, :] is (dy2, dx2)
        generator = torch.Generator()
        generator.manual_seed(seed)
        pair_offsets = (torch.rand(num_pairs, 2, 2, generator=generator) - 0.5) * 2 * pattern_scale
        self.register_buffer("pair_offsets", pair_offsets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute dense BAD descriptor map.

        Args:
            x: Input grayscale image tensor of shape (B, 1, H, W).
               Values can be in any range (typically [0, 255] or [0, 1]).

        Returns:
            Descriptor map of shape (B, num_pairs, H, W).
            If binarize=False: raw differences (float).
            If binarize=True: binary descriptors (0 or 1, or soft values).
        """
        B, C, H, W = x.shape
        device = x.device
        dtype = x.dtype
        num_pairs = self.num_pairs

        # Step 1: Compute box-averaged image using avg_pool2d
        # Pad to maintain spatial dimensions
        pad = self.box_size // 2
        x_padded = F.pad(x, (pad, pad, pad, pad), mode="replicate")
        box_avg = F.avg_pool2d(x_padded, kernel_size=self.box_size, stride=1)
        # box_avg: [B, 1, H, W]

        # Step 2: Create normalized coordinate grids for sampling
        # grid_sample expects coordinates in [-1, 1] range with align_corners=True
        # where (-1, -1) is top-left and (1, 1) is bottom-right

        # Create base normalized coordinates
        y_coords = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
        x_coords = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
        base_y, base_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
        # base_y, base_x: [H, W]

        # Convert pixel offsets to normalized coordinate offsets
        # For align_corners=True: pixel offset of 1 = 2/(dim-1) in normalized space
        # Add small epsilon to prevent division by zero for 1-pixel dimensions
        scale_y = 2.0 / (H - 1 + 1e-8)
        scale_x = 2.0 / (W - 1 + 1e-8)

        # Extract and scale offsets for both sampling positions
        # pair_offsets: [num_pairs, 2, 2] -> (dy, dx) for position 1 and 2
        offsets1 = self.pair_offsets[:, 0, :]  # [num_pairs, 2] (dy1, dx1)
        offsets2 = self.pair_offsets[:, 1, :]  # [num_pairs, 2] (dy2, dx2)

        # Scale to normalized coordinates
        dy1_norm = offsets1[:, 0:1, None] * scale_y  # [num_pairs, 1, 1]
        dx1_norm = offsets1[:, 1:2, None] * scale_x
        dy2_norm = offsets2[:, 0:1, None] * scale_y
        dx2_norm = offsets2[:, 1:2, None] * scale_x

        # Compute sampling grids for all pairs
        # Add offsets to base grid: [num_pairs, H, W]
        grid1_y = base_y.unsqueeze(0) + dy1_norm
        grid1_x = base_x.unsqueeze(0) + dx1_norm
        grid2_y = base_y.unsqueeze(0) + dy2_norm
        grid2_x = base_x.unsqueeze(0) + dx2_norm

        # Stack to create grids: [num_pairs, H, W, 2]
        # grid_sample expects (x, y) order
        grid1 = torch.stack([grid1_x, grid1_y], dim=-1)
        grid2 = torch.stack([grid2_x, grid2_y], dim=-1)

        # Step 3: Prepare tensors for batched grid_sample
        # We need to sample box_avg[b] with grid[p] for all (b, p) combinations
        # Use repeat_interleave/repeat for clarity and efficiency

        # Repeat box_avg for each pair: [B, 1, H, W] -> [B*num_pairs, 1, H, W]
        # Each batch element is repeated num_pairs times consecutively
        box_avg_batched = box_avg.repeat_interleave(num_pairs, dim=0)

        # Repeat grids for each batch: [num_pairs, H, W, 2] -> [B*num_pairs, H, W, 2]
        # The entire grid sequence is repeated B times
        grid1_batched = grid1.repeat(B, 1, 1, 1)
        grid2_batched = grid2.repeat(B, 1, 1, 1)

        # Step 4: Sample using bilinear interpolation
        sample1 = F.grid_sample(
            box_avg_batched,
            grid1_batched,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        sample2 = F.grid_sample(
            box_avg_batched,
            grid2_batched,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        # sample1, sample2: [B * num_pairs, 1, H, W]

        # Reshape back to [B, num_pairs, H, W]
        sample1 = sample1.reshape(B, num_pairs, H, W)
        sample2 = sample2.reshape(B, num_pairs, H, W)

        # Step 5: Compute difference (raw descriptor)
        diff = sample1 - sample2  # [B, num_pairs, H, W]

        # Step 6: Optional binarization
        # Python if/elif/else is fine for ONNX since conditions resolve at trace time
        if not self.binarize:
            return diff
        elif self.soft_binarize:
            # Soft binarization using sigmoid for differentiable output
            return torch.sigmoid(diff * self.temperature)
        else:
            # Hard binarization: maps to {0, 1}
            return (diff > 0).to(diff.dtype)


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
