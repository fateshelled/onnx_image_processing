"""
Angle Estimation Module for Feature Points.

This module provides an ONNX-exportable implementation of angle/orientation
estimation for feature points detected by methods like Shi-Tomasi.

The angle calculation follows the AKAZE implementation approach:
1. Apply Gaussian-weighted kernels to the local patch around each pixel
2. Compute intensity moments (m10 and m01)
3. Calculate orientation using atan2(m01, m10)

This is designed as a standalone module that can be used with any feature
detector (e.g., Shi-Tomasi, Harris, etc.) in the pipeline:
    Shi-Tomasi → AngleEstimator → Sparse BAD → Sinkhorn

Reference:
    Based on the orientation estimation approach from AKAZE:
    Fast Explicit Diffusion for Accelerated Features in Nonlinear Scale Spaces
    Pablo F. Alcantarilla, Jesús Nuevo, Adrien Bartoli
    BMVC 2013
"""

import torch
from torch import nn
import torch.nn.functional as F


class AngleEstimator(nn.Module):
    """
    Estimate dominant orientation/angle for feature points using Gaussian-weighted
    intensity moments.

    This module computes the orientation at each pixel location using a method
    similar to AKAZE's orientation estimator. It applies Gaussian-weighted
    convolution kernels to compute intensity moments, then calculates the
    dominant orientation using the arctangent of the moment ratio.

    The computation is fully parallelized across all pixels using tensor operations,
    making it suitable for ONNX export and deployment.

    Args:
        patch_size: Size of the local patch for orientation computation.
                   Must be odd. Larger values provide more stable orientation
                   at the cost of reduced localization. Default: 15.
        sigma: Standard deviation of the Gaussian weighting function.
               Controls how much weight is given to pixels based on distance
               from the center. Smaller values give more weight to nearby pixels.
               Default: 2.5.

    Input:
        image: Grayscale image tensor of shape (N, 1, H, W).
               Values typically in range [0, 255] or [0, 1].

    Output:
        angles: Orientation map of shape (N, 1, H, W) in radians.
               Range: [-π, π] where:
               - 0 radians points to the right (positive x-axis)
               - π/2 radians points down (positive y-axis)
               - -π/2 radians points up (negative y-axis)

    Example:
        >>> # Use with Shi-Tomasi feature detector
        >>> from pytorch_model.corner.shi_tomasi import ShiTomasiScore
        >>> from pytorch_model.orientation.angle_estimation import AngleEstimator
        >>>
        >>> detector = ShiTomasiScore(block_size=5)
        >>> angle_estimator = AngleEstimator(patch_size=15, sigma=2.5)
        >>>
        >>> img = torch.randn(1, 1, 480, 640)
        >>> scores = detector(img)  # Detect feature points
        >>> angles = angle_estimator(img)  # Compute angles at all pixels
        >>>
        >>> # Pipeline: Shi-Tomasi → Angle Estimation → Sparse BAD → Sinkhorn
        >>> print(scores.shape)  # [1, 1, 480, 640]
        >>> print(angles.shape)  # [1, 1, 480, 640]

    Notes:
        - All operations use pure tensor operations without dynamic control flow
          for full ONNX compatibility.
        - The Gaussian weighting ensures smooth orientation estimates and
          reduces sensitivity to noise.
        - The moment-based approach is computationally efficient, requiring
          only two convolutions followed by atan2.
    """

    def __init__(self, patch_size: int = 15, sigma: float = 2.5):
        super().__init__()

        if patch_size % 2 == 0:
            raise ValueError(f"patch_size must be odd, got {patch_size}")
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")

        self.patch_size = patch_size
        self.sigma = sigma

        # Create Gaussian-weighted coordinate grids
        # These represent the (x, y) coordinates relative to patch center
        half_size = patch_size // 2
        y, x = torch.meshgrid(
            torch.arange(-half_size, half_size + 1, dtype=torch.float32),
            torch.arange(-half_size, half_size + 1, dtype=torch.float32),
            indexing='ij'
        )

        # Compute Gaussian weights: w(x,y) = exp(-(x² + y²) / (2σ²))
        # This gives higher weight to pixels near the center
        gaussian = torch.exp(-(x**2 + y**2) / (2 * sigma**2))

        # Create moment computation kernels
        # m10 = Σ(x * w(x,y) * I(x,y)) - weighted x-coordinate moment
        # m01 = Σ(y * w(x,y) * I(x,y)) - weighted y-coordinate moment
        #
        # Fused kernel approach: single 2-output-channel convolution
        # computes both moments simultaneously in one kernel launch.
        weight_x = (x * gaussian).view(1, 1, patch_size, patch_size)
        weight_y = (y * gaussian).view(1, 1, patch_size, patch_size)

        # Register as buffer (non-trainable parameters saved with model)
        # Shape: (2, 1, patch_size, patch_size)
        self.register_buffer('moment_kernels', torch.cat([weight_x, weight_y], dim=0))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Compute orientation/angle map for the entire image.

        The dominant orientation at each pixel is computed using:
            θ = atan2(m01, m10)

        where m10 and m01 are Gaussian-weighted intensity moments:
            m10 = Σ(x * w(x,y) * I(x,y))
            m01 = Σ(y * w(x,y) * I(x,y))
            w(x,y) = exp(-(x² + y²) / (2σ²))

        The moments capture the distribution of intensity in the local patch,
        and their ratio gives the dominant orientation direction.

        Args:
            image: Input grayscale image tensor of shape (N, 1, H, W).

        Returns:
            Orientation map of shape (N, 1, H, W) in radians [-π, π].
            Each pixel contains the dominant orientation angle at that location.

        Algorithm:
            1. Apply Gaussian-weighted moment kernels via convolution
            2. Extract m10 (x-moment) and m01 (y-moment) from output channels
            3. Compute θ = atan2(m01, m10) to get orientation angle

        Computational Complexity:
            - Single 2-channel convolution: O(N * H * W * patch_size²)
            - atan2 operation: O(N * H * W)
            - Total: O(N * H * W * patch_size²)
        """
        padding = self.patch_size // 2

        # Apply Gaussian-weighted moment kernels
        # Input:  (N, 1, H, W)
        # Kernel: (2, 1, patch_size, patch_size)
        # Output: (N, 2, H, W) where channel 0 = m10, channel 1 = m01
        moments = F.conv2d(image, self.moment_kernels, padding=padding)

        # Extract moments from channels
        m10 = moments[:, 0:1]  # x-weighted moment (N, 1, H, W)
        m01 = moments[:, 1:2]  # y-weighted moment (N, 1, H, W)

        # Compute dominant orientation angle
        # atan2(y, x) returns angle in radians [-π, π]
        # Convention: 0° points right, 90° points down (image coordinates)
        orientation = torch.atan2(m01, m10)  # (N, 1, H, W)

        return orientation


class AngleEstimatorMultiScale(nn.Module):
    """
    Multi-scale angle estimation using Gaussian pyramid.

    **WARNING: This is an experimental feature with incomplete implementation.**
    The multi-scale score-based selection logic is not yet implemented.
    Currently, this class always returns orientations from scale 0.

    This variant is intended to compute orientations at multiple scales and select
    the orientation from the scale with the strongest response. This could
    provide more robust orientation estimates for features at different scales.

    Args:
        num_scales: Number of scale levels (image downsampling stages).
        patch_size: Size of the local patch for orientation computation (must be odd).
        sigma: Standard deviation of the Gaussian weighting.
        pooling_factor: Downsampling factor between scales (default: 2).

    Note:
        For production use, please use the single-scale `AngleEstimator` instead.
        This multi-scale variant is provided for experimental purposes only.

    Example:
        >>> # Note: This is experimental and incomplete
        >>> estimator = AngleEstimatorMultiScale(num_scales=3, patch_size=15)
        >>> img = torch.randn(1, 1, 480, 640)
        >>> angles, scales = estimator(img)
        >>> print(angles.shape)  # [1, 1, 480, 640]
        >>> print(scales.shape)  # [1, 1, 480, 640] - always 0 (not implemented)
    """

    def __init__(
        self,
        num_scales: int = 3,
        patch_size: int = 15,
        sigma: float = 2.5,
        pooling_factor: int = 2
    ):
        super().__init__()

        self.num_scales = num_scales
        self.pooling_factor = pooling_factor

        # Create angle estimators for each scale
        # Using ModuleList to ensure proper registration
        self.estimators = nn.ModuleList([
            AngleEstimator(patch_size=patch_size, sigma=sigma)
            for _ in range(num_scales)
        ])

    def forward(
        self,
        image: torch.Tensor,
        scores: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute multi-scale orientation with optional score-based selection.

        **WARNING: Score-based selection is not implemented yet.**
        This method currently always returns orientations from scale 0.

        Args:
            image: Input image of shape (N, 1, H, W).
            scores: Optional feature scores of shape (N, 1, H, W).
                   **Currently ignored - not implemented.**

        Returns:
            Tuple of:
                - orientations: Orientation map from scale 0 (N, 1, H, W).
                - scale_indices: All zeros (N, 1, H, W) - selection not implemented.
        """
        orientation_list = []
        current_image = image

        # Compute orientations at each scale
        for i in range(self.num_scales):
            # Compute orientation at current scale
            orientation = self.estimators[i](current_image)

            # Upsample back to original resolution if needed
            if i > 0:
                orientation = F.interpolate(
                    orientation,
                    size=(image.shape[2], image.shape[3]),
                    mode='nearest'
                )

            orientation_list.append(orientation)

            # Downsample for next scale (except last iteration)
            if i < self.num_scales - 1:
                current_image = F.avg_pool2d(
                    current_image,
                    kernel_size=self.pooling_factor,
                    stride=self.pooling_factor
                )

        # Stack all orientations: (num_scales, N, 1, H, W)
        all_orientations = torch.stack(orientation_list, dim=0)

        # NOTE: Multi-scale selection logic is not implemented yet.
        # Currently always returns scale 0 regardless of scores parameter.
        # TODO: Implement proper scale selection based on feature scores:
        #   1. Compute or accept pre-computed scores at each scale
        #   2. Find scale with maximum score at each pixel
        #   3. Select orientation from that scale
        # For reference, see AKAZE implementation in akaze.py lines 435-453

        if scores is not None:
            import warnings
            warnings.warn(
                "AngleEstimatorMultiScale: score-based selection not implemented. "
                "Returning scale 0 orientations. Use AngleEstimator for production.",
                UserWarning
            )

        # Use first scale (not implemented multi-scale selection)
        selected_orientations = all_orientations[0]
        scale_indices = torch.zeros_like(all_orientations[0])

        return selected_orientations, scale_indices
