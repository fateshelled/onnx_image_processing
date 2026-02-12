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

from ..corner.shi_tomasi import ShiTomasiScore
from ..orientation.angle_estimation import AngleEstimator


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
    Complete pipeline: Shi-Tomasi + Angle + Sparse BAD descriptor.

    This module implements the feature detection and description pipeline:
        1. Shi-Tomasi corner detection
        2. Angle estimation for rotation invariance
        3. Sparse BAD descriptor computation with rotation compensation

    This is similar to AKAZESparseBADSinkhornMatcher but uses Shi-Tomasi
    instead of AKAZE for feature detection.

    Args:
        block_size: Shi-Tomasi block size (default: 5).
        patch_size: Angle estimation patch size (default: 15).
        sigma: Angle estimation sigma (default: 2.5).
        descriptor_mode: BAD descriptor output mode ('raw', 'hard', 'soft').
        temperature: Temperature for soft binary encoding (default: 1.0).

    Example:
        >>> model = ShiTomasiAngleSparseBAD()
        >>> img1 = torch.randn(1, 1, 480, 640)
        >>> img2 = torch.randn(1, 1, 480, 640)
        >>>
        >>> # Get keypoints first (using NMS or top-k)
        >>> scores, angles = model.detect_and_orient(img1)
        >>> # ... select top-k keypoints ...
        >>> # Then compute rotation-aware descriptors
        >>> descriptors = model.describe(img1, keypoints, angles)
    """

    def __init__(
        self,
        block_size: int = 5,
        patch_size: int = 15,
        sigma: float = 2.5,
        descriptor_mode: str = 'soft',
        temperature: float = 1.0
    ):
        super().__init__()

        # Feature detection + orientation
        self.detector = ShiTomasiWithAngle(
            block_size=block_size,
            patch_size=patch_size,
            sigma=sigma
        )

        # TODO: Import and initialize BAD descriptor
        # This would require creating a rotation-aware sparse BAD module
        # similar to what's done in akaze_sparse_bad_sinkhorn.py
        self.descriptor_mode = descriptor_mode
        self.temperature = temperature

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

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returns scores and angles.

        Full descriptor computation would require keypoint selection
        and sparse descriptor extraction.
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
