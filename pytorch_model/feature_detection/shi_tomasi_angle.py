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
    **EXPERIMENTAL - INCOMPLETE IMPLEMENTATION**

    Placeholder for future Shi-Tomasi + Angle + Sparse BAD descriptor pipeline.

    This module is intended to implement the complete feature detection and
    description pipeline:
        1. Shi-Tomasi corner detection
        2. Angle estimation for rotation invariance
        3. Sparse BAD descriptor computation with rotation compensation

    **Current Status:**
    - ✓ Feature detection (Shi-Tomasi) - implemented
    - ✓ Angle estimation - implemented
    - ✗ Sparse BAD descriptor - NOT IMPLEMENTED
    - ✗ Descriptor rotation compensation - NOT IMPLEMENTED

    This is intended to be similar to AKAZESparseBADSinkhornMatcher but using
    Shi-Tomasi instead of AKAZE for feature detection.

    Args:
        block_size: Shi-Tomasi block size (default: 5).
        patch_size: Angle estimation patch size (default: 15).
        sigma: Angle estimation sigma (default: 2.5).
        descriptor_mode: BAD descriptor output mode (not used - not implemented).
        temperature: Temperature for soft binary encoding (not used - not implemented).

    Note:
        For production use, manually combine ShiTomasiWithAngle with BAD descriptor.
        See akaze_sparse_bad_sinkhorn.py for reference implementation.

    Example:
        >>> # Currently only supports detect_and_orient()
        >>> model = ShiTomasiAngleSparseBAD()
        >>> img = torch.randn(1, 1, 480, 640)
        >>> scores, angles = model.detect_and_orient(img)
        >>> # Descriptor computation not yet available
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

        # Feature detection + orientation (implemented)
        self.detector = ShiTomasiWithAngle(
            block_size=block_size,
            patch_size=patch_size,
            sigma=sigma
        )

        # Store parameters for future descriptor implementation
        # NOTE: BAD descriptor integration is not yet implemented.
        # TODO: Import and initialize rotation-aware sparse BAD descriptor
        # This would require:
        #   1. Import BADDescriptor from descriptor/bad.py
        #   2. Create sparse sampling logic with keypoint coordinates
        #   3. Implement rotation compensation using angle information
        #   4. Follow pattern from akaze_sparse_bad_sinkhorn.py
        # For now, these parameters are stored but unused.
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
        Forward pass returns scores and angles only.

        **Note:** This does NOT compute descriptors. Full descriptor computation
        is not yet implemented and would require:
        - Keypoint selection (NMS or top-k)
        - Sparse descriptor extraction
        - Rotation compensation using angle information

        Returns:
            Tuple of (scores, angles) both with shape (N, 1, H, W).
        """
        return self.detect_and_orient(image)

    def describe(self, image: torch.Tensor, keypoints: torch.Tensor, angles: torch.Tensor):
        """
        Compute rotation-aware descriptors at keypoints.

        **NOT IMPLEMENTED - This method will raise NotImplementedError.**

        Args:
            image: Input image (N, 1, H, W).
            keypoints: Keypoint coordinates (N, K, 2) as (x, y).
            angles: Orientation angles (N, K) in radians.

        Raises:
            NotImplementedError: This method is not yet implemented.
        """
        raise NotImplementedError(
            "ShiTomasiAngleSparseBAD.describe() is not yet implemented. "
            "This class currently only provides feature detection and angle estimation. "
            "For descriptor computation, please manually combine:\n"
            "  1. ShiTomasiWithAngle for detection and orientation\n"
            "  2. BADDescriptor for descriptor computation\n"
            "  3. Rotation compensation (see akaze_sparse_bad_sinkhorn.py for reference)"
        )


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
