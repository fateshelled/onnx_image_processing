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

from ..detector.shi_tomasi import ShiTomasiScore
from ..orientation.angle_estimation import AngleEstimator
from ..descriptor.bad import SparseBAD
from ..utils import apply_nms_maxpool, select_topk_keypoints


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

        # Feature detection + orientation
        self.detector = ShiTomasiWithAngle(
            block_size=block_size,
            patch_size=patch_size,
            sigma=sigma
        )

        # Sparse BAD descriptor computation with orientation support
        self.descriptor = SparseBAD(
            num_pairs=num_pairs,
            binarize=binarize,
            soft_binarize=soft_binarize,
            temperature=temperature,
            normalize_descriptors=normalize_descriptors,
            sampling_mode=sampling_mode,
        )

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
        return self.descriptor(image, keypoints, orientation)

    def forward(
        self,
        image: torch.Tensor,
        keypoints: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Detect features, estimate orientations, and compute descriptors.

        Args:
            image: Input grayscale image of shape (B, 1, H, W).
            keypoints: Keypoint coordinates of shape (B, K, 2) in (y, x) format.

        Returns:
            Tuple of:
                - scores: Feature point scores of shape (B, 1, H, W).
                - angles: Orientation map of shape (B, 1, H, W) in radians.
                - descriptors: Rotation-aware BAD descriptors of shape
                  (B, K, num_pairs) at keypoint locations.
        """
        # Detect feature points and compute orientations
        scores, angles = self.detect_and_orient(image)

        # Compute rotation-aware BAD descriptors at keypoint locations
        descriptors = self.describe(image, keypoints, angles)

        return scores, angles, descriptors


class ShiTomasiAngleSparseBADDetector(nn.Module):
    """
    Complete detector with NMS and top-k selection for ONNX export.

    This module provides a single forward pass that:
    1. Detects features and computes orientations
    2. Applies NMS and selects top-k keypoints
    3. Computes rotation-aware descriptors at selected keypoints

    Designed for single-image ONNX export with end-to-end detection and
    description. For two-image matching with Sinkhorn, see
    ShiTomasiAngleSparseBADSinkhornMatcher.

    Args:
        max_keypoints: Maximum number of keypoints to detect per image.
                       Output will be padded to this size.
        block_size: Block size for Shi-Tomasi corner detection (must be odd).
                   Default is 5.
        patch_size: Patch size for angle estimation (must be odd). Default is 15.
        sigma: Gaussian sigma for angle estimation. Default is 2.5.
        num_pairs: Number of BAD descriptor comparison pairs (descriptor
                   dimensionality). Must be 256 or 512. Default is 256.
        binarize: If True, output binarized BAD descriptors. Default is False.
        soft_binarize: If True and binarize=True, use sigmoid for soft
                       binarization. Default is True.
        temperature: Temperature for soft sigmoid binarization. Default is 10.0.
        normalize_descriptors: If True, L2-normalize descriptors before
                              matching. Default is True.
        sampling_mode: Sampling mode for descriptor grid sampling.
                       Choose 'nearest' or 'bilinear'. Default is 'nearest'.
        nms_radius: Radius for non-maximum suppression. Default is 3.
        score_threshold: Minimum score threshold for keypoint selection.
                        Default is 0.0.

    Example:
        >>> detector = ShiTomasiAngleSparseBADDetector(max_keypoints=512)
        >>> img = torch.randn(1, 1, 480, 640)
        >>> kpts, scores, desc = detector(img)
        >>> print(kpts.shape)   # [1, 512, 2]
        >>> print(scores.shape) # [1, 512]
        >>> print(desc.shape)   # [1, 512, 256]
    """

    def __init__(
        self,
        max_keypoints: int,
        block_size: int = 5,
        patch_size: int = 15,
        sigma: float = 2.5,
        num_pairs: int = 256,
        binarize: bool = False,
        soft_binarize: bool = True,
        temperature: float = 10.0,
        normalize_descriptors: bool = True,
        sampling_mode: str = "nearest",
        nms_radius: int = 3,
        score_threshold: float = 0.0,
    ) -> None:
        super().__init__()

        self.max_keypoints = max_keypoints
        self.nms_radius = nms_radius
        self.score_threshold = score_threshold

        # Feature detection + orientation + description
        self.model = ShiTomasiAngleSparseBAD(
            block_size=block_size,
            patch_size=patch_size,
            sigma=sigma,
            num_pairs=num_pairs,
            binarize=binarize,
            soft_binarize=soft_binarize,
            temperature=temperature,
            normalize_descriptors=normalize_descriptors,
            sampling_mode=sampling_mode,
        )

    def forward(
        self,
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Detect keypoints and compute descriptors for a single image.

        Args:
            image: Input grayscale image of shape (B, 1, H, W).

        Returns:
            Tuple of:
                - keypoints: Detected keypoints of shape (B, K, 2) in (y, x)
                  format. Invalid keypoints are (-1, -1).
                - scores: Keypoint scores of shape (B, K).
                - descriptors: Rotation-aware BAD descriptors of shape
                  (B, K, num_pairs) at keypoint locations.
        """
        # 1. Detect features and compute orientations
        score_map, angles = self.model.detect_and_orient(image)
        score_map = score_map.squeeze(1)  # (B, H, W)

        # 2. Apply NMS
        nms_mask = apply_nms_maxpool(score_map, self.nms_radius)

        # 3. Select top-k keypoints
        keypoints, scores = select_topk_keypoints(
            score_map, nms_mask, self.max_keypoints, self.score_threshold
        )

        # 4. Compute rotation-aware descriptors
        descriptors = self.model.describe(image, keypoints, angles)

        return keypoints, scores, descriptors


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
