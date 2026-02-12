"""
AKAZE (Accelerated-KAZE) Feature Detector in PyTorch.

This module provides an ONNX-exportable implementation of the AKAZE algorithm
using pure tensor operations. The implementation includes:
- Non-linear diffusion using Fast Explicit Diffusion (FED) scheme
- Hessian-based feature point detection
- Non-maximum suppression using MaxPool2d
- Orientation estimation using intensity centroid method

All operations are implemented using tensor operations without dynamic control
flow (no if statements or for loops in forward pass) for ONNX compatibility.

Reference:
    Fast Explicit Diffusion for Accelerated Features in Nonlinear Scale Spaces
    Pablo F. Alcantarilla, Jesús Nuevo, Adrien Bartoli
    BMVC 2013
"""

import torch
from torch import nn
import torch.nn.functional as F
import math


class NonLinearDiffusion(nn.Module):
    """
    Non-linear diffusion filter using Fast Explicit Diffusion (FED) scheme.

    Implements anisotropic diffusion with Perona-Malik conduction function
    to preserve edges while smoothing. The FED scheme uses a fixed number
    of iterations with predetermined time steps.

    Args:
        num_iterations: Number of diffusion iterations (fixed for ONNX export).
        kappa: Contrast parameter for the conduction function.
               Controls edge preservation sensitivity.

    The diffusion equation is:
        ∂L/∂t = div(c(|∇L|) * ∇L)
    where c(|∇L|) is the conduction function.
    """

    def __init__(self, num_iterations: int = 3, kappa: float = 0.05):
        super().__init__()
        self.num_iterations = num_iterations
        self.kappa = kappa

        # Fused Sobel kernels: 2-output-channel conv computes both gradients
        # in a single kernel launch.
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3) / 8.0

        sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3) / 8.0

        # Gradient: (1ch input) -> (2ch output) via single conv
        self.register_buffer('sobel_xy', torch.cat([sobel_x, sobel_y], dim=0))
        # Divergence: (2ch input) -> (2ch output) via groups=2 conv,
        # then sum channels to get scalar divergence.
        self.register_buffer('sobel_xy_grouped', torch.cat([sobel_x, sobel_y], dim=0))

        # Time step for FED scheme (using fixed value for stability)
        self.dt = 0.25

    def compute_gradient(self, image: torch.Tensor) -> torch.Tensor:
        """
        Compute image gradients using fused Sobel convolution.

        Args:
            image: Input image tensor of shape (N, 1, H, W).

        Returns:
            Gradient tensor of shape (N, 2, H, W) where channel 0 is grad_x
            and channel 1 is grad_y.
        """
        return F.conv2d(image, self.sobel_xy, padding=1)  # (N, 2, H, W)

    def conduction_function(self, gradient_magnitude: torch.Tensor) -> torch.Tensor:
        """
        Perona-Malik conduction function (g2 variant).

        c(|∇L|) = 1 / (1 + (|∇L|/κ)^2)

        Args:
            gradient_magnitude: Magnitude of image gradient, shape (N, 1, H, W).

        Returns:
            Conduction coefficient, shape (N, 1, H, W).
        """
        return 1.0 / (1.0 + (gradient_magnitude / self.kappa) ** 2)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply non-linear diffusion filtering.

        Args:
            image: Input image tensor of shape (N, 1, H, W).

        Returns:
            Diffused image of shape (N, 1, H, W).
        """
        result = image

        # Fixed number of iterations (unrolled for ONNX export)
        for _ in range(self.num_iterations):
            # Compute gradients: single fused conv -> (N, 2, H, W)
            grads = self.compute_gradient(result)

            # Compute gradient magnitude from 2-channel gradient
            grad_mag = torch.sqrt((grads * grads).sum(dim=1, keepdim=True) + 1e-8)

            # Compute conduction coefficients
            c = self.conduction_function(grad_mag)  # (N, 1, H, W)

            # Compute diffusion flux: c * ∇L (broadcast c over 2 grad channels)
            flux = c * grads  # (N, 2, H, W)

            # Compute divergence via groups=2 conv (fused) + channel sum
            div_xy = F.conv2d(flux, self.sobel_xy_grouped, padding=1, groups=2)
            divergence = div_xy.sum(dim=1, keepdim=True)  # (N, 1, H, W)

            # Update: L_new = L_old + dt * div(c * ∇L)
            result = result + self.dt * divergence

        return result


class HessianDetector(nn.Module):
    """
    Hessian-based feature point detector.

    Computes the determinant of the Hessian matrix for feature point detection.
    The Hessian matrix captures second-order intensity variations.

    Args:
        threshold: Response threshold for feature detection.
        nms_size: Size of the non-maximum suppression window (must be odd).
    """

    def __init__(self, threshold: float = 0.001, nms_size: int = 5):
        super().__init__()
        self.threshold = threshold
        self.nms_size = nms_size

        # Fused second derivative kernels: single 3-output-channel conv
        # computes Lxx, Lyy, Lxy in one kernel launch.
        kernel_xx = torch.tensor([
            [1, -2, 1],
            [2, -4, 2],
            [1, -2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3) / 16.0

        kernel_yy = torch.tensor([
            [1, 2, 1],
            [-2, -4, -2],
            [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3) / 16.0

        kernel_xy = torch.tensor([
            [1, 0, -1],
            [0, 0, 0],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3) / 4.0

        self.register_buffer('hessian_kernels', torch.cat([kernel_xx, kernel_yy, kernel_xy], dim=0))

    def compute_hessian_response(self, image: torch.Tensor) -> torch.Tensor:
        """
        Compute Hessian determinant response.

        The Hessian matrix is:
            H = [Lxx  Lxy]
                [Lxy  Lyy]

        Response = det(H) = Lxx * Lyy - Lxy^2

        Args:
            image: Input image tensor of shape (N, 1, H, W).

        Returns:
            Hessian response map of shape (N, 1, H, W).
        """
        # Single fused conv: (N, 1, H, W) -> (N, 3, H, W) [Lxx, Lyy, Lxy]
        hessian = F.conv2d(image, self.hessian_kernels, padding=1)
        Lxx = hessian[:, 0:1]
        Lyy = hessian[:, 1:2]
        Lxy = hessian[:, 2:3]

        # Compute determinant: det(H) = Lxx * Lyy - Lxy^2
        response = Lxx * Lyy - Lxy * Lxy

        return response

    def non_maximum_suppression(self, response: torch.Tensor) -> torch.Tensor:
        """
        Apply non-maximum suppression using MaxPool2d.

        A pixel is a local maximum if it equals the maximum value in its
        neighborhood after max pooling.

        Args:
            response: Response map of shape (N, 1, H, W).

        Returns:
            Binary mask of local maxima, shape (N, 1, H, W).
        """
        # Apply max pooling
        padding = self.nms_size // 2
        max_pooled = F.max_pool2d(
            response,
            kernel_size=self.nms_size,
            stride=1,
            padding=padding
        )

        # A pixel is a local maximum if it equals the max-pooled value
        local_maxima = (response == max_pooled).float()

        return local_maxima

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Detect feature points using Hessian response.

        Args:
            image: Input image tensor of shape (N, 1, H, W).

        Returns:
            Feature point score map of shape (N, 1, H, W).
            Higher values indicate stronger feature responses.
        """
        # Compute Hessian response
        response = self.compute_hessian_response(image)

        # Apply non-maximum suppression
        local_maxima = self.non_maximum_suppression(response)

        # Apply threshold and combine with local maxima
        above_threshold = (response > self.threshold).float()
        feature_mask = local_maxima * above_threshold

        # Return weighted response (score map)
        scores = response * feature_mask

        # Clamp to non-negative values
        scores = torch.clamp(scores, min=0.0)

        return scores


class OrientationEstimator(nn.Module):
    """
    Estimate dominant orientation using intensity centroid method.

    Computes the orientation at each pixel using Gaussian-weighted
    intensity moments. The orientation is computed in parallel for
    all pixels using tensor operations.

    Args:
        patch_size: Size of the patch for orientation computation (must be odd).
        sigma: Standard deviation of the Gaussian weighting.
    """

    def __init__(self, patch_size: int = 15, sigma: float = 2.5):
        super().__init__()
        self.patch_size = patch_size
        self.sigma = sigma

        # Create Gaussian-weighted coordinate grids
        half_size = patch_size // 2
        y, x = torch.meshgrid(
            torch.arange(-half_size, half_size + 1, dtype=torch.float32),
            torch.arange(-half_size, half_size + 1, dtype=torch.float32),
            indexing='ij'
        )

        # Compute Gaussian weights
        gaussian = torch.exp(-(x**2 + y**2) / (2 * sigma**2))

        # Fused moment kernels: single 2-output-channel conv computes
        # both m10 and m01 in one kernel launch.
        weight_x = (x * gaussian).view(1, 1, patch_size, patch_size)
        weight_y = (y * gaussian).view(1, 1, patch_size, patch_size)

        self.register_buffer('moment_kernels', torch.cat([weight_x, weight_y], dim=0))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Compute orientation map for the entire image.

        The orientation is computed as:
            θ = atan2(m01, m10)
        where m10 and m01 are the first-order moments.

        Args:
            image: Input image tensor of shape (N, 1, H, W).

        Returns:
            Orientation map of shape (N, 1, H, W) in radians [-π, π].
        """
        padding = self.patch_size // 2

        # Single fused conv: (N, 1, H, W) -> (N, 2, H, W) [m10, m01]
        moments = F.conv2d(image, self.moment_kernels, padding=padding)

        # Compute orientation: θ = atan2(m01, m10)
        orientation = torch.atan2(moments[:, 1:2], moments[:, 0:1])

        return orientation


class AKAZE(nn.Module):
    """
    Complete AKAZE feature detector with ONNX export support.

    Implements the full AKAZE pipeline:
    1. Non-linear diffusion for scale-space construction
    2. Hessian-based feature point detection with NMS
    3. Orientation estimation using intensity centroid

    All operations use pure tensor operations without dynamic control flow
    for full ONNX compatibility.

    Args:
        num_scales: Number of scale levels (diffusion iterations per scale).
        diffusion_iterations: Number of FED iterations per scale.
        kappa: Contrast parameter for diffusion (edge preservation).
        threshold: Feature detection threshold.
        nms_size: Non-maximum suppression window size (must be odd).
        orientation_patch_size: Patch size for orientation computation (must be odd).
        orientation_sigma: Gaussian sigma for orientation weighting.

    Example:
        >>> # Compatible with Shi-Tomasi interface
        >>> model = AKAZE(num_scales=3, threshold=0.001)
        >>> img = torch.randn(1, 1, 480, 640)
        >>> scores, orientations = model(img)
        >>> print(scores.shape)        # [1, 1, 480, 640] - same as Shi-Tomasi
        >>> print(orientations.shape)  # [1, 1, 480, 640] - extra angle info
    """

    def __init__(
        self,
        num_scales: int = 3,
        diffusion_iterations: int = 3,
        kappa: float = 0.05,
        threshold: float = 0.001,
        nms_size: int = 5,
        orientation_patch_size: int = 15,
        orientation_sigma: float = 2.5,
    ):
        super().__init__()

        self.num_scales = num_scales

        # Create diffusion modules for each scale
        # Using ModuleList to ensure proper registration
        self.diffusion_layers = nn.ModuleList([
            NonLinearDiffusion(
                num_iterations=diffusion_iterations,
                kappa=kappa
            )
            for _ in range(num_scales)
        ])

        # Feature detector
        self.detector = HessianDetector(
            threshold=threshold,
            nms_size=nms_size
        )

        # Orientation estimator
        self.orientation_estimator = OrientationEstimator(
            patch_size=orientation_patch_size,
            sigma=orientation_sigma
        )

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Detect AKAZE features and compute orientations.

        This method follows the same interface as Shi-Tomasi corner detector
        but additionally provides orientation information. This makes it easy
        to swap between different detectors or combine with other modules.

        Args:
            image: Input grayscale image tensor of shape (N, 1, H, W).
                   Values should be in range [0, 255] or [0, 1].

        Returns:
            Tuple of:
                - scores: Feature point score map of shape (N, 1, H, W).
                  Higher values indicate stronger feature responses.
                  Compatible with Shi-Tomasi score output format.
                - orientations: Dominant orientation map of shape (N, 1, H, W).
                  Values in radians [-π, π]. Each pixel's orientation
                  corresponds to the scale where maximum response was detected.
                  This is the additional information not available in Shi-Tomasi.

        Example:
            >>> model = AKAZE()
            >>> img = torch.randn(1, 1, 480, 640)
            >>> scores, orientations = model(img)
            >>> # scores can be used like Shi-Tomasi scores
            >>> # orientations provide extra rotation information
        """
        # Lists to store scores and orientations at each scale
        scale_scores_list = []
        scale_orientations_list = []

        current_scale = image

        # Process each scale (unrolled loop using ModuleList)
        for i in range(self.num_scales):
            # Apply non-linear diffusion
            current_scale = self.diffusion_layers[i](current_scale)

            # Detect features at this scale
            scale_scores = self.detector(current_scale)

            # Compute orientations at this scale
            scale_orientations = self.orientation_estimator(current_scale)

            # Store for later selection
            scale_scores_list.append(scale_scores)
            scale_orientations_list.append(scale_orientations)

        # Stack all scales along a new dimension
        # Shape: (num_scales, N, 1, H, W)
        all_scores = torch.stack(scale_scores_list, dim=0)
        all_orientations = torch.stack(scale_orientations_list, dim=0)

        # Find the maximum score at each pixel across scales.
        # amax produces only ReduceMax (no ArgMax) in ONNX, which is
        # fully supported by TensorRT.
        scores = all_scores.amax(dim=0)  # (N, 1, H, W)

        # Select orientations from the scale with maximum response.
        # Build a selection mask by comparing each scale to the max,
        # avoiding ArgMax + one_hot which TensorRT cannot parse.
        mask = (all_scores == scores.unsqueeze(0)).float()  # (num_scales, N, 1, H, W)
        # Normalize to handle ties (e.g. multiple scales at 0)
        mask = mask / mask.sum(dim=0, keepdim=True).clamp(min=1.0)

        orientations = (all_orientations * mask).sum(dim=0)  # (N, 1, H, W)

        return scores, orientations
