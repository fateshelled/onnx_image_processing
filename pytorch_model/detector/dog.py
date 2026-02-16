import math
import torch
from torch import nn
import torch.nn.functional as F


def create_gaussian_kernel(sigma: float, kernel_size: int) -> torch.Tensor:
    """
    Create a 2D Gaussian kernel.

    Args:
        sigma: Standard deviation of the Gaussian.
        kernel_size: Size of the kernel (must be odd).

    Returns:
        Gaussian kernel of shape (1, 1, kernel_size, kernel_size).
    """
    half = kernel_size // 2
    x = torch.arange(-half, half + 1, dtype=torch.float32)
    y = torch.arange(-half, half + 1, dtype=torch.float32)

    # Create meshgrid
    yy, xx = torch.meshgrid(y, x, indexing='ij')

    # Compute Gaussian
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()

    return kernel.unsqueeze(0).unsqueeze(0)


class DoGDetector(nn.Module):
    """
    Difference of Gaussians (DoG) feature point detector.

    The DoG detector approximates the Laplacian of Gaussian (LoG) by computing
    the difference between two Gaussian-blurred versions of an image at different
    scales. This is computationally efficient and suitable for ONNX export.

    The implementation uses fixed scales and static operations to ensure
    compatibility with ONNX export and efficient parallel processing on GPU.

    Args:
        num_scales: Number of scale levels in the pyramid. Default is 5.
        sigma_base: Base sigma value for the first scale. Default is 1.6.
        sigma_ratio: Ratio between consecutive sigma values. Default is sqrt(2).
        kernel_size: Size of the Gaussian kernel. If None, computed as
                     6 * sigma_max + 1. Default is None.

    Example:
        >>> detector = DoGDetector(num_scales=5, sigma_base=1.6)
        >>> image = torch.rand(1, 1, 256, 256)
        >>> dog_responses = detector(image)
        >>> print(dog_responses.shape)  # (1, 4, 256, 256)
    """

    def __init__(
        self,
        num_scales: int = 5,
        sigma_base: float = 1.6,
        sigma_ratio: float = math.sqrt(2),
        kernel_size: int = None,
    ) -> None:
        super().__init__()

        if num_scales < 2:
            raise ValueError(f"num_scales must be at least 2, got {num_scales}")

        self.num_scales = num_scales
        self.sigma_base = sigma_base
        self.sigma_ratio = sigma_ratio

        # Compute sigma values for each scale
        self.sigmas = [sigma_base * (sigma_ratio ** i) for i in range(num_scales)]

        # Determine kernel size
        if kernel_size is None:
            max_sigma = self.sigmas[-1]
            kernel_size = int(6 * max_sigma + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1

        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        # Create Gaussian kernels for each scale
        # Stack all kernels into a single tensor for efficient processing
        kernels = []
        for sigma in self.sigmas:
            kernel = create_gaussian_kernel(sigma, kernel_size)
            kernels.append(kernel)

        # Shape: (num_scales, 1, kernel_size, kernel_size)
        gaussian_kernels = torch.cat(kernels, dim=0)
        self.register_buffer('gaussian_kernels', gaussian_kernels)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Compute Difference of Gaussians response maps.

        Args:
            image: Input grayscale image tensor of shape (N, 1, H, W).
                   Values should be in range [0, 255] or [0, 1].

        Returns:
            dog_responses: DoG response maps of shape (N, num_scales-1, H, W).
                          Each channel represents the difference between
                          consecutive Gaussian-blurred scales. Higher absolute
                          values indicate stronger blob-like features.
        """
        N, C, H, W = image.shape

        if C != 1:
            raise ValueError(f"Input must be grayscale (1 channel), got {C} channels")

        img = image.float()

        # Pad the image once for all convolutions
        img_padded = F.pad(img, (self.padding, self.padding, self.padding, self.padding), mode='replicate')

        # Apply all Gaussian filters in parallel
        # Repeat input for each scale: (N, num_scales, H_pad, W_pad)
        img_repeated = img_padded.repeat(1, self.num_scales, 1, 1)

        # Apply grouped convolution to process all scales in parallel
        # Each group processes one scale with its corresponding Gaussian kernel
        gaussian_pyramid = F.conv2d(
            img_repeated,
            self.gaussian_kernels,
            groups=self.num_scales
        )  # Shape: (N, num_scales, H, W)

        # Compute Difference of Gaussians between consecutive scales
        # DoG[i] = Gaussian[i+1] - Gaussian[i]
        # We compute all differences in parallel using slicing
        dog_responses = gaussian_pyramid[:, 1:, :, :] - gaussian_pyramid[:, :-1, :, :]
        # Shape: (N, num_scales-1, H, W)

        return dog_responses


class DoGDetectorWithScore(nn.Module):
    """
    DoG detector that outputs a single score map combining all scale responses.

    This variant computes the maximum absolute response across all DoG scales
    to produce a single-channel score map suitable for keypoint detection.

    Args:
        num_scales: Number of scale levels in the pyramid. Default is 5.
        sigma_base: Base sigma value for the first scale. Default is 1.6.
        sigma_ratio: Ratio between consecutive sigma values. Default is sqrt(2).
        kernel_size: Size of the Gaussian kernel. If None, computed as
                     6 * sigma_max + 1. Default is None.

    Example:
        >>> detector = DoGDetectorWithScore(num_scales=5)
        >>> image = torch.rand(1, 1, 256, 256)
        >>> score_map = detector(image)
        >>> print(score_map.shape)  # (1, 1, 256, 256)
    """

    def __init__(
        self,
        num_scales: int = 5,
        sigma_base: float = 1.6,
        sigma_ratio: float = math.sqrt(2),
        kernel_size: int = None,
    ) -> None:
        super().__init__()

        self.dog_detector = DoGDetector(
            num_scales=num_scales,
            sigma_base=sigma_base,
            sigma_ratio=sigma_ratio,
            kernel_size=kernel_size,
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Compute DoG score map.

        Args:
            image: Input grayscale image tensor of shape (N, 1, H, W).
                   Values should be in range [0, 255] or [0, 1].

        Returns:
            score_map: DoG score map of shape (N, 1, H, W).
                      Higher values indicate stronger blob-like features
                      (maximal response across all scales).
        """
        # Get DoG responses for all scales
        dog_responses = self.dog_detector(image)  # (N, num_scales-1, H, W)

        # Take absolute values to detect both dark and bright blobs
        abs_responses = torch.abs(dog_responses)

        # Compute maximum response across all scales
        score_map, _ = torch.max(abs_responses, dim=1, keepdim=True)  # (N, 1, H, W)

        return score_map
