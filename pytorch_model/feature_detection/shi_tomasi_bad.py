"""
Shi-Tomasi + BAD Feature Detector.

This module provides a unified feature detection model that combines
Shi-Tomasi corner detection with BAD (Box Average Difference) descriptor
extraction. The model takes a grayscale image as input and outputs
corner scores and dense descriptors in a single forward pass.

Designed for ONNX export as a single integrated model.
"""

import torch
from torch import nn
import torch.nn.functional as F

from pytorch_model.detector.shi_tomasi import ShiTomasiScore
from pytorch_model.descriptor.bad import BADDescriptor


class ShiTomasiBADDetector(nn.Module):
    """
    Unified feature detection model combining Shi-Tomasi corner detection
    and BAD descriptor extraction.

    Runs both Shi-Tomasi score computation and BAD descriptor extraction
    on the same input image, returning corner scores and dense descriptors
    in a single forward pass. This is more efficient than running two
    separate models and ensures consistent processing.

    Args:
        block_size: Block size for Shi-Tomasi structure tensor computation.
                    Must be a positive odd integer. Default is 3.
        sobel_size: Sobel kernel size for gradient computation.
                    Currently only supports 3. Default is 3.
        num_pairs: Number of BAD descriptor comparison pairs (descriptor
                   dimensionality). Default is 256.
        binarize: If True, output binarized BAD descriptors. Default is False.
        soft_binarize: If True and binarize=True, use sigmoid for soft
                       binarization. Default is True.
        temperature: Temperature for soft sigmoid binarization. Default is 10.0.

    Example:
        >>> model = ShiTomasiBADDetector(block_size=3, num_pairs=256)
        >>> img = torch.randn(1, 1, 480, 640)  # Grayscale image
        >>> scores, descriptors = model(img)
        >>> print(scores.shape)       # [1, 1, 480, 640]
        >>> print(descriptors.shape)  # [1, 256, 480, 640]
    """

    def __init__(
        self,
        block_size: int = 3,
        sobel_size: int = 3,
        num_pairs: int = 256,
        binarize: bool = False,
        soft_binarize: bool = True,
        temperature: float = 10.0,
    ) -> None:
        super().__init__()

        self.corner_detector = ShiTomasiScore(
            block_size=block_size,
            sobel_size=sobel_size,
        )
        self.descriptor = BADDescriptor(
            num_pairs=num_pairs,
            binarize=binarize,
            soft_binarize=soft_binarize,
            temperature=temperature,
        )

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Detect corners and compute descriptors for the input image.

        Args:
            image: Input grayscale image tensor of shape (N, 1, H, W).
                   Values should be in range [0, 255] or [0, 1].

        Returns:
            Tuple of:
                - scores: Shi-Tomasi corner score map of shape (N, 1, H, W).
                  Higher values indicate stronger corner responses.
                - descriptors: Dense BAD descriptor map of shape
                  (N, num_pairs, H, W).
        """
        scores = self.corner_detector(image)
        descriptors = self.descriptor(image)
        return scores, descriptors
