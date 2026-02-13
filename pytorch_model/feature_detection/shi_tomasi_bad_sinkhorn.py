"""
Shi-Tomasi + BAD + Sinkhorn Feature Matcher.

This module provides a unified feature matching model that combines
Shi-Tomasi corner detection, BAD (Box Average Difference) descriptor
extraction, and Sinkhorn matching in a single forward pass. The model
takes two grayscale images as input and outputs matched keypoints with
their matching probability matrix.

Designed for ONNX export as a single integrated model.
"""

import torch
from torch import nn
import torch.nn.functional as F

from pytorch_model.feature_detection.shi_tomasi_bad import ShiTomasiBADDetector
from pytorch_model.matching.sinkhorn import SinkhornMatcher
from pytorch_model.descriptor.bad import extract_descriptors_at_keypoints_subpixel
from pytorch_model.utils import apply_nms_maxpool, select_topk_keypoints


class ShiTomasiBADSinkhornMatcher(nn.Module):
    """
    Unified feature matching model combining Shi-Tomasi corner detection,
    BAD descriptor extraction, and Sinkhorn matching.

    Takes two grayscale images as input and performs end-to-end feature
    matching: detects keypoints using Shi-Tomasi corner detection, extracts
    BAD descriptors at those keypoints, and computes matching probabilities
    using the Sinkhorn algorithm. Keypoint selection uses non-maximum
    suppression (NMS) and top-k filtering.

    This model is fully ONNX-exportable with fixed output shapes, making it
    suitable for deployment on NPU/GPU hardware.

    Args:
        max_keypoints: Maximum number of keypoints to detect per image.
                       Output will be padded to this size.
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
        sinkhorn_iterations: Number of Sinkhorn iterations for matching.
                            Default is 20.
        epsilon: Entropy regularization parameter for Sinkhorn. Default is 1.0.
        unused_score: Score for dustbin entries in Sinkhorn (controls match
                      threshold). Default is 1.0.
        distance_type: Distance metric for Sinkhorn cost matrix. Either 'l1'
                       or 'l2'. Default is 'l2'.
        nms_radius: Radius for non-maximum suppression. A keypoint is kept
                    only if it's the local maximum within a
                    (2*nms_radius+1) x (2*nms_radius+1) neighborhood.
                    Default is 3.
        score_threshold: Minimum corner score threshold for keypoint selection.
                        Keypoints with scores below this are discarded.
                        Default is 0.0 (no filtering).
        normalize_descriptors: If True, L2-normalize descriptors before matching.
                              This is strongly recommended when using raw (non-binary)
                              descriptors to ensure stable matching. Default is True.

    Example:
        >>> model = ShiTomasiBADSinkhornMatcher(max_keypoints=512)
        >>> img1 = torch.randn(1, 1, 480, 640)  # Grayscale image 1
        >>> img2 = torch.randn(1, 1, 480, 640)  # Grayscale image 2
        >>> kpts1, kpts2, probs = model(img1, img2)
        >>> print(kpts1.shape)  # [1, 512, 2]
        >>> print(kpts2.shape)  # [1, 512, 2]
        >>> print(probs.shape)  # [1, 513, 513]
    """

    def __init__(
        self,
        max_keypoints: int,
        block_size: int = 3,
        sobel_size: int = 3,
        num_pairs: int = 256,
        binarize: bool = False,
        soft_binarize: bool = True,
        temperature: float = 10.0,
        sinkhorn_iterations: int = 20,
        epsilon: float = 1.0,
        unused_score: float = 1.0,
        distance_type: str = "l2",
        nms_radius: int = 3,
        score_threshold: float = 0.0,
        normalize_descriptors: bool = True,
    ) -> None:
        super().__init__()

        self.max_keypoints = max_keypoints
        self.nms_radius = nms_radius
        self.score_threshold = score_threshold
        self.normalize_descriptors = normalize_descriptors

        # Feature detector: Shi-Tomasi + BAD
        self.detector = ShiTomasiBADDetector(
            block_size=block_size,
            sobel_size=sobel_size,
            num_pairs=num_pairs,
            binarize=binarize,
            soft_binarize=soft_binarize,
            temperature=temperature,
        )

        # Feature matcher: Sinkhorn
        self.matcher = SinkhornMatcher(
            iterations=sinkhorn_iterations,
            epsilon=epsilon,
            unused_score=unused_score,
            distance_type=distance_type,
        )

    def _extract_descriptors_at_keypoints_batched(
        self,
        descriptor_map: torch.Tensor,
        keypoints: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract descriptors at keypoint locations.

        Uses subpixel interpolation to extract descriptor vectors at the
        specified keypoint coordinates. Handles invalid keypoints (at -1, -1)
        by setting their descriptors to zero vectors.

        Args:
            descriptor_map: Dense descriptor map of shape (B, D, H, W).
            keypoints: Keypoint coordinates of shape (B, K, 2) in (y, x) format.

        Returns:
            Descriptors of shape (B, K, D), with zero vectors for invalid
            keypoints.
        """
        B, D, H, W = descriptor_map.shape

        # Create validity mask (keypoints with y >= 0 are valid)
        valid_mask = (keypoints[:, :, 0] >= 0).float()  # (B, K)

        # Clamp keypoints to valid range for extraction
        # (invalid ones will be masked out anyway)
        y_coords = torch.clamp(keypoints[:, :, 0], min=0.0, max=float(H - 1))
        x_coords = torch.clamp(keypoints[:, :, 1], min=0.0, max=float(W - 1))
        keypoints_clamped = torch.stack([y_coords, x_coords], dim=-1)

        # Extract descriptors using subpixel interpolation
        descriptors = extract_descriptors_at_keypoints_subpixel(
            descriptor_map,
            keypoints_clamped,
        )  # (B, K, D)

        # Zero out descriptors for invalid keypoints
        descriptors = descriptors * valid_mask.unsqueeze(-1)

        return descriptors

    def forward(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Detect keypoints and compute matches between two images.

        Args:
            image1: First grayscale image of shape (B, 1, H, W).
                    Values should be in range [0, 255] or [0, 1].
            image2: Second grayscale image of shape (B, 1, H, W).
                    Values should be in range [0, 255] or [0, 1].

        Returns:
            Tuple of:
                - keypoints1: Detected keypoints in first image of shape
                  (B, K, 2) in (y, x) format. Invalid keypoints are marked
                  with (-1, -1).
                - keypoints2: Detected keypoints in second image of shape
                  (B, K, 2) in (y, x) format. Invalid keypoints are marked
                  with (-1, -1).
                - matching_probs: Matching probability matrix of shape
                  (B, K+1, K+1). Entry [i, j] is the probability that
                  keypoint i in image1 matches keypoint j in image2.
                  The last row/column represents the dustbin (unmatched).
        """
        # 1. Detect features in both images
        scores1, desc_map1 = self.detector(image1)  # (B, 1, H, W), (B, D, H, W)
        scores2, desc_map2 = self.detector(image2)
        scores1 = scores1.squeeze(1)  # (B, H, W)
        scores2 = scores2.squeeze(1)

        # 2. Apply NMS to both score maps
        nms_mask1 = apply_nms_maxpool(scores1, self.nms_radius)
        nms_mask2 = apply_nms_maxpool(scores2, self.nms_radius)

        # 3. Select top-k keypoints from both images
        keypoints1, _ = select_topk_keypoints(
            scores1, nms_mask1, self.max_keypoints, self.score_threshold,
        )
        keypoints2, _ = select_topk_keypoints(
            scores2, nms_mask2, self.max_keypoints, self.score_threshold,
        )

        # 4. Extract descriptors at keypoint locations
        desc1 = self._extract_descriptors_at_keypoints_batched(desc_map1, keypoints1)
        desc2 = self._extract_descriptors_at_keypoints_batched(desc_map2, keypoints2)

        # 5. Normalize descriptors if enabled (recommended for raw descriptors)
        if self.normalize_descriptors:
            desc1 = F.normalize(desc1, p=2, dim=-1)
            desc2 = F.normalize(desc2, p=2, dim=-1)

        # 6. Perform Sinkhorn matching
        matching_probs = self.matcher(desc1, desc2)  # (B, K+1, K+1)

        return keypoints1, keypoints2, matching_probs
