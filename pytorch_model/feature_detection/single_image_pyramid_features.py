"""
Single-image feature extractor built on the pyramid Sinkhorn matcher.

The pair model (``ShiTomasiAngleSparseBADSinkhornMatcherPyramid``) detects
keypoints and computes descriptors independently per image, then runs one
Sinkhorn pass over the concatenated descriptors of both images. This
wrapper exposes the *per-image* part only: one image in, fixed set of
keypoints + descriptors out.

This is used to cache per-frame features (keypoint detection cost is
O(K) frames instead of O(K^2) pairs); matching is then done separately
(e.g. numpy Sinkhorn re-implementation) on the cached descriptors.

Static-shape friendly for ONNX export (fixed K, no dynamic branching).
"""

import torch
from torch import nn


class SingleImagePyramidFeatures(nn.Module):
    """Extract(keypoints, descriptors) for a single image.

    Wraps an existing ``ShiTomasiAngleSparseBADSinkhornMatcherPyramid``
    instance and reuses its detector / descriptor / sharpener so exported
    weights stay identical to the pair model.

    Example:
        >>> pair = ShiTomasiAngleSparseBADSinkhornMatcherPyramid(
        ...     max_keypoints=512, num_levels=2)
        >>> model = SingleImagePyramidFeatures(pair)
        >>> kps, descs = model(torch.randn(1, 1, 480, 640))
        >>> kps.shape   # [1, 512, 2]  (y, x), full-resolution; invalid (-1,-1)
        >>> descs.shape # [1, 512, num_pairs]
    """

    def __init__(self, matcher: nn.Module) -> None:
        super().__init__()
        self.model = matcher

    def forward(
        self,
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            image: Grayscale image of shape (B, 1, H, W).

        Returns:
            Tuple of:
                - keypoints: (B, K, 2) in (y, x) format, full-resolution
                  coordinates. Invalid keypoints are (-1, -1).
                - descriptors: (B, K, num_pairs).
        """
        if self.model.sharpener is not None:
            image = self.model.sharpener(image)

        kps: list[torch.Tensor] = []
        ds: list[torch.Tensor] = []
        for level in range(self.model.num_levels):
            kp_l, d_l, _ = self.model._level_keypoints(image, level)
            kps.append(kp_l)
            ds.append(d_l)

        keypoints = torch.cat(kps, dim=1)
        descriptors = torch.cat(ds, dim=1)
        return keypoints, descriptors


class PairWithDescriptors(nn.Module):
    """Pair model that also exposes the per-image descriptors.

    Wraps ``ShiTomasiAngleSparseBADSinkhornMatcherPyramid`` and returns the
    standard outputs *plus* the descriptors the matcher already computes
    internally (``desc1`` / ``desc2``). No extra computation: descriptors
    are detected/described per image independently inside the pair model
    anyway, so emission is free.

    Example:
        >>> pair = ShiTomasiAngleSparseBADSinkhornMatcherPyramid(
        ...     max_keypoints=512, num_levels=2)
        >>> model = PairWithDescriptors(pair)
        >>> k1, k2, d1, d2, P = model(img1, img2)
        >>> d1.shape  # [1, 512, num_pairs]
        >>> P.shape   # [1, 513, 513]
    """

    def __init__(self, matcher: nn.Module) -> None:
        super().__init__()
        self.model = matcher

    def forward(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            image1/image2: Grayscale images (B, 1, H, W).

        Returns:
            Tuple of:
                - keypoints1: (B, K, 2) (y, x), full-resolution; invalid (-1,-1)
                - keypoints2: same for image2
                - descriptors1: (B, K, num_pairs)
                - descriptors2: (B, K, num_pairs)
                - matching_probs: (B, K+1, K+1)
        """
        if self.model.sharpener is not None:
            image1 = self.model.sharpener(image1)
            image2 = self.model.sharpener(image2)

        kps1: list[torch.Tensor] = []
        ds1: list[torch.Tensor] = []
        kps2: list[torch.Tensor] = []
        ds2: list[torch.Tensor] = []
        for level in range(self.model.num_levels):
            kp1_l, d1_l, _ = self.model._level_keypoints(image1, level)
            kp2_l, d2_l, _ = self.model._level_keypoints(image2, level)
            kps1.append(kp1_l)
            ds1.append(d1_l)
            kps2.append(kp2_l)
            ds2.append(d2_l)

        keypoints1 = torch.cat(kps1, dim=1)
        keypoints2 = torch.cat(kps2, dim=1)
        desc1 = torch.cat(ds1, dim=1)
        desc2 = torch.cat(ds2, dim=1)
        matching_probs = self.model.matcher(desc1, desc2)
        return keypoints1, keypoints2, desc1, desc2, matching_probs
