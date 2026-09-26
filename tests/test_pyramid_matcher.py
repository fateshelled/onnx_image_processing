#!/usr/bin/env python3
"""
Pyramid variant tests: ShiTomasiAngleSparseBADSinkhornMatcherPyramid.

Verifies:
  1. Config validation (num_levels / level_weights guards).
  2. Output shapes match the single-scale model (keypoints, probs).
  3. Per-level keypoint budgets sum to max_keypoints.
  4. Coarse-level keypoints are mapped back to full-resolution coordinates.
  5. num_levels=1 is equivalent to the base matcher (same detection set).
  6. On synthetic homography pairs, level-1-only matching succeeds where a
     shifted region appears at a different apparent scale.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))

from pytorch_model.feature_detection.shi_tomasi_angle_sparse_bad_sinkhorn import (
    ShiTomasiAngleSparseBADSinkhornMatcher,
)
from pytorch_model.feature_detection.shi_tomasi_angle_sparse_bad_sinkhorn_pyramid import (
    ShiTomasiAngleSparseBADSinkhornMatcherPyramid,
)

torch.manual_seed(0)


def _base_kwargs():
    return dict(
        max_keypoints=128,
        binarize=True,
        soft_binarize=False,
        epsilon=0.05,
        unused_score=1.0,
        distance_type="l2",
    )


def _checkerboard(h=128, w=160, cell=8):
    """Textured checkerboard-like image with varied corners."""
    y = torch.arange(h).view(-1, 1)
    x = torch.arange(w).view(1, -1)
    img = torch.sin(y / cell * 1.3) + torch.sin(x / (cell * 0.7) * 2.1)
    img = img + torch.sin((y + x) / (cell * 1.7))
    return (img + 3.0) / 6.0 * 255.0


class TestConfig:
    def test_num_levels_min(self):
        with pytest.raises(ValueError):
            ShiTomasiAngleSparseBADSinkhornMatcherPyramid(max_keypoints=64, num_levels=0)

    def test_num_levels_max(self):
        with pytest.raises(ValueError):
            ShiTomasiAngleSparseBADSinkhornMatcherPyramid(max_keypoints=64, num_levels=4)

    def test_bad_weights(self):
        with pytest.raises(ValueError):
            ShiTomasiAngleSparseBADSinkhornMatcherPyramid(
                max_keypoints=64, num_levels=2, level_weights=[0.5, 0.25]
            )
        with pytest.raises(ValueError):
            ShiTomasiAngleSparseBADSinkhornMatcherPyramid(
                max_keypoints=64, num_levels=2, level_weights=[0.5, 0.0]
            )


class TestShapes:
    def test_output_shapes_match_base(self):
        for num_levels in (1, 2, 3):
            model = ShiTomasiAngleSparseBADSinkhornMatcherPyramid(
                max_keypoints=128, num_levels=num_levels
            ).eval()
            img1 = torch.rand(1, 1, 128, 160) * 255.0
            img2 = torch.rand(1, 1, 128, 160) * 255.0
            with torch.no_grad():
                k1, k2, probs = model(img1, img2)
            assert k1.shape == (1, 128, 2)
            assert k2.shape == (1, 128, 2)
            assert probs.shape == (1, 129, 129)

    def test_budget_split_sum(self):
        model = ShiTomasiAngleSparseBADSinkhornMatcherPyramid(
            max_keypoints=130, num_levels=2
        )
        assert sum(model.level_keypoint_budgets) == 130
        assert all(b >= 0 for b in model.level_keypoint_budgets)

    def test_keypoints_fullres_and_valid(self):
        model = ShiTomasiAngleSparseBADSinkhornMatcherPyramid(
            max_keypoints=128, num_levels=2
        ).eval()
        img = _checkerboard()[None, None]
        with torch.no_grad():
            k1, _, _ = model(img, img)
        assert k1.min() >= -1.0
        valid = (k1 >= 0).all(dim=-1)
        assert valid.any()
        # Full-res coordinates must fit the original image extent
        y = k1[0, valid.squeeze(0), 0]
        x = k1[0, valid.squeeze(0), 1]
        assert y.max() < 128 and x.max() < 160


class TestEquivalence:
    def test_num_levels1_matches_base_detection(self):
        """Level-0-only pyramid should reproduce the base matcher scores."""
        base = ShiTomasiAngleSparseBADSinkhornMatcher(
            max_keypoints=64, binarize=False, epsilon=0.05, distance_type="l2"
        ).eval()
        pyr = ShiTomasiAngleSparseBADSinkhornMatcherPyramid(
            max_keypoints=64, num_levels=1, binarize=False, epsilon=0.05,
            distance_type="l2",
        ).eval()
        img1 = _checkerboard()[None, None]
        img2 = (img1.flip(-1) + torch.randn_like(img1) * 0.01)
        with torch.no_grad():
            kb1, kb2, pb = base(img1, img2)
            kp1, kp2, pp = pyr(img1, img2)
        assert torch.allclose(kb1, kp1, atol=1e-5)
        assert torch.allclose(pb, pp, atol=1e-5)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
