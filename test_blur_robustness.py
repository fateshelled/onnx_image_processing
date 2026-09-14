#!/usr/bin/env python3
"""
Blur robustness tests: Contrast Adaptive Sharpening (CAS) module and its
integration into the feature matching pipeline.

Verifies:
  1. CAS module correctness (shape/dtype/scale preservation, flat-region
     no-op, gradient contrast restoration on blurred input).
  2. CAS-integrated pipeline runs and produces consistent output shapes.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from pytorch_model.enhance import ContrastAdaptiveSharpening
from pytorch_model.feature_detection.shi_tomasi_angle_sparse_bad_sinkhorn import (
    ShiTomasiAngleSparseBADSinkhornMatcher,
)


def _defocus(x: torch.Tensor, factor: int = 6) -> torch.Tensor:
    """Defocus-like blur: downscale then upscale (pure lowpass, no noise)."""
    return F.interpolate(
        F.interpolate(x, scale_factor=1.0 / factor, mode="bilinear",
                      align_corners=False),
        size=x.shape[-2:], mode="bilinear", align_corners=False,
    )


class TestContrastAdaptiveSharpening:
    def test_shape_dtype_scale_preserved(self):
        cas = ContrastAdaptiveSharpening(sharpness=0.5)
        img = torch.rand(1, 1, 64, 80) * 255.0
        out = cas(img)
        assert out.shape == img.shape
        assert out.dtype == img.dtype
        assert out.min() >= 0.0 and out.max() <= 255.0

    def test_flat_image_is_noop(self):
        """Flat regions: amp~0, kernel degenerates to passthrough."""
        cas = ContrastAdaptiveSharpening(sharpness=0.7)
        flat = torch.full((1, 1, 32, 32), 128.0)
        out = cas(flat)
        assert torch.allclose(out, flat, atol=1e-3)

    def test_sharpness_zero_is_mild_filter(self):
        """sharpness=0 maps to FidelityFX peak -1/8 (mild low-ringing filter),
        not a passthrough. It must, however, stay stable and close to input
        on low-frequency content."""
        cas = ContrastAdaptiveSharpening(sharpness=0.0)
        img = torch.rand(1, 1, 32, 32) * 255.0
        out = cas(img)
        assert out.shape == img.shape
        # Mild filter: mean absolute deviation must be small relative to range
        assert (out - img).abs().mean() < 0.1 * 255.0

    def test_invalid_params_raise(self):
        with pytest.raises(ValueError):
            ContrastAdaptiveSharpening(sharpness=1.5)
        with pytest.raises(ValueError):
            ContrastAdaptiveSharpening(input_scale=0.0)
        with pytest.raises(ValueError):
            ContrastAdaptiveSharpening(padding_mode="wrap")

    def test_gradient_contrast_restoration_on_blur(self):
        """CAS must increase gradient magnitude of blurred input (its core
        claim: restore edge contrast lost to blur)."""
        torch.manual_seed(0)
        cas = ContrastAdaptiveSharpening(sharpness=0.5)
        img = torch.rand(1, 1, 64, 80) * 255.0
        blurred = F.avg_pool2d(img, 5, stride=1, padding=2)

        gx = lambda x: (x[:, :, :, 2:] - x[:, :, :, :-2]).abs().mean()
        assert gx(cas(blurred)) > gx(blurred)


class TestPipelineIntegration:
    def test_cas_enabled_runs_and_same_shapes(self):
        """Pipeline with cas_sharpness > 0 must run and keep output contract."""
        torch.manual_seed(1)
        model = ShiTomasiAngleSparseBADSinkhornMatcher(
            max_keypoints=64, cas_sharpness=0.5
        )
        model.eval()
        img1 = torch.rand(1, 1, 80, 100) * 255.0
        img2 = torch.roll(img1, shifts=4, dims=3)
        with torch.no_grad():
            k1, k2, probs = model(img1, img2)
        assert k1.shape == (1, 64, 2)
        assert k2.shape == (1, 64, 2)
        assert probs.shape == (1, 65, 65)
        assert torch.isfinite(probs).all()

    def test_cas_disabled_matches_default(self):
        """cas_sharpness=0 must be identical to the default (no sharpener)."""
        torch.manual_seed(2)
        m_flag = ShiTomasiAngleSparseBADSinkhornMatcher(
            max_keypoints=32, cas_sharpness=0.0
        )
        m_default = ShiTomasiAngleSparseBADSinkhornMatcher(max_keypoints=32)
        m_flag.eval()
        m_default.eval()
        assert m_flag.sharpener is None
        assert m_default.sharpener is None

        img1 = torch.rand(1, 1, 60, 80) * 255.0
        img2 = torch.roll(img1, shifts=3, dims=3)
        with torch.no_grad():
            k1a, k2a, pa = m_flag(img1, img2)
            k1b, k2b, pb = m_default(img1, img2)
        assert torch.equal(k1a, k1b)
        assert torch.equal(pa, pb)

    def test_cas_changes_output_when_enabled(self):
        """Enabling CAS must actually alter results (wired into forward)."""
        torch.manual_seed(3)
        m_off = ShiTomasiAngleSparseBADSinkhornMatcher(
            max_keypoints=32, cas_sharpness=0.0
        )
        m_on = ShiTomasiAngleSparseBADSinkhornMatcher(
            max_keypoints=32, cas_sharpness=0.7
        )
        m_off.eval()
        m_on.eval()
        img1 = torch.rand(1, 1, 60, 80) * 255.0
        img2 = torch.roll(img1, shifts=3, dims=3)
        with torch.no_grad():
            _, _, p_off = m_off(img1, img2)
            _, _, p_on = m_on(img1, img2)
        assert not torch.allclose(p_off, p_on)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
