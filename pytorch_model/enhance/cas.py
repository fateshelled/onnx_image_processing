"""
Contrast Adaptive Sharpening (CAS) for blur robustness.

Implements AMD FidelityFX-style Contrast Adaptive Sharpening as an
ONNX-exportable PyTorch module. Applying CAS before feature detection
restores edge contrast on blurred frames (motion blur, defocus, video
compression), improving Shi-Tomasi gradient statistics and BAD descriptor
discriminability without amplifying noise as aggressively as a fixed
unsharp mask.

Reference:
    AMD FidelityFX Contrast Adaptive Sharpening (CAS), no-scaling path:
    https://github.com/GPUOpen-Effects/FidelityFX-CAS

Algorithm (3x3 neighborhood, cross-shaped kernel):
    a b c
    d e f
    g h i
    - Soft min/max over the {b, d, e, f, h} cross window
    - amp = sqrt(sat(min(mn, 1 - mx) / mx))  (input in [0, 1])
    - w = amp * peak, where peak = -1 / lerp(8, 5, sharpness)
    - out = sat((w*(b + d + f + h) + e) / (1 + 4*w))

    - Negative-lobe kernel differentiation means most flat regions are
      algebraic passthroughs (not amp ~ 0): amp reaches ~0 only near signal
      limits (v near 0 or 1), where mn or 1 - mx approaches 0. Flat mid-tone
      regions have amp ~ 1 but the kernel weights cancel exactly
      (w*(4v) + v = v(1 + 4w)), so flat noise IS amplified in proportion to
      sharpness (up to ~4x at sharpness=1), unlike a quality-gated CAS.
      This is still milder than a fixed unsharp mask, which also boosts
      edge-adjacent ringing everywhere.

Design notes for ONNX export (opset 14+):
    - Only standard ops: pad (replicate) and tensor slicing for the
      neighborhood, plus arithmetic. No dynamic control flow.
    - Input expected in [0, 1] (like the FidelityFX reference). Images in
      [0, 255] must be scaled by the caller or via ``input_scale``.

Example:
    >>> sharpener = ContrastAdaptiveSharpening(sharpness=0.4)
    >>> img = torch.rand(1, 1, 480, 640)
    >>> out = sharpener(img)  # same shape, sharpened
"""

import torch
from torch import nn
import torch.nn.functional as F


class ContrastAdaptiveSharpening(nn.Module):
    """
    Contrast Adaptive Sharpening (FidelityFX CAS, no-scaling path).

    Args:
        sharpness: Sharpening strength in [0, 1]. 0 = lower ringing,
                   1 = maximum. Maps to the FidelityFX ``peak`` constant
                   ``-1 / lerp(8, 5, sharpness)``. Note that 0 does not fully
        disable sharpening: it still applies a mild fixed kernel
        (peak = -1/8). Structure the caller to skip the module entirely
        (e.g., ``cas_sharpness > 0``) to bypass CAS. Default is 0.4.
        input_scale: Divisor applied to the input before filtering.
                     Use 255.0 for images in [0, 255] range, 1.0 for
                     images already in [0, 1]. Default is 255.0.
        padding_mode: Padding mode for the neighborhood extraction.
                      'replicate' avoids dark borders. Default is 'replicate'.

    Example:
        >>> cas = ContrastAdaptiveSharpening(sharpness=0.4)
        >>> img = torch.rand(1, 1, 480, 640) * 255.0
        >>> sharp = cas(img)
        >>> sharp.shape
        torch.Size([1, 1, 480, 640])
    """

    def __init__(
        self,
        sharpness: float = 0.4,
        input_scale: float = 255.0,
        padding_mode: str = "replicate",
    ) -> None:
        super().__init__()

        if not 0.0 <= sharpness <= 1.0:
            raise ValueError(f"sharpness must be in [0, 1], got {sharpness}")
        if input_scale <= 0:
            raise ValueError(f"input_scale must be positive, got {input_scale}")
        if padding_mode not in ("replicate", "constant", "reflect"):
            raise ValueError(
                f"padding_mode must be 'replicate', 'constant' or 'reflect', "
                f"got {padding_mode}"
            )

        # FidelityFX CasSetup: peak = -1 / lerp(8, 5, sat(sharpness))
        lerp = 8.0 + (5.0 - 8.0) * min(max(sharpness, 0.0), 1.0)
        self.peak = -1.0 / lerp
        self.input_scale = input_scale
        self.padding_mode = padding_mode

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply contrast adaptive sharpening.

        Args:
            image: Input grayscale image of shape (B, 1, H, W).

        Returns:
            Sharpened image of shape (B, 1, H, W), same dtype and scale
            as the input (input_scale is applied internally and undone).
        """
        img = image.float()

        # Work in [0, 1] domain like the FidelityFX reference
        if self.input_scale != 1.0:
            img = img / self.input_scale

        # 3x3 neighborhood via replicate padding
        padded = F.pad(img, (1, 1, 1, 1), mode=self.padding_mode)

        # Cross-shaped 5-tap window: {up, left, center, right, down}
        b = padded[:, :, :-2, 1:-1]   # up
        d = padded[:, :, 1:-1, :-2]   # left
        e = padded[:, :, 1:-1, 1:-1]  # center
        f = padded[:, :, 1:-1, 2:]    # right
        h = padded[:, :, 2:, 1:-1]    # down

        # Soft min/max over the cross window (no diagonals, fast path)
        mx = torch.max(torch.max(d, e), torch.max(f, torch.max(b, h)))
        mn = torch.min(torch.min(d, e), torch.min(f, torch.min(b, h)))

        # Smooth minimum distance to signal limit divided by smooth max
        amp = torch.clamp(torch.minimum(mn, 1.0 - mx) / (mx + 1e-8), 0.0, 1.0)
        # Shaping amount of sharpening
        amp = torch.sqrt(amp)

        # Negative-lobe cross kernel: {0 w 0 / w 1 w / 0 w 0}
        w = amp * self.peak
        out = (b * w + d * w + f * w + h * w + e) / (1.0 + 4.0 * w)
        out = torch.clamp(out, 0.0, 1.0)

        # Restore the input scale
        if self.input_scale != 1.0:
            out = out * self.input_scale

        return out.to(image.dtype)
