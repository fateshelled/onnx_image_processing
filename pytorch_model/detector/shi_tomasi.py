import torch
from torch import nn
import torch.nn.functional as F


class ShiTomasiScore(nn.Module):
    """
    Shi-Tomasi corner detection score calculator.

    Computes the Shi-Tomasi score (minimum eigenvalue of structure tensor)
    for every pixel in the input image. The output is a score image with
    the same spatial dimensions as the input.

    The Shi-Tomasi score is defined as:
        score = min(lambda1, lambda2)

    where lambda1 and lambda2 are eigenvalues of the structure tensor M:
        M = | sum(Ix*Ix)  sum(Ix*Iy) |
            | sum(Ix*Iy)  sum(Iy*Iy) |

    The minimum eigenvalue can be computed analytically:
        lambda_min = (Ixx + Iyy) / 2 - sqrt(((Ixx - Iyy) / 2)^2 + Ixy^2)

    Args:
        block_size: Size of the neighborhood window for computing
                    the structure tensor sums. Default is 3.
        sobel_size: Size of the Sobel kernel for gradient computation.
                    Currently only supports 3. Default is 3.

    Raises:
        ValueError: If sobel_size is not 3.
    """

    def __init__(self, block_size: int = 3, sobel_size: int = 3) -> None:
        super().__init__()

        if sobel_size != 3:
            raise ValueError(f"sobel_size must be 3, got {sobel_size}")

        if block_size <= 0 or block_size % 2 == 0:
            raise ValueError(f"block_size must be a positive odd integer, got {block_size}")
        self.block_size = block_size
        self.sobel_size = sobel_size

        # Fused Sobel kernels: single 2-output-channel conv computes
        # both Ix and Iy gradients in one kernel launch.
        sobel_x = torch.tensor([
            [-1., 0., 1.],
            [-2., 0., 2.],
            [-1., 0., 1.]
        ]).unsqueeze(0).unsqueeze(0)  # Shape: 1x1x3x3

        sobel_y = torch.tensor([
            [-1., -2., -1.],
            [0., 0., 0.],
            [1., 2., 1.]
        ]).unsqueeze(0).unsqueeze(0)  # Shape: 1x1x3x3

        self.register_buffer('sobel_xy', torch.cat([sobel_x, sobel_y], dim=0))  # (2,1,3,3)

        # Fused sum kernel: groups=3 conv computes sum_Ixx, sum_Iyy, sum_Ixy
        # in one kernel launch over the 3-channel stacked input.
        sum_kernel = torch.ones(1, 1, block_size, block_size)
        self.register_buffer('sum_kernel_grouped', sum_kernel.repeat(3, 1, 1, 1))  # (3,1,bs,bs)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Compute Shi-Tomasi score for each pixel.

        Args:
            image: Input grayscale image tensor of shape (N, 1, H, W).
                   Values should be in range [0, 255] or [0, 1].

        Returns:
            score: Shi-Tomasi score image of shape (N, 1, H, W).
                   Higher values indicate stronger corner responses.
        """
        img = image.float()

        # Compute gradients using fused Sobel conv: 1ch -> 2ch (Ix, Iy)
        sobel_pad = self.sobel_size // 2
        img_padded = F.pad(img, (sobel_pad, sobel_pad, sobel_pad, sobel_pad), mode='replicate')
        grads = F.conv2d(img_padded, self.sobel_xy)  # (N, 2, H, W)
        Ix = grads[:, 0:1]
        Iy = grads[:, 1:2]

        # Compute products of gradients and stack into 3 channels
        products = torch.cat([Ix * Ix, Iy * Iy, Ix * Iy], dim=1)  # (N, 3, H, W)

        # Sum over block neighborhood using fused groups=3 conv
        block_pad = self.block_size // 2
        products_padded = F.pad(products, (block_pad, block_pad, block_pad, block_pad), mode='replicate')
        sums = F.conv2d(products_padded, self.sum_kernel_grouped, groups=3)  # (N, 3, H, W)
        sum_Ixx = sums[:, 0:1]
        sum_Iyy = sums[:, 1:2]
        sum_Ixy = sums[:, 2:3]

        # Compute minimum eigenvalue of structure tensor
        # For 2x2 matrix [[a, b], [b, c]], eigenvalues are:
        # lambda = (a + c) / 2 +/- sqrt(((a - c) / 2)^2 + b^2)
        # minimum eigenvalue = (a + c) / 2 - sqrt(((a - c) / 2)^2 + b^2)
        half_trace = (sum_Ixx + sum_Iyy) / 2
        diff_half = (sum_Ixx - sum_Iyy) / 2
        discriminant = diff_half * diff_half + sum_Ixy * sum_Ixy
        sqrt_discriminant = torch.sqrt(discriminant + 1e-10)  # Small epsilon for numerical stability

        lambda_min = half_trace - sqrt_discriminant

        # Ensure non-negative scores (eigenvalues should be non-negative for positive semi-definite matrix)
        score = torch.clamp(lambda_min, min=0.0)

        return score
