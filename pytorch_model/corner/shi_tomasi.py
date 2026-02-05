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

        self.block_size = block_size
        self.sobel_size = sobel_size

        # Sobel kernels for gradient computation (3x3)
        # Sobel kernel for horizontal gradient (dI/dx)
        sobel_x = torch.tensor([
            [-1., 0., 1.],
            [-2., 0., 2.],
            [-1., 0., 1.]
        ]).unsqueeze(0).unsqueeze(0)  # Shape: 1x1x3x3

        # Sobel kernel for vertical gradient (dI/dy)
        sobel_y = torch.tensor([
            [-1., -2., -1.],
            [0., 0., 0.],
            [1., 2., 1.]
        ]).unsqueeze(0).unsqueeze(0)  # Shape: 1x1x3x3

        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

        # Sum kernel for block neighborhood
        sum_kernel = torch.ones(1, 1, block_size, block_size)
        self.register_buffer('sum_kernel', sum_kernel)

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

        # Compute gradients using Sobel filters with replicate padding
        sobel_pad = self.sobel_size // 2
        img_padded = F.pad(img, (sobel_pad, sobel_pad, sobel_pad, sobel_pad), mode='replicate')
        Ix = F.conv2d(img_padded, self.sobel_x)
        Iy = F.conv2d(img_padded, self.sobel_y)

        # Compute products of gradients
        Ixx = Ix * Ix
        Iyy = Iy * Iy
        Ixy = Ix * Iy

        # Sum over block neighborhood
        block_pad = self.block_size // 2
        Ixx_padded = F.pad(Ixx, (block_pad, block_pad, block_pad, block_pad), mode='replicate')
        Iyy_padded = F.pad(Iyy, (block_pad, block_pad, block_pad, block_pad), mode='replicate')
        Ixy_padded = F.pad(Ixy, (block_pad, block_pad, block_pad, block_pad), mode='replicate')
        sum_Ixx = F.conv2d(Ixx_padded, self.sum_kernel)
        sum_Iyy = F.conv2d(Iyy_padded, self.sum_kernel)
        sum_Ixy = F.conv2d(Ixy_padded, self.sum_kernel)

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
