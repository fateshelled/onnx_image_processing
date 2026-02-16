import torch
from torch import nn
import torch.nn.functional as F


class FASTScore(nn.Module):
    """
    FAST (Features from Accelerated Segment Test) corner detection score calculator.

    Implements the "Faster than FAST" optimization strategy using binary encoding
    to eliminate dynamic loops and conditional branches, making it ONNX-friendly
    and GPU-efficient.

    The FAST algorithm detects corners by examining a circle of 16 pixels around
    a candidate point. A point is classified as a corner if there exists a set of
    9 contiguous pixels in the circle that are all brighter or all darker than
    the center pixel by more than a threshold.

    This implementation uses a binary encoding strategy:
    1. Encode the 16 pixel differences into a 32-bit buffer (16 bits for dark, 16 for bright)
    2. Create 24-bit circular buffers to handle wraparound
    3. Use bitwise operations to detect 9 consecutive bits in a single state
    4. Completely avoids if-else branches and dynamic loops

    Args:
        threshold: Intensity difference threshold. A pixel is considered
                   significantly different if |I_circle - I_center| > threshold.
                   Default is 20.
        use_nms: Whether to apply non-maximum suppression. Default is False.
        nms_radius: Radius for non-maximum suppression. Default is 3.

    Reference:
        Binary encoding optimization strategy for GPU-efficient FAST implementation
    """

    def __init__(
        self,
        threshold: int = 20,
        use_nms: bool = False,
        nms_radius: int = 3
    ) -> None:
        super().__init__()

        self.threshold = threshold
        self.use_nms = use_nms
        self.nms_radius = nms_radius

        # Bresenham circle of radius 3: 16 pixel offsets
        # Ordered clockwise starting from (0, -3)
        circle_offsets = torch.tensor([
            [0, -3], [1, -3], [2, -2], [3, -1],
            [3, 0], [3, 1], [2, 2], [1, 3],
            [0, 3], [-1, 3], [-2, 2], [-3, 1],
            [-3, 0], [-3, -1], [-2, -2], [-1, -3]
        ], dtype=torch.long)  # Shape: (16, 2) - (dy, dx)

        self.register_buffer('circle_offsets', circle_offsets)

        # Precompute powers of 2 for binary encoding
        powers_of_2 = torch.tensor(
            [1 << i for i in range(16)],
            dtype=torch.int32
        )
        self.register_buffer('powers_of_2', powers_of_2.view(1, 1, 1, 16))

    def _sample_circle_pixels(self, image: torch.Tensor) -> torch.Tensor:
        """
        Sample 16 pixels around each point in the image using the Bresenham circle.

        Args:
            image: Input image of shape (N, 1, H, W)

        Returns:
            circle_pixels: Sampled pixels of shape (N, H, W, 16)
        """
        N, C, H, W = image.shape
        device = image.device

        # Pad image to handle border pixels
        # We need padding of 3 pixels on all sides for radius-3 circle
        padded = F.pad(image, (3, 3, 3, 3), mode='replicate')  # (N, 1, H+6, W+6)

        # Create pixel coordinate grids for the original image
        y_coords = torch.arange(H, device=device).view(-1, 1).expand(H, W)  # (H, W)
        x_coords = torch.arange(W, device=device).view(1, -1).expand(H, W)  # (H, W)

        # Add padding offset (3 pixels)
        y_coords = y_coords + 3
        x_coords = x_coords + 3

        # Sample all 16 circle positions
        circle_pixels = []
        for i in range(16):
            dy, dx = self.circle_offsets[i]
            sampled_y = y_coords + dy
            sampled_x = x_coords + dx

            # Use advanced indexing to gather pixels
            # Create batch indices
            batch_indices = torch.arange(N, device=device).view(N, 1, 1).expand(N, H, W)

            # Gather pixels: padded[batch_indices, 0, sampled_y, sampled_x]
            sampled = padded[batch_indices, 0, sampled_y, sampled_x]  # (N, H, W)
            circle_pixels.append(sampled)

        # Stack along last dimension: (N, H, W, 16)
        circle_pixels = torch.stack(circle_pixels, dim=-1)

        return circle_pixels

    def _binary_encoding(
        self,
        center_pixels: torch.Tensor,
        circle_pixels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode pixel differences into binary dark/bright bits.

        Args:
            center_pixels: Center pixel values of shape (N, H, W)
            circle_pixels: Circle pixel values of shape (N, H, W, 16)

        Returns:
            dark_bits: 16-bit dark state encoded as int32, shape (N, H, W)
            bright_bits: 16-bit bright state encoded as int32, shape (N, H, W)
        """
        # Compute differences: Dt = I_circle - I_center
        center_expanded = center_pixels.unsqueeze(-1)  # (N, H, W, 1)
        diff = circle_pixels - center_expanded  # (N, H, W, 16)

        threshold = float(self.threshold)

        # Binary encoding using conditional operators (no if-else branches)
        # Dark: Dt >= threshold -> bit = 1
        # Bright: Dt <= -threshold -> bit = 1
        dark_mask = (diff >= threshold).to(torch.int32)  # (N, H, W, 16)
        bright_mask = (diff <= -threshold).to(torch.int32)  # (N, H, W, 16)

        # Convert bit masks to 16-bit integers using precomputed powers of 2
        # Encode: sum of (bit_i * 2^i)
        dark_bits = (dark_mask * self.powers_of_2).sum(dim=-1)  # (N, H, W)
        bright_bits = (bright_mask * self.powers_of_2).sum(dim=-1)  # (N, H, W)

        return dark_bits, bright_bits

    def _detect_9_consecutive(self, bits_16: torch.Tensor) -> torch.Tensor:
        """
        Detect if there are 9 consecutive bits set in a 16-bit circular pattern.

        Uses fully vectorized operations with arithmetic-only operations for
        full ONNX/TensorRT compatibility (no bitwise operations).

        Args:
            bits_16: 16-bit encoded states of shape (N, H, W)

        Returns:
            detected: Boolean tensor of shape (N, H, W) indicating detection
        """
        # For circular detection, we need to handle wraparound
        # Create a 24-bit buffer by appending the lower 8 bits to the upper end
        # This allows us to detect patterns like [..., 15, 16, 1, 2, ...]

        # Extract lower 8 bits using modulo instead of bitwise AND
        # lower_8 = bits_16 & 0xFF -> bits_16 % 256
        lower_8 = bits_16 % 256  # bits [0:7]

        # Create 24-bit buffer: upper_16_bits | (lower_8_bits << 16)
        # Since the bits don't overlap, OR can be replaced with addition
        # This represents: [bit0, bit1, ..., bit15, bit0, bit1, ..., bit7]
        buffer_24 = bits_16.to(torch.int32) + (lower_8.to(torch.int32) * 65536)

        # Vectorized check: all 16 possible starting positions for 9 consecutive bits
        # mask for 9 consecutive bits: 0b111111111 = 0x1FF = 511

        # Create division factors tensor: [2^0, 2^1, 2^2, ..., 2^15] with shape (16, 1, 1, 1)
        divisors = torch.tensor(
            [2 ** i for i in range(16)],
            device=bits_16.device,
            dtype=torch.int32
        )
        divisors = divisors.view(16, *([1] * bits_16.ndim))

        # Expand buffer_24 to (1, N, H, W) for broadcasting
        buffer_expanded = buffer_24.unsqueeze(0)  # (1, N, H, W)

        # Apply all divisions in parallel and mask to 9 bits using modulo
        # shifted = (buffer_expanded // divisors) & 0x1FF -> % 512
        shifted = (buffer_expanded // divisors) % 512  # (16, N, H, W)

        # Check if all 9 bits are set for each shift position: (16, N, H, W)
        is_all_set = (shifted == 511)  # 0x1FF = 511

        # Combine: detected if ANY of the 16 positions has 9 consecutive bits
        detected = is_all_set.any(dim=0)  # (N, H, W)

        return detected

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Compute FAST corner detection score for each pixel.

        Args:
            image: Input grayscale image tensor of shape (N, 1, H, W).
                   Values should be in range [0, 255].

        Returns:
            score: FAST detection score of shape (N, 1, H, W).
                   Binary output: 1.0 for detected corners, 0.0 otherwise.
                   If use_nms=True, returns suppressed scores.
        """
        N, C, H, W = image.shape

        # Convert to float for processing
        img = image.float()

        # Sample 16 circle pixels for each position
        circle_pixels = self._sample_circle_pixels(img)  # (N, H, W, 16)

        # Get center pixel values
        center_pixels = img.squeeze(1)  # (N, H, W)

        # Binary encoding
        dark_bits, bright_bits = self._binary_encoding(center_pixels, circle_pixels)

        # Detect 9 consecutive bits for dark and bright
        dark_detected = self._detect_9_consecutive(dark_bits)  # (N, H, W)
        bright_detected = self._detect_9_consecutive(bright_bits)  # (N, H, W)

        # Combine: corner if either dark OR bright has 9 consecutive
        corner_detected = dark_detected | bright_detected  # (N, H, W)

        # Convert to float score
        score = corner_detected.float().unsqueeze(1)  # (N, 1, H, W)

        # Optional: Non-maximum suppression
        if self.use_nms:
            score = self._apply_nms(score)

        return score

    def _apply_nms(self, score: torch.Tensor) -> torch.Tensor:
        """
        Apply non-maximum suppression using max pooling.

        Args:
            score: Score map of shape (N, 1, H, W)

        Returns:
            suppressed: NMS-suppressed score map of shape (N, 1, H, W)
        """
        # Use max pooling to find local maxima
        kernel_size = 2 * self.nms_radius + 1
        padding = self.nms_radius

        # Find local maximum in neighborhood
        max_pooled = F.max_pool2d(
            score,
            kernel_size=kernel_size,
            stride=1,
            padding=padding
        )

        # Keep only pixels that are local maxima
        suppressed = torch.where(score == max_pooled, score, torch.zeros_like(score))

        return suppressed
