import torch
from torch import nn


class DepthImage2Point(nn.Module):
    DTYPE = torch.float32
    def __init__(self, scale: float, width: int, height: int, cx: float, cy: float, fx: float, fy: float) -> None:
        super().__init__()
        self.scale = float(scale)
        u = torch.zeros([height, width, 1], dtype=self.DTYPE)
        v = torch.zeros_like(u)
        for w in range(width):
            u[:, w, 0] = w
        for h in range(height):
            v[h, :, 0] = h
        u -= cx
        u /= fx
        v -= cy
        v /= fy
        ones = torch.ones_like(u)
        self.uv = torch.concatenate([u, v, ones], dim=2)
        self.uv *= self.scale

    def forward(self, depth_image: torch.Tensor):
        return depth_image * self.uv  # (height, width, 3)


class Point2Point(nn.Module):
    DTYPE = torch.float32
    def __init__(self, rotation: torch.Tensor, translation: torch.Tensor) -> None:
        super().__init__()
        self.rotation = rotation.reshape(3, 3).to(dtype=self.DTYPE)
        self.translation = translation.reshape(3).to(dtype=self.DTYPE)

    def forward(self, point_image: torch.Tensor):
        return point_image @ self.rotation + self.translation


class Point2Pixel(nn.Module):
    DTYPE = torch.float32
    def __init__(self, scale: float, width: int, height: int, cx: float, cy: float, fx: float, fy: float) -> None:
        super().__init__()
        self.scale = float(scale)
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    def forward(self, point_image: torch.Tensor):
        x = point_image[:, :, 0] # (h, w)
        y = point_image[:, :, 1] # (h, w)
        depth = point_image[:, :, 2] # (h, w)
        x = x / depth * self.fx + self.cx
        y = y / depth * self.fy + self.cy
        x[depth == 0.0] = 0.0
        y[depth == 0.0] = 0.0
        # return torch.concatenate([x.unsqueeze(2), y.unsqueeze(2)], dim=2)
        return x, y


class DepthAlignment(nn.Module):
    DTYPE = torch.float32
    def __init__(self, scale: float, width: int, height: int, depth_cx: float, depth_cy: float, depth_fx: float, depth_fy: float, rgb_cx: float, rgb_cy: float, rgb_fx: float, rgb_fy: float, rotation: torch.Tensor, translation: torch.Tensor) -> None:
        super().__init__()
        self.width = width
        self.height = height
        self.dp2pt = DepthImage2Point(scale, width, height, depth_cx, depth_cy, depth_fx, depth_fy)
        self.pt2pt = Point2Point(rotation, translation)
        self.pt2px = Point2Pixel(scale, width, height, rgb_cx, rgb_cy, rgb_fx, rgb_fy)

        self.u = torch.zeros([height, width], dtype=torch.int64)
        self.v = torch.zeros_like(self.u)
        for w in range(width):
            self.u[:, w] = w
        for h in range(height):
            self.v[h, :] = h
        self.u = self.u.reshape(-1)
        self.v = self.v.reshape(-1)

    def forward(self, depth_image: torch.Tensor) -> torch.Tensor:
        depth_pt = self.dp2pt(depth_image)  # (h, w, 3)
        rgb_pt = self.pt2pt(depth_pt)       # (h, w, 3)
        aligned_px_x, aligned_px_y = self.pt2px(rgb_pt) # (h, w) x 2

        p_x = aligned_px_x.reshape(-1)
        p_y = aligned_px_y.reshape(-1)
        mask_x = torch.bitwise_or(p_x < 0, p_x >= self.width)
        mask_y = torch.bitwise_or(p_y < 0, p_y >= self.height)
        mask = torch.bitwise_or(mask_x, mask_y)
        p_x[mask] = torch.tensor(0.0)
        p_y[mask] = torch.tensor(0.0)

        p_x0 = (p_x + torch.tensor(-0.5)).to(torch.int64)
        p_x1 = (p_x + torch.tensor(0.5)).to(torch.int64)
        p_y0 = (p_y + torch.tensor(-0.5)).to(torch.int64)
        p_y1 = (p_y + torch.tensor(0.5)).to(torch.int64)
        p_x = p_x.to(torch.int64)
        p_y = p_y.to(torch.int64)

        fill_value = torch.tensor(10000.0)
        align0 = torch.full_like(depth_image, fill_value=fill_value)
        align1 = torch.full_like(depth_image, fill_value=fill_value)
        align2 = torch.full_like(depth_image, fill_value=fill_value)
        align3 = torch.full_like(depth_image, fill_value=fill_value)
        align_val = depth_image[self.v, self.u]
        align0[p_y0, p_x0] = align_val
        align1[p_y0, p_x1] = align_val
        align2[p_y1, p_x0] = align_val
        align3[p_y1, p_x1] = align_val
        align = torch.minimum(align0, align1)
        align = torch.minimum(align, align2)
        align = torch.minimum(align, align3)
        align[align == fill_value] = torch.tensor(0.0)
        return align
