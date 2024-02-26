import torch
from torch import nn


class DepthToPointCloud(nn.Module):
    def __init__(self, scale, width, height, cx, cy, fx, fy) -> None:
        super().__init__()
        self.scale = float(scale)
        u = torch.zeros([height, width, 1], dtype=torch.float32)
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

    def forward(self, depth):
        return depth * self.uv
