from depth2pointcloud import DepthToPointCloud
import torch
from torch import nn
import torch.nn.functional as F


class DepthToPointCloudWithNormal(nn.Module):
    def __init__(self, scale: float, width: int, height: int, cx: float, cy: float, fx: float, fy: float) -> None:
        super().__init__()
        self.base_model = DepthToPointCloud(
            scale, width, height, cx, cy, fx, fy)
        self.sobel_v = torch.FloatTensor(
            [[1, 0, -1],
             [2, 0, -2],
             [1, 0, -1]]).expand(1, 3, 3, 3)    # NCHW 1x3x3x3
        self.sobel_h = torch.FloatTensor(
            [[1, 2, 1],
             [0, 0, 0],
             [-1, -2, -1]]).expand(1, 3, 3, 3)  # NCHW 1x3x3x3

        self.ones = torch.full([1, 1, height, width], -1.0, dtype=torch.float32)

    def forward(self, depth: torch.Tensor):
        pcd = self.base_model.forward(depth)  # hwc HxWx3
        pcd_nchw = pcd.permute(2, 0, 1).unsqueeze(0)  # nchw 1x3xHxW
        dx = F.conv2d(pcd_nchw, self.sobel_v, padding=1)  # 1x1xHxW
        dy = F.conv2d(pcd_nchw, self.sobel_h, padding=1)  # 1x1xHxW
        vec = torch.concatenate([dx, dy, self.ones], dim=1)  # 1x3xHxW
        vec = vec.squeeze().permute(1, 2, 0)  # 1x3xHxW -> 3xHxW -> HxWx3
        # normalize
        norm = torch.sqrt(torch.sum(vec ** 2, dim=2, keepdim=True))  # HxWx1
        norm = torch.divide(vec, norm)  # HxWx3 / HxWx1 -> HxWx3
        return pcd, norm
