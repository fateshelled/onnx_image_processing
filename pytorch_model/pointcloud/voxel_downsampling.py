import torch
from torch import nn


class VoxelDownsampling(nn.Module):
    DTYPE = torch.float32

    def __init__(self) -> None:
        super().__init__()

    def forward(self, points: torch.Tensor, leaf_size: torch.Tensor):
        # points: (N, 3), leaf_size: scalar
        N = points.shape[0]

        # 1. Compute voxel indices by dividing coordinates by leaf size
        voxel_coords = torch.floor(points / leaf_size).to(torch.int64)
        voxel_min = voxel_coords.min(dim=0).values
        voxel_coords = voxel_coords - voxel_min

        # Compute 1D voxel keys for sorting
        voxel_max = voxel_coords.max(dim=0).values
        dim1 = voxel_max[1] + 1
        dim2 = voxel_max[2] + 1
        voxel_keys = (voxel_coords[:, 0] * dim1 * dim2
                    + voxel_coords[:, 1] * dim2
                    + voxel_coords[:, 2])

        # 2. Sort coordinates by voxel keys
        sorted_indices = torch.argsort(voxel_keys)
        sorted_keys = voxel_keys[sorted_indices]
        sorted_points = points[sorted_indices]

        # 3. Compute group IDs using prefix sum of boundary flags
        is_new_group = torch.cat([
            torch.tensor([True]),
            sorted_keys[1:] != sorted_keys[:-1]
        ])
        group_ids = torch.cumsum(is_new_group.to(torch.int64), dim=0) - 1

        # 4. Aggregate coordinates per voxel using scatter_add
        group_ids_expanded = group_ids.unsqueeze(1).expand_as(sorted_points)
        group_sum = torch.zeros(N, 3, dtype=self.DTYPE).scatter_add(
            0, group_ids_expanded, sorted_points
        )
        group_count = torch.zeros(N, 1, dtype=self.DTYPE).scatter_add(
            0, group_ids.unsqueeze(1), torch.ones(N, 1, dtype=self.DTYPE)
        )

        # 5. Compute mean coordinates and validity mask
        output_points = group_sum / torch.clamp(group_count, min=1.0)
        mask = group_count.squeeze(1) > 0

        return output_points, mask
