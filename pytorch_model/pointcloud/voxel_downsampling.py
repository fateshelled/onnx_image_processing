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

        # 3. Compute group boundaries and group IDs using prefix sum
        is_new_group = torch.cat([
            torch.tensor([True]),
            sorted_keys[1:] != sorted_keys[:-1]
        ])
        is_group_end = torch.cat([
            sorted_keys[:-1] != sorted_keys[1:],
            torch.tensor([True])
        ])
        group_ids = torch.cumsum(is_new_group.to(torch.int64), dim=0) - 1

        # 4. Prefix sums of sorted coordinates and counts
        prefix_sum = torch.cumsum(sorted_points, dim=0)
        prefix_count = torch.cumsum(torch.ones(N, 1, dtype=self.DTYPE), dim=0)

        # Shifted prefix sums (cumsum[i-1], with 0 at position 0)
        shifted_prefix_sum = torch.zeros_like(prefix_sum)
        shifted_prefix_sum[1:] = prefix_sum[:-1]
        shifted_prefix_count = torch.zeros_like(prefix_count)
        shifted_prefix_count[1:] = prefix_count[:-1]

        # 5. Compute per-group offsets via group start positions
        group_start_indices = torch.nonzero(is_new_group).squeeze(1)
        group_end_indices = torch.nonzero(is_group_end).squeeze(1)

        # For each position, get the offset (prefix sum before its group start)
        start_per_pos = group_start_indices[group_ids]
        offset_sum = shifted_prefix_sum[start_per_pos]
        offset_count = shifted_prefix_count[start_per_pos]

        # 6. Group-local prefix sums (reset at group boundaries)
        local_prefix_sum = prefix_sum - offset_sum
        local_prefix_count = prefix_count - offset_count

        # At group end positions, local prefix sum equals the group total
        group_total_sum = local_prefix_sum[group_end_indices]
        group_total_count = local_prefix_count[group_end_indices]

        # 7. Compute mean coordinates per voxel
        group_means = group_total_sum / group_total_count

        # 8. Build output: means at [0..M-1], zeros at [M..N-1]
        M = group_means.shape[0]
        padding_pts = torch.zeros(N - M, 3, dtype=self.DTYPE)
        output_points = torch.cat([group_means, padding_pts], dim=0)

        mask_valid = torch.ones(M, dtype=torch.bool)
        mask_pad = torch.zeros(N - M, dtype=torch.bool)
        mask = torch.cat([mask_valid, mask_pad], dim=0)

        return output_points, mask
