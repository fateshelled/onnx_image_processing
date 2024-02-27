import torch
from torch import nn


class OtsuThreshold(nn.Module):
    def __init__(self, width: int, height: int, min_val: int, max_val: int, dtype=torch.int32, device="cpu") -> None:
        super().__init__()
        self.width = width
        self.height = height
        self.min_val = min_val
        self.max_val = max_val
        self.d = max_val - min_val - 1
        self.dtype = dtype
        self.zero = torch.tensor(0, dtype=self.dtype)
        self.index = torch.zeros([height, width, self.d], dtype=dtype, device=device)
        for i in range(min_val, max_val - 1, 1):
            self.index[:, :, i] = i

    def forward(self, img_HxW: torch.Tensor):
        stacked = torch.concat([img_HxW.unsqueeze(2)] * self.d, dim=2)
        class_b = (stacked <= self.index) # class black HxWx(max_val-min_val-1)
        class_w = ~class_b                # class white HxWx(max_val-min_val-1)

        hw = self.width * self.height
        masked_b = torch.where(class_b, stacked, self.zero)
        masked_w = torch.where(class_w, stacked, self.zero)
        sum_b = torch.sum(masked_b.reshape(hw, self.d), dim=0, dtype=torch.float32)
        sum_w = torch.sum(masked_w.reshape(hw, self.d), dim=0, dtype=torch.float32)
        num_b = torch.sum(class_b.reshape(hw, self.d), dim=0, dtype=torch.float32)
        num_w = hw - num_b

        mean_b = sum_b / num_b
        mean_w = sum_w / num_w

        var_hist = num_b * num_w * ((mean_b - mean_w) ** 2)
        thresh = torch.argmax(var_hist).to(self.dtype)

        bin_img = torch.where(
            img_HxW <= thresh,
            torch.tensor(self.min_val, dtype=self.dtype),
            torch.tensor(self.max_val, dtype=self.dtype))
        return thresh, bin_img
