import torch
from torch import nn


class OtsuThreshold(nn.Module):
    def __init__(self, min_val: int, max_val: int, dtype=torch.int32, device="cpu") -> None:
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.BINS = max_val - min_val + 1
        self.dtype = dtype
        self.device = device

        self.mask_bk = torch.tril(torch.ones([self.BINS, self.BINS], dtype=torch.int32, device=self.device))
        self.mask_wh = torch.ones_like(self.mask_bk, device=self.device) - self.mask_bk

    def forward(self, img_HxW: torch.Tensor):
        # calc histgram
        indices = img_HxW.view(-1).to(torch.int64)
        hist = torch.zeros(self.BINS, device=self.device, dtype=torch.int64)
        hist = hist.scatter_add(0, indices, torch.ones_like(indices))

        hist_class = hist * torch.arange(self.min_val, self.max_val + 1, 1, dtype=torch.int64, device=self.device)

        # black class
        fc_bk = hist_class * self.mask_bk
        hist_bk = hist * self.mask_bk
        fc_bk_sum = torch.sum(fc_bk, dim=1)
        num_bk = torch.sum(hist_bk, dim=1)
        mean_bk = fc_bk_sum / num_bk

        # white class
        fc_wh = hist_class * self.mask_wh
        hist_wh = hist * self.mask_wh
        fc_wh_sum = torch.sum(fc_wh, dim=1)
        num_wh = torch.sum(hist_wh, dim=1)
        mean_wh = fc_wh_sum / num_wh

        # betweeb class variance
        var_hist = num_bk * num_wh * ((mean_bk - mean_wh) ** 2)
        var_hist = torch.where(torch.isnan(var_hist), torch.tensor(0, dtype=torch.float32), var_hist)
        thresh = torch.argmax(var_hist)

        bin_img = torch.where(
            img_HxW <= thresh,
            torch.tensor(self.min_val, dtype=self.dtype),
            torch.tensor(self.max_val, dtype=self.dtype))
        return thresh, bin_img
