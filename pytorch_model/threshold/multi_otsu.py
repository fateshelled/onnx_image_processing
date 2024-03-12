import torch
from torch import nn
import itertools
import math

class MultiOtsuThreshold(torch.nn.Module):
    # DTYPE = torch.float64
    DTYPE = torch.float32
    def __init__(self, min_val: int, max_val: int, device="cpu", n_class = 3, calc_hist=False) -> None:
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.BINS = max_val - min_val
        self.device = device
        self.n_class = n_class
        self.calc_hist = calc_hist

        self.COMBINATIONS = math.comb(self.BINS - 1, self.n_class - 1)
        self.ZERO = torch.tensor(0, dtype=self.DTYPE, device=self.device)

        mask_idx = torch.zeros([self.COMBINATIONS, self.BINS], dtype=self.DTYPE, device=device)
        for i, thresholds in enumerate(itertools.combinations(range(1, self.BINS), self.n_class - 1)):
            for j, th in enumerate(thresholds):
                mask_idx[i, th:] = j + 1

        mask = []
        for i in range(self.n_class):
            mask.append((mask_idx == i).to(self.DTYPE).unsqueeze(0))
        self.mask = torch.concatenate(mask, dim=0)

        self.threshold_indices = [
            torch.count_nonzero(mask[0], dim=2)[0] - 1
        ]
        for i in range(1, self.n_class - 1):
            self.threshold_indices.append(
                torch.count_nonzero(mask[i], dim=2)[0] + self.threshold_indices[-1]
            )

    def calc_histogram(self, img_HxW: torch.Tensor):
        indices = img_HxW.view(-1)
        if img_HxW.dtype != torch.int64:
            indices = indices.to(torch.int64)
        hist = torch.zeros(self.BINS, device=self.device, dtype=torch.int64)
        hist = hist.scatter_add(0, indices, torch.ones_like(indices)).to(self.DTYPE)
        return hist

    def forward(self, input: torch.Tensor):
        if self.calc_hist:
            hist = self.calc_histogram(input)
        else:
            hist = input

        cls_val = torch.arange(self.min_val, self.max_val, 1, dtype=self.DTYPE, device=self.device)
        masked_hist_class = hist * cls_val * self.mask
        masked_hist = hist * self.mask
        fc_sum = torch.sum(masked_hist_class, dim=2)
        num = torch.sum(masked_hist, dim=2)
        mean = fc_sum / num

        var_hist = torch.zeros(self.COMBINATIONS, dtype=self.DTYPE, device=self.device)
        for i, j in itertools.combinations(range(self.n_class), 2):
            var_hist += num[i, :] * num[j, :] * ((mean[i, :] - mean[j, :]) ** 2)

        var_hist = torch.where(torch.isnan(var_hist), self.ZERO, var_hist)
        thresh_idx = torch.argmax(var_hist)

        thresholds = []
        for i in range(self.n_class - 1):
            thresholds.append(self.threshold_indices[i][thresh_idx])
        return thresholds
