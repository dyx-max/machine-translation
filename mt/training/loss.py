"""
损失函数
"""
import torch
import torch.nn as nn


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, ignore_index=0, eos_idx=2, eos_weight=1.0):
        super().__init__()
        self.classes = classes
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.eos_idx = eos_idx
        self.eos_weight = eos_weight

    def forward(self, log_probs, target):
        with torch.no_grad():
            true_dist = torch.full_like(log_probs, self.smoothing / (self.classes - 1))
            # 正常 token
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
            # PAD 清零
            true_dist[target == self.ignore_index] = 0
            # EOS 权重调整（可选）
            if self.eos_weight != 1.0:
                eos_mask = (target == self.eos_idx)
                true_dist[eos_mask, self.eos_idx] *= self.eos_weight

        loss = torch.sum(-true_dist * log_probs, dim=1)
        valid_mask = (target != self.ignore_index)
        return (loss * valid_mask).sum() / valid_mask.sum()


