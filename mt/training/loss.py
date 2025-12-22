"""
损失函数
"""
import torch
import torch.nn as nn


class LabelSmoothingLoss(nn.Module):
    """标签平滑损失"""
    def __init__(self, classes, smoothing=0.1, ignore_index=0, eos_idx=2, eos_weight=2.0):
        super().__init__()
        self.classes = classes
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.eos_idx = eos_idx
        self.eos_weight = eos_weight

    def forward(self, log_probs, target):
        """
        Args:
            log_probs: [B*L, V] log概率
            target: [B*L] 目标token ID
        Returns:
            平均损失
        """
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.classes - 2))
            mask = (target == self.ignore_index)
            safe_target = target.clone()
            safe_target[mask] = 0
            true_dist.scatter_(1, safe_target.unsqueeze(1), 1.0 - self.smoothing)
            true_dist[mask] = 0

        loss = torch.sum(-true_dist * log_probs, dim=1)

        # 对EOS token加权
        eos_mask = (target == self.eos_idx)
        loss[eos_mask] = loss[eos_mask] * self.eos_weight

        return torch.mean(loss)

