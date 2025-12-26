"""mt.models.gcn

GCN 组件（V2：归一化移入模型内部 + 支持多种归一化方案）

变更点
------
- 预处理阶段不再输出归一化后的邻接矩阵；Dataset/Collate 返回的是 0/1 邻接（含自环）。
- 在 GCNLayer 内部根据 normalize 参数对邻接矩阵进行归一化。

normalize 可选：
- None: 不归一化
- "sym": D^{-1/2} A D^{-1/2}
- "row": D^{-1} A

padding mask
------------
pad_mask: [B, L]，True 表示 padding。
会在归一化前将与 padding 相关的边置 0，并保证 padding 自环为 0。
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def _normalize_adj(adj: torch.Tensor, normalize: Optional[str]) -> torch.Tensor:
    """归一化邻接矩阵。

    Args:
        adj: [B, L, L]
        normalize: None | "sym" | "row"
    """
    if normalize is None:
        return adj

    deg = adj.sum(dim=-1)  # [B,L]

    if normalize == "sym":
        deg_inv_sqrt = deg.clamp_min(1e-12).pow(-0.5)
        return adj * deg_inv_sqrt.unsqueeze(2) * deg_inv_sqrt.unsqueeze(1)

    if normalize == "row":
        deg_inv = deg.clamp_min(1e-12).pow(-1.0)
        return adj * deg_inv.unsqueeze(2)

    raise ValueError(f"Unknown normalize mode: {normalize}")


class GCNLayer(nn.Module):
    """单层GCN（改进版）"""

    def __init__(self, d_model: int, dropout: float = 0.1, *, normalize: Optional[str] = "sym"):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()
        self.norm = nn.LayerNorm(d_model)

        # 边权重学习（可选，用于调整邻接矩阵的影响）
        self.edge_weight = nn.Parameter(torch.ones(1))

        self.normalize = normalize

    def forward(self, x: torch.Tensor, adj: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向

        Args:
            x:        [B, L, d_model]
            adj:      [B, L, L]（0/1，通常含自环；无需预归一化）
            pad_mask: [B, L]（True 表示 padding）
        """
        # 1) 可学习边权重
        adj_w = adj * self.edge_weight

        # 2) padding mask：mask 掉 padding 相关连接
        if pad_mask is not None:
            keep = (~pad_mask).to(dtype=adj_w.dtype)  # [B,L]
            mask_row = keep.unsqueeze(2)
            mask_col = keep.unsqueeze(1)
            adj_w = adj_w * (mask_row * mask_col)

        # 3) 在层内部归一化
        adj_n = _normalize_adj(adj_w, self.normalize)

        # 4) GCN 聚合
        agg = torch.bmm(adj_n, x)  # [B, L, d_model]

        # 5) 线性 + 非线性
        out = self.linear(agg)
        out = self.dropout(self.act(out))

        # 6) 输出缩放（防止过大）
        out = out * (x.size(-1) ** -0.5)

        # 7) 残差 + LN
        return self.norm(x + out)


class SyntaxGCN(nn.Module):
    """语法GCN网络"""

    def __init__(
        self,
        d_model: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        *,
        normalize: Optional[str] = "sym",
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [GCNLayer(d_model, dropout, normalize=normalize) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, adj, pad_mask)
        return x
