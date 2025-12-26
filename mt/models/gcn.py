"""
图卷积网络（GCN）组件（改进版：内部归一化 + 可配置归一化方案）
"""
from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn


GCNNormType = Literal["none", "sym", "left"]


def normalize_adjacency(adj: torch.Tensor, norm_type: GCNNormType = "sym") -> torch.Tensor:
    """在模型内部对邻接矩阵进行归一化。

    Args:
        adj: [B, L, L] 或 [L, L] 的邻接矩阵，要求非负且包含自环。
        norm_type:
            - "none": 不做归一化，直接使用 A
            - "sym":  对称归一化 D^{-1/2} A D^{-1/2}
            - "left": 左归一化 D^{-1} A

    Returns:
        归一化后的邻接矩阵（形状与输入相同）。
    """
    if norm_type == "none":
        return adj

    if adj.dim() == 2:
        adj = adj.unsqueeze(0)  # [1, L, L]
        squeeze_back = True
    else:
        squeeze_back = False

    # 度向量: [B, L]
    deg = adj.sum(dim=-1)

    if norm_type == "sym":
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        D_inv_sqrt = torch.diag_embed(deg_inv_sqrt)  # [B, L, L]
        out = D_inv_sqrt @ adj @ D_inv_sqrt
    elif norm_type == "left":
        deg_inv = deg.pow(-1.0)
        deg_inv[deg_inv == float("inf")] = 0
        D_inv = torch.diag_embed(deg_inv)
        out = D_inv @ adj
    else:
        raise ValueError(f"未知的 GCN 归一化类型: {norm_type}")

    if squeeze_back:
        out = out.squeeze(0)
    return out


class GCNLayer(nn.Module):
    """单层GCN（内部度归一化，可学习边权重）"""

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        norm_type: GCNNormType = "sym",
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()
        self.norm = nn.LayerNorm(d_model)

        # 边权重学习（可选，用于调整邻接矩阵的整体影响）
        self.edge_weight = nn.Parameter(torch.ones(1))

        # 邻接矩阵归一化方式
        self.norm_type: GCNNormType = norm_type

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """前向传播。

        Args:
            x:        [B, L, d_model]
            adj:      [B, L, L]（未归一化）或 [L, L]
            pad_mask: [B, L]，True 表示 padding 位置，需要在图中屏蔽。

        Returns:
            out: [B, L, d_model]
        """
        # 确保 adj 具有 batch 维度
        if adj.dim() == 2:
            adj = adj.unsqueeze(0).expand(x.size(0), -1, -1)

        # 1. 先应用 padding mask，将 padding 行列置零
        if pad_mask is not None:
            # pad_mask: True 表示 padding
            valid = (~pad_mask).float()  # [B, L]
            mask_row = valid.unsqueeze(2)  # [B, L, 1]
            mask_col = valid.unsqueeze(1)  # [B, 1, L]
            adj = adj * mask_row * mask_col

        # 2. 加自环（确保图中存在自环）
        #    使用广播在 batch 维度上添加单位矩阵
        eye = torch.eye(adj.size(-1), device=adj.device, dtype=adj.dtype)
        adj = adj + eye.unsqueeze(0)

        # 3. 归一化邻接矩阵
        adj_norm = normalize_adjacency(adj, norm_type=self.norm_type)

        # 4. 应用全局可学习边权重
        adj_weighted = adj_norm * self.edge_weight

        # 5. GCN 聚合
        agg = torch.bmm(adj_weighted, x)  # [B, L, d_model]

        # 6. 线性 + 激活 + dropout
        out = self.linear(agg)
        out = self.dropout(self.act(out))

        # 7. 输出缩放，防止数值过大
        scale = x.size(-1) ** -0.5
        out = out * scale

        # 8. 残差连接 + LayerNorm
        return self.norm(x + out)


class SyntaxGCN(nn.Module):
    """语法GCN网络（支持多层与可配置归一化方案）。"""

    def __init__(
        self,
        d_model: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        norm_type: GCNNormType = "sym",
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [GCNLayer(d_model, dropout=dropout, norm_type=norm_type) for _ in range(num_layers)]
        )

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Args:
            x:        [B, L, d_model]
            adj:      [B, L, L] 未归一化邻接矩阵或 [L, L]
            pad_mask: [B, L]，True 表示 padding 位置

        Returns:
            [B, L, d_model]
        """
        for layer in self.layers:
            x = layer(x, adj, pad_mask)
        return x
