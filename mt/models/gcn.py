"""
图卷积网络（GCN）组件
"""
import torch
import torch.nn as nn


class GCNLayer(nn.Module):
    """单层GCN"""
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, adj):
        """
        Args:
            x:   [B, L, d_model]
            adj: [B, L, L] (包含自环或外部已归一化)
        Returns:
            [B, L, d_model]
        """
        agg = torch.bmm(adj, x)  # [B, L, d_model]
        out = self.linear(agg)
        out = self.dropout(self.act(out))
        return self.norm(x + out)


class SyntaxGCN(nn.Module):
    """语法GCN网络"""
    def __init__(self, d_model: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([GCNLayer(d_model, dropout) for _ in range(num_layers)])

    def forward(self, x, adj):
        """
        Args:
            x:   [B, L, d_model]
            adj: [B, L, L]
        Returns:
            [B, L, d_model]
        """
        for layer in self.layers:
            x = layer(x, adj)
        return x

