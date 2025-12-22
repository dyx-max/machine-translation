"""
图卷积网络（GCN）组件（改进版：边权重、padding mask、输出scale）
"""
import torch
import torch.nn as nn


class GCNLayer(nn.Module):
    """单层GCN（改进版）"""
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()
        self.norm = nn.LayerNorm(d_model)
        
        # 边权重学习（可选，用于调整邻接矩阵的影响）
        self.edge_weight = nn.Parameter(torch.ones(1))

    def forward(self, x, adj, pad_mask=None):
        """
        Args:
            x:        [B, L, d_model]
            adj:      [B, L, L] (包含自环或外部已归一化)
            pad_mask: [B, L] (True表示padding位置，需要mask掉)
        Returns:
            [B, L, d_model]
        """
        # 1. 应用边权重（可学习的全局缩放）
        adj_weighted = adj * self.edge_weight
        
        # 2. 应用padding mask（mask掉padding位置的连接）
        if pad_mask is not None:
            # pad_mask: [B, L] -> [B, L, 1] 和 [B, 1, L]
            # 如果i或j是padding，则adj[i,j]=0
            mask_2d = (~pad_mask).float()  # [B, L]，True->0, False->1
            mask_row = mask_2d.unsqueeze(2)  # [B, L, 1]
            mask_col = mask_2d.unsqueeze(1)  # [B, 1, L]
            adj_mask = mask_row * mask_col   # [B, L, L]
            adj_weighted = adj_weighted * adj_mask
        
        # 3. GCN聚合
        agg = torch.bmm(adj_weighted, x)  # [B, L, d_model]
        
        # 4. 线性变换
        out = self.linear(agg)
        out = self.dropout(self.act(out))
        
        # 5. 输出scale（防止GCN输出过大）
        # 使用sqrt(d_model)进行缩放，类似Transformer的attention scaling
        scale = (x.size(-1) ** -0.5)
        out = out * scale
        
        # 6. 残差连接 + LayerNorm
        return self.norm(x + out)


class SyntaxGCN(nn.Module):
    """语法GCN网络（改进版）"""
    def __init__(self, d_model: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([GCNLayer(d_model, dropout) for _ in range(num_layers)])

    def forward(self, x, adj, pad_mask=None):
        """
        Args:
            x:        [B, L, d_model]
            adj:      [B, L, L]
            pad_mask: [B, L] (True表示padding位置)
        Returns:
            [B, L, d_model]
        """
        for layer in self.layers:
            x = layer(x, adj, pad_mask)
        return x


