"""
融合模块：用于融合Transformer和GCN的输出
"""
import torch
import torch.nn as nn


class ParallelFusion(nn.Module):
    """并行融合模块"""
    def __init__(self, d_model: int, mode: str = "concat", dropout: float = 0.1):
        super().__init__()
        assert mode in ("concat", "gate"), "fusion mode must be 'concat' or 'gate'"
        self.mode = mode
        self.dropout = nn.Dropout(dropout)
        if mode == "concat":
            self.proj = nn.Linear(2 * d_model, d_model)
            self.norm = nn.LayerNorm(d_model)
        else:
            self.gate = nn.Linear(2 * d_model, d_model)
            self.norm = nn.LayerNorm(d_model)

    def forward(self, t_out, g_out):
        """
        Args:
            t_out: Transformer输出 [B, L, d_model]
            g_out: GCN输出 [B, L, d_model]
        Returns:
            融合后的输出 [B, L, d_model]
        """
        if self.mode == "concat":
            fused = torch.cat([t_out, g_out], dim=-1)
            fused = self.dropout(self.proj(fused))
            return self.norm(fused)
        else:
            gate = torch.sigmoid(self.gate(torch.cat([t_out, g_out], dim=-1)))
            fused = gate * t_out + (1 - gate) * g_out
            fused = self.dropout(fused)
            return self.norm(fused)

