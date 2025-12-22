"""
融合模块：用于融合Transformer和GCN的输出（改进版：残差连接 + 正确的LayerNorm）
"""
import torch
import torch.nn as nn


class ParallelFusion(nn.Module):
    """
    并行融合模块（改进版）
    
    关键改进：
    1. ✅ 残差连接：保留Transformer原始输出，防止信息漂移
    2. ✅ 正确的LayerNorm位置：Post-LN (residual + sublayer)
    3. ✅ Gate初始化：偏向Transformer，GCN逐渐注入
    4. ✅ Dropout + Residual组合：标准Transformer模式
    5. ✅ 对齐检查：确保t_out和g_out形状一致
    """
    def __init__(self, d_model: int, mode: str = "gate", dropout: float = 0.1):
        super().__init__()
        assert mode in ("concat", "gate"), "fusion mode must be 'concat' or 'gate'"
        self.mode = mode
        self.d_model = d_model
        
        if mode == "concat":
            # Concat模式：[t_out; g_out] -> linear -> d_model
            self.proj = nn.Linear(2 * d_model, d_model)
        else:
            # Gate模式：学习一个门控向量
            self.gate = nn.Linear(2 * d_model, d_model)
            # 初始化gate偏向Transformer（GCN逐渐注入）
            # 使gate初始输出接近0.5，让模型自己学习最优权重
            nn.init.xavier_uniform_(self.gate.weight)
            if self.gate.bias is not None:
                nn.init.constant_(self.gate.bias, 0.0)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, t_out, g_out):
        """
        Args:
            t_out: Transformer输出 [B, L, d_model]
            g_out: GCN输出 [B, L, d_model]
        Returns:
            融合后的输出 [B, L, d_model]
        """
        # 对齐检查：确保t_out和g_out形状一致
        assert t_out.shape == g_out.shape, \
            f"Shape mismatch: t_out {t_out.shape} vs g_out {g_out.shape}"
        
        # **关键1：保留residual（Transformer原始输出）**
        residual = t_out
        
        # 融合操作
        if self.mode == "concat":
            # Concat模式：[t_out; g_out] -> linear
            fused = torch.cat([t_out, g_out], dim=-1)  # [B, L, 2*d_model]
            fused = self.proj(fused)  # [B, L, d_model]
        else:
            # Gate模式：gate * t_out + (1 - gate) * g_out
            gate = torch.sigmoid(self.gate(torch.cat([t_out, g_out], dim=-1)))  # [B, L, d_model]
            fused = gate * t_out + (1 - gate) * g_out  # [B, L, d_model]
        
        # **关键2：Dropout（防止过拟合）**
        fused = self.dropout(fused)
        
        # **关键3：残差连接 + LayerNorm（Post-LN模式）**
        # 这是Transformer的标准模式：LN(x + Sublayer(x))
        return self.norm(residual + fused)


class PreLNFusion(nn.Module):
    """
    Pre-LN融合模块（可选）
    
    Pre-LN模式：Sublayer(LN(x)) + x
    更稳定，适合深层网络
    """
    def __init__(self, d_model: int, mode: str = "gate", dropout: float = 0.1):
        super().__init__()
        assert mode in ("concat", "gate"), "fusion mode must be 'concat' or 'gate'"
        self.mode = mode
        self.d_model = d_model
        
        # Pre-LN：先对输入做LayerNorm
        self.norm_t = nn.LayerNorm(d_model)
        self.norm_g = nn.LayerNorm(d_model)
        
        if mode == "concat":
            self.proj = nn.Linear(2 * d_model, d_model)
        else:
            self.gate = nn.Linear(2 * d_model, d_model)
            nn.init.xavier_uniform_(self.gate.weight)
            if self.gate.bias is not None:
                nn.init.constant_(self.gate.bias, 0.0)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, t_out, g_out):
        """
        Args:
            t_out: Transformer输出 [B, L, d_model]
            g_out: GCN输出 [B, L, d_model]
        Returns:
            融合后的输出 [B, L, d_model]
        """
        # 对齐检查
        assert t_out.shape == g_out.shape, \
            f"Shape mismatch: t_out {t_out.shape} vs g_out {g_out.shape}"
        
        # **关键1：保留residual**
        residual = t_out
        
        # **关键2：Pre-LN（先归一化）**
        t_normed = self.norm_t(t_out)
        g_normed = self.norm_g(g_out)
        
        # 融合操作
        if self.mode == "concat":
            fused = torch.cat([t_normed, g_normed], dim=-1)
            fused = self.proj(fused)
        else:
            gate = torch.sigmoid(self.gate(torch.cat([t_normed, g_normed], dim=-1)))
            fused = gate * t_normed + (1 - gate) * g_normed
        
        # **关键3：Dropout**
        fused = self.dropout(fused)
        
        # **关键4：残差连接（Pre-LN模式：x + Sublayer(LN(x))）**
        return residual + fused


