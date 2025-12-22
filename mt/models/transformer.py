"""
Transformer核心组件
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, dim_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, dim_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim_model, 2, dtype=torch.float32) * (-math.log(10000.0) / dim_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: [B, L, d_model]
        return x + self.pe[:x.size(1)].unsqueeze(0)


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q, k, v: [B, L*, d_model]
            mask:    [B, 1 or H, Lq, Lk], float tensor with values 0 or -inf
        Returns:
            [B, Lq, d_model]
        """
        B, Lq, _ = q.size()
        B, Lk, _ = k.size()
        B, Lv, _ = v.size()

        q = self.W_q(q).view(B, Lq, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(k).view(B, Lk, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(v).view(B, Lv, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            mask = mask.to(scores.device)
            scores = scores + mask

        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)  # [B, H, Lq, d_k]
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        out = self.W_o(out)
        return out


class AddNorm(nn.Module):
    """残差连接和层归一化"""
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sub_out):
        return self.norm(x + self.dropout(sub_out))


class FFN(nn.Module):
    """前馈神经网络"""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.ff2(self.dropout(self.relu(self.ff1(x))))


class EncoderLayer(nn.Module):
    """编码器层"""
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.add1 = AddNorm(d_model, dropout)
        self.ffn = FFN(d_model, d_ff, dropout)
        self.add2 = AddNorm(d_model, dropout)

    def forward(self, x, src_attn_mask=None):
        x2 = self.self_attention(x, x, x, src_attn_mask)
        x = self.add1(x, x2)
        x2 = self.ffn(x)
        x = self.add2(x, x2)
        return x


class DecoderLayer(nn.Module):
    """解码器层"""
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.add1 = AddNorm(d_model, dropout)

        self.enc_dec_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.add2 = AddNorm(d_model, dropout)

        self.ffn = FFN(d_model, d_ff, dropout)
        self.add3 = AddNorm(d_model, dropout)

    def forward(self, x, memory, tgt_attn_mask=None, memory_attn_mask=None):
        x2 = self.self_attention(x, x, x, tgt_attn_mask)
        x = self.add1(x, x2)

        x2 = self.enc_dec_attention(x, memory, memory, memory_attn_mask)
        x = self.add2(x, x2)

        x2 = self.ffn(x)
        x = self.add3(x, x2)
        return x

