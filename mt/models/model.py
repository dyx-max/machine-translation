"""
主模型：Transformer + GCN（严格三段式）
1. Transformer Encoder 得到 H
2. 用 H 作为 GCN 输入，得到 H'
3. 融合 H 和 H' 作为 Encoder 输出
4. Decoder 保持标准 Transformer 结构

V2 变更：
- GCN 的邻接归一化逻辑移入 mt.models.gcn（GCNLayer 内部）。
- 因此 forward/encode 期望 adj_src 为 0/1 邻接（通常含自环），无需预归一化。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mt.models.transformer import EncoderLayer, DecoderLayer, PositionalEncoding
from mt.models.gcn import SyntaxGCN
from mt.models.fusion import ParallelFusion
from mt.utils.masks import subsequent_mask, make_pad_attn_mask


class TransformerGCN(nn.Module):
    """Transformer + GCN 严格三段式融合模型"""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        src_vocab_size: int,
        tgt_vocab_size: int,
        max_len: int,
        pad_idx: int = 0,
        dropout: float = 0.1,
        gcn_layers: int = 2,
        fusion_mode: str = "gate",
        *,
        gcn_normalize: str | None = "sym",
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.d_model = d_model

        # 1. 词嵌入和位置编码
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)

        # 2. 编码器和解码器
        self.encoder = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.decoder = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        # 3. GCN 和融合模块（仅用于编码器）
        self.src_syntax_gcn = SyntaxGCN(
            d_model,
            num_layers=gcn_layers,
            dropout=dropout,
            normalize=gcn_normalize,
        )
        self.enc_fusion = ParallelFusion(d_model, mode=fusion_mode, dropout=dropout)

        # 4. 输出层
        self.generator = nn.Linear(d_model, tgt_vocab_size)

        self._init_weights()

    def _init_weights(self):
        """初始化模型权重"""
        nn.init.normal_(self.src_embed.weight, mean=0, std=self.d_model**-0.5)
        nn.init.normal_(self.tgt_embed.weight, mean=0, std=self.d_model**-0.5)

        nn.init.xavier_uniform_(self.generator.weight)
        if self.generator.bias is not None:
            nn.init.constant_(self.generator.bias, 0)

    def build_masks(self, src: torch.Tensor, tgt: torch.Tensor):
        """构建各种mask"""
        device = src.device
        B, S = src.size()
        B, T = tgt.size()

        src_pad = src == self.pad_idx
        tgt_pad = tgt == self.pad_idx

        src_attn_mask = make_pad_attn_mask(S, src_pad).to(device)
        tgt_attn_pad_mask = make_pad_attn_mask(T, tgt_pad).to(device)
        causal = subsequent_mask(T, device=device).to(device)
        tgt_attn_mask = tgt_attn_pad_mask + causal
        memory_attn_mask = make_pad_attn_mask(T, src_pad).to(device)

        return src_attn_mask, tgt_attn_mask, memory_attn_mask

    def encode(self, src: torch.Tensor, src_attn_mask: torch.Tensor, adj_src: torch.Tensor):
        """编码（严格三段式）"""
        src_emb = self.dropout(self.pos_encoder(self.src_embed(src)))  # [B,S,d_model]

        h = src_emb
        for layer in self.encoder:
            h = layer(h, src_attn_mask)

        adj_src = adj_src.to(src.device)
        src_pad_mask = src == self.pad_idx
        h_prime = self.src_syntax_gcn(h, adj_src, pad_mask=src_pad_mask)

        enc_out = self.enc_fusion(h, h_prime)
        return enc_out

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_attn_mask: torch.Tensor,
        memory_attn_mask: torch.Tensor,
    ):
        """标准 Transformer 解码器"""
        tgt_emb = self.dropout(self.pos_encoder(self.tgt_embed(tgt)))

        dec_out = tgt_emb
        for layer in self.decoder:
            dec_out = layer(dec_out, memory, tgt_attn_mask, memory_attn_mask)

        return dec_out

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, adj_src: torch.Tensor):
        """前向传播

        Args:
            src: 源序列 [B, S]
            tgt: 目标序列 [B, T]
            adj_src: 源语言邻接矩阵 [B, S, S]（0/1，通常含自环；无需预归一化）
        """
        src_attn_mask, tgt_attn_mask, memory_attn_mask = self.build_masks(src, tgt)
        memory = self.encode(src, src_attn_mask, adj_src)
        dec_out = self.decode(tgt, memory, tgt_attn_mask, memory_attn_mask)

        logits = self.generator(dec_out)
        return F.log_softmax(logits, dim=-1)

    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        adj_src: torch.Tensor,
        max_len: int = 100,
        bos_idx: int = 2,
        eos_idx: int = 3,
    ):
        """自回归生成（推理时使用）"""
        self.eval()
        device = next(self.parameters()).device

        src_attn_mask = (src == self.pad_idx).to(device)
        src_attn_mask = make_pad_attn_mask(src.size(1), src_attn_mask).to(device)
        memory = self.encode(src, src_attn_mask, adj_src)

        tgt = torch.ones(1, 1, dtype=torch.long, device=device) * bos_idx

        for _ in range(max_len):
            tgt_attn_mask = subsequent_mask(tgt.size(1), device=device)
            memory_attn_mask = make_pad_attn_mask(tgt.size(1), (src == self.pad_idx).to(device))

            dec_out = self.decode(tgt, memory, tgt_attn_mask, memory_attn_mask)
            next_token_logits = self.generator(dec_out[:, -1, :])
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)

            tgt = torch.cat([tgt, next_token], dim=1)
            if next_token.item() == eos_idx:
                break

        return tgt[0, 1:]
