"""
主模型：Transformer + GCN
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer import EncoderLayer, DecoderLayer, PositionalEncoding
from models.gcn import SyntaxGCN
from models.fusion import ParallelFusion
from utils.masks import subsequent_mask, make_pad_attn_mask


class TransformerGCN(nn.Module):
    """Transformer + GCN 并行融合模型"""
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
        gcn_layers_src: int = 2,
        gcn_layers_tgt: int = 2,
        fusion_mode: str = "concat",  # or "gate"
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.d_model = d_model

        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)

        self.encoder = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.src_syntax_gcn = SyntaxGCN(d_model, num_layers=gcn_layers_src, dropout=dropout)
        self.tgt_syntax_gcn = SyntaxGCN(d_model, num_layers=gcn_layers_tgt, dropout=dropout)

        self.enc_fusion = ParallelFusion(d_model, mode=fusion_mode, dropout=dropout)
        self.dec_fusion = ParallelFusion(d_model, mode=fusion_mode, dropout=dropout)

        self.generator = nn.Linear(d_model, tgt_vocab_size)

    def build_masks(self, src: torch.Tensor, tgt: torch.Tensor):
        """构建各种mask"""
        device = src.device
        B, S = src.size()
        B, T = tgt.size()

        src_pad = (src == self.pad_idx)
        tgt_pad = (tgt == self.pad_idx)

        src_attn_mask = make_pad_attn_mask(S, src_pad).to(device)
        tgt_attn_pad_mask = make_pad_attn_mask(T, tgt_pad).to(device)
        causal = subsequent_mask(T, device=device).to(device)
        tgt_attn_mask = tgt_attn_pad_mask + causal
        memory_attn_mask = make_pad_attn_mask(T, src_pad).to(device)

        return src_attn_mask, tgt_attn_mask, memory_attn_mask

    def encode(self, src: torch.Tensor, src_attn_mask: torch.Tensor, adj_src: torch.Tensor):
        """编码"""
        src_emb = self.dropout(self.pos_encoder(self.src_embed(src)))  # [B,S,d_model]

        t_out = src_emb
        for layer in self.encoder:
            t_out = layer(t_out, src_attn_mask)

        adj_src = adj_src.to(src.device)
        g_out = self.src_syntax_gcn(src_emb, adj_src)  # [B,S,d_model]

        enc_out = self.enc_fusion(t_out, g_out)
        return enc_out

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_attn_mask: torch.Tensor,
               memory_attn_mask: torch.Tensor, adj_tgt: torch.Tensor):
        """解码"""
        tgt_emb = self.dropout(self.pos_encoder(self.tgt_embed(tgt)))  # [B,T,d_model]

        t_out = tgt_emb
        for layer in self.decoder:
            t_out = layer(t_out, memory, tgt_attn_mask, memory_attn_mask)

        adj_tgt = adj_tgt.to(tgt.device)
        g_out = self.tgt_syntax_gcn(tgt_emb, adj_tgt)  # [B,T,d_model]

        dec_out = self.dec_fusion(t_out, g_out)
        return dec_out

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, adj_src: torch.Tensor, adj_tgt: torch.Tensor):
        """前向传播"""
        src_attn_mask, tgt_attn_mask, memory_attn_mask = self.build_masks(src, tgt)
        memory = self.encode(src, src_attn_mask, adj_src)
        dec_out = self.decode(tgt, memory, tgt_attn_mask, memory_attn_mask, adj_tgt)
        logits = self.generator(dec_out)  # [B,T,V]
        return F.log_softmax(logits, dim=-1)

