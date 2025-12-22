"""
纯Transformer基线模型（用于对比，禁用GCN）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from mt.models.transformer import EncoderLayer, DecoderLayer, PositionalEncoding
from mt.utils.masks import subsequent_mask, make_pad_attn_mask


class TransformerBaseline(nn.Module):
    """纯Transformer模型（无GCN）"""
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

        self.generator = nn.Linear(d_model, tgt_vocab_size)
        
        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化模型权重"""
        # Embedding层：normal初始化
        nn.init.normal_(self.src_embed.weight, mean=0, std=self.d_model ** -0.5)
        nn.init.normal_(self.tgt_embed.weight, mean=0, std=self.d_model ** -0.5)
        
        # Generator层：Xavier初始化
        nn.init.xavier_uniform_(self.generator.weight)
        if self.generator.bias is not None:
            nn.init.constant_(self.generator.bias, 0)

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

    def encode(self, src: torch.Tensor, src_attn_mask: torch.Tensor, adj_src: torch.Tensor = None):
        """编码（adj_src参数保留以兼容接口，但不会使用）"""
        src_emb = self.dropout(self.pos_encoder(self.src_embed(src)))  # [B,S,d_model]

        t_out = src_emb
        for layer in self.encoder:
            t_out = layer(t_out, src_attn_mask)

        return t_out

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_attn_mask: torch.Tensor,
               memory_attn_mask: torch.Tensor, adj_tgt: torch.Tensor = None):
        """解码（adj_tgt参数保留以兼容接口，但不会使用）"""
        tgt_emb = self.dropout(self.pos_encoder(self.tgt_embed(tgt)))  # [B,T,d_model]

        t_out = tgt_emb
        for layer in self.decoder:
            t_out = layer(t_out, memory, tgt_attn_mask, memory_attn_mask)

        return t_out

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, adj_src: torch.Tensor = None, adj_tgt: torch.Tensor = None):
        """前向传播（adj参数保留以兼容接口，但不会使用）"""
        src_attn_mask, tgt_attn_mask, memory_attn_mask = self.build_masks(src, tgt)
        memory = self.encode(src, src_attn_mask, adj_src)
        dec_out = self.decode(tgt, memory, tgt_attn_mask, memory_attn_mask, adj_tgt)
        logits = self.generator(dec_out)  # [B,T,V]
        return F.log_softmax(logits, dim=-1)

