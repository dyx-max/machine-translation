"""
主模型：Transformer + GCN
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from mt.models.transformer import EncoderLayer, DecoderLayer, PositionalEncoding
from mt.models.gcn import SyntaxGCN
from mt.models.fusion import ParallelFusion
from mt.utils.masks import subsequent_mask, make_pad_attn_mask


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

    def encode(self, src: torch.Tensor, src_attn_mask: torch.Tensor, adj_src: torch.Tensor):
        """编码（改进版：GCN使用Transformer第一层输出，统一特征空间）"""
        src_emb = self.dropout(self.pos_encoder(self.src_embed(src)))  # [B,S,d_model]

        # Transformer编码
        t_out = src_emb
        for i, layer in enumerate(self.encoder):
            t_out = layer(t_out, src_attn_mask)
            # 让GCN使用Transformer第一层的输出，而不是初始embedding
            # 这样GCN和Transformer的特征空间更匹配
            if i == 0:
                transformer_first_out = t_out

        # GCN处理（使用Transformer第一层输出，而不是初始embedding）
        adj_src = adj_src.to(src.device)
        # 使用Transformer第一层输出作为GCN输入，添加残差连接确保信息流
        g_out = self.src_syntax_gcn(transformer_first_out, adj_src)  # [B,S,d_model]
        
        # 融合：Transformer深层输出 + GCN输出（基于Transformer第一层）
        enc_out = self.enc_fusion(t_out, g_out)
        return enc_out

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_attn_mask: torch.Tensor,
               memory_attn_mask: torch.Tensor, adj_tgt: torch.Tensor = None, use_tgt_gcn: bool = True):
        """
        解码（改进版：GCN使用Transformer第一层输出，统一特征空间）
        
        Args:
            tgt: 目标序列
            memory: 编码器输出
            tgt_attn_mask: 目标序列mask
            memory_attn_mask: 记忆mask
            adj_tgt: 目标语言邻接矩阵（可选，None时禁用GCN）
            use_tgt_gcn: 是否使用target端GCN（默认True，解码时可设为False）
        """
        tgt_emb = self.dropout(self.pos_encoder(self.tgt_embed(tgt)))  # [B,T,d_model]

        # Transformer解码
        t_out = tgt_emb
        for i, layer in enumerate(self.decoder):
            t_out = layer(t_out, memory, tgt_attn_mask, memory_attn_mask)
            # 让GCN使用Transformer第一层的输出
            if i == 0:
                transformer_first_out = t_out

        # GCN处理（可选：解码时可禁用target端GCN）
        if use_tgt_gcn and adj_tgt is not None:
            adj_tgt = adj_tgt.to(tgt.device)
            g_out = self.tgt_syntax_gcn(transformer_first_out, adj_tgt)  # [B,T,d_model]
            # 融合：Transformer深层输出 + GCN输出（基于Transformer第一层）
            dec_out = self.dec_fusion(t_out, g_out)
        else:
            # 禁用target端GCN，只使用Transformer输出
            dec_out = t_out
        
        return dec_out

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, adj_src: torch.Tensor, adj_tgt: torch.Tensor = None, use_tgt_gcn: bool = True):
        """
        前向传播
        
        Args:
            src: 源序列
            tgt: 目标序列
            adj_src: 源语言邻接矩阵
            adj_tgt: 目标语言邻接矩阵（可选）
            use_tgt_gcn: 是否使用target端GCN（默认True，训练时使用，推理时可禁用）
        """
        src_attn_mask, tgt_attn_mask, memory_attn_mask = self.build_masks(src, tgt)
        memory = self.encode(src, src_attn_mask, adj_src)
        dec_out = self.decode(tgt, memory, tgt_attn_mask, memory_attn_mask, adj_tgt, use_tgt_gcn=use_tgt_gcn)
        logits = self.generator(dec_out)  # [B,T,V]
        return F.log_softmax(logits, dim=-1)

