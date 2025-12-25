"""
主模型：Transformer + GCN（严格三段式）
1. Transformer Encoder 得到 H
2. 用 H 作为 GCN 输入，得到 H'
3. 融合 H 和 H' 作为 Encoder 输出
4. Decoder 保持标准 Transformer 结构
"""
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
        gcn_layers: int = 2,  # 只需要一个 GCN 层数参数
        fusion_mode: str = "gate",  # 默认使用 gate 融合
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
        self.encoder = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        # 3. GCN 和融合模块（仅用于编码器）
        self.src_syntax_gcn = SyntaxGCN(d_model, num_layers=gcn_layers, dropout=dropout)
        self.enc_fusion = ParallelFusion(d_model, mode=fusion_mode, dropout=dropout)

        # 4. 输出层
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
        """
        编码（严格三段式）
        1. Transformer Encoder 得到 H
        2. 用 H 作为 GCN 输入，得到 H'
        3. 融合 H 和 H' 作为最终输出
        """
        # 1. 词嵌入 + 位置编码
        src_emb = self.dropout(self.pos_encoder(self.src_embed(src)))  # [B,S,d_model]

        # 2. Transformer 编码得到 H
        h = src_emb
        for layer in self.encoder:
            h = layer(h, src_attn_mask)  # [B,S,d_model]

        # 3. GCN 处理得到 H'
        adj_src = adj_src.to(src.device)
        src_pad_mask = (src == self.pad_idx)  # [B, S]，True表示padding位置
        h_prime = self.src_syntax_gcn(h, adj_src, pad_mask=src_pad_mask)  # [B,S,d_model]
        
        # 4. 融合 H 和 H'
        enc_out = self.enc_fusion(h, h_prime)  # [B,S,d_model]
        return enc_out

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_attn_mask: torch.Tensor,
               memory_attn_mask: torch.Tensor):
        """
        标准 Transformer 解码器
        不包含 GCN 或融合操作
        """
        # 1. 词嵌入 + 位置编码
        tgt_emb = self.dropout(self.pos_encoder(self.tgt_embed(tgt)))  # [B,T,d_model]

        # 2. 标准 Transformer 解码
        dec_out = tgt_emb
        for layer in self.decoder:
            dec_out = layer(
                dec_out,  # [B,T,d_model]
                memory,   # [B,S,d_model]
                tgt_attn_mask,    # [B,1,T,T]
                memory_attn_mask  # [B,1,1,S]
            )
        
        return dec_out  # [B,T,d_model]

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, adj_src: torch.Tensor):
        """
        前向传播
        
        Args:
            src: 源序列 [B, S]
            tgt: 目标序列 [B, T]
            adj_src: 源语言邻接矩阵 [B, S, S]
        """
        # 1. 构建各种mask
        src_attn_mask, tgt_attn_mask, memory_attn_mask = self.build_masks(src, tgt)
        
        # 2. 编码（包含GCN和融合）
        memory = self.encode(src, src_attn_mask, adj_src)  # [B,S,d_model]
        
        # 3. 解码（标准Transformer）
        dec_out = self.decode(tgt, memory, tgt_attn_mask, memory_attn_mask)  # [B,T,d_model]
        
        # 4. 输出层
        logits = self.generator(dec_out)  # [B,T,V]
        return F.log_softmax(logits, dim=-1)
    
    @torch.no_grad()
    def generate(self, src: torch.Tensor, adj_src: torch.Tensor, max_len: int = 100, 
                bos_idx: int = 2, eos_idx: int = 3):
        """
        自回归生成（推理时使用）
        
        Args:
            src: 源序列 [1, S]
            adj_src: 源语言邻接矩阵 [1, S, S]
            max_len: 最大生成长度
            bos_idx: 开始标记索引
            eos_idx: 结束标记索引
        """
        self.eval()
        device = next(self.parameters()).device
        
        # 1. 编码
        src_attn_mask = (src == self.pad_idx).to(device)  # [1, S]
        src_attn_mask = make_pad_attn_mask(src.size(1), src_attn_mask).to(device)  # [1,1,S,S]
        memory = self.encode(src, src_attn_mask, adj_src)  # [1,S,d_model]
        
        # 2. 初始化目标序列（以BOS开始）
        tgt = torch.ones(1, 1, dtype=torch.long, device=device) * bos_idx  # [1,1]
        
        # 3. 自回归生成
        for _ in range(max_len):
            # 构建目标mask
            tgt_attn_mask = subsequent_mask(tgt.size(1), device=device)  # [1,T,T]
            memory_attn_mask = make_pad_attn_mask(tgt.size(1), (src == self.pad_idx).to(device))  # [1,1,1,S]
            
            # 解码
            dec_out = self.decode(tgt, memory, tgt_attn_mask, memory_attn_mask)  # [1,T,d_model]
            next_token_logits = self.generator(dec_out[:, -1, :])  # [1,V]
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)  # [1,1]
            
            # 添加到序列
            tgt = torch.cat([tgt, next_token], dim=1)  # [1,T+1]
            
            # 如果生成了EOS，提前结束
            if next_token.item() == eos_idx:
                break
                
        return tgt[0, 1:]  # 去掉开头的BOS