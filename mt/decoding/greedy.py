"""
贪心解码器

V2 变更
------
- 与训练 pipeline 保持一致：推理时构建的是 0/1 邻接（含自环）且不做归一化。
- 归一化由模型内部 GCN 完成（mt.models.gcn）。
"""

import torch
import torch.nn.functional as F

from mt.data.tokenizer import decode_sp
from mt.data.dependency import build_dep_edges, edges_to_adjacency


def greedy_decode(
    model,
    src_ids,
    sp_src,
    sp_tgt,
    device,
    max_len=64,
    pad_idx=0,
    repetition_penalty=1.0,
):
    """
    贪心解码器（改进版：支持重复惩罚）

    Args:
        model: 训练好的模型
        src_ids: 源序列ID [L]
        sp_src: 源语言SentencePiece处理器
        sp_tgt: 目标语言SentencePiece处理器
        device: 设备
        max_len: 最大长度
        pad_idx: padding索引
        repetition_penalty: 重复惩罚系数（>1.0惩罚重复，1.0不惩罚）

    Returns:
        解码后的文本字符串
    """
    model.eval()
    with torch.no_grad():
        src_ids = src_ids.unsqueeze(0).to(device) if src_ids.dim() == 1 else src_ids.to(device)

        # 构建源语言邻接矩阵（如果需要）
        needs_adj = hasattr(model, 'src_syntax_gcn')

        if needs_adj:
            try:
                text = decode_sp(sp_src, src_ids[0].cpu().tolist())
                edges = build_dep_edges([text], lang="zh", max_len=src_ids.size(1))[0]
                adj_src = edges_to_adjacency(
                    [edges],
                    max_len=src_ids.size(1),
                    add_self_loops=True,
                    normalize=None,
                    dtype=torch.float32,
                    device=device,
                    directed=True,
                )
            except Exception:
                adj_src = torch.eye(src_ids.size(1), device=device, dtype=torch.float32).unsqueeze(0)
        else:
            adj_src = None

        # 构建memory
        src_attn_mask, _, _ = model.build_masks(src_ids, src_ids)
        if adj_src is not None:
            memory = model.encode(src_ids, src_attn_mask, adj_src)
        else:
            memory = model.encode(src_ids, src_attn_mask)

        # 初始化：从BOS开始
        tgt_ids = torch.tensor([[1]], device=device)  # BOS token

        for step in range(max_len - 1):
            # 检查是否已生成EOS
            if tgt_ids[0, -1].item() == 2:  # EOS
                break

            _, tgt_mask, mem_mask = model.build_masks(src_ids, tgt_ids)

            # 标准 Transformer 解码
            dec_out = model.decode(tgt_ids, memory, tgt_mask, mem_mask)

            logits = model.generator(dec_out[:, -1:, :])  # [1,1,V]
            log_probs = F.log_softmax(logits, dim=-1)  # [1,1,V]

            # 应用重复惩罚
            if repetition_penalty > 1.0 and step > 0:
                current_tokens = tgt_ids[0, 1:].cpu().tolist()
                for token_id in current_tokens:
                    if token_id in [0, 1, 2]:
                        continue
                    if token_id < log_probs.size(-1):
                        log_probs[0, 0, token_id] /= repetition_penalty

            next_token = log_probs[0, 0].argmax().item()
            tgt_ids = torch.cat([tgt_ids, torch.tensor([[next_token]], device=device)], dim=1)

            if next_token == 2:
                break

        return decode_sp(sp_tgt, tgt_ids[0].cpu().tolist())
