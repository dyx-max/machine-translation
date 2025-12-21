"""
解码器相关功能
"""
import torch
import torch.nn.functional as F

from data.tokenizer import decode_sp
from data.dependency import build_dep_adj


def beam_search_decode(model, src_ids, sp_src, sp_tgt, device, max_len=64, pad_idx=0, beam_size=4):
    """
    Beam search解码器
    Args:
        model: 训练好的模型
        src_ids: 源序列ID [L]
        sp_src: 源语言SentencePiece处理器
        sp_tgt: 目标语言SentencePiece处理器
        device: 设备
        max_len: 最大长度
        pad_idx: padding索引
        beam_size: beam大小
    Returns:
        解码后的文本字符串
    """
    model.eval()
    with torch.no_grad():
        src_ids = src_ids.unsqueeze(0).to(device)     # [1,S]
        # 注意：build_dep_adj 现在使用spaCy自己分词，不再需要sp参数
        adj_src = build_dep_adj([decode_sp(sp_src, src_ids[0].cpu().tolist())],
                                lang="zh", max_len=src_ids.size(1)).to(device)

        # 构建memory
        src_attn_mask, _, _ = model.build_masks(src_ids, src_ids)
        memory = model.encode(src_ids, src_attn_mask, adj_src)

        # beam初始化
        beams = [(torch.tensor([[1]], device=device), 0.0)]  # (序列, 累积log概率)

        for _ in range(max_len-1):
            new_beams = []
            for seq, score in beams:
                if seq[0, -1].item() == 2:  # EOS，保持不扩展
                    new_beams.append((seq, score))
                    continue

                # 注意：build_dep_adj 现在使用spaCy自己分词，不再需要sp参数
                adj_tgt = build_dep_adj([decode_sp(sp_tgt, seq[0].cpu().tolist())],
                                        lang="en", max_len=seq.size(1)).to(device)
                _, tgt_mask, mem_mask = model.build_masks(src_ids, seq)
                dec_out = model.decode(seq, memory, tgt_mask, mem_mask, adj_tgt)
                logits = model.generator(dec_out[:, -1:, :])  # [1,1,V]
                log_probs = F.log_softmax(logits, dim=-1).squeeze(0).squeeze(0)  # [V]

                # 取topk
                topk_log_probs, topk_ids = torch.topk(log_probs, beam_size)
                for k in range(beam_size):
                    next_seq = torch.cat([seq, topk_ids[k].view(1,1)], dim=1)
                    new_beams.append((next_seq, score + topk_log_probs[k].item()))

            # 保留前beam_size个
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]

            # 如果所有beam都以EOS结尾，提前停止
            if all(seq[0, -1].item() == 2 for seq, _ in beams):
                break

        # 选择得分最高的序列
        best_seq = beams[0][0][0].cpu().tolist()
        return decode_sp(sp_tgt, best_seq)

