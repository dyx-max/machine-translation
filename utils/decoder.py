"""
解码器相关功能
"""
import torch
import torch.nn.functional as F

from data.tokenizer import decode_sp
from data.dependency import build_dep_adj


def greedy_decode(model, src_ids, sp_src, sp_tgt, device, max_len=64, pad_idx=0):
    """
    贪心解码器（简单高效，适合训练初期）
    Args:
        model: 训练好的模型
        src_ids: 源序列ID [L]
        sp_src: 源语言SentencePiece处理器
        sp_tgt: 目标语言SentencePiece处理器
        device: 设备
        max_len: 最大长度
        pad_idx: padding索引
    Returns:
        解码后的文本字符串
    """
    model.eval()
    with torch.no_grad():
        src_ids = src_ids.unsqueeze(0).to(device)     # [1,S]
        
        # 构建源语言邻接矩阵（如果需要）
        try:
            adj_src = build_dep_adj([decode_sp(sp_src, src_ids[0].cpu().tolist())],
                                    lang="zh", max_len=src_ids.size(1)).to(device)
        except:
            adj_src = None

        # 构建memory
        src_attn_mask, _, _ = model.build_masks(src_ids, src_ids)
        if adj_src is not None:
            memory = model.encode(src_ids, src_attn_mask, adj_src)
        else:
            memory = model.encode(src_ids, src_attn_mask)

        # 初始化：从BOS开始
        tgt_ids = torch.tensor([[1]], device=device)  # BOS token

        for _ in range(max_len-1):
            # 检查是否已生成EOS
            if tgt_ids[0, -1].item() == 2:  # EOS
                break

            # 构建目标语言邻接矩阵（如果需要）
            try:
                adj_tgt = build_dep_adj([decode_sp(sp_tgt, tgt_ids[0].cpu().tolist())],
                                        lang="en", max_len=tgt_ids.size(1)).to(device)
            except:
                adj_tgt = None

            _, tgt_mask, mem_mask = model.build_masks(src_ids, tgt_ids)
            if adj_tgt is not None:
                dec_out = model.decode(tgt_ids, memory, tgt_mask, mem_mask, adj_tgt)
            else:
                dec_out = model.decode(tgt_ids, memory, tgt_mask, mem_mask)
            
            logits = model.generator(dec_out[:, -1:, :])  # [1,1,V]
            log_probs = F.log_softmax(logits, dim=-1)
            
            # 贪心选择：选择概率最高的token
            next_token = log_probs.argmax(dim=-1).item()
            tgt_ids = torch.cat([tgt_ids, torch.tensor([[next_token]], device=device)], dim=1)
            
            # 如果生成EOS，立即停止
            if next_token == 2:  # EOS
                break

        # 解码序列
        return decode_sp(sp_tgt, tgt_ids[0].cpu().tolist())


def beam_search_decode(model, src_ids, sp_src, sp_tgt, device, max_len=64, pad_idx=0, beam_size=4, length_penalty=0.6):
    """
    Beam search解码器（改进版：修复EOS处理，添加长度归一化）
    Args:
        model: 训练好的模型
        src_ids: 源序列ID [L]
        sp_src: 源语言SentencePiece处理器
        sp_tgt: 目标语言SentencePiece处理器
        device: 设备
        max_len: 最大长度
        pad_idx: padding索引
        beam_size: beam大小
        length_penalty: 长度惩罚系数（0.6是常用值，值越小越偏向短序列）
    Returns:
        解码后的文本字符串
    """
    model.eval()
    with torch.no_grad():
        src_ids = src_ids.unsqueeze(0).to(device)     # [1,S]
        # 注意：build_dep_adj 现在使用spaCy自己分词，不再需要sp参数
        # 检查模型是否需要adj_src（兼容纯Transformer基线）
        try:
            adj_src = build_dep_adj([decode_sp(sp_src, src_ids[0].cpu().tolist())],
                                    lang="zh", max_len=src_ids.size(1)).to(device)
        except:
            adj_src = None

        # 构建memory
        src_attn_mask, _, _ = model.build_masks(src_ids, src_ids)
        if adj_src is not None:
            memory = model.encode(src_ids, src_attn_mask, adj_src)
        else:
            memory = model.encode(src_ids, src_attn_mask)

        # beam初始化：(序列, 累积log概率, 是否已完成)
        beams = [(torch.tensor([[1]], device=device), 0.0, False)]  # BOS token

        for step in range(max_len-1):
            new_beams = []
            for seq, score, finished in beams:
                # 如果已完成（已生成EOS），直接保留
                if finished:
                    new_beams.append((seq, score, True))
                    continue

                # 检查当前序列是否包含EOS（不应该发生，但保险起见）
                if seq[0, -1].item() == 2:  # EOS
                    new_beams.append((seq, score, True))
                    continue

                # 构建目标语言邻接矩阵（如果需要）
                try:
                    adj_tgt = build_dep_adj([decode_sp(sp_tgt, seq[0].cpu().tolist())],
                                            lang="en", max_len=seq.size(1)).to(device)
                except:
                    adj_tgt = None

                _, tgt_mask, mem_mask = model.build_masks(src_ids, seq)
                if adj_tgt is not None:
                    dec_out = model.decode(seq, memory, tgt_mask, mem_mask, adj_tgt)
                else:
                    dec_out = model.decode(seq, memory, tgt_mask, mem_mask)
                logits = model.generator(dec_out[:, -1:, :])  # [1,1,V]
                log_probs = F.log_softmax(logits, dim=-1).squeeze(0).squeeze(0)  # [V]

                # 取topk
                topk_log_probs, topk_ids = torch.topk(log_probs, beam_size)
                for k in range(beam_size):
                    next_token = topk_ids[k].item()
                    next_seq = torch.cat([seq, topk_ids[k].view(1, 1)], dim=1)
                    
                    # 计算归一化得分（长度惩罚）
                    new_score = score + topk_log_probs[k].item()
                    seq_len = next_seq.size(1)
                    normalized_score = new_score / (seq_len ** length_penalty)
                    
                    # 检查是否生成EOS
                    is_finished = (next_token == 2)
                    
                    new_beams.append((next_seq, normalized_score, is_finished))

            # 按归一化得分排序，保留前beam_size个
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]

            # 如果所有beam都已完成，提前停止
            if all(finished for _, _, finished in beams):
                break

        # 选择得分最高的序列（优先选择已完成的）
        completed_beams = [(seq, score) for seq, score, finished in beams if finished]
        if completed_beams:
            best_seq = max(completed_beams, key=lambda x: x[1])[0][0].cpu().tolist()
        else:
            # 如果没有完成的，选择得分最高的
            best_seq = max(beams, key=lambda x: x[1])[0][0].cpu().tolist()
        
        return decode_sp(sp_tgt, best_seq)

