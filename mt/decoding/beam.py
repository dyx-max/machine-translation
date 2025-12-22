"""
Beam Search解码器（改进版：修正长度惩罚，禁用target端GCN）
"""
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional

from mt.data.tokenizer import decode_sp
from mt.data.dependency import build_dep_adj


def beam_search_decode(
    model,
    src_ids,
    sp_src,
    sp_tgt,
    device,
    max_len=64,
    pad_idx=0,
    beam_size=4,
    length_penalty=0.6,
    repetition_penalty=1.0,
    n_best=1,
    early_stop=True,
    debug=False,
    disable_tgt_gcn=True  # 解码时禁用target端GCN
):
    """
    Beam search解码器（修正版：正确实现长度惩罚）
    
    Args:
        model: 训练好的模型
        src_ids: 源序列ID [L]
        sp_src: 源语言SentencePiece处理器
        sp_tgt: 目标语言SentencePiece处理器
        device: 设备
        max_len: 最大长度
        pad_idx: padding索引
        beam_size: beam大小
        length_penalty: 长度惩罚系数（NMT标准公式：score / ((5+len)/6)^alpha）
                       alpha=0.6是常用值，值越小越偏向短序列
        repetition_penalty: 重复惩罚系数（>1.0惩罚重复，1.0不惩罚）
        n_best: 返回top-n个结果
        early_stop: 是否在所有beam完成后提前停止
        debug: 是否打印调试信息
        disable_tgt_gcn: 是否禁用target端GCN（解码时推荐True）
    
    Returns:
        如果n_best=1，返回字符串；否则返回字符串列表
    """
    model.eval()
    with torch.no_grad():
        src_ids = src_ids.unsqueeze(0).to(device) if src_ids.dim() == 1 else src_ids.to(device)
        
        # 检查模型是否需要adj_src
        needs_adj_src = hasattr(model, 'src_syntax_gcn')
        
        if needs_adj_src:
            try:
                adj_src = build_dep_adj(
                    [decode_sp(sp_src, src_ids[0].cpu().tolist())],
                    lang="zh", 
                    max_len=src_ids.size(1)
                ).to(device)
            except:
                adj_src = torch.eye(src_ids.size(1), device=device, dtype=torch.float32)
        else:
            adj_src = None

        # 构建memory
        src_attn_mask, _, _ = model.build_masks(src_ids, src_ids)
        if adj_src is not None:
            memory = model.encode(src_ids, src_attn_mask, adj_src)
        else:
            memory = model.encode(src_ids, src_attn_mask)

        # beam初始化：(序列, 累积raw_log_prob, 长度, 是否已完成, 已生成的token集合)
        # 注意：分开存储raw_log_prob和length，排序时才应用length_penalty
        beams = [(torch.tensor([[1]], device=device), 0.0, 1, False, set())]  # BOS token

        # 定义长度惩罚函数（NMT标准公式）
        def compute_normalized_score(raw_score, length):
            """计算归一化得分（应用长度惩罚）"""
            if length_penalty == 0:
                return raw_score / length  # 简单的平均
            else:
                # NMT标准公式：normalized_score = raw_score / ((5 + length) / 6) ^ length_penalty
                return raw_score / (((5 + length) / 6) ** length_penalty)

        for step in range(max_len-1):
            new_beams = []
            for seq, raw_score, length, finished, seen_tokens in beams:
                # 如果已完成（已生成EOS），直接保留
                if finished:
                    new_beams.append((seq, raw_score, length, True, seen_tokens))
                    continue

                # 检查当前序列是否包含EOS
                if seq[0, -1].item() == 2:  # EOS
                    new_beams.append((seq, raw_score, length, True, seen_tokens))
                    continue

                # 构建目标语言邻接矩阵（解码时禁用target端GCN）
                if disable_tgt_gcn:
                    adj_tgt = None
                else:
                    needs_adj_tgt = hasattr(model, 'tgt_syntax_gcn')
                    if needs_adj_tgt:
                        try:
                            adj_tgt = build_dep_adj(
                                [decode_sp(sp_tgt, seq[0].cpu().tolist())],
                                lang="en", 
                                max_len=seq.size(1)
                            ).to(device)
                        except:
                            adj_tgt = torch.eye(seq.size(1), device=device, dtype=torch.float32)
                    else:
                        adj_tgt = None

                _, tgt_mask, mem_mask = model.build_masks(src_ids, seq)
                
                # 解码：如果禁用target端GCN，设置use_tgt_gcn=False
                if disable_tgt_gcn:
                    # 禁用target端GCN，只使用Transformer
                    dec_out = model.decode(seq, memory, tgt_mask, mem_mask, adj_tgt=None, use_tgt_gcn=False)
                else:
                    # 使用target端GCN（如果adj_tgt可用）
                    dec_out = model.decode(seq, memory, tgt_mask, mem_mask, adj_tgt, use_tgt_gcn=True)
                
                logits = model.generator(dec_out[:, -1:, :])  # [1,1,V]
                log_probs = F.log_softmax(logits, dim=-1)  # [1,1,V]
                
                # 修复：确保正确提取[V]向量
                if log_probs.dim() == 3:
                    log_probs = log_probs.squeeze(0).squeeze(0)  # [V]
                elif log_probs.dim() == 2:
                    log_probs = log_probs.squeeze(0)  # [V]

                # 应用重复惩罚
                if repetition_penalty > 1.0:
                    # 使用seen_tokens集合（已去重），对每个唯一token只惩罚一次
                    for token_id in seen_tokens:
                        # 对已生成的token降低概率（只对唯一token惩罚一次）
                        if token_id < log_probs.size(-1):
                            log_probs[token_id] /= repetition_penalty

                # 取topk
                topk_log_probs, topk_ids = torch.topk(log_probs, beam_size)  # topk_ids: [beam_size]
                
                # 调试：检查log_probs的多样性
                if debug and step == 0:
                    print(f"[DEBUG Beam Search] Step {step}, log_probs stats: mean={log_probs.mean().item():.4f}, std={log_probs.std().item():.4f}")
                    print(f"[DEBUG Beam Search] Top-{beam_size} tokens: {topk_ids.cpu().tolist()}, probs: {torch.exp(topk_log_probs).cpu().tolist()}")
                
                for k in range(beam_size):
                    next_token = topk_ids[k].item()  # 标量
                    next_token_tensor = torch.tensor([[next_token]], device=device)
                    next_seq = torch.cat([seq, next_token_tensor], dim=1)
                    
                    # 更新已生成的token集合
                    new_seen_tokens = seen_tokens.copy()
                    if next_token not in [0, 1, 2]:  # 不记录特殊token
                        new_seen_tokens.add(next_token)
                    
                    # 更新raw log概率和长度（不应用长度惩罚）
                    new_raw_score = raw_score + topk_log_probs[k].item()
                    new_length = length + 1
                    
                    # 检查是否生成EOS
                    is_finished = (next_token == 2)
                    
                    new_beams.append((next_seq, new_raw_score, new_length, is_finished, new_seen_tokens))

            # 应用长度惩罚并排序（使用已定义的compute_normalized_score函数）
            # 计算归一化得分并排序
            beams_with_scores = [
                (seq, raw_score, length, finished, seen_tokens, compute_normalized_score(raw_score, length))
                for seq, raw_score, length, finished, seen_tokens in new_beams
            ]
            beams = sorted(beams_with_scores, key=lambda x: x[5], reverse=True)[:beam_size]
            # 移除归一化得分，保持原始格式
            beams = [(seq, raw_score, length, finished, seen_tokens) for seq, raw_score, length, finished, seen_tokens, _ in beams]

            # 如果所有beam都已完成且early_stop，提前停止
            if early_stop and all(finished for _, _, _, finished, _ in beams):
                break

        # 选择得分最高的序列（优先选择已完成的）
        # 再次应用长度惩罚进行最终排序
        completed_beams = [
            (seq, raw_score, length, compute_normalized_score(raw_score, length))
            for seq, raw_score, length, finished, _ in beams if finished
        ]
        if completed_beams:
            best_beams = sorted(completed_beams, key=lambda x: x[3], reverse=True)[:n_best]
            best_seqs = [seq for seq, _, _, _ in best_beams]
        else:
            # 如果没有完成的，选择得分最高的
            incomplete_beams = [
                (seq, raw_score, length, compute_normalized_score(raw_score, length))
                for seq, raw_score, length, finished, _ in beams
            ]
            best_beams = sorted(incomplete_beams, key=lambda x: x[3], reverse=True)[:n_best]
            best_seqs = [seq for seq, _, _, _ in best_beams]
        
        # 解码序列
        results = [decode_sp(sp_tgt, seq[0].cpu().tolist()) for seq in best_seqs]
        
        if debug:
            print(f"[DEBUG Beam Search] Final beams: {len(beams)}, completed: {len(completed_beams) if completed_beams else 0}")
            if completed_beams:
                print(f"[DEBUG Beam Search] Best normalized scores: {[f'{score:.4f}' for _, _, _, score in best_beams[:3]]}")
        
        if n_best == 1:
            return results[0]
        else:
            return results
