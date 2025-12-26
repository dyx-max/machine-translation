import torch
import torch.nn.functional as F
from typing import List, Tuple

from mt.data.tokenizer import decode_sp
from mt.data.dependency import build_dep_edges, edges_to_adjacency


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
    repetition_penalty=1.2,   # >1.0 惩罚重复
    n_best=1,
    early_stop=True,
    min_length=10,            # 新增：最小生成长度
    ngram_size=3,             # 新增：重复惩罚的 n-gram 大小
    debug=False
):
    model.eval()
    with torch.no_grad():
        src_ids = src_ids.unsqueeze(0).to(device) if src_ids.dim() == 1 else src_ids.to(device)

        # 构建源端邻接矩阵（如果模型需要）
        # 重要：与训练 pipeline 保持一致，这里构建的是 0/1 邻接（含自环）且不做归一化；
        # 归一化在模型内部完成。
        needs_adj_src = hasattr(model, 'src_syntax_gcn')
        if needs_adj_src:
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
                )
            except Exception:
                adj_src = torch.eye(src_ids.size(1), device=device, dtype=torch.float32).unsqueeze(0)
        else:
            adj_src = None

        src_attn_mask, _, _ = model.build_masks(src_ids, src_ids)
        memory = model.encode(src_ids, src_attn_mask, adj_src) if adj_src is not None else model.encode(src_ids, src_attn_mask)

        # beam 初始化
        beams = [(torch.tensor([[1]], device=device), 0.0, 1, False, [])]  # BOS token

        def compute_normalized_score(raw_score, length):
            return raw_score / (((5 + length) / 6) ** length_penalty)

        for step in range(max_len - 1):
            new_beams = []
            for seq, raw_score, length, finished, history in beams:
                if finished:
                    new_beams.append((seq, raw_score, length, True, history))
                    continue

                # EOS 过早生成时忽略
                if seq[0, -1].item() == 2 and length >= min_length:
                    new_beams.append((seq, raw_score, length, True, history))
                    continue

                _, tgt_mask, mem_mask = model.build_masks(src_ids, seq)
                dec_out = model.decode(seq, memory, tgt_mask, mem_mask)
                logits = model.generator(dec_out[:, -1:, :])
                log_probs = F.log_softmax(logits, dim=-1).squeeze(0).squeeze(0)  # [V]

                # n-gram 重复惩罚
                if repetition_penalty > 1.0 and len(history) >= ngram_size - 1:
                    recent_ngram = tuple(history[-(ngram_size - 1):])
                    for token_id in range(log_probs.size(-1)):
                        candidate_ngram = recent_ngram + (token_id,)
                        if candidate_ngram in [tuple(history[i:i+ngram_size]) for i in range(len(history)-ngram_size+1)]:
                            log_probs[token_id] /= repetition_penalty

                topk_log_probs, topk_ids = torch.topk(log_probs, beam_size)
                for k in range(beam_size):
                    next_token = topk_ids[k].item()
                    next_seq = torch.cat([seq, torch.tensor([[next_token]], device=device)], dim=1)
                    new_history = history + [next_token]
                    new_raw_score = raw_score + topk_log_probs[k].item()
                    new_length = length + 1
                    is_finished = (next_token == 2 and new_length >= min_length)
                    new_beams.append((next_seq, new_raw_score, new_length, is_finished, new_history))

            # 排序并截断
            beams = sorted(
                [(seq, raw_score, length, finished, history, compute_normalized_score(raw_score, length))
                 for seq, raw_score, length, finished, history in new_beams],
                key=lambda x: x[5], reverse=True
            )[:beam_size]
            beams = [(seq, raw_score, length, finished, history) for seq, raw_score, length, finished, history, _ in beams]

            if early_stop and all(finished for _, _, _, finished, _ in beams):
                break

        # 选择得分最高的序列（优先选择已完成的）
        completed_beams = [
            (seq, raw_score, length, compute_normalized_score(raw_score, length))
            for seq, raw_score, length, finished, _ in beams if finished
        ]
        if completed_beams:
            best_beams = sorted(completed_beams, key=lambda x: x[3], reverse=True)[:n_best]
        else:
            best_beams = sorted(
                [(seq, raw_score, length, compute_normalized_score(raw_score, length))
                 for seq, raw_score, length, finished, _ in beams],
                key=lambda x: x[3], reverse=True
            )[:n_best]

        # 解码序列
        best_seqs = [seq for seq, _, _, _ in best_beams]
        results = [decode_sp(sp_tgt, seq[0].cpu().tolist()) for seq in best_seqs]

        if debug:
            print(f"[DEBUG Beam Search] Final beams: {len(beams)}, completed: {len(completed_beams)}")
            print(f"[DEBUG Beam Search] Best normalized scores: {[f'{score:.4f}' for _, _, _, score in best_beams]}")

        return results[0] if n_best == 1 else results
