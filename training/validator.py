"""
验证相关功能
"""
import torch

from data.tokenizer import decode_sp
from training.loss import LabelSmoothingLoss
from utils.decoder import beam_search_decode


def run_validation(model, dataloader, sp_src, sp_tgt, device, max_len=64, num_examples=2, pad_idx=0, debug=False):
    """
    运行验证（改进版：添加调试输出）
    Args:
        model: 模型
        dataloader: 验证数据加载器
        sp_src: 源语言SentencePiece处理器
        sp_tgt: 目标语言SentencePiece处理器
        device: 设备
        max_len: 最大长度
        num_examples: 打印示例数量
        pad_idx: padding索引
        debug: 是否打印调试信息
    """
    model.eval()
    criterion = LabelSmoothingLoss(classes=sp_tgt.vocab_size(), smoothing=0.1, ignore_index=pad_idx)
    total_loss, count = 0, 0

    with torch.no_grad():
        for src_ids, tgt_ids, adj_src, adj_tgt_in in dataloader:
            src_ids, tgt_ids = src_ids.to(device), tgt_ids.to(device)
            tgt_in, tgt_out = tgt_ids[:, :-1], tgt_ids[:, 1:]

            # 使用DataLoader中预计算的邻接矩阵
            adj_src = adj_src.to(device)
            adj_tgt_in = adj_tgt_in.to(device)

            log_probs = model(src_ids, tgt_in, adj_src, adj_tgt_in)
            loss = criterion(log_probs.reshape(-1, log_probs.size(-1)), tgt_out.reshape(-1))
            total_loss += loss.item()
            count += 1

            if count <= num_examples:
                src_text = decode_sp(sp_src, src_ids[0].cpu().tolist())
                tgt_text = decode_sp(sp_tgt, tgt_ids[0].cpu().tolist())
                # 验证阶段打印样例：使用beam search
                pred_text = beam_search_decode(model, src_ids[0].cpu(), sp_src, sp_tgt, device, max_len, pad_idx, beam_size=4)
                print("-"*80)
                print(f"SOURCE:    {src_text}")
                print(f"TARGET:    {tgt_text}")
                print(f"PREDICTED: {pred_text}")
                
                # 调试输出
                if debug and hasattr(model, 'encode'):
                    # 检查logits分布
                    logits = log_probs[0]  # [T, V]
                    eos_log_prob = logits[:, 2].max().item()  # EOS token的log概率
                    top5_probs = torch.topk(torch.exp(logits), 5, dim=-1)
                    
                    print(f"\n[DEBUG] EOS log prob: {eos_log_prob:.4f}")
                    print(f"[DEBUG] Logits stats - mean: {logits.mean().item():.4f}, std: {logits.std().item():.4f}")
                    print(f"[DEBUG] Top-5 token probs (first 3 positions):")
                    for i in range(min(3, logits.size(0))):
                        top_tokens = top5_probs.indices[i].cpu().tolist()
                        top_probs = top5_probs.values[i].cpu().tolist()
                        print(f"  Pos {i}: {list(zip(top_tokens, [f'{p:.3f}' for p in top_probs]))}")
                    
                    # 检查GCN输出（如果模型有GCN）
                    if hasattr(model, 'src_syntax_gcn'):
                        # 获取编码器中间输出
                        src_emb = model.dropout(model.pos_encoder(model.src_embed(src_ids)))
                        src_attn_mask, _, _ = model.build_masks(src_ids, src_ids)
                        t_out = src_emb
                        for i, layer in enumerate(model.encoder):
                            t_out = layer(t_out, src_attn_mask)
                            if i == 0:
                                transformer_first = t_out
                        
                        g_out = model.src_syntax_gcn(transformer_first, adj_src)
                        print(f"\n[DEBUG] Transformer output stats - mean: {t_out.mean().item():.4f}, std: {t_out.std().item():.4f}")
                        print(f"[DEBUG] GCN output stats - mean: {g_out.mean().item():.4f}, std: {g_out.std().item():.4f}")
                        print(f"[DEBUG] Output difference - mean: {(t_out - g_out).mean().item():.4f}, std: {(t_out - g_out).std().item():.4f}")

    print(f"Validation average loss: {total_loss/max(count,1):.4f}")

