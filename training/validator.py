"""
验证相关功能
"""
import torch

from data.tokenizer import decode_sp
from data.dependency import build_dep_adj
from training.loss import LabelSmoothingLoss
from utils.decoder import beam_search_decode


def run_validation(model, dataloader, sp_src, sp_tgt, device, max_len=64, num_examples=2, pad_idx=0):
    """
    运行验证
    Args:
        model: 模型
        dataloader: 验证数据加载器
        sp_src: 源语言SentencePiece处理器
        sp_tgt: 目标语言SentencePiece处理器
        device: 设备
        max_len: 最大长度
        num_examples: 打印示例数量
        pad_idx: padding索引
    """
    model.eval()
    criterion = LabelSmoothingLoss(classes=sp_tgt.vocab_size(), smoothing=0.1, ignore_index=pad_idx)
    total_loss, count = 0, 0

    with torch.no_grad():
        for src_ids, tgt_ids, adj_src, adj_tgt in dataloader:
            src_ids, tgt_ids = src_ids.to(device), tgt_ids.to(device)
            tgt_in, tgt_out = tgt_ids[:, :-1], tgt_ids[:, 1:]

            # 构建依存树邻接矩阵
            src_texts = [decode_sp(sp_src, ids.cpu().tolist()) for ids in src_ids]
            tgt_texts_in = [decode_sp(sp_tgt, ids.cpu().tolist()) for ids in tgt_in]

            adj_src = build_dep_adj(src_texts, sp_src, lang="zh", max_len=src_ids.size(1)).to(device)
            adj_tgt = build_dep_adj(tgt_texts_in, sp_tgt, lang="en", max_len=tgt_in.size(1)).to(device)

            log_probs = model(src_ids, tgt_in, adj_src, adj_tgt)
            loss = criterion(log_probs.reshape(-1, log_probs.size(-1)), tgt_out.reshape(-1))
            total_loss += loss.item()
            count += 1

            if count <= num_examples:
                src_text = decode_sp(sp_src, src_ids[0].cpu().tolist())
                tgt_text = decode_sp(sp_tgt, tgt_ids[0].cpu().tolist())
                # 使用beam search解码
                pred_text = beam_search_decode(model, src_ids[0].cpu(), sp_src, sp_tgt, device, max_len, pad_idx, beam_size=4)
                print("-"*80)
                print(f"SOURCE:    {src_text}")
                print(f"TARGET:    {tgt_text}")
                print(f"PREDICTED: {pred_text}")

    print(f"Validation average loss: {total_loss/max(count,1):.4f}")

