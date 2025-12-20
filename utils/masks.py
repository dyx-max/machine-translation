"""
Mask工具函数
"""
import torch


def subsequent_mask(T: int, device=None):
    """
    生成因果mask（causal mask）
    Args:
        T: 序列长度
        device: 设备
    Returns:
        [1, 1, T, T] 的mask，0表示保留，-inf表示屏蔽
    """
    m = torch.triu(torch.ones(T, T, device=device), diagonal=1)
    m = m.masked_fill(m == 1, float("-inf")).masked_fill(m == 0, 0.0)
    return m.unsqueeze(0).unsqueeze(1)  # [1, 1, T, T]


def make_pad_attn_mask(q_len: int, pad_mask_k: torch.Tensor):
    """
    生成padding mask
    Args:
        q_len: query序列长度
        pad_mask_k: [B, Lk] bool tensor，True表示padding位置
    Returns:
        [B, 1, Lq, Lk] float mask，0表示保留，-inf表示屏蔽
    """
    B, Lk = pad_mask_k.size()
    mask = pad_mask_k.unsqueeze(1).unsqueeze(2).expand(B, 1, q_len, Lk)  # [B,1,Lq,Lk]
    mask = mask.masked_fill(mask == True, float("-inf")).masked_fill(mask == False, 0.0)
    return mask

