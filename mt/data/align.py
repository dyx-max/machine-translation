"""
subword↔word 对齐工具
用于在子词级别和词级别之间进行特征对齐
"""
from typing import List, Dict, Tuple
import torch


def word_to_subword_map(text: str, tokenizer) -> Dict[int, List[int]]:
    """
    构建词到子词的索引映射
    
    Args:
        text: 原始文本（词级别，空格分隔）
        tokenizer: SentencePiece分词器
    
    Returns:
        mapping: {word_idx: [subword_idx1, subword_idx2, ...]}
        例如: {0: [0, 1], 1: [2], 2: [3, 4]} 表示第0个词对应子词0和1
    """
    words = text.split()
    mapping = {}
    subword_idx = 0
    
    for word_idx, word in enumerate(words):
        # 编码单个词（不包括BOS/EOS）
        subword_ids = tokenizer.encode(word, out_type=int)
        # 移除BOS和EOS（如果存在）
        subword_ids = [sid for sid in subword_ids if sid not in (1, 2)]
        
        if subword_ids:
            mapping[word_idx] = list(range(subword_idx, subword_idx + len(subword_ids)))
            subword_idx += len(subword_ids)
        else:
            # 如果词被编码为空，仍然记录一个映射
            mapping[word_idx] = [subword_idx]
            subword_idx += 1
    
    return mapping


def pool_subwords_to_words(hidden_states: torch.Tensor, mapping: Dict[int, List[int]], 
                           mode: str = 'first') -> torch.Tensor:
    """
    将子词级别的hidden states聚合到词级别
    
    Args:
        hidden_states: [B, L_subword, d_model] 子词级别的hidden states
        mapping: 词到子词的映射（从word_to_subword_map获得）
        mode: 聚合模式
            - 'first': 取每个词的第一个子词
            - 'mean': 取每个词的所有子词的平均值
            - 'max': 取每个词的所有子词的最大值
    
    Returns:
        [B, L_word, d_model] 词级别的hidden states
    """
    B, L_subword, d_model = hidden_states.shape
    num_words = len(mapping)
    
    if num_words == 0:
        return torch.empty(B, 0, d_model, device=hidden_states.device, dtype=hidden_states.dtype)
    
    word_states = []
    
    for word_idx in range(num_words):
        subword_indices = mapping.get(word_idx, [])
        if not subword_indices:
            # 如果没有对应的子词，使用零向量
            word_states.append(torch.zeros(B, d_model, device=hidden_states.device, dtype=hidden_states.dtype))
            continue
        
        # 获取该词的所有子词表示
        subword_reprs = hidden_states[:, subword_indices, :]  # [B, num_subwords, d_model]
        
        if mode == 'first':
            word_repr = subword_reprs[:, 0, :]  # [B, d_model]
        elif mode == 'mean':
            word_repr = subword_reprs.mean(dim=1)  # [B, d_model]
        elif mode == 'max':
            word_repr = subword_reprs.max(dim=1)[0]  # [B, d_model]
        else:
            raise ValueError(f"Unknown pooling mode: {mode}")
        
        word_states.append(word_repr)
    
    return torch.stack(word_states, dim=1)  # [B, L_word, d_model]


def expand_words_to_subwords(word_states: torch.Tensor, mapping: Dict[int, List[int]]) -> torch.Tensor:
    """
    将词级别的表示扩展回子词级别
    
    Args:
        word_states: [B, L_word, d_model] 词级别的hidden states
        mapping: 词到子词的映射（从word_to_subword_map获得）
    
    Returns:
        [B, L_subword, d_model] 子词级别的hidden states
    """
    B, L_word, d_model = word_states.shape
    
    # 计算最大子词长度
    max_subword_idx = max(max(indices) for indices in mapping.values()) if mapping else 0
    L_subword = max_subword_idx + 1
    
    subword_states = torch.zeros(B, L_subword, d_model, 
                                 device=word_states.device, dtype=word_states.dtype)
    
    for word_idx in range(L_word):
        subword_indices = mapping.get(word_idx, [])
        if subword_indices:
            # 将词表示复制到所有对应的子词位置
            word_repr = word_states[:, word_idx, :]  # [B, d_model]
            for subword_idx in subword_indices:
                subword_states[:, subword_idx, :] = word_repr
    
    return subword_states

