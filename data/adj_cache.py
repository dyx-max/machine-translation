"""
邻接矩阵预计算与缓存（优化版：单进程顺序处理）
"""
from __future__ import annotations
import os
import json
from typing import Tuple, Optional
import torch
from tqdm import tqdm

from data.dependency import build_dep_adj


def _chunks(lst, n):
    """将列表分割成指定大小的块"""
    for i in range(0, len(lst), n):
        yield lst[i:i+n]


def compute_adj_sequential(texts, lang: str, max_len: int, chunk_size: int = 3000, 
                           desc: str = "计算邻接矩阵") -> torch.Tensor:
    """
    单进程顺序计算依存邻接矩阵（避免重复加载spaCy模型和IPC开销）
    
    Args:
        texts: 文本列表
        lang: 语言代码 ("zh" 或 "en")
        max_len: 最大序列长度
        chunk_size: 批处理大小（建议2000-5000）
        desc: 进度条描述
    
    Returns:
        [N, L, L] float32 tensor（调用方可再转dtype）
    """
    if len(texts) == 0:
        return torch.empty(0, max_len, max_len, dtype=torch.float32)
    
    # 在主进程中预加载spaCy模型（只加载一次，后续复用）
    from data.dependency import _get_nlp
    print(f"  预加载spaCy模型 ({lang})...", end=" ", flush=True)
    nlp = _get_nlp(lang)  # 第一次调用会加载模型，后续调用直接返回已加载的模型
    print("✓")
    
    all_adjs = []
    chunks = list(_chunks(texts, chunk_size))
    total_chunks = len(chunks)
    
    # 使用tqdm显示进度（按chunk显示，包含总数和速度信息）
    for chunk in tqdm(chunks, desc=desc, unit="chunk", total=total_chunks, 
                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'):
        with torch.no_grad():
            # 批量计算，spaCy模型已加载，不会重复加载
            # 注意：build_dep_adj 现在使用spaCy自己分词，不再需要sp参数
            adj_batch = build_dep_adj(chunk, lang=lang, max_len=max_len)  # [B, L, L]
            all_adjs.append(adj_batch.cpu())
    
    # 合并所有批次
    result = torch.cat(all_adjs, dim=0)
    print(f"  ✓ 完成，共 {len(texts)} 条文本，矩阵形状: {result.shape}")
    return result


def ensure_adj_cache(ds, src_lang: str, tgt_lang: str, max_src_len: int, max_tgt_in_len: int,
                     cache_dir: str, chunk_size: int = 3000, dtype: torch.dtype=torch.float16,
                     force_recompute: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    生成或加载邻接矩阵缓存（单进程顺序处理，带进度条）
    
    Args:
        ds: 数据集
        src_lang: 源语言代码
        tgt_lang: 目标语言代码
        max_src_len: 源语言最大长度
        max_tgt_in_len: 目标语言输入最大长度
        cache_dir: 缓存目录
        chunk_size: 批处理大小（建议2000-5000，默认3000）
        dtype: 保存的数据类型（默认float16）
        force_recompute: 是否强制重新计算（忽略已有缓存）
    
    Returns:
        (adj_src, adj_tgt_in) 两个tensor
    """
    os.makedirs(cache_dir, exist_ok=True)
    f_src = os.path.join(cache_dir, 'adj_src.pt')
    f_tgt = os.path.join(cache_dir, 'adj_tgt_in.pt')
    f_meta = os.path.join(cache_dir, 'meta.json')

    # 检查缓存是否存在
    if not force_recompute and os.path.exists(f_src) and os.path.exists(f_tgt):
        print(f"加载已存在的缓存: {cache_dir}")
        adj_src = torch.load(f_src, map_location='cpu')
        adj_tgt_in = torch.load(f_tgt, map_location='cpu')
        return adj_src, adj_tgt_in

    # 收集文本
    print(f"收集文本数据（共 {len(ds)} 条）...")
    src_texts = [ex['translation'][src_lang] for ex in ds]
    tgt_texts = [ex['translation'][tgt_lang] for ex in ds]

    # 单进程顺序计算（避免重复加载spaCy模型和IPC开销）
    print(f"开始计算邻接矩阵（chunk_size={chunk_size}）...")
    
    # 计算源语言邻接矩阵
    adj_src = compute_adj_sequential(
        src_texts, src_lang, max_src_len, 
        chunk_size=chunk_size, 
        desc=f"计算 {src_lang} 邻接矩阵"
    )
    
    # 计算目标语言邻接矩阵
    adj_tgt_in = compute_adj_sequential(
        tgt_texts, tgt_lang, max_tgt_in_len,
        chunk_size=chunk_size,
        desc=f"计算 {tgt_lang} 邻接矩阵"
    )

    # 降精度保存（节省磁盘空间）
    print(f"保存缓存到 {cache_dir}...")
    adj_src_fp16 = adj_src.to(dtype)
    adj_tgt_in_fp16 = adj_tgt_in.to(dtype)

    torch.save(adj_src_fp16, f_src)
    torch.save(adj_tgt_in_fp16, f_tgt)
    
    # 保存元数据
    with open(f_meta, 'w', encoding='utf-8') as f:
        json.dump({
            'count': len(ds),
            'src_lang': src_lang,
            'tgt_lang': tgt_lang,
            'max_src_len': max_src_len,
            'max_tgt_in_len': max_tgt_in_len,
            'dtype': str(dtype),
            'chunk_size': chunk_size
        }, f, ensure_ascii=False, indent=2)

    print(f"✓ 缓存保存完成: {cache_dir}")
    return adj_src, adj_tgt_in

