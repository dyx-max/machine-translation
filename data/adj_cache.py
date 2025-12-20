"""
邻接矩阵预计算与缓存
"""
from __future__ import annotations
import os
import json
from typing import Tuple, Optional
import torch
from multiprocessing import Pool, cpu_count

from data.dependency import build_dep_adj


def _chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]


def _compute_chunk(args):
    texts, lang, max_len = args
    # 在子进程中调用，确保各自加载spaCy模型
    with torch.no_grad():
        adj = build_dep_adj(texts, sp=None, lang=lang, max_len=max_len)  # [B, L, L]
        return adj.cpu()


def compute_adj_parallel(texts, lang: str, max_len: int, num_workers: Optional[int]=None, chunk_size: int=256) -> torch.Tensor:
    """
    使用多进程批量计算依存邻接矩阵
    Returns: [N, L, L] float32 tensor（调用方可再转dtype）
    """
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    # 避免过多进程在Windows上开销过大
    num_workers = max(1, min(num_workers, 8))

    tasks = [(chunk, lang, max_len) for chunk in _chunks(texts, chunk_size)]
    if num_workers == 1:
        parts = [_compute_chunk(t) for t in tasks]
    else:
        with Pool(processes=num_workers) as pool:
            parts = list(pool.map(_compute_chunk, tasks))
    return torch.cat(parts, dim=0)


def ensure_adj_cache(ds, src_lang: str, tgt_lang: str, max_src_len: int, max_tgt_in_len: int,
                     cache_dir: str, num_workers: Optional[int]=None, dtype: torch.dtype=torch.float16) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    生成或加载邻接矩阵缓存。
    保存为：cache_dir/adj_src.pt, cache_dir/adj_tgt_in.pt 以及 meta.json
    """
    os.makedirs(cache_dir, exist_ok=True)
    f_src = os.path.join(cache_dir, 'adj_src.pt')
    f_tgt = os.path.join(cache_dir, 'adj_tgt_in.pt')
    f_meta = os.path.join(cache_dir, 'meta.json')

    if os.path.exists(f_src) and os.path.exists(f_tgt):
        adj_src = torch.load(f_src, map_location='cpu')
        adj_tgt_in = torch.load(f_tgt, map_location='cpu')
        return adj_src, adj_tgt_in

    # 收集文本
    src_texts = [ex['translation'][src_lang] for ex in ds]
    tgt_texts = [ex['translation'][tgt_lang] for ex in ds]

    # 多进程计算
    adj_src = compute_adj_parallel(src_texts, src_lang, max_src_len, num_workers=num_workers)
    adj_tgt_in = compute_adj_parallel(tgt_texts, tgt_lang, max_tgt_in_len, num_workers=num_workers)

    # 降精度保存
    adj_src = adj_src.to(dtype)
    adj_tgt_in = adj_tgt_in.to(dtype)

    torch.save(adj_src, f_src)
    torch.save(adj_tgt_in, f_tgt)
    with open(f_meta, 'w', encoding='utf-8') as f:
        json.dump({
            'count': len(ds),
            'src_lang': src_lang,
            'tgt_lang': tgt_lang,
            'max_src_len': max_src_len,
            'max_tgt_in_len': max_tgt_in_len,
            'dtype': str(dtype)
        }, f, ensure_ascii=False, indent=2)

    return adj_src, adj_tgt_in

