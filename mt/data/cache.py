"""
邻接矩阵预计算与缓存（优化版：多进程分批处理）
"""
from __future__ import annotations
import os
import json
from typing import Tuple, Optional
import torch
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import tempfile
import shutil

from mt.data.dependency import build_dep_adj, _get_nlp


def _chunks(lst, n):
    """将列表分割成指定大小的块"""
    for i in range(0, len(lst), n):
        yield lst[i:i+n]


def _process_chunk(args):
    """处理单个chunk的辅助函数，用于多进程"""
    chunk_idx, chunk, lang, max_len, temp_dir = args
    nlp = _get_nlp(lang)
    adj_chunk = build_dep_adj(chunk, lang=lang, max_len=max_len, nlp=nlp)
    
    # 将结果保存到临时文件
    chunk_file = os.path.join(temp_dir, f'chunk_{chunk_idx:04d}.pt')
    torch.save(adj_chunk.cpu().half(), chunk_file)
    return chunk_idx, len(chunk), chunk_file


def compute_adj_parallel(texts, lang: str, max_len: int, chunk_size: int = 1000, 
                        max_workers: int = None, desc: str = "计算邻接矩阵") -> torch.Tensor:
    """
    多进程并行计算依存邻接矩阵
    
    Args:
        texts: 文本列表
        lang: 语言代码 ("zh" 或 "en")
        max_len: 最大序列长度
        chunk_size: 每个chunk的文本数量
        max_workers: 最大工作进程数，None表示使用CPU核心数
        desc: 进度条描述
    
    Returns:
        [N, L, L] float16 tensor
    """
    if len(texts) == 0:
        return torch.empty(0, max_len, max_len, dtype=torch.float16)
    
    if max_workers is None:
        max_workers = min(mp.cpu_count(), 8)  # 默认最多8个worker
    
    # 创建临时目录存储chunk结果
    temp_dir = tempfile.mkdtemp(prefix='adj_cache_')
    try:
        # 预加载spaCy模型（避免子进程重复加载）
        print(f"  预加载spaCy模型 ({lang})...", end=" ", flush=True)
        _ = _get_nlp(lang)
        print("✓")
        
        # 将文本分块
        chunks = list(_chunks(texts, chunk_size))
        total_chunks = len(chunks)
        
        # 准备参数
        task_args = [(i, chunk, lang, max_len, temp_dir) 
                    for i, chunk in enumerate(chunks)]
        
        # 使用ProcessPoolExecutor并行处理
        print(f"  使用 {max_workers} 个工作进程处理 {total_chunks} 个chunks...")
        completed = 0
        chunk_files = [None] * total_chunks
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_process_chunk, args): i 
                      for i, args in enumerate(task_args)}
            
            # 使用tqdm显示进度
            with tqdm(total=len(texts), desc=desc, unit="样本") as pbar:
                for future in as_completed(futures):
                    chunk_idx, chunk_len, chunk_file = future.result()
                    chunk_files[chunk_idx] = chunk_file
                    completed += 1
                    pbar.update(chunk_len)
                    pbar.set_postfix({
                        '进度': f'{completed}/{total_chunks} chunks',
                        '状态': '处理中...'
                    })
        
        # 按顺序加载并合并所有chunk
        print("  合并chunk结果...")
        result = []
        for chunk_file in chunk_files:
            if chunk_file and os.path.exists(chunk_file):
                result.append(torch.load(chunk_file, map_location='cpu'))
        
        if not result:
            raise RuntimeError("没有有效的chunk结果")
            
        result = torch.cat(result, dim=0)
        print(f"  ✓ 完成，共 {len(texts)} 条文本，矩阵形状: {result.shape}")
        return result
        
    finally:
        # 清理临时文件
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


def compute_adj_sequential(texts, lang: str, max_len: int, chunk_size: int = 3000, 
                          desc: str = "计算邻接矩阵") -> torch.Tensor:
    """
    单进程顺序计算依存邻接矩阵（兼容旧版）
    
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
            adj_batch = build_dep_adj(chunk, lang=lang, max_len=max_len, nlp=nlp)  # [B, L, L]
            all_adjs.append(adj_batch.cpu())
    
    # 合并所有批次
    result = torch.cat(all_adjs, dim=0)
    print(f"  ✓ 完成，共 {len(texts)} 条文本，矩阵形状: {result.shape}")
    return result


def ensure_adj_cache(ds, src_lang: str, tgt_lang: str, max_src_len: int, max_tgt_in_len: int,
                     cache_dir: str, chunk_size: int = 3000, max_workers: int = None,
                     dtype: torch.dtype = torch.float16, force_recompute: bool = False,
                     use_parallel: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    生成或加载邻接矩阵缓存（支持多进程并行处理）
    
    Args:
        ds: 数据集
        src_lang: 源语言代码
        tgt_lang: 目标语言代码
        max_src_len: 源语言最大长度
        max_tgt_in_len: 目标语言输入最大长度
        cache_dir: 缓存目录
        chunk_size: 批处理大小（默认3000）
        max_workers: 最大工作进程数，None表示自动设置
        dtype: 保存的数据类型（默认float16）
        force_recompute: 是否强制重新计算（忽略已有缓存）
        use_parallel: 是否使用多进程并行处理（默认True）
    
    Returns:
        (adj_src, adj_tgt_in) 两个tensor
    """
    os.makedirs(cache_dir, exist_ok=True)
    f_src = os.path.join(cache_dir, 'adj_src.pt')
    f_tgt = os.path.join(cache_dir, 'adj_tgt_in.pt')
    f_meta = os.path.join(cache_dir, 'meta.json')

    # 检查缓存是否存在且大小匹配
    if not force_recompute and os.path.exists(f_src) and os.path.exists(f_tgt):
        # 检查元数据是否存在
        cache_valid = False
        if os.path.exists(f_meta):
            try:
                with open(f_meta, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                    cached_count = meta.get('count', 0)
                    # 检查数据集大小是否匹配
                    if cached_count == len(ds):
                        cache_valid = True
                    else:
                        print(f"警告: 缓存大小 ({cached_count}) 与数据集大小 ({len(ds)}) 不匹配，将重新计算")
            except Exception as e:
                print(f"警告: 读取缓存元数据失败: {e}，将重新计算")
        else:
            # 如果没有元数据，检查tensor大小
            try:
                adj_src_test = torch.load(f_src, map_location='cpu')
                if adj_src_test.shape[0] == len(ds):
                    cache_valid = True
                else:
                    print(f"警告: 缓存tensor大小 ({adj_src_test.shape[0]}) 与数据集大小 ({len(ds)}) 不匹配，将重新计算")
            except Exception as e:
                print(f"警告: 检查缓存失败: {e}，将重新计算")
        
        if cache_valid:
            print(f"加载已存在的缓存: {cache_dir} (大小: {len(ds)})")
            adj_src = torch.load(f_src, map_location='cpu')
            adj_tgt_in = torch.load(f_tgt, map_location='cpu')
            return adj_src, adj_tgt_in
        else:
            print(f"缓存无效，将重新计算...")

    # 收集文本
    print(f"收集文本数据（共 {len(ds)} 条）...")
    src_texts = [ex['translation'][src_lang] for ex in ds]
    tgt_texts = [ex['translation'][tgt_lang] for ex in ds]

    # 选择计算方式
    compute_fn = compute_adj_parallel if use_parallel else compute_adj_sequential
    compute_kwargs = {
        'chunk_size': chunk_size,
        'desc': f"计算 {src_lang} 邻接矩阵"
    }
    if use_parallel:
        compute_kwargs['max_workers'] = max_workers
    
    print("\n" + "=" * 60)
    print(f"开始计算邻接矩阵（{'多进程' if use_parallel else '单进程'}模式）...")
    if use_parallel:
        print(f"  工作进程数: {max_workers or '自动'}, 每chunk大小: {chunk_size}")
    print("=" * 60)
    
    # 计算源语言邻接矩阵
    print(f"\n计算源语言 ({src_lang}) 邻接矩阵...")
    adj_src = compute_fn(
        src_texts, src_lang, max_src_len, **compute_kwargs
    )
    
    # 计算目标语言邻接矩阵
    print(f"\n计算目标语言 ({tgt_lang}) 邻接矩阵...")
    compute_kwargs['desc'] = f"计算 {tgt_lang} 邻接矩阵"
    adj_tgt_in = compute_fn(
        tgt_texts, tgt_lang, max_tgt_in_len, **compute_kwargs
    )

    # 保存缓存
    print(f"\n保存缓存到 {cache_dir}...")
    
    # 转换为指定精度
    if adj_src.dtype != dtype:
        adj_src = adj_src.to(dtype)
    if adj_tgt_in.dtype != dtype:
        adj_tgt_in = adj_tgt_in.to(dtype)
    
    # 保存到临时文件，然后原子重命名，避免写入过程中断导致损坏
    temp_dir = os.path.join(cache_dir, 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # 保存源语言邻接矩阵
        temp_src = os.path.join(temp_dir, 'adj_src.pt.tmp')
        torch.save(adj_src, temp_src)
        
        # 保存目标语言邻接矩阵
        temp_tgt = os.path.join(temp_dir, 'adj_tgt_in.pt.tmp')
        torch.save(adj_tgt_in, temp_tgt)
        
        # 保存元数据
        meta = {
            'count': len(ds),
            'src_lang': src_lang,
            'tgt_lang': tgt_lang,
            'max_src_len': max_src_len,
            'max_tgt_in_len': max_tgt_in_len,
            'dtype': str(dtype),
            'chunk_size': chunk_size,
            'use_parallel': use_parallel,
            'max_workers': max_workers
        }
        with open(os.path.join(temp_dir, 'meta.json.tmp'), 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        
        # 原子重命名临时文件
        for src, dst in [
            (temp_src, f_src),
            (temp_tgt, f_tgt),
            (os.path.join(temp_dir, 'meta.json.tmp'), f_meta)
        ]:
            if os.path.exists(dst):
                os.remove(dst)
            os.rename(src, dst)
            
    finally:
        # 清理临时目录
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

    print(f"✓ 缓存保存完成: {cache_dir}")
    return adj_src, adj_tgt_in

