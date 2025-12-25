"""
邻接矩阵预计算与缓存（优化版：多进程分批处理）
"""
from __future__ import annotations
import os
import json
import time
from typing import Tuple, Optional
import torch
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import tempfile
import shutil

from mt.data.dependency import build_dep_adj, _get_nlp

# --- 全局变量，用于在工作进程中存储模型 ---
g_nlp = None


def init_worker(lang: str):
    """工作进程初始化函数，用于加载spaCy模型"""
    global g_nlp
    print(f"  [Worker {os.getpid()}] Initializing spaCy model for '{lang}'...")
    g_nlp = _get_nlp(lang)
    print(f"  [Worker {os.getpid()}] Model for '{lang}' initialized.")


def _chunks(lst, n):
    """将列表分割成指定大小的块"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def _process_chunk(args):
    """处理单个chunk的辅助函数，用于多进程"""
    chunk_idx, chunk, lang, max_len, temp_dir = args
    global g_nlp
    if g_nlp is None:
        # 这是一个后备措施，正常情况下initializer应该已经加载了模型
        print(f"  [Worker {os.getpid()}] Fallback: Initializing spaCy model for '{lang}'...")
        g_nlp = _get_nlp(lang)

    adj_chunk = build_dep_adj(chunk, lang=lang, max_len=max_len, nlp=g_nlp)

    # 将结果保存到临时文件
    chunk_file = os.path.join(temp_dir, f'chunk_{chunk_idx:04d}.pt')
    temp_file = f"{chunk_file}.tmp"

    max_retries = 3
    retry_delay = 1  # 秒

    for attempt in range(max_retries):
        try:
            # 保存到临时文件（使用旧格式以提高稳定性）
            torch.save(adj_chunk.cpu().half(), temp_file, _use_new_zipfile_serialization=False)

            # 原子重命名
            if os.path.exists(chunk_file):
                os.remove(chunk_file)
            os.rename(temp_file, chunk_file)

            # 验证文件是否完整
            try:
                loaded = torch.load(chunk_file, map_location='cpu')
                if loaded.dim() != 3:  # 检查张量维度
                    raise ValueError("Invalid tensor dimensions")
                return chunk_idx, len(chunk), chunk_file
            except Exception as e:
                print(f"警告: 文件验证失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if os.path.exists(chunk_file):
                    os.remove(chunk_file)
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                if attempt == max_retries - 1:
                    raise
                time.sleep(retry_delay * (attempt + 1))
                continue

        except Exception as e:
            print(f"警告: 处理chunk {chunk_idx} 时出错 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(retry_delay * (attempt + 1))

    raise RuntimeError(f"无法处理chunk {chunk_idx}，已达到最大重试次数")


def compute_adj_parallel(texts, lang: str, max_len: int, chunk_size: int = 1000,
                         max_workers: int = None, desc: str = "计算邻接矩阵") -> torch.Tensor:
    """
    多进程并行计算依存邻接矩阵
    """
    if len(texts) == 0:
        return torch.empty(0, max_len, max_len, dtype=torch.float16)

    if max_workers is None:
        max_workers = min(mp.cpu_count(), 8)  # 默认最多8个worker

    temp_dir = tempfile.mkdtemp(prefix='adj_cache_')
    try:
        chunks = list(_chunks(texts, chunk_size))
        total_chunks = len(chunks)
        task_args = [(i, chunk, lang, max_len, temp_dir) for i, chunk in enumerate(chunks)]

        print(f"  使用 {max_workers} 个工作进程处理 {total_chunks} 个chunks...")
        completed = 0
        chunk_files = [None] * total_chunks

        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=init_worker,
            initargs=(lang,)
        ) as executor:
            futures = {executor.submit(_process_chunk, args): i for i, args in enumerate(task_args)}

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

        print("  合并chunk结果...")
        result = []
        valid_chunks = 0
        for chunk_file in chunk_files:
            if not chunk_file or not os.path.exists(chunk_file):
                print(f"  警告: 跳过不存在的chunk文件: {chunk_file}")
                continue
            try:
                chunk_data = torch.load(chunk_file, map_location='cpu')
                if chunk_data.dim() != 3:
                    print(f"  警告: 跳过无效的chunk文件 {chunk_file} (维度错误: {chunk_data.dim()})")
                    continue
                result.append(chunk_data)
                valid_chunks += 1
            except Exception as e:
                print(f"  警告: 加载chunk文件 {chunk_file} 时出错: {e}")
                continue

        if not result:
            raise RuntimeError("没有有效的chunk结果")

        print(f"  成功加载 {valid_chunks}/{len(chunk_files)} 个chunk文件")
        result = torch.cat(result, dim=0)
        print(f"  ✓ 完成，共 {len(texts)} 条文本，矩阵形状: {result.shape}")
        return result

    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


def compute_adj_sequential(texts, lang: str, max_len: int, chunk_size: int = 3000, 
                           desc: str = "计算邻接矩阵") -> torch.Tensor:
    """
    单进程顺序计算依存邻接矩阵（兼容旧版）
    """
    if len(texts) == 0:
        return torch.empty(0, max_len, max_len, dtype=torch.float32)
    
    print(f"  预加载spaCy模型 ({lang})...", end=" ", flush=True)
    nlp = _get_nlp(lang)
    print("✓")
    
    all_adjs = []
    chunks = list(_chunks(texts, chunk_size))
    total_chunks = len(chunks)
    
    for chunk in tqdm(chunks, desc=desc, unit="chunk", total=total_chunks, 
                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'):
        with torch.no_grad():
            adj_batch = build_dep_adj(chunk, lang=lang, max_len=max_len, nlp=nlp)
            all_adjs.append(adj_batch.cpu())
    
    result = torch.cat(all_adjs, dim=0)
    print(f"  ✓ 完成，共 {len(texts)} 条文本，矩阵形状: {result.shape}")
    return result


def ensure_adj_cache(ds, src_lang: str, max_src_len: int,
                     cache_dir: str, chunk_size: int = 3000, max_workers: int = None,
                     dtype: torch.dtype = torch.float16, force_recompute: bool = False,
                     use_parallel: bool = True) -> torch.Tensor:
    """
    生成或加载源语言邻接矩阵缓存（支持多进程并行处理）
    """
    os.makedirs(cache_dir, exist_ok=True)
    f_src = os.path.join(cache_dir, 'adj_src.pt')
    f_meta = os.path.join(cache_dir, 'meta.json')

    if not force_recompute and os.path.exists(f_src):
        cache_valid = False
        if os.path.exists(f_meta):
            try:
                with open(f_meta, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                    if meta.get('count', 0) == len(ds):
                        cache_valid = True
                    else:
                        print(f"警告: 缓存大小 ({meta.get('count', 0)}) 与数据集大小 ({len(ds)}) 不匹配，将重新计算")
            except Exception as e:
                print(f"警告: 读取缓存元数据失败: {e}，将重新计算")
        else:
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
            return adj_src
        else:
            print(f"缓存无效，将重新计算...")

    src_texts = [ex['translation'][src_lang] for ex in ds]

    compute_fn = compute_adj_parallel if use_parallel else compute_adj_sequential
    compute_kwargs = {'chunk_size': chunk_size}
    if use_parallel:
        compute_kwargs['max_workers'] = max_workers

    print("\n" + "=" * 60)
    print(f"开始计算邻接矩阵（{'多进程' if use_parallel else '单进程'}模式）...")
    if use_parallel:
        print(f"  工作进程数: {max_workers or '自动'}, 每chunk大小: {chunk_size}")
    print("=" * 60)

    print(f"\n计算源语言 ({src_lang}) 邻接矩阵...")
    adj_src = compute_fn(src_texts, src_lang, max_src_len, desc=f"计算 {src_lang} 邻接矩阵", **compute_kwargs)

    print(f"\n保存缓存到 {cache_dir}...")
    if adj_src.dtype != dtype:
        adj_src = adj_src.to(dtype)

    temp_dir = os.path.join(cache_dir, 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    try:
        temp_src = os.path.join(temp_dir, 'adj_src.pt.tmp')
        torch.save(adj_src, temp_src, _use_new_zipfile_serialization=False)


        meta = {
            'count': len(ds),
            'src_lang': src_lang,
            'max_src_len': max_src_len,
            'dtype': str(dtype),
            'chunk_size': chunk_size,
            'use_parallel': use_parallel,
            'max_workers': max_workers
        }
        with open(os.path.join(temp_dir, 'meta.json.tmp'), 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        for src, dst in [
            (temp_src, f_src),
            (os.path.join(temp_dir, 'meta.json.tmp'), f_meta)
        ]:
            if os.path.exists(dst):
                os.remove(dst)
            os.rename(src, dst)
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

    print(f"✓ 缓存保存完成: {cache_dir}")
    return adj_src
