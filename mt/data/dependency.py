"""
依存分析相关功能（V9：有向依存边 + 局部窗口边 + 修复长句切分错位）

变更摘要
--------
- build_dep_edges:
  1) 依存边默认保留方向信息（head -> dep），不再强制无向。
  2) 添加局部窗口边（自环 + 相邻双向边）以提升长句信息传播。
  3) 长句 chunk 解析时，仅保留 chunk 内部依存边，并修复索引错位。
- compute_edges_parallel 仍采用 chunk 落盘策略，避免内存/共享内存爆炸。

边列表格式
----------
每个样本对应一个 LongTensor，形状为 [E, 2]，每行是一条有向边 (src, dst)。
- 默认：依存边为 head->dep（有向）。
- 局部窗口边：自环 (i,i) + 相邻双向 (i,i-1),(i-1,i),(i,i+1),(i+1,i)。

注意
----
- edges_to_adjacency 当前仍会把边列表当作“无向”写入 adj[i,j] 与 adj[j,i]。
  如果你希望真正用有向图训练，需要同步修改 edges_to_adjacency 只写单向。
  这次按你的要求先修复 build_dep_edges。
"""

from __future__ import annotations

import os
import spacy
import torch
from typing import List, Optional, Tuple

from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import tempfile
import shutil
import time


# --- 全局加载spaCy模型 ---
_nlp_zh = None
_nlp_en = None

# --- 软切分配置 ---
SOFT_SPLIT_CONJUNCTIONS = {
    "zh": {"和", "并且", "但是", "因为", "所以", "如果", "虽然", "同时", "以及", "或", "或者"},
    "en": {"and", "but", "because", "although", "while", "which", "that", "or", "if", "so", "when"},
}


def _get_nlp(lang: str):
    """获取spaCy模型（延迟加载，并确保有sentencizer）"""
    global _nlp_zh, _nlp_en
    if lang == "zh":
        if _nlp_zh is None:
            _nlp_zh = spacy.load("zh_core_web_sm")
            if not _nlp_zh.has_pipe("sentencizer"):
                _nlp_zh.add_pipe("sentencizer")
        return _nlp_zh
    else:
        if _nlp_en is None:
            _nlp_en = spacy.load("en_core_web_sm")
            if not _nlp_en.has_pipe("sentencizer"):
                _nlp_en.add_pipe("sentencizer")
        return _nlp_en


def _find_soft_split_point(tokens: List[spacy.tokens.Token], split_point: int, lang: str, window: int = 5) -> int:
    """在切分点附近寻找最佳的软切分位置（优先连词，其次标点）"""
    start = max(0, split_point - window)
    end = min(len(tokens), split_point + window)
    search_window = tokens[start:end]
    conjunctions = SOFT_SPLIT_CONJUNCTIONS.get(lang, set())

    for i, token in enumerate(search_window):
        if token.text.lower() in conjunctions:
            return start + i + 1

    for i, token in enumerate(search_window):
        if token.is_punct and i > 0:
            return start + i + 1

    return -1


def _split_long_sentence(sent: spacy.tokens.span.Span, lang: str, limit: int) -> List[spacy.tokens.span.Span]:
    """对超过长度限制的单个句子进行智能软切分（修复了死循环问题）"""
    tokens = list(sent)
    if len(tokens) <= limit:
        return [sent]

    chunks = []
    offset = 0

    while offset < len(tokens):
        if offset + limit >= len(tokens):
            chunks.append(sent[offset:])
            break

        soft_split_point = _find_soft_split_point(tokens, offset + limit, lang)

        if soft_split_point > offset:
            actual_split = soft_split_point
        else:
            actual_split = offset + limit

        if actual_split <= offset:
            actual_split = offset + limit

        chunks.append(sent[offset:actual_split])
        offset = actual_split

    return chunks


def build_dep_edges(
    texts: List[str],
    lang: str = "zh",
    max_len: int = 64,
    spacy_parse_limit: int = 64,
    nlp=None,
) -> List[torch.Tensor]:
    """构建依存树“边列表”（有向 + 局部窗口边 + 过滤跨句/跨chunk依存）。

    依存边方向：默认保留 head -> dep。

    Returns
    -------
    List[torch.Tensor]
        长度为 B 的列表，每个元素是 LongTensor [E, 2]，表示有向边 (src, dst)。
    """
    parser = nlp if nlp is not None else _get_nlp(lang)
    batch_edges: List[torch.Tensor] = []

    for text in texts:
        doc = parser(text)
        num_tokens = len(doc)
        effective_len = min(num_tokens, max_len)

        # 现在 edges_set 存储有向边 (src, dst)
        edges_set: set[tuple[int, int]] = set()

        def _add_edge(i: int, j: int, directed: bool = True):
            """添加边。

            默认保留方向 (i -> j)。若 directed=False，则同时加反向边。
            """
            if i >= effective_len or j >= effective_len or i < 0 or j < 0 or i == j:
                return
            edges_set.add((i, j))
            if not directed:
                edges_set.add((j, i))

        # 1) 依存边（保留方向 head -> dep）
        for sent in doc.sents:
            if len(sent) > spacy_parse_limit:
                # 软切分：chunk 只是原 doc 的 span
                chunks = _split_long_sentence(sent, lang, spacy_parse_limit)
                for chunk in chunks:
                    chunk_start = chunk.start
                    chunk_len = len(chunk)
                    chunk_end = chunk_start + chunk_len

                    # 对 chunk 文本重新 parse（token.i 是 chunk_doc 内部索引）
                    chunk_doc = parser(chunk.text)
                    for token in chunk_doc:
                        # 只保留 chunk 内部依存：避免 head 跑到 chunk 外导致错位
                        # 使用 token.i / token.head.i （均是 chunk_doc 内索引）来判断
                        if 0 <= token.head.i < chunk_len:
                            # chunk_doc 索引 -> 原 doc 索引
                            dep_abs = chunk_start + token.i
                            head_abs = chunk_start + token.head.i
                            # 保留方向：head -> dep
                            _add_edge(head_abs, dep_abs, directed=True)
            else:
                for token in sent:
                    if token.sent == token.head.sent:
                        # 保留方向：head -> dep
                        _add_edge(token.head.i, token.i, directed=True)

        # 2) 局部窗口边（自环 + 相邻双向），长句信息传播救星
        for i in range(effective_len):
            edges_set.add((i, i))
            if i > 0:
                edges_set.add((i, i - 1))
                edges_set.add((i - 1, i))
            if i < effective_len - 1:
                edges_set.add((i, i + 1))
                edges_set.add((i + 1, i))

        if not edges_set:
            edges = torch.empty((0, 2), dtype=torch.long)
        else:
            edges = torch.tensor(sorted(edges_set), dtype=torch.long)

        batch_edges.append(edges)

    return batch_edges


def edges_to_adjacency(
    edges_list: List[torch.Tensor],
    max_len: int,
    *,
    add_self_loops: bool = True,
    normalize: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
    directed: bool = True,
) -> torch.Tensor:
    """将边列表转换为稠密邻接矩阵。

    Parameters
    ----------
    edges_list:
        List[Tensor[E,2]]，每个元素是一条样本的边集合，边格式为 (src, dst)。
    directed:
        - True: 仅写入 adj[src, dst]（有向邻接）
        - False: 同时写入 adj[src, dst] 与 adj[dst, src]（无向邻接）

    注意
    ----
    - 你现在的 build_dep_edges 会生成：
      1) 依存边 head->dep（有向）
      2) 局部窗口边（包含双向相邻边）
      3) 自环
      因此一般建议 directed=True。
    """
    bsz = len(edges_list)
    adj = torch.zeros((bsz, max_len, max_len), dtype=dtype, device=device)

    for b, edges in enumerate(edges_list):
        if edges is None or edges.numel() == 0:
            continue
        if edges.dtype != torch.long:
            edges = edges.long()
        if device is not None:
            edges = edges.to(device)

        i = edges[:, 0].clamp_(0, max_len - 1)
        j = edges[:, 1].clamp_(0, max_len - 1)
        adj[b, i, j] = 1
        if not directed:
            adj[b, j, i] = 1

    if add_self_loops:
        eye = torch.eye(max_len, dtype=dtype, device=adj.device).unsqueeze(0)
        adj = adj + eye

    if normalize is None:
        return adj

    deg = adj.sum(dim=-1)

    if normalize == "sym":
        deg_inv_sqrt = deg.clamp_min(1e-12).pow(-0.5)
        return adj * deg_inv_sqrt.unsqueeze(2) * deg_inv_sqrt.unsqueeze(1)

    if normalize == "row":
        deg_inv = deg.clamp_min(1e-12).pow(-1.0)
        return adj * deg_inv.unsqueeze(2)

    raise ValueError(f"Unknown normalize mode: {normalize}")


def build_dep_adj(
    texts,
    sp=None,
    lang: str = "zh",
    max_len: int = 64,
    spacy_parse_limit: int = 64,
    nlp=None,
):
    """兼容旧接口：构建稠密邻接矩阵（默认返回 sym 归一化）。"""
    edges_list = build_dep_edges(
        list(texts),
        lang=lang,
        max_len=max_len,
        spacy_parse_limit=spacy_parse_limit,
        nlp=nlp,
    )
    return edges_to_adjacency(
        edges_list,
        max_len=max_len,
        add_self_loops=True,
        normalize="sym",
        dtype=torch.float32,
        device=None,
    )


# -----------------------
# 批量计算：sequential / parallel
# -----------------------

def _chunks(lst: List[str], n: int):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


# 全局变量，用于在工作进程中存储模型
g_nlp = None


def init_worker(lang: str):
    global g_nlp
    print(f"  [Worker {os.getpid()}] Initializing spaCy model for '{lang}'...")
    g_nlp = _get_nlp(lang)
    print(f"  [Worker {os.getpid()}] Model for '{lang}' initialized.")


def _process_edges_chunk_to_file(args) -> Tuple[int, int, str]:
    """worker 计算一个 chunk 的边列表，并写入临时文件。

    Returns: (chunk_idx, chunk_len, chunk_file)
    """
    chunk_idx, chunk_texts, lang, max_len, spacy_parse_limit, temp_dir = args
    global g_nlp
    if g_nlp is None:
        g_nlp = _get_nlp(lang)

    edges_chunk = build_dep_edges(
        chunk_texts,
        lang=lang,
        max_len=max_len,
        spacy_parse_limit=spacy_parse_limit,
        nlp=g_nlp,
    )

    chunk_file = os.path.join(temp_dir, f"chunk_{chunk_idx:04d}.pt")
    temp_file = f"{chunk_file}.tmp"

    max_retries = 3
    retry_delay = 1

    for attempt in range(max_retries):
        try:
            torch.save(edges_chunk, temp_file, _use_new_zipfile_serialization=False)

            if os.path.exists(chunk_file):
                os.remove(chunk_file)
            os.rename(temp_file, chunk_file)

            loaded = torch.load(chunk_file, map_location="cpu")
            if not isinstance(loaded, list):
                raise ValueError("Invalid edges chunk format")

            return chunk_idx, len(chunk_texts), chunk_file

        except Exception:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception:
                    pass
            if attempt == max_retries - 1:
                raise
            time.sleep(retry_delay * (attempt + 1))

    raise RuntimeError(f"Failed to process edges chunk {chunk_idx}")


def compute_edges_sequential(
    texts: List[str],
    lang: str,
    max_len: int,
    *,
    spacy_parse_limit: int = 64,
    chunk_size: int = 3000,
    desc: str = "计算边列表",
) -> List[torch.Tensor]:
    """单进程顺序计算边列表。"""
    if len(texts) == 0:
        return []

    print(f"  预加载spaCy模型 ({lang})...", end=" ", flush=True)
    nlp = _get_nlp(lang)
    print("✓")

    all_edges: List[torch.Tensor] = []
    chunks = list(_chunks(texts, chunk_size))

    for chunk in tqdm(chunks, desc=desc, unit="chunk", total=len(chunks)):
        with torch.no_grad():
            edges_batch = build_dep_edges(
                chunk,
                lang=lang,
                max_len=max_len,
                spacy_parse_limit=spacy_parse_limit,
                nlp=nlp,
            )
        all_edges.extend(edges_batch)

    print(f"  ✓ 完成，共 {len(texts)} 条文本，边列表条目数: {len(all_edges)}")
    return all_edges


def compute_edges_parallel(
    texts: List[str],
    lang: str,
    max_len: int,
    *,
    spacy_parse_limit: int = 64,
    chunk_size: int = 1000,
    max_workers: Optional[int] = None,
    desc: str = "计算边列表",
) -> List[torch.Tensor]:
    """多进程并行计算边列表（chunk 落盘版）。"""
    if len(texts) == 0:
        return []

    if max_workers is None:
        max_workers = min(mp.cpu_count(), 8)

    temp_dir = tempfile.mkdtemp(prefix="edges_cache_")
    try:
        chunks = list(_chunks(texts, chunk_size))
        task_args = [(i, chunk, lang, max_len, spacy_parse_limit, temp_dir) for i, chunk in enumerate(chunks)]

        print(f"  使用 {max_workers} 个工作进程处理 {len(chunks)} 个chunks...")

        chunk_files: List[Optional[str]] = [None] * len(chunks)

        with ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker, initargs=(lang,)) as ex:
            futures = {ex.submit(_process_edges_chunk_to_file, args): args[0] for args in task_args}
            with tqdm(total=len(texts), desc=desc, unit="样本") as pbar:
                for fut in as_completed(futures):
                    chunk_idx, chunk_len, chunk_file = fut.result()
                    chunk_files[chunk_idx] = chunk_file
                    pbar.update(chunk_len)

        result: List[torch.Tensor] = []
        for cf in chunk_files:
            if not cf or not os.path.exists(cf):
                raise RuntimeError(f"Missing chunk file: {cf}")
            part = torch.load(cf, map_location="cpu")
            if not isinstance(part, list):
                raise ValueError(f"Invalid chunk content: {cf}")
            result.extend(part)

        print(f"  ✓ 完成，共 {len(texts)} 条文本，边列表条目数: {len(result)}")
        return result

    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
