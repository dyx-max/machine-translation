"""
依存分析相关功能（V7：边列表缓存 + 运行时转邻接矩阵）

变更摘要
--------
- 预处理阶段不再直接生成稠密邻接矩阵，而是生成“边列表”以便缓存与迁移。
- 提供 compute_edges_sequential / compute_edges_parallel 以批量生成边列表。
- 提供 edges_to_adjacency 以在运行时（DataLoader / 二次预处理）将边列表转为
  [B, L, L] 的邻接矩阵（可选择是否加自环、是否做归一化）。

兼容性
------
为了不破坏既有模块（如 decoding / tests / 旧缓存），本文件仍保留 build_dep_adj，
但其内部实现改为：build_dep_edges -> edges_to_adjacency ->（可选）归一化。

边列表格式
----------
每个样本对应一个 LongTensor，形状为 [E, 2]，每行是一条无向边 (i, j)。
- i, j 为 token 索引（0-based），且均满足 i < max_len, j < max_len。
- 默认不包含自环；是否加自环由 edges_to_adjacency 控制。

注意
----
spaCy 的 tokenization 与 SentencePiece 无关；这里的图结构基于 spaCy token。
本项目的 GCN 是在“固定长度 max_len 的 token 序列表示”上使用该图。
"""

from __future__ import annotations

import os
import spacy
import torch
from typing import List, Optional, Tuple

from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed


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
    """构建依存树“边列表”（优先软切分 + 过滤跨句依存边）。

    Returns
    -------
    List[torch.Tensor]
        长度为 B 的列表，每个元素是 LongTensor [E, 2]，表示无向边 (i, j)。
        边会被裁剪到 max_len 以内。
    """
    parser = nlp if nlp is not None else _get_nlp(lang)
    batch_edges: List[torch.Tensor] = []

    for text in texts:
        doc = parser(text)
        num_tokens = len(doc)
        effective_len = min(num_tokens, max_len)

        edges_set = set()  # (i, j) with i < j

        def _add_edge(i: int, j: int):
            if i >= effective_len or j >= effective_len:
                return
            if i == j:
                return
            a, b = (i, j) if i < j else (j, i)
            edges_set.add((a, b))

        for sent in doc.sents:
            if len(sent) > spacy_parse_limit:
                chunks = _split_long_sentence(sent, lang, spacy_parse_limit)
                for chunk in chunks:
                    chunk_text = chunk.text
                    chunk_doc = parser(chunk_text)
                    for token in chunk_doc:
                        if token.sent == token.head.sent:
                            chunk_start = chunk.start
                            i = chunk_start + token.i
                            j = chunk_start + token.head.i
                            _add_edge(i, j)
            else:
                for token in sent:
                    if token.sent == token.head.sent:
                        _add_edge(token.i, token.head.i)

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
) -> torch.Tensor:
    """将边列表转换为稠密邻接矩阵。

    Parameters
    ----------
    edges_list:
        List[Tensor[E,2]]，每个元素是一条样本的无向边 (i,j) (i<j)。
    max_len:
        输出邻接矩阵大小 L。
    add_self_loops:
        是否添加单位阵自环。
    normalize:
        - None: 不归一化，仅 0/1 邻接（可含自环）
        - "sym": D^{-1/2} A D^{-1/2}
        - "row": D^{-1} A
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
    """兼容旧接口：构建稠密邻接矩阵。

    旧逻辑中该函数会输出“已归一化”的邻接矩阵。
    现在默认仍返回 sym 归一化结果，以尽量保持历史行为。

    新 pipeline 请优先使用：build_dep_edges + edges_to_adjacency，并在模型内部归一化。
    """
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


def _process_edges_chunk(args) -> Tuple[int, List[torch.Tensor]]:
    chunk_idx, chunk_texts, lang, max_len, spacy_parse_limit = args
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
    return chunk_idx, edges_chunk


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
    """多进程并行计算边列表。"""
    if len(texts) == 0:
        return []

    if max_workers is None:
        max_workers = min(mp.cpu_count(), 8)

    chunks = list(_chunks(texts, chunk_size))
    task_args = [(i, chunk, lang, max_len, spacy_parse_limit) for i, chunk in enumerate(chunks)]

    print(f"  使用 {max_workers} 个工作进程处理 {len(chunks)} 个chunks...")

    edges_out: List[Optional[List[torch.Tensor]]] = [None] * len(chunks)
    with ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker, initargs=(lang,)) as ex:
        futures = {ex.submit(_process_edges_chunk, args): args[0] for args in task_args}
        with tqdm(total=len(texts), desc=desc, unit="样本") as pbar:
            for fut in as_completed(futures):
                chunk_idx, edges_chunk = fut.result()
                edges_out[chunk_idx] = edges_chunk
                pbar.update(len(edges_chunk))

    result: List[torch.Tensor] = []
    for part in edges_out:
        if part:
            result.extend(part)

    print(f"  ✓ 完成，共 {len(texts)} 条文本，边列表条目数: {len(result)}")
    return result
