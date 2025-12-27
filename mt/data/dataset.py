"""mt.data.dataset

Dataset + collate 负责：
- 文本 -> SentencePiece token ids
- 图结构：支持两种来源
  1) 旧：稠密邻接矩阵缓存 adj_src_cache: Tensor[N,L,L]
  2) 新：边列表缓存 edges_src_cache: List[Tensor[E,2]]

注意：
- 新 pipeline 默认在 collate 内把边列表转为稠密邻接矩阵（训练前一步）。
- 归一化不再在预处理做；由模型内部完成。
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset

from mt.data.tokenizer import encode_sp
from mt.data.dependency import build_dep_edges, edges_to_adjacency


class WMTDataset(Dataset):
    """WMT数据集

    Parameters
    ----------
    adj_src_cache:
        旧格式：预计算的稠密邻接矩阵缓存 [N, L, L]。
    edges_src_cache:
        新格式：预计算的边列表缓存，长度为 N；每个元素是 LongTensor[E,2]。
    skip_adj:
        为 True 时跳过依存图（纯Transformer），保持接口兼容。
    """

    def __init__(
        self,
        ds,
        sp_src,
        sp_tgt,
        max_src_len: int = 64,
        max_tgt_len: int = 64,
        *,
        adj_src_cache: Optional[torch.Tensor] = None,
        edges_src_cache: Optional[List[torch.Tensor]] = None,
        skip_adj: bool = False,
    ):
        self.ds = ds
        self.sp_src = sp_src
        self.sp_tgt = sp_tgt
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

        self.adj_src_cache = adj_src_cache
        self.edges_src_cache = edges_src_cache
        self.skip_adj = skip_adj

        if self.adj_src_cache is not None and self.edges_src_cache is not None:
            raise ValueError("Provide only one of adj_src_cache or edges_src_cache")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        item = self.ds[idx]
        zh, en = item["translation"]["zh"], item["translation"]["en"]
        src_ids = torch.tensor(encode_sp(self.sp_src, zh, self.max_src_len))
        tgt_ids = torch.tensor(encode_sp(self.sp_tgt, en, self.max_tgt_len))

        if self.skip_adj:
            # 训练脚本/模型不会使用，但为了接口统一仍返回占位
            adj_or_edges: Union[torch.Tensor, torch.Tensor] = torch.eye(self.max_src_len, dtype=torch.float32)
            return src_ids, tgt_ids, adj_or_edges

        # 旧：直接给稠密矩阵
        if self.adj_src_cache is not None:
            adj_src = self.adj_src_cache[idx].to(dtype=torch.float32)
            return src_ids, tgt_ids, adj_src

        # 新：边列表
        if self.edges_src_cache is not None:
            edges = self.edges_src_cache[idx]
            if not isinstance(edges, torch.Tensor):
                edges = torch.tensor(edges, dtype=torch.long)
            return src_ids, tgt_ids, edges

        # 无缓存：在线构建边列表（较慢）
        edges = build_dep_edges([zh], lang="zh", max_len=self.max_src_len)[0]
        return src_ids, tgt_ids, edges


def collate_batch(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    max_src_len: Optional[int] = None,
    edges_to_adj: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """批处理函数

    - 若第三个元素是 [L,L]，认为是稠密邻接矩阵，直接 stack。
    - 若第三个元素是 [E,2]，认为是边列表：默认在这里转为 [B,L,L]。

    之所以放在 collate：
    - 训练时 DataLoader 已经拿到 batch，转稠密矩阵更自然。
    - 便于在 GPU 上（pin_memory 后）快速搬运。
    """

    src, tgt, graph = zip(*batch)
    src_t = torch.stack(src)
    tgt_t = torch.stack(tgt)

    g0 = graph[0]

    # case 1) edges list: [E,2]
    # 必须在 "dense adjacency" 判断之前处理，否则会误判为 [L,L] 并 stack 失败。
    if g0.dim() == 2 and g0.size(-1) == 2:
        if not edges_to_adj:
            raise ValueError("edges_to_adj=False is not supported in current collate")
        if max_src_len is None:
            raise ValueError("max_src_len is required when edges_to_adj=True")

        edges_list = list(graph)
        adj = edges_to_adjacency(
            edges_list,
            max_len=max_src_len,
            add_self_loops=True,
            normalize=None,  # 归一化移到模型内部
            dtype=torch.float32,
        )
        return src_t, tgt_t, adj

    # case 2) dense adjacency: [L,L]
    if g0.dim() == 2:
        return src_t, tgt_t, torch.stack([g.to(dtype=torch.float32) for g in graph])

    raise ValueError(f"Unknown graph tensor shape in batch: {tuple(g0.shape)}")
