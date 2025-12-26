"""
数据集相关功能
"""
from __future__ import annotations

from typing import List, Optional, Literal, Tuple

import torch
from torch.utils.data import Dataset

from mt.data.tokenizer import encode_sp
from mt.data.dependency import build_dep_adj
from mt.data.cache import edges_to_adjacency


EdgeFormat = Literal["adjacency", "edge_list"]


class WMTDataset(Dataset):
    """WMT数据集

    支持两种图结构表示形式：
    - 稠密邻接矩阵（旧格式）：adj_src_cache: Tensor [N, L, L]
    - 边列表（新格式）：edges_src_cache: List[Tensor[num_edges, 2]]

    如果 skip_adj=True，则跳过图结构，返回单位矩阵（模型可忽略）。
    在使用边列表格式时，Dataset 会在 __getitem__ 中按需将单句
    的边列表转换为邻接矩阵（未归一化），归一化由 GCN 模型内部完成。
    """

    def __init__(
        self,
        ds,
        sp_src,
        sp_tgt,
        max_src_len: int = 64,
        max_tgt_len: int = 64,
        adj_src_cache: Optional[torch.Tensor] = None,
        edges_src_cache: Optional[List[torch.Tensor]] = None,
        skip_adj: bool = False,
    ) -> None:
        if adj_src_cache is not None and edges_src_cache is not None:
            raise ValueError("adj_src_cache 和 edges_src_cache 不能同时提供，只能二选一")

        self.ds = ds
        self.sp_src = sp_src
        self.sp_tgt = sp_tgt
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

        # 旧格式：稠密邻接矩阵 [N, L, L]
        self.adj_src_cache: Optional[torch.Tensor] = adj_src_cache

        # 新格式：边列表缓存（长度为 N 的列表，每个元素为 [E_i, 2] long tensor）
        self.edges_src_cache: Optional[List[torch.Tensor]] = edges_src_cache

        self.skip_adj: bool = skip_adj  # 是否跳过图结构（用于纯Transformer）

        if self.adj_src_cache is not None:
            if len(self.adj_src_cache) != len(self.ds):
                raise ValueError(
                    f"邻接矩阵缓存大小 ({len(self.adj_src_cache)}) 与数据集大小 ({len(self.ds)}) 不匹配"
                )

        if self.edges_src_cache is not None:
            if len(self.edges_src_cache) != len(self.ds):
                raise ValueError(
                    f"边列表缓存大小 ({len(self.edges_src_cache)}) 与数据集大小 ({len(self.ds)}) 不匹配"
                )

    def __len__(self) -> int:
        return len(self.ds)

    def _get_graph_from_cache(self, idx: int) -> torch.Tensor:
        """根据缓存类型返回单条样本的图结构（邻接矩阵，未归一化）。"""
        if self.adj_src_cache is not None:
            # 旧格式：直接取出稠密邻接矩阵并提升到 float32
            return self.adj_src_cache[idx].to(dtype=torch.float32)

        if self.edges_src_cache is not None:
            # 新格式：将边列表转换为邻接矩阵（这里不做度归一化）
            edges = self.edges_src_cache[idx]
            return edges_to_adjacency(edges, max_len=self.max_src_len, normalized=False)

        # 无缓存场景下，退回到在线构建（使用旧的 build_dep_adj 接口）
        item = self.ds[idx]
        zh = item["translation"]["zh"]
        # 注意：build_dep_adj 内部已使用 spaCy 分词，与 SentencePiece 无关
        adj = build_dep_adj([zh], lang="zh", max_len=self.max_src_len)[0]
        return adj.to(dtype=torch.float32)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        item = self.ds[idx]
        zh, en = item["translation"]["zh"], item["translation"]["en"]

        src_ids = torch.tensor(encode_sp(self.sp_src, zh, self.max_src_len))
        tgt_ids = torch.tensor(encode_sp(self.sp_tgt, en, self.max_tgt_len))

        if self.skip_adj:
            # 跳过图结构（用于纯Transformer模型）
            # 返回单位矩阵以保持接口兼容性，但模型可以选择忽略
            adj_src = torch.eye(self.max_src_len, dtype=torch.float32)
        else:
            adj_src = self._get_graph_from_cache(idx)

        return src_ids, tgt_ids, adj_src


def collate_batch(batch):
    """批处理函数

    Args:
        batch: List[Tuple[src_ids, tgt_ids, adj_src]]，其中
            - src_ids: [L_src]
            - tgt_ids: [L_tgt]
            - adj_src: [L_src, L_src]

    Returns:
        src:     [B, L_src]
        tgt:     [B, L_tgt]
        adj_src: [B, L_src, L_src]
    """
    src, tgt, adj_src = zip(*batch)
    return torch.stack(src), torch.stack(tgt), torch.stack(adj_src)
