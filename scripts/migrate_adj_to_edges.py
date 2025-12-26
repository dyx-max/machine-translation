"""将旧的稠密邻接矩阵缓存转换为新的边列表缓存格式。

用法示例：

    python -m scripts.migrate_adj_to_edges \
        --old-cache ./cache/dep/train \
        --new-cache ./cache/dep/train_edges \
        --max-len 64

假设旧缓存目录中包含：
    - adj_src.pt
    - meta.json

新缓存目录将生成：
    - edges_src.pt  （List[Tensor[num_edges, 2]]）
    - meta_edges.json
"""
from __future__ import annotations

import argparse
import json
import os
from typing import List

import torch

from mt.data.cache import edges_to_adjacency


def adjacency_to_edges(adj: torch.Tensor, max_len: int) -> torch.Tensor:
    """将单个 [L, L] 邻接矩阵转换为边列表 [E, 2]。

    注意：这里假设输入邻接矩阵已经包含自环；若不确定，可在外部先补自环。
    """
    if adj.dim() != 2:
        raise ValueError(f"期望 2D 邻接矩阵，实际维度: {adj.dim()}")

    L = min(adj.size(0), max_len)
    # 只取前 L x L 子矩阵
    sub_adj = (adj[:L, :L] > 0).to(torch.bool)

    src_idx, dst_idx = torch.nonzero(sub_adj, as_tuple=True)
    if src_idx.numel() == 0:
        return torch.empty(0, 2, dtype=torch.long)

    edges = torch.stack([src_idx, dst_idx], dim=-1).to(torch.long)
    return edges


def migrate_one_cache(old_cache_dir: str, new_cache_dir: str, max_len: int) -> None:
    os.makedirs(new_cache_dir, exist_ok=True)

    f_old_adj = os.path.join(old_cache_dir, "adj_src.pt")
    f_old_meta = os.path.join(old_cache_dir, "meta.json")

    if not os.path.exists(f_old_adj):
        raise FileNotFoundError(f"未找到旧邻接矩阵缓存文件: {f_old_adj}")

    print(f"加载旧邻接矩阵缓存: {f_old_adj}")
    adj_src: torch.Tensor = torch.load(f_old_adj, map_location="cpu")
    if adj_src.dim() != 3:
        raise ValueError(f"期望 3D 邻接矩阵缓存 [N, L, L]，实际维度: {adj_src.dim()}")

    num_samples = adj_src.size(0)
    print(f"  共 {num_samples} 条样本，矩阵形状: {tuple(adj_src.shape)}")

    edges_list: List[torch.Tensor] = []
    for i in range(num_samples):
        adj_i = adj_src[i].to(torch.float32)
        edges_i = adjacency_to_edges(adj_i, max_len=max_len)
        edges_list.append(edges_i)

    # 保存新的边列表缓存
    f_new_edges = os.path.join(new_cache_dir, "edges_src.pt")
    f_new_meta = os.path.join(new_cache_dir, "meta_edges.json")

    print(f"保存新的边列表缓存到: {f_new_edges}")
    torch.save(edges_list, f_new_edges, _use_new_zipfile_serialization=False)

    meta = {
        "count": num_samples,
        "max_src_len": max_len,
        "format": "edge_list",
        "source": "migrated_from_dense_adjacency",
        "old_cache_dir": os.path.abspath(old_cache_dir),
    }

    # 如果存在旧 meta.json，则复制部分信息
    if os.path.exists(f_old_meta):
        try:
            with open(f_old_meta, "r", encoding="utf-8") as f:
                old_meta = json.load(f)
            meta.update({
                k: old_meta[k]
                for k in ["src_lang", "chunk_size", "use_parallel", "max_workers"]
                if k in old_meta
            })
        except Exception as e:
            print(f"警告: 读取旧 meta.json 失败，将仅写入基础元数据: {e}")

    with open(f_new_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("迁移完成。")


def main() -> None:
    parser = argparse.ArgumentParser(description="将稠密邻接矩阵缓存迁移为边列表缓存")
    parser.add_argument("--old-cache", type=str, required=True, help="旧缓存目录（包含 adj_src.pt）")
    parser.add_argument("--new-cache", type=str, required=True, help="新缓存目录（输出 edges_src.pt）")
    parser.add_argument("--max-len", type=int, required=True, help="源句子最大长度（与训练配置一致）")

    args = parser.parse_args()

    migrate_one_cache(args.old_cache, args.new_cache, max_len=args.max_len)


if __name__ == "__main__":
    main()

