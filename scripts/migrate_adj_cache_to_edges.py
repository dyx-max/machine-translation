"""scripts.migrate_adj_cache_to_edges

将已有的“稠密邻接矩阵缓存”迁移为“边列表缓存”。

输入目录结构（旧）
-----------------
cache_root/
  train/
    adj_src.pt
    meta.json (可选)
  valid/
    adj_src.pt
    meta.json (可选)

输出目录结构（新）
-----------------
cache_root/
  train/
    edges_src.pt
    meta.json (更新 format=edge_list)
  valid/
    edges_src.pt
    meta.json

使用
----
python scripts/migrate_adj_cache_to_edges.py --cache_root cache --splits train valid

注意
----
- 该脚本不会删除 adj_src.pt；只会新增 edges_src.pt，并更新 meta.json。
- 对称边：从 A 的上三角提取 (i<j) 的边；默认忽略自环。
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List

import torch


def dense_adj_to_edges(adj: torch.Tensor) -> torch.Tensor:
    """单个样本：从 [L,L] 0/1 邻接矩阵提取边列表 [E,2] (i<j)，忽略自环。"""
    if adj.dim() != 2 or adj.size(0) != adj.size(1):
        raise ValueError(f"adj must be [L,L], got {tuple(adj.shape)}")

    a = adj
    if a.dtype != torch.bool:
        a = a > 0

    # 只取上三角 (i<j)
    triu = torch.triu(a, diagonal=1)
    idx = triu.nonzero(as_tuple=False)  # [E,2]
    return idx.to(dtype=torch.long)


def migrate_one_split(split_dir: str):
    f_adj = os.path.join(split_dir, "adj_src.pt")
    f_edges = os.path.join(split_dir, "edges_src.pt")
    f_meta = os.path.join(split_dir, "meta.json")

    if not os.path.exists(f_adj):
        print(f"[SKIP] {split_dir}: adj_src.pt not found")
        return

    print(f"[LOAD] {f_adj}")
    adj_all = torch.load(f_adj, map_location="cpu")
    if adj_all.dim() != 3:
        raise ValueError(f"adj_src.pt must be [N,L,L], got {tuple(adj_all.shape)}")

    edges_all: List[torch.Tensor] = []
    for i in range(adj_all.size(0)):
        edges_all.append(dense_adj_to_edges(adj_all[i]))

    print(f"[SAVE] {f_edges} (N={len(edges_all)})")
    torch.save(edges_all, f_edges, _use_new_zipfile_serialization=False)

    meta = {}
    if os.path.exists(f_meta):
        try:
            with open(f_meta, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            meta = {}

    meta.update(
        {
            "count": int(adj_all.size(0)),
            "max_src_len": int(adj_all.size(1)),
            "format": "edge_list",
            "source": "migrated_from_dense_adj",
        }
    )

    with open(f_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] {split_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_root", type=str, required=True, help="缓存根目录")
    ap.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "valid"],
        help="需要迁移的 split 子目录列表",
    )
    args = ap.parse_args()

    for sp in args.splits:
        split_dir = os.path.join(args.cache_root, sp)
        migrate_one_split(split_dir)


if __name__ == "__main__":
    main()

