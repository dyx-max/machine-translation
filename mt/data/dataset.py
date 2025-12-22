"""
数据集相关功能
"""
import torch
from torch.utils.data import Dataset

from mt.data.tokenizer import encode_sp
from mt.data.dependency import build_dep_adj


class WMTDataset(Dataset):
    """WMT数据集
    可选：提供预计算的邻接矩阵缓存，避免运行时重复构建
    如果 skip_adj=True，则跳过邻接矩阵计算（用于纯Transformer模型）
    """
    def __init__(self, ds, sp_src, sp_tgt, max_src_len=64, max_tgt_len=64,
                 adj_src_cache: torch.Tensor | None = None,
                 adj_tgt_in_cache: torch.Tensor | None = None,
                 skip_adj: bool = False):
        self.ds = ds
        self.sp_src = sp_src
        self.sp_tgt = sp_tgt
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.adj_src_cache = adj_src_cache  # [N, max_src_len, max_src_len] (fp16)
        # 注意：训练时decoder输入长度为 max_tgt_len-1
        self.adj_tgt_in_cache = adj_tgt_in_cache  # [N, max_tgt_len-1, max_tgt_len-1] (fp16)
        self.skip_adj = skip_adj  # 是否跳过邻接矩阵计算（用于纯Transformer）

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        zh, en = item["translation"]["zh"], item["translation"]["en"]
        src_ids = torch.tensor(encode_sp(self.sp_src, zh, self.max_src_len))
        tgt_ids = torch.tensor(encode_sp(self.sp_tgt, en, self.max_tgt_len))

        if self.skip_adj:
            # 跳过邻接矩阵计算（用于纯Transformer模型）
            # 返回单位矩阵以保持接口兼容性，但模型不会使用
            adj_src = torch.eye(self.max_src_len, dtype=torch.float32)
            adj_tgt_in = torch.eye(self.max_tgt_len - 1, dtype=torch.float32)
        elif self.adj_src_cache is not None and self.adj_tgt_in_cache is not None:
            # 从缓存读取，提升为float32以匹配后续计算精度
            adj_src = self.adj_src_cache[idx].to(dtype=torch.float32)
            adj_tgt_in = self.adj_tgt_in_cache[idx].to(dtype=torch.float32)
        else:
            # 在线构建（较慢，仅用于无缓存情况）
            # 注意：build_dep_adj 现在使用spaCy自己分词，不再需要sp参数
            adj_src = build_dep_adj([zh], lang="zh", max_len=self.max_src_len)[0]
            adj_tgt_in = build_dep_adj([en], lang="en", max_len=self.max_tgt_len-1)[0]

        return src_ids, tgt_ids, adj_src, adj_tgt_in


def collate_batch(batch):
    """批处理函数"""
    src, tgt, adj_src, adj_tgt_in = zip(*batch)
    return torch.stack(src), torch.stack(tgt), torch.stack(adj_src), torch.stack(adj_tgt_in)

