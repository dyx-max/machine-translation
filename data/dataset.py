"""
数据集相关功能
"""
import torch
from torch.utils.data import Dataset

from data.tokenizer import encode_sp
from data.dependency import build_dep_adj


class WMTDataset(Dataset):
    """WMT数据集"""
    def __init__(self, ds, sp_src, sp_tgt, max_src_len=64, max_tgt_len=64):
        self.ds = ds
        self.sp_src = sp_src
        self.sp_tgt = sp_tgt
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        zh, en = item["translation"]["zh"], item["translation"]["en"]
        src_ids = torch.tensor(encode_sp(self.sp_src, zh, self.max_src_len))
        tgt_ids = torch.tensor(encode_sp(self.sp_tgt, en, self.max_tgt_len))

        # 构建依存邻接矩阵
        adj_src = build_dep_adj([zh], self.sp_src, lang="zh", max_len=self.max_src_len)[0]
        adj_tgt = build_dep_adj([en], self.sp_tgt, lang="en", max_len=self.max_tgt_len)[0]

        return src_ids, tgt_ids, adj_src, adj_tgt


def collate_batch(batch):
    """批处理函数"""
    src, tgt, adj_src, adj_tgt = zip(*batch)
    return torch.stack(src), torch.stack(tgt), torch.stack(adj_src), torch.stack(adj_tgt)

