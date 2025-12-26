"""
数据处理模块
"""

from mt.data.tokenizer import train_or_load_spm, encode_sp, decode_sp
from mt.data.dependency import build_dep_adj, build_dep_edges, edges_to_adjacency
from mt.data.dataset import WMTDataset, collate_batch
from mt.data.cache import ensure_adj_cache, ensure_edge_cache, compute_adj_sequential
from mt.data.align import word_to_subword_map, pool_subwords_to_words, expand_words_to_subwords

__all__ = [
    "train_or_load_spm",
    "encode_sp",
    "decode_sp",
    "build_dep_adj",
    "build_dep_edges",
    "edges_to_adjacency",
    "WMTDataset",
    "collate_batch",
    "ensure_adj_cache",
    "ensure_edge_cache",
    "compute_adj_sequential",
    "word_to_subword_map",
    "pool_subwords_to_words",
    "expand_words_to_subwords",
]
