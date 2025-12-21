"""
独立的预计算脚本：提前生成邻接矩阵缓存
在训练前运行此脚本，可以提前计算并保存邻接矩阵，训练时直接加载，完全没有开销。
"""
import os
import nltk
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

import torch
from datasets import load_dataset

from config import Config
from data.adj_cache import ensure_adj_cache


def main():
    """主函数：预计算并缓存邻接矩阵"""
    config = Config()
    
    print("=" * 60)
    print("邻接矩阵预计算脚本")
    print("=" * 60)
    print(f"训练集大小: {config.train_size}")
    print(f"源语言最大长度: {config.max_src_len}")
    print(f"目标语言最大长度: {config.max_tgt_len}")
    print(f"批处理大小 (chunk_size): {config.precompute_chunk_size}")
    print("=" * 60)
    print()

    # 加载数据集
    print("加载数据集...")
    wmt = load_dataset("wmt17", "zh-en")
    ds_train = wmt["train"].select(range(config.train_size))
    ds_valid = wmt["validation"]
    print(f"✓ 训练集: {len(ds_train)} 条")
    print(f"✓ 验证集: {len(ds_valid)} 条")
    print()

    # 预计算训练集缓存
    print("=" * 60)
    print("预计算训练集邻接矩阵")
    print("=" * 60)
    cache_train_dir = os.path.join(config.cache_root, "train")
    adj_src_train, adj_tgt_in_train = ensure_adj_cache(
        ds_train, 
        src_lang="zh", 
        tgt_lang="en",
        max_src_len=config.max_src_len, 
        max_tgt_in_len=config.max_tgt_len-1,
        cache_dir=cache_train_dir, 
        chunk_size=config.precompute_chunk_size, 
        dtype=torch.float16,
    )
    print(f"✓ 训练集缓存完成")
    print(f"  - 源语言矩阵形状: {adj_src_train.shape}")
    print(f"  - 目标语言矩阵形状: {adj_tgt_in_train.shape}")
    print()

    # 预计算验证集缓存
    print("=" * 60)
    print("预计算验证集邻接矩阵")
    print("=" * 60)
    cache_valid_dir = os.path.join(config.cache_root, "valid")
    adj_src_valid, adj_tgt_in_valid = ensure_adj_cache(
        ds_valid, 
        src_lang="zh", 
        tgt_lang="en",
        max_src_len=config.max_src_len, 
        max_tgt_in_len=config.max_tgt_len-1,
        cache_dir=cache_valid_dir, 
        chunk_size=config.precompute_chunk_size, 
        dtype=torch.float16,
    )
    print(f"✓ 验证集缓存完成")
    print(f"  - 源语言矩阵形状: {adj_src_valid.shape}")
    print(f"  - 目标语言矩阵形状: {adj_tgt_in_valid.shape}")
    print()

    print("=" * 60)
    print("✓ 所有缓存预计算完成！")
    print("=" * 60)
    print(f"缓存位置:")
    print(f"  - 训练集: {cache_train_dir}")
    print(f"  - 验证集: {cache_valid_dir}")
    print()
    print("现在可以运行 train.py 开始训练，训练时会直接加载这些缓存。")
    print("=" * 60)


if __name__ == "__main__":
    main()

