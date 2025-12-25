"""
预计算邻接矩阵缓存脚本（使用YAML配置）
"""
import os
import argparse
import torch
from datasets import load_dataset

from mt.utils.config_loader import load_config
from mt.data.cache import ensure_adj_cache


def main(config_path="configs/gcn_fusion.yaml"):
    """主函数"""
    # 加载YAML配置
    config = load_config(config_path)
    
    # 数据配置
    data_config = config['data']
    train_size = data_config['train_size']
    max_src_len = data_config['max_src_len']
    max_tgt_len = data_config['max_tgt_len']
    cache_root = data_config['cache_root']
    precompute_chunk_size = data_config['precompute_chunk_size']
    precompute_max_workers = data_config.get('precompute_max_workers')  # 使用get避免老配置报错
    precompute_use_parallel = data_config.get('precompute_use_parallel', True)
    
    print("=" * 60)
    print("预计算邻接矩阵缓存")
    print("=" * 60)
    print(f"训练集大小: {train_size}")
    print(f"源语言最大长度: {max_src_len}")
    print(f"目标语言最大长度: {max_tgt_len}")
    print(f"批处理大小 (chunk_size): {precompute_chunk_size}")
    print(f"使用多进程: {precompute_use_parallel}")
    if precompute_use_parallel:
        print(f"最大工作进程数: {precompute_max_workers or '自动'}")
    print("=" * 60)
    
    # 加载数据集
    print("\n加载数据集...")
    wmt = load_dataset("wmt17", "zh-en")
    ds_train = wmt["train"].select(range(train_size))
    ds_valid = wmt["validation"]
    
    print(f"训练集: {len(ds_train)} 条")
    print(f"验证集: {len(ds_valid)} 条")
    
    # 预计算训练集缓存
    print("\n" + "=" * 60)
    print("预计算训练集邻接矩阵缓存...")
    print("=" * 60)
    cache_train_dir = os.path.join(cache_root, "train")
    adj_src_train = ensure_adj_cache(
        ds_train, src_lang="zh",
        max_src_len=max_src_len, 
        cache_dir=cache_train_dir, 
        chunk_size=precompute_chunk_size, 
        max_workers=precompute_max_workers,
        use_parallel=precompute_use_parallel,
        dtype=torch.float16,
    )
    print(f"✓ 训练集缓存完成: {cache_train_dir}")
    
    # 预计算验证集缓存
    print("\n" + "=" * 60)
    print("预计算验证集邻接矩阵缓存...")
    print("=" * 60)
    cache_valid_dir = os.path.join(cache_root, "valid")
    adj_src_valid = ensure_adj_cache(
        ds_valid, src_lang="zh",
        max_src_len=max_src_len, 
        cache_dir=cache_valid_dir, 
        chunk_size=precompute_chunk_size, 
        max_workers=precompute_max_workers,
        use_parallel=precompute_use_parallel,
        dtype=torch.float16,
    )
    print(f"✓ 验证集缓存完成: {cache_valid_dir}")
    
    print("\n" + "=" * 60)
    print("所有缓存预计算完成！")
    print("=" * 60)


if __name__ == "__main__":
    import multiprocessing as mp

    # Set the start method to 'spawn' for spaCy compatibility
    # This should be done only once, at the beginning of the script execution
    try:
        mp.set_start_method('spawn', force=True)
        print("Multiprocessing start method set to 'spawn'.")
    except RuntimeError:
        # The start method can only be set once.
        # If it's already set, we can ignore the error.
        pass
    parser = argparse.ArgumentParser(description="预计算邻接矩阵缓存")
    parser.add_argument("--config", type=str, default="configs/gcn_fusion.yaml",
                       help="配置文件路径（默认: configs/gcn_fusion.yaml）")
    args = parser.parse_args()
    
    main(args.config)
