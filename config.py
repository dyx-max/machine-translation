"""
配置文件
"""
import os


class Config:
    """训练配置"""
    # 数据配置
    max_src_len = 64
    max_tgt_len = 64
    train_size = 50000  # 训练集大小
    
    # 模型配置
    d_model = 512
    num_heads = 8
    num_layers = 4
    d_ff = 1024
    dropout = 0.1
    gcn_layers_src = 2
    gcn_layers_tgt = 2
    fusion_mode = "concat"  # "concat" 或 "gate"
    
    # 训练配置
    batch_size = 32
    epochs = 10
    pad_idx = 0
    
    # 分词器配置
    vocab_size = 8000
    
    # 设备
    device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"

    # DataLoader 加速
    dataloader_workers = max(1, (os.cpu_count() or 2) // 2)
    pin_memory = True
    persistent_workers = True

    # 预计算/缓存设置
    cache_root = "cache"
    precompute_workers = max(1, (os.cpu_count() or 2) - 1)
    
    # 文件路径
    zh_corpus = "zh.txt"
    en_corpus = "en.txt"
    spm_zh_prefix = "spm_zh"
    spm_en_prefix = "spm_en"

