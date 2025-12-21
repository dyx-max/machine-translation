"""
主训练脚本
"""
import os
import nltk
nltk.download('wordnet')
nltk.download('punkt')

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

from config import Config
from models.model import TransformerGCN
from data.tokenizer import train_or_load_spm
from data.dataset import WMTDataset, collate_batch
from data.adj_cache import ensure_adj_cache
from training.trainer import Trainer


def prepare_corpus(ds_train, zh_corpus, en_corpus, train_size):
    """准备语料文件"""
    if not os.path.exists(zh_corpus):
        print(f"生成语料文件: {zh_corpus}, {en_corpus}")
        with open(zh_corpus, "w", encoding="utf-8") as fzh, \
             open(en_corpus, "w", encoding="utf-8") as fen:
            for i in range(train_size):
                fzh.write(ds_train[i]["translation"]["zh"] + "\n")
                fen.write(ds_train[i]["translation"]["en"] + "\n")
    else:
        print(f"语料文件已存在: {zh_corpus}, {en_corpus}")


def main():
    """主函数"""
    config = Config()
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载数据集
    print("加载数据集...")
    wmt = load_dataset("wmt17", "zh-en")
    ds_train = wmt["train"].select(range(config.train_size))
    ds_valid = wmt["validation"]

    # 准备语料文件
    prepare_corpus(ds_train, config.zh_corpus, config.en_corpus, config.train_size)

    # 训练或加载SentencePiece模型
    print("加载/训练SentencePiece模型...")
    sp_src = train_or_load_spm(config.zh_corpus, config.spm_zh_prefix, config.vocab_size)
    sp_tgt = train_or_load_spm(config.en_corpus, config.spm_en_prefix, config.vocab_size)

    # 预计算邻接矩阵缓存（单进程顺序处理，带进度条）
    print("预计算并缓存邻接矩阵...")
    cache_train_dir = os.path.join(config.cache_root, "train")
    cache_valid_dir = os.path.join(config.cache_root, "valid")
    adj_src_train, adj_tgt_in_train = ensure_adj_cache(
        ds_train, src_lang="zh", tgt_lang="en",
        max_src_len=config.max_src_len, max_tgt_in_len=config.max_tgt_len-1,
        cache_dir=cache_train_dir, chunk_size=config.precompute_chunk_size, dtype=torch.float16,
    )
    adj_src_valid, adj_tgt_in_valid = ensure_adj_cache(
        ds_valid, src_lang="zh", tgt_lang="en",
        max_src_len=config.max_src_len, max_tgt_in_len=config.max_tgt_len-1,
        cache_dir=cache_valid_dir, chunk_size=config.precompute_chunk_size, dtype=torch.float16,
    )

    # 创建数据加载器（使用缓存 + DataLoader加速参数）
    print("创建数据加载器...")
    train_loader = DataLoader(
        WMTDataset(ds_train, sp_src, sp_tgt, config.max_src_len, config.max_tgt_len,
                   adj_src_cache=adj_src_train, adj_tgt_in_cache=adj_tgt_in_train),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=config.dataloader_workers,
        pin_memory=(config.pin_memory and device.type=='cuda'),
        persistent_workers=config.persistent_workers if config.dataloader_workers > 0 else False,
    )
    valid_loader = DataLoader(
        WMTDataset(ds_valid, sp_src, sp_tgt, config.max_src_len, config.max_tgt_len,
                   adj_src_cache=adj_src_valid, adj_tgt_in_cache=adj_tgt_in_valid),
        batch_size=1,
        collate_fn=collate_batch,
        num_workers=max(1, config.dataloader_workers//2),
        pin_memory=(config.pin_memory and device.type=='cuda'),
        persistent_workers=(config.dataloader_workers//2) > 0 and config.persistent_workers,
    )

    # 创建模型
    print("创建模型...")
    model = TransformerGCN(
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        d_ff=config.d_ff,
        src_vocab_size=sp_src.vocab_size(),
        tgt_vocab_size=sp_tgt.vocab_size(),
        max_len=max(config.max_src_len, config.max_tgt_len),
        pad_idx=config.pad_idx,
        dropout=config.dropout,
        gcn_layers_src=config.gcn_layers_src,
        gcn_layers_tgt=config.gcn_layers_tgt,
        fusion_mode=config.fusion_mode,
    ).to(device)

    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        sp_src=sp_src,
        sp_tgt=sp_tgt,
        device=device,
        config={
            'd_model': config.d_model,
            'max_tgt_len': config.max_tgt_len,
            'pad_idx': config.pad_idx,
        }
    )

    # 开始训练
    print("开始训练...")
    trainer.train(config.epochs)

    print("训练完成！")
    return model, sp_src, sp_tgt


if __name__ == "__main__":
    model, sp_src, sp_tgt = main()
