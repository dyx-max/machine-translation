"""
纯Transformer基线训练脚本（用于对比）
使用方法：python train_baseline.py
"""
import os
import nltk
nltk.download('wordnet')
nltk.download('punkt')

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

from config import Config
from models.transformer_baseline import TransformerBaseline
from data.tokenizer import train_or_load_spm
from data.dataset import WMTDataset, collate_batch
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
    print("="*80)
    print("训练纯Transformer基线模型（无GCN）")
    print("="*80)

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

    # 纯Transformer不需要邻接矩阵，跳过缓存计算以节省时间和存储
    print("纯Transformer模型不需要邻接矩阵，跳过缓存计算...")

    # 创建数据加载器（skip_adj=True 跳过邻接矩阵计算）
    print("创建数据加载器...")
    train_loader = DataLoader(
        WMTDataset(ds_train, sp_src, sp_tgt, config.max_src_len, config.max_tgt_len,
                   skip_adj=True),  # 跳过邻接矩阵计算
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=config.dataloader_workers,
        pin_memory=(config.pin_memory and device.type=='cuda'),
        persistent_workers=config.persistent_workers if config.dataloader_workers > 0 else False,
    )
    valid_loader = DataLoader(
        WMTDataset(ds_valid, sp_src, sp_tgt, config.max_src_len, config.max_tgt_len,
                   skip_adj=True),  # 跳过邻接矩阵计算
        batch_size=1,
        collate_fn=collate_batch,
        num_workers=max(1, config.dataloader_workers//2),
        pin_memory=(config.pin_memory and device.type=='cuda'),
        persistent_workers=(config.dataloader_workers//2) > 0 and config.persistent_workers,
    )

    # 创建纯Transformer模型
    print("创建纯Transformer模型...")
    model = TransformerBaseline(
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        d_ff=config.d_ff,
        src_vocab_size=sp_src.vocab_size(),
        tgt_vocab_size=sp_tgt.vocab_size(),
        max_len=max(config.max_src_len, config.max_tgt_len),
        pad_idx=config.pad_idx,
        dropout=config.dropout,
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
    print("开始训练纯Transformer基线...")
    trainer.train(config.epochs)

    print("训练完成！")
    return model, sp_src, sp_tgt


if __name__ == "__main__":
    model, sp_src, sp_tgt = main()

