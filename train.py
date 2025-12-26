"""
主训练脚本（使用YAML配置）

V2 变更（新 pipeline：边列表缓存）
--------------------------------
- 预处理阶段不再生成/缓存稠密邻接矩阵，而是缓存边列表（edges_src.pt）。
- DataLoader collate 阶段将边列表转换为稠密邻接矩阵（0/1 + 自环，未归一化）。
- GCN 模型内部完成邻接矩阵归一化（见 mt.models.gcn）。
"""

from __future__ import annotations

import os
import argparse

import nltk

nltk.download("wordnet")
nltk.download("punkt")

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

from mt.utils.config_loader import load_config
from mt.models.model import TransformerGCN
from mt.data.tokenizer import train_or_load_spm
from mt.data.dataset import WMTDataset, collate_batch
from mt.data.cache import ensure_edge_cache
from mt.training.trainer import Trainer


def prepare_corpus(ds_train, zh_corpus: str, en_corpus: str, train_size: int):
    """准备语料文件"""
    if not os.path.exists(zh_corpus):
        print(f"生成语料文件: {zh_corpus}, {en_corpus}")
        with open(zh_corpus, "w", encoding="utf-8") as fzh, open(en_corpus, "w", encoding="utf-8") as fen:
            for i in range(train_size):
                fzh.write(ds_train[i]["translation"]["zh"] + "\n")
                fen.write(ds_train[i]["translation"]["en"] + "\n")
    else:
        print(f"语料文件已存在: {zh_corpus}, {en_corpus}")


def main(config_path: str = "configs/gcn_fusion.yaml"):
    config = load_config(config_path)

    device_type = config.get("device", {}).get("type", "auto")
    if device_type in ("auto", "cuda"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print(f"使用设备: {device}")

    data_config = config["data"]
    max_src_len = data_config["max_src_len"]
    max_tgt_len = data_config["max_tgt_len"]
    train_size = data_config["train_size"]
    vocab_size = data_config["vocab_size"]
    cache_root = data_config["cache_root"]
    precompute_chunk_size = data_config["precompute_chunk_size"]
    zh_corpus = data_config["zh_corpus"]
    en_corpus = data_config["en_corpus"]
    spm_zh_prefix = data_config["spm_zh_prefix"]
    spm_en_prefix = data_config["spm_en_prefix"]

    print("加载数据集...")
    wmt = load_dataset("wmt17", "zh-en")
    ds_train = wmt["train"].select(range(train_size))
    ds_valid = wmt["validation"]

    prepare_corpus(ds_train, zh_corpus, en_corpus, train_size)

    print("加载/训练SentencePiece模型...")
    sp_src = train_or_load_spm(zh_corpus, spm_zh_prefix, vocab_size)
    sp_tgt = train_or_load_spm(en_corpus, spm_en_prefix, vocab_size)

    # ---- 新：预计算边列表缓存（不生成稠密矩阵） ----
    print("预计算并缓存边列表（edges）...")
    cache_train_dir = os.path.join(cache_root, "train")
    cache_valid_dir = os.path.join(cache_root, "valid")

    edges_src_train = ensure_edge_cache(
        ds_train,
        src_lang="zh",
        max_src_len=max_src_len,
        cache_dir=cache_train_dir,
        chunk_size=precompute_chunk_size,
        force_recompute=False,
        use_parallel=True,
    )
    edges_src_valid = ensure_edge_cache(
        ds_valid,
        src_lang="zh",
        max_src_len=max_src_len,
        cache_dir=cache_valid_dir,
        chunk_size=precompute_chunk_size,
        force_recompute=False,
        use_parallel=True,
    )

    training_config = config["training"]
    batch_size = training_config["batch_size"]
    epochs = training_config["epochs"]
    dataloader_workers = training_config.get("dataloader_workers")
    if dataloader_workers is None:
        dataloader_workers = max(1, (os.cpu_count() or 2) // 2)
    pin_memory = training_config.get("pin_memory", True)
    persistent_workers = training_config.get("persistent_workers", True)

    print("创建数据加载器...")

    def _collate(b):
        return collate_batch(b, max_src_len=max_src_len, edges_to_adj=True)

    train_loader = DataLoader(
        WMTDataset(
            ds_train,
            sp_src,
            sp_tgt,
            max_src_len,
            max_tgt_len,
            edges_src_cache=edges_src_train,
        ),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate,
        num_workers=dataloader_workers,
        pin_memory=(pin_memory and device.type == "cuda"),
        persistent_workers=persistent_workers if dataloader_workers > 0 else False,
    )

    valid_loader = DataLoader(
        WMTDataset(
            ds_valid,
            sp_src,
            sp_tgt,
            max_src_len,
            max_tgt_len,
            edges_src_cache=edges_src_valid,
        ),
        batch_size=1,
        collate_fn=_collate,
        num_workers=max(1, dataloader_workers // 2),
        pin_memory=(pin_memory and device.type == "cuda"),
        persistent_workers=(dataloader_workers // 2) > 0 and persistent_workers,
    )

    model_config = config["model"]

    print("创建模型...")
    model = TransformerGCN(
        d_model=model_config["d_model"],
        num_heads=model_config["num_heads"],
        num_layers=model_config["num_layers"],
        d_ff=model_config["d_ff"],
        src_vocab_size=sp_src.vocab_size(),
        tgt_vocab_size=sp_tgt.vocab_size(),
        max_len=max(max_src_len, max_tgt_len),
        pad_idx=model_config["pad_idx"],
        dropout=model_config["dropout"],
        gcn_layers=model_config["gcn_layers"],
        fusion_mode=model_config["fusion_mode"],
        gcn_normalize=model_config.get("gcn_normalize", "sym"),
    ).to(device)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        sp_src=sp_src,
        sp_tgt=sp_tgt,
        device=device,
        config={
            "d_model": model_config["d_model"],
            "max_tgt_len": max_tgt_len,
            "pad_idx": model_config["pad_idx"],
        },
    )

    from mt.training.hooks import create_validation_hook

    validation_hook = create_validation_hook(
        model,
        valid_loader,
        sp_src,
        sp_tgt,
        device,
        config={"max_tgt_len": max_tgt_len, "pad_idx": model_config["pad_idx"]},
        num_samples=2,
        decode_method="beam_search",
    )
    trainer.hooks.register_on_validation_end(validation_hook)

    print("开始训练...")
    trainer.train(epochs)

    print("训练完成！")
    return model, sp_src, sp_tgt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练Transformer+GCN模型")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/gcn_fusion.yaml",
        help="配置文件路径（默认: configs/gcn_fusion.yaml）",
    )
    args = parser.parse_args()

    model, sp_src, sp_tgt = main(args.config)
