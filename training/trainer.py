"""
训练器
"""
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.tokenizer import decode_sp
from training.loss import LabelSmoothingLoss
from training.scheduler import NoamOpt
from training.validator import run_validation


class Trainer:
    """训练器类"""
    def __init__(self, model, train_loader, valid_loader, sp_src, sp_tgt, device, config):
        """
        Args:
            model: 模型
            train_loader: 训练数据加载器
            valid_loader: 验证数据加载器
            sp_src: 源语言SentencePiece处理器
            sp_tgt: 目标语言SentencePiece处理器
            device: 设备
            config: 配置字典
        """
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.sp_src = sp_src
        self.sp_tgt = sp_tgt
        self.device = device
        self.config = config

        # 优化器和调度器
        base_optimizer = torch.optim.Adam(
            model.parameters(), 
            betas=(0.9, 0.98), 
            eps=1e-9
        )
        self.scheduler = NoamOpt(
            model_size=config['d_model'], 
            factor=2.0, 
            warmup=4000, 
            optimizer=base_optimizer
        )

        # 损失函数
        self.criterion = LabelSmoothingLoss(
            classes=sp_tgt.vocab_size(), 
            smoothing=0.1, 
            ignore_index=config['pad_idx']
        )

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        batch_iterator = tqdm(self.train_loader, desc=f"Epoch {epoch:02d}")
        
        for src_ids, tgt_ids, adj_src, adj_tgt_in in batch_iterator:
            src_ids, tgt_ids = src_ids.to(self.device), tgt_ids.to(self.device)
            tgt_in, tgt_out = tgt_ids[:, :-1], tgt_ids[:, 1:]

            # 使用预计算的邻接矩阵，避免运行时重复构建
            adj_src = adj_src.to(self.device)
            adj_tgt_in = adj_tgt_in.to(self.device)

            log_probs = self.model(src_ids, tgt_in, adj_src, adj_tgt_in)  # [B,T,V]
            loss = self.criterion(log_probs.reshape(-1, log_probs.size(-1)), tgt_out.reshape(-1))

            self.scheduler.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scheduler.step()

            total_loss += loss.item()
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

        avg_loss = total_loss / len(self.train_loader)
        print(f"Epoch {epoch} average loss: {avg_loss:.4f}")
        return avg_loss

    def validate(self, epoch, debug=False):
        """验证"""
        run_validation(
            self.model, 
            self.valid_loader, 
            self.sp_src, 
            self.sp_tgt, 
            self.device, 
            max_len=self.config['max_tgt_len'], 
            num_examples=2, 
            pad_idx=self.config['pad_idx'],
            debug=debug
        )

    def train(self, epochs):
        """完整训练流程"""
        for epoch in range(1, epochs + 1):
            self.train_epoch(epoch)
            self.validate(epoch)
        
        return self.model

