"""
训练器（纯粹化：只负责训练循环，验证和日志通过hooks解耦）
"""
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mt.training.loss import LabelSmoothingLoss
from mt.training.scheduler import NoamOpt
from mt.training.hooks import TrainingHooks


class Trainer:
    """训练器类（纯粹化版本）"""
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
        
        # 钩子系统
        self.hooks = TrainingHooks()

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        batch_iterator = tqdm(self.train_loader, desc=f"Epoch {epoch:02d}")
        
        for batch_idx, (src_ids, tgt_ids, adj_src) in enumerate(batch_iterator):
            src_ids, tgt_ids = src_ids.to(self.device), tgt_ids.to(self.device)
            tgt_in, tgt_out = tgt_ids[:, :-1], tgt_ids[:, 1:]

            # 使用预计算的邻接矩阵，避免运行时重复构建
            adj_src = adj_src.to(self.device)

            log_probs = self.model(src_ids, tgt_in, adj_src)  # [B,T,V]
            loss = self.criterion(log_probs.reshape(-1, log_probs.size(-1)), tgt_out.reshape(-1))

            self.scheduler.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.scheduler.step()
            
            # 获取当前学习率
            current_lr = self.scheduler.optimizer.param_groups[0]['lr']

            total_loss += loss.item()
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
            
            # 调用batch结束钩子
            self.hooks.call_on_batch_end(batch_idx, loss.item(), grad_norm.item(), current_lr)

        avg_loss = total_loss / len(self.train_loader)
        print(f"Epoch {epoch} average loss: {avg_loss:.4f}")
        
        # 调用epoch结束钩子
        self.hooks.call_on_epoch_end(epoch, avg_loss)
        
        return avg_loss

    def validate(self, epoch, compute_loss=True):
        """验证（纯粹化：只计算损失，不做解码）
        
        Args:
            epoch: epoch编号
            compute_loss: 是否计算损失
        
        Returns:
            valid_loss: 验证损失（如果compute_loss=True）
            samples: 样例列表（可选，用于hooks）
        """
        self.model.eval()
        total_loss, count = 0, 0
        samples = []

        with torch.no_grad():
            for src_ids, tgt_ids, adj_src in self.valid_loader:
                src_ids, tgt_ids = src_ids.to(self.device), tgt_ids.to(self.device)
                tgt_in, tgt_out = tgt_ids[:, :-1], tgt_ids[:, 1:]

                adj_src = adj_src.to(self.device)

                if compute_loss:
                    log_probs = self.model(src_ids, tgt_in, adj_src)
                    loss = self.criterion(log_probs.reshape(-1, log_probs.size(-1)), tgt_out.reshape(-1))
                    total_loss += loss.item()
                    count += 1

        valid_loss = total_loss / max(count, 1) if compute_loss else 0.0
        
        # 调用验证结束钩子
        self.hooks.call_on_validation_end(epoch, valid_loss, samples)
        
        return valid_loss

    def train(self, epochs):
        """完整训练流程"""
        for epoch in range(1, epochs + 1):
            self.train_epoch(epoch)
            self.validate(epoch)
        
        return self.model

