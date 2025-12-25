"""
训练/验证回调钩子（Hooks）
用于解耦训练循环和验证/日志记录逻辑
"""
from typing import Dict, List, Optional, Callable
import torch
from torch.utils.data import DataLoader

from mt.decoding.greedy import greedy_decode
from mt.decoding.beam import beam_search_decode
from mt.data.tokenizer import decode_sp


class TrainingHooks:
    """训练回调钩子集合"""
    
    def __init__(self):
        self.on_batch_end_hooks: List[Callable] = []
        self.on_epoch_end_hooks: List[Callable] = []
        self.on_validation_end_hooks: List[Callable] = []
    
    def register_on_batch_end(self, hook: Callable):
        """注册batch结束时的回调
        
        Args:
            hook: 回调函数，接收 (batch_idx, loss, grad_norm, lr) 参数
        """
        self.on_batch_end_hooks.append(hook)
    
    def register_on_epoch_end(self, hook: Callable):
        """注册epoch结束时的回调
        
        Args:
            hook: 回调函数，接收 (epoch, avg_loss) 参数
        """
        self.on_epoch_end_hooks.append(hook)
    
    def register_on_validation_end(self, hook: Callable):
        """注册验证结束时的回调
        
        Args:
            hook: 回调函数，接收 (epoch, valid_loss, samples) 参数
        """
        self.on_validation_end_hooks.append(hook)
    
    def call_on_batch_end(self, batch_idx: int, loss: float, grad_norm: float, lr: float):
        """调用所有batch结束回调"""
        for hook in self.on_batch_end_hooks:
            hook(batch_idx, loss, grad_norm, lr)
    
    def call_on_epoch_end(self, epoch: int, avg_loss: float):
        """调用所有epoch结束回调"""
        for hook in self.on_epoch_end_hooks:
            hook(epoch, avg_loss)
    
    def call_on_validation_end(self, epoch: int, valid_loss: float, samples: Optional[List[Dict]] = None):
        """调用所有验证结束回调"""
        for hook in self.on_validation_end_hooks:
            hook(epoch, valid_loss, samples)


def create_validation_hook(
    model, 
    valid_loader, 
    sp_src, 
    sp_tgt, 
    device, 
    config,
    num_samples=2,
    decode_method="greedy"
) -> Callable:
    """创建验证钩子，负责解码和打印样例"""
    
    def hook(epoch: int, valid_loss: float, _):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch} 验证结果:")
        print(f"Validation average loss: {valid_loss:.4f}")
        print(f"{'='*80}")
        
        model.eval()
        count = 0
        samples = []
        
        with torch.no_grad():
            for src_ids, tgt_ids, _ in valid_loader:
                if count >= num_samples:
                    break
                
                # 确保src_ids和tgt_ids是CPU上的tensor
                src_ids_cpu = src_ids[0].cpu() if src_ids.dim() > 1 else src_ids.cpu()
                tgt_ids_cpu = tgt_ids[0].cpu() if tgt_ids.dim() > 1 else tgt_ids.cpu()
                
                # 解码源文本和目标文本
                src_text = decode_sp(sp_src, src_ids_cpu.tolist())
                tgt_text = decode_sp(sp_tgt, tgt_ids_cpu.tolist())
                
                # 解码预测文本
                if decode_method == "greedy":
                    pred_text = greedy_decode(
                        model, src_ids_cpu, sp_src, sp_tgt, device,
                        max_len=config['max_tgt_len'],
                        pad_idx=config['pad_idx']
                    )
                elif decode_method == "beam_search":
                    pred_text = beam_search_decode(
                        model, src_ids_cpu, sp_src, sp_tgt, device,
                        max_len=config['max_tgt_len'],
                        pad_idx=config['pad_idx'],
                        beam_size=5  # 可从配置中读取
                    )
                else:
                    raise ValueError(f"Unknown decode_method: {decode_method}")
                
                print(f"\n样例 {count + 1}:")
                print(f"SOURCE:    {src_text}")
                print(f"TARGET:    {tgt_text}")
                print(f"PREDICTED ({decode_method}): {pred_text}")
                
                samples.append({
                    'source': src_text,
                    'target': tgt_text,
                    'predicted': pred_text
                })
                count += 1
        
        print(f"{'='*80}\n")
        
        # 可以在这里调用其他钩子，如保存样例
        # dump_hook = create_sample_dump_hook(...)
        # dump_hook(epoch, valid_loss, samples)
    
    return hook


def create_gradient_norm_hook() -> Callable:
    """创建梯度范数记录钩子"""
    def hook(batch_idx: int, loss: float, grad_norm: float, lr: float):
        if batch_idx % 100 == 0:  # 每100个batch记录一次
            print(f"  Batch {batch_idx}: grad_norm={grad_norm:.4f}, lr={lr:.2e}")
    return hook


def create_checkpoint_hook(checkpoint_dir: str, save_interval: int = 1) -> Callable:
    """创建检查点保存钩子"""
    import os
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    def hook(epoch: int, avg_loss: float, model: torch.nn.Module, optimizer, scheduler):
        if epoch % save_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}")
    
    return hook


def create_sample_dump_hook(output_dir: str, num_samples: int = 5) -> Callable:
    """创建样例翻译保存钩子"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    def hook(epoch: int, valid_loss: float, samples: Optional[List[Dict]] = None):
        if samples:
            output_file = os.path.join(output_dir, f"valid_epoch_{epoch}.txt")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"Epoch {epoch}, Validation Loss: {valid_loss:.4f}\n")
                f.write("=" * 80 + "\n")
                for i, sample in enumerate(samples[:num_samples]):
                    f.write(f"\nSample {i+1}:\n")
                    f.write(f"SOURCE:    {sample.get('source', '')}\n")
                    f.write(f"TARGET:    {sample.get('target', '')}\n")
                    f.write(f"PREDICTED: {sample.get('predicted', '')}\n")
            print(f"  Samples saved: {output_file}")
    
    return hook
