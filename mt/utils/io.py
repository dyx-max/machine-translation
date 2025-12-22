"""
IO工具函数
"""
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional
import torch


def save_json(data: Any, file_path: str, indent: int = 2, ensure_ascii: bool = False):
    """保存JSON文件"""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)


def load_json(file_path: str) -> Any:
    """加载JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_pickle(data: Any, file_path: str):
    """保存pickle文件"""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(file_path: str) -> Any:
    """加载pickle文件"""
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    file_path: str,
    **kwargs
):
    """保存模型检查点"""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        **kwargs
    }
    
    torch.save(checkpoint, file_path)


def load_checkpoint(
    file_path: str,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """加载模型检查点"""
    checkpoint = torch.load(file_path, map_location=device)
    
    if model is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint

