"""
保存样例翻译与注意力权重
"""
from typing import List, Dict, Optional
import torch
import json
from pathlib import Path


def dump_translation_samples(
    samples: List[Dict[str, str]],
    output_file: str,
    epoch: Optional[int] = None
):
    """
    保存翻译样例
    
    Args:
        samples: 样例列表，每个样例包含 'source', 'target', 'predicted' 等字段
        output_file: 输出文件路径
        epoch: epoch编号（可选）
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        if epoch is not None:
            f.write(f"Epoch {epoch}\n")
            f.write("=" * 80 + "\n\n")
        
        for i, sample in enumerate(samples):
            f.write(f"Sample {i+1}:\n")
            f.write(f"SOURCE:    {sample.get('source', '')}\n")
            f.write(f"TARGET:    {sample.get('target', '')}\n")
            f.write(f"PREDICTED: {sample.get('predicted', '')}\n")
            if 'attention' in sample:
                f.write(f"ATTENTION: {sample['attention']}\n")
            f.write("\n")


def extract_attention_weights(
    model: torch.nn.Module,
    src_ids: torch.Tensor,
    tgt_ids: torch.Tensor,
    adj_src: torch.Tensor,
    adj_tgt: torch.Tensor,
    layer_idx: int = 0,
    head_idx: int = 0
) -> Optional[torch.Tensor]:
    """
    提取注意力权重（需要模型支持）
    
    Args:
        model: 模型
        src_ids: 源序列
        tgt_ids: 目标序列
        adj_src: 源语言邻接矩阵
        adj_tgt: 目标语言邻接矩阵
        layer_idx: 层索引
        head_idx: 头索引
    
    Returns:
        注意力权重矩阵 [B, H, Lq, Lk] 或 None（如果模型不支持）
    """
    # 注意：这需要模型支持返回注意力权重
    # 当前模型架构可能不支持，这里提供一个接口
    # 实际使用时需要修改模型forward方法以返回注意力权重
    
    model.eval()
    with torch.no_grad():
        try:
            # 尝试获取注意力权重（需要模型支持）
            # 这里只是示例，实际实现需要根据模型架构调整
            pass
        except:
            return None
    
    return None


def dump_attention_visualization(
    attention_weights: torch.Tensor,
    src_tokens: List[str],
    tgt_tokens: List[str],
    output_file: str
):
    """
    保存注意力权重可视化数据（JSON格式）
    
    Args:
        attention_weights: 注意力权重 [Lq, Lk]
        src_tokens: 源语言token列表
        tgt_tokens: 目标语言token列表
        output_file: 输出文件路径
    """
    # 转换为numpy并保存为JSON
    import numpy as np
    
    attn_np = attention_weights.cpu().numpy() if isinstance(attention_weights, torch.Tensor) else attention_weights
    
    data = {
        "source_tokens": src_tokens,
        "target_tokens": tgt_tokens,
        "attention_weights": attn_np.tolist()
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def dump_graph_statistics(
    adj_matrices: List[torch.Tensor],
    output_file: str
):
    """
    保存图统计信息（度分布、连通分量等）
    
    Args:
        adj_matrices: 邻接矩阵列表
        output_file: 输出文件路径
    """
    import numpy as np
    
    stats = {
        "num_graphs": len(adj_matrices),
        "avg_degree": [],
        "max_degree": [],
        "min_degree": [],
        "density": []
    }
    
    for adj in adj_matrices:
        adj_np = adj.cpu().numpy() if isinstance(adj, torch.Tensor) else adj
        # 计算度（忽略自环）
        degrees = (adj_np.sum(axis=1) - adj_np.diagonal())
        stats["avg_degree"].append(float(degrees.mean()))
        stats["max_degree"].append(float(degrees.max()))
        stats["min_degree"].append(float(degrees.min()))
        # 计算密度
        n = adj_np.shape[0]
        num_edges = (adj_np.sum() - n) / 2  # 无向图，减去自环
        density = num_edges / (n * (n - 1) / 2) if n > 1 else 0
        stats["density"].append(float(density))
    
    # 计算平均值
    stats["avg_degree_mean"] = float(np.mean(stats["avg_degree"]))
    stats["avg_density"] = float(np.mean(stats["density"]))
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

