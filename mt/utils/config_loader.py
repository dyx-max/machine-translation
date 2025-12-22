"""
配置加载器：从YAML文件加载配置
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: str, base_config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    加载YAML配置文件，支持继承base配置
    
    Args:
        config_path: 配置文件路径
        base_config_path: 基础配置文件路径（如果为None，自动查找base.yaml）
    
    Returns:
        配置字典
    """
    config_dir = Path(config_path).parent
    
    # 加载base配置
    if base_config_path is None:
        base_config_path = config_dir / "base.yaml"
    
    base_config = {}
    if os.path.exists(base_config_path):
        with open(base_config_path, 'r', encoding='utf-8') as f:
            base_config = yaml.safe_load(f) or {}
    
    # 加载当前配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}
    
    # 深度合并配置（当前配置覆盖base配置）
    merged_config = _deep_merge(base_config, config)
    
    # 处理特殊值
    merged_config = _process_special_values(merged_config)
    
    return merged_config


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """深度合并两个字典"""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _process_special_values(config: Dict) -> Dict:
    """处理特殊值（如null表示自动计算）"""
    import os
    
    # 处理dataloader_workers
    if config.get('training', {}).get('dataloader_workers') is None:
        config.setdefault('training', {})['dataloader_workers'] = max(1, (os.cpu_count() or 2) // 2)
    
    # 处理device
    device_config = config.get('device', {})
    if device_config.get('type') == 'auto':
        device_config['type'] = 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'
    
    return config


def save_config(config: Dict[str, Any], output_path: str):
    """保存配置到YAML文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

