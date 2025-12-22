#!/bin/bash
# 训练脚本
# 用法: ./scripts/train.sh <config_file> [run_name]

set -e

if [ $# -lt 1 ]; then
    echo "用法: $0 <config_file> [run_name]"
    echo "示例: $0 configs/gcn_fusion.yaml my_experiment"
    exit 1
fi

CONFIG_FILE="$1"
RUN_NAME="${2:-$(date +%Y%m%d_%H%M%S)}"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# 创建运行目录
RUN_DIR="runs/$RUN_NAME"
mkdir -p "$RUN_DIR"

# 复制配置文件
cp "$CONFIG_FILE" "$RUN_DIR/config.yaml"

# 运行训练
echo "开始训练..."
echo "配置: $CONFIG_FILE"
echo "运行目录: $RUN_DIR"
python train.py --config "$CONFIG_FILE" --run_dir "$RUN_DIR"

echo "训练完成！结果保存在: $RUN_DIR"

