#!/bin/bash
# 解码批处理脚本
# 用法: ./scripts/decode.sh <checkpoint> <src_file> <output_file> [config_file]

set -e

if [ $# -lt 3 ]; then
    echo "用法: $0 <checkpoint> <src_file> <output_file> [config_file]"
    echo "示例: $0 runs/my_exp/checkpoints/epoch_10.pt test.zh test.en.hyp configs/decode_beam5.yaml"
    exit 1
fi

CHECKPOINT="$1"
SRC_FILE="$2"
OUTPUT_FILE="$3"
CONFIG_FILE="${4:-configs/decode_beam5.yaml}"

if [ ! -f "$CHECKPOINT" ]; then
    echo "错误: 检查点不存在: $CHECKPOINT"
    exit 1
fi

if [ ! -f "$SRC_FILE" ]; then
    echo "错误: 源文件不存在: $SRC_FILE"
    exit 1
fi

# 运行解码
echo "开始解码..."
echo "检查点: $CHECKPOINT"
echo "源文件: $SRC_FILE"
echo "输出文件: $OUTPUT_FILE"
python decode.py --checkpoint "$CHECKPOINT" --src_file "$SRC_FILE" --output_file "$OUTPUT_FILE" --config "$CONFIG_FILE"

echo "解码完成！结果保存在: $OUTPUT_FILE"

