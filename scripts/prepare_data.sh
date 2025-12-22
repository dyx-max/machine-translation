#!/bin/bash
# 数据准备脚本：下载语料、句法模型，预计算缓存
# 用法: ./scripts/prepare_data.sh

set -e

echo "=== 数据准备脚本 ==="

# 检查spaCy模型
echo "检查spaCy模型..."
python -c "import spacy; spacy.load('zh_core_web_sm'); spacy.load('en_core_web_sm')" 2>/dev/null || {
    echo "spaCy模型未安装，开始下载..."
    python -m spacy download zh_core_web_sm
    python -m spacy download en_core_web_sm
}

# 检查NLTK数据
echo "检查NLTK数据..."
python -c "import nltk; nltk.data.find('tokenizers/punkt'); nltk.data.find('corpora/wordnet')" 2>/dev/null || {
    echo "NLTK数据未下载，开始下载..."
    python -c "import nltk; nltk.download('wordnet'); nltk.download('punkt')"
}

# 预计算缓存（如果存在precompute_cache.py）
if [ -f "precompute_cache.py" ]; then
    echo "预计算邻接矩阵缓存..."
    python precompute_cache.py
else
    echo "注意: precompute_cache.py 不存在，跳过缓存预计算"
fi

echo "=== 数据准备完成 ==="

