#!/bin/bash

echo "========================================"
echo "机器翻译项目环境配置脚本"
echo "========================================"
echo ""

echo "[1/6] 创建虚拟环境..."
if [ -d "venv" ]; then
    echo "虚拟环境已存在，跳过创建"
else
    python3 -m venv venv
    echo "虚拟环境创建完成"
fi
echo ""

echo "[2/6] 激活虚拟环境..."
source venv/bin/activate
echo ""

echo "[3/6] 升级pip..."
python -m pip install --upgrade pip
echo ""

echo "[4/6] 安装依赖包..."
echo "这可能需要几分钟时间，请耐心等待..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "依赖安装失败，尝试手动安装..."
    pip install "numpy>=1.24.0,<2.0.0"
    pip install datasets sentencepiece sacrebleu nltk pyter3 tqdm spacy --upgrade
    pip install torch torchvision torchaudio
fi
echo ""

echo "[5/6] 下载spaCy语言模型..."
python -m spacy download zh_core_web_sm
python -m spacy download en_core_web_sm
echo ""

echo "[6/6] 下载NLTK数据..."
python -c "import nltk; nltk.download('wordnet'); nltk.download('punkt')"
echo ""

echo "========================================"
echo "环境配置完成！"
echo "========================================"
echo ""
echo "下一步："
echo "1. 运行 python test_installation.py 验证安装"
echo "2. 运行 python train.py 开始训练"
echo ""

