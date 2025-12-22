"""
验证依赖安装的测试脚本
"""
print("=" * 50)
print("检查依赖安装...")
print("=" * 50)

# 检查Python版本
import sys
print(f"✓ Python版本: {sys.version}")

# 检查PyTorch
try:
    import torch
    print(f"✓ PyTorch版本: {torch.__version__}")
    print(f"✓ CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ CUDA版本: {torch.version.cuda}")
        print(f"✓ GPU设备: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("✗ PyTorch未安装")

# 检查其他依赖
dependencies = [
    'datasets', 'sentencepiece', 'sacrebleu', 
    'nltk', 'tqdm', 'spacy', 'numpy'
]

for dep in dependencies:
    try:
        mod = __import__(dep)
        version = getattr(mod, '__version__', '未知')
        print(f"✓ {dep}: {version}")
    except ImportError:
        print(f"✗ {dep}: 未安装")

# 检查spaCy模型
try:
    import spacy
    nlp_zh = spacy.load("zh_core_web_sm")
    nlp_en = spacy.load("en_core_web_sm")
    print("✓ spaCy中文模型: 已加载")
    print("✓ spaCy英文模型: 已加载")
except Exception as e:
    print(f"✗ spaCy模型: {e}")

# 检查NLTK数据
try:
    import nltk
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
    print("✓ NLTK数据: 已下载")
except LookupError:
    print("✗ NLTK数据: 未下载（运行 nltk.download()）")

# 检查项目模块（使用新的mt包）
try:
    from mt.models.model import TransformerGCN
    from mt.data.dataset import WMTDataset
    from mt.training.trainer import Trainer
    print("✓ 项目模块: 导入成功")
except Exception as e:
    print(f"✗ 项目模块: {e}")

print("=" * 50)
print("检查完成！")
print("=" * 50)

