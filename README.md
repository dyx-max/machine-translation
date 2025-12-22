# 机器翻译项目 - Transformer + GCN

基于Transformer和图卷积网络（GCN）的中英机器翻译系统，支持句法信息融合。

## 📋 目录

- [快速开始](#快速开始)
- [环境要求](#环境要求)
- [安装步骤](#安装步骤)
- [使用方法](#使用方法)
- [项目结构](#项目结构)
- [配置说明](#配置说明)
- [关键特性](#关键特性)
- [常见问题](#常见问题)
- [开发指南](#开发指南)

---

## 🚀 快速开始

### 一键安装（Windows）

```powershell
# 运行安装脚本
.\setup.bat

# 或手动执行
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download zh_core_web_sm en_core_web_sm
python -c "import nltk; nltk.download('wordnet'); nltk.download('punkt')"
```

### 一键安装（Linux/macOS）

```bash
# 运行安装脚本
chmod +x setup.sh
./setup.sh

# 或手动执行
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download zh_core_web_sm en_core_web_sm
python -c "import nltk; nltk.download('wordnet'); nltk.download('punkt')"
```

### 开始训练

```bash
# 激活虚拟环境后
# 使用默认配置（configs/gcn_fusion.yaml）
python train.py

# 或指定配置文件
python train.py --config configs/gcn_fusion.yaml

# 或使用脚本（推荐）
./scripts/train.sh configs/gcn_fusion.yaml my_experiment
```

---

## 🔧 环境要求

- **Python**: 3.10 或 3.11（推荐）
- **操作系统**: Windows 10/11, Linux, macOS
- **内存**: 建议至少8GB RAM
- **GPU**: 可选，有CUDA支持的GPU可加速训练（推荐）

---

## 📦 安装步骤

### 1. 创建虚拟环境

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### 2. 安装依赖

```bash
# 升级pip
python -m pip install --upgrade pip

# 安装依赖（注意numpy版本兼容性）
pip install "numpy>=1.24.0,<2.0.0"
pip install -r requirements.txt

# 安装PyTorch（根据你的CUDA版本）
# CPU版本：
pip install torch torchvision torchaudio

# CUDA 11.8：
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1：
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. 下载spaCy语言模型

```bash
python -m spacy download zh_core_web_sm
python -m spacy download en_core_web_sm
```

### 4. 下载NLTK数据

```bash
python -c "import nltk; nltk.download('wordnet'); nltk.download('punkt')"
```

### 5. 验证安装

```bash
python test_installation.py
```

---

## 🎯 使用方法

### 训练模型

```bash
# Transformer + GCN（默认配置）
python train.py

# 指定配置文件
python train.py --config configs/gcn_fusion.yaml

# 纯Transformer基线（用于对比）
python train_baseline.py --config configs/gcn_fusion.yaml

# 使用脚本（推荐）
./scripts/train.sh configs/gcn_fusion.yaml my_experiment
```

### 预计算缓存（可选）

在训练前预计算邻接矩阵缓存，可以加速后续训练：

```bash
python precompute_cache.py --config configs/gcn_fusion.yaml
```

### 解码

```bash
# 使用脚本
./scripts/decode.sh runs/my_exp/checkpoints/epoch_10.pt test.zh test.en.hyp

# 或直接使用Python
python decode.py --checkpoint runs/my_exp/checkpoints/epoch_10.pt --src_file test.zh --output_file test.en.hyp
```

### 评估

```python
from mt.eval.sacrebleu_eval import evaluate_from_files

metrics = evaluate_from_files("test.en.hyp", "test.en.ref")
print(f"BLEU: {metrics['BLEU']:.2f}")
print(f"chrF: {metrics['chrF']:.2f}")
```

### 训练流程

训练脚本会自动执行：

1. **加载数据集**: 从HuggingFace下载WMT17中英数据集
2. **准备语料**: 生成训练语料文件（`zh.txt`, `en.txt`）
3. **训练分词器**: 训练或加载SentencePiece模型
4. **预计算缓存**: 计算并缓存依存句法邻接矩阵（如果不存在）
5. **创建数据加载器**: 准备训练和验证数据
6. **初始化模型**: 创建Transformer+GCN模型
7. **开始训练**: 执行训练循环，每个epoch后验证

### 训练输出示例

```
使用设备: cuda
加载数据集...
加载/训练SentencePiece模型...
预计算并缓存邻接矩阵...
创建数据加载器...
创建模型...
开始训练...
Epoch 01: 100%|████████| 1562/1562 [10:30<00:00, loss: 5.234]
Epoch 1 average loss: 5.234
--------------------------------------------------------------------------------
SOURCE:    加利福尼亚州水务工程的新问题
TARGET:    New Questions Over California Water Project
PREDICTED (greedy): New questions about California water project
Validation average loss: 4.876
```

---

## 📁 项目结构

```
machine-translation/
├── configs/                # YAML配置文件（唯一配置方式）
│   ├── base.yaml           # 基础配置（数据、模型、训练通用项）
│   ├── gcn_fusion.yaml     # GCN融合实验配置（继承base.yaml）
│   └── decode_beam5.yaml   # Beam search解码配置
│
├── mt/                     # 源码主包
│   ├── models/             # 模型定义
│   │   ├── transformer.py  # Transformer核心组件
│   │   ├── gcn.py          # 图卷积网络
│   │   ├── fusion.py       # 融合模块
│   │   ├── model.py        # 主模型（TransformerGCN）
│   │   └── transformer_baseline.py  # 纯Transformer基线
│   │
│   ├── data/               # 数据处理
│   │   ├── tokenizer.py    # SentencePiece分词器
│   │   ├── dataset.py      # WMT数据集
│   │   ├── dependency.py   # 依存分析
│   │   ├── cache.py        # 邻接矩阵缓存
│   │   └── align.py        # subword↔word对齐工具
│   │
│   ├── training/           # 训练相关
│   │   ├── trainer.py      # 训练器（纯粹化）
│   │   ├── loss.py         # 标签平滑损失
│   │   ├── scheduler.py    # Noam学习率调度器
│   │   └── hooks.py        # 训练回调机制
│   │
│   ├── decoding/           # 解码模块
│   │   ├── beam.py         # Beam search（改进版）
│   │   └── greedy.py       # 贪心解码（改进版）
│   │
│   ├── eval/               # 评估模块
│   │   ├── sacrebleu_eval.py  # SacreBLEU评估
│   │   └── dump_samples.py    # 样例保存
│   │
│   └── utils/             # 工具模块
│       ├── masks.py        # Mask工具
│       ├── config_loader.py # YAML配置加载器
│       ├── logging.py      # 日志工具
│       └── io.py           # IO工具
│
├── scripts/                # 脚本目录
│   ├── train.sh           # 训练入口脚本
│   ├── decode.sh          # 解码批处理脚本
│   └── prepare_data.sh    # 数据准备脚本
│
├── tests/                  # 测试目录
│   ├── test_align.py      # 对齐工具测试
│   ├── test_beam.py       # Beam search测试
│   └── test_dependency.py # 依存分析测试
│
├── train.py               # 主训练脚本
├── train_baseline.py       # 基线训练脚本
├── precompute_cache.py    # 预计算邻接矩阵缓存脚本
├── requirements.txt       # 依赖列表
├── pyproject.toml         # 项目配置
└── README.md              # 本文档
```

---

## ⚙️ 配置说明

### YAML配置系统（唯一配置方式）

项目**完全使用YAML配置文件**，不再支持Python类配置。所有配置通过 `configs/` 目录下的YAML文件管理。

#### 配置文件结构

**configs/base.yaml** - 基础配置（所有实验继承）：
```yaml
# 数据配置
data:
  max_src_len: 64
  max_tgt_len: 64
  train_size: 50000
  vocab_size: 8000
  cache_root: "cache"
  precompute_chunk_size: 3000

# 模型配置
model:
  d_model: 512
  num_heads: 8
  num_layers: 4
  d_ff: 1024
  dropout: 0.1
  gcn_layers_src: 2
  gcn_layers_tgt: 2
  fusion_mode: "gate"  # "concat" 或 "gate"
  pad_idx: 0

# 训练配置
training:
  batch_size: 32
  epochs: 10
  dataloader_workers: null  # null表示自动计算
  pin_memory: true
  persistent_workers: true

# 设备配置
device:
  type: "auto"  # "auto", "cuda", "cpu"
```

**configs/gcn_fusion.yaml** - GCN融合实验配置（继承base.yaml）：
```yaml
# 继承base.yaml，覆盖特定项
model:
  fusion_mode: "gate"  # 或 "concat"
  gcn_layers_src: 2
  gcn_layers_tgt: 2

# 句法图配置
dependency:
  edge_strategy: "full"
  normalization: "sym"
```

**configs/decode_beam5.yaml** - Beam search解码配置：
```yaml
decoding:
  method: "beam_search"
  beam_size: 5
  length_penalty: 0.6  # NMT标准公式的alpha参数
  repetition_penalty: 1.2  # >1.0惩罚重复
  max_len: 64
  early_stop: true
  n_best: 1
```

#### 使用配置

```python
from mt.utils.config_loader import load_config

# 加载配置（自动继承base.yaml）
config = load_config("configs/gcn_fusion.yaml")

# 访问配置
d_model = config['model']['d_model']
batch_size = config['training']['batch_size']
beam_size = config.get('decoding', {}).get('beam_size', 5)
```

#### 创建新实验配置

1. 复制 `configs/gcn_fusion.yaml` 为新文件
2. 修改需要覆盖的配置项
3. 运行训练时指定新配置文件：`python train.py --config configs/my_experiment.yaml`

---

## 🔍 关键特性

### 1. Transformer + GCN融合

- **Transformer**: 标准的编码器-解码器架构
- **GCN**: 处理依存句法信息
- **融合方式**: 支持concat和gate两种模式
- **特征对齐**: GCN使用Transformer第一层输出，统一特征空间
- **训练/推理一致性**: 训练时使用source端GCN + target端GCN，推理时禁用target端GCN以提高速度

### 2. 改进的解码器

#### Beam Search（修正版）

**关键改进**：
- ✅ **长度惩罚修正**: 使用NMT标准公式，raw log_prob和length分开存储，只在排序时应用
- ✅ **重复惩罚**: 支持重复惩罚（repetition_penalty），避免重复生成
- ✅ **禁用target端GCN**: 解码时自动禁用target端GCN，只使用Transformer，提高速度
- ✅ **n-best输出**: 支持返回top-n个结果
- ✅ **调试模式**: 支持详细的调试输出

**长度惩罚公式**（NMT标准）：
```
normalized_score = raw_score / ((5 + length) / 6) ^ length_penalty
```

**使用示例**：

```python
from mt.decoding.beam import beam_search_decode

# 使用改进的beam search
pred_text = beam_search_decode(
    model, src_ids[0].cpu(), sp_src, sp_tgt, device,
    max_len=64, pad_idx=0, beam_size=5,
    length_penalty=0.6,           # NMT标准公式的alpha参数
    repetition_penalty=1.2,       # >1.0惩罚重复
    disable_tgt_gcn=True,          # 禁用target端GCN（默认True）
    debug=False                    # 启用调试模式
)
```

#### 贪心解码（改进版）

```python
from mt.decoding.greedy import greedy_decode

pred_text = greedy_decode(
    model, src_ids[0].cpu(), sp_src, sp_tgt, device,
    max_len=64, pad_idx=0,
    repetition_penalty=1.1,       # 轻微惩罚重复
    disable_tgt_gcn=True          # 禁用target端GCN（默认True）
)
```

**重复惩罚参数说明**：
- `1.0`: 不惩罚重复（默认）
- `1.1-1.3`: 轻微惩罚，适合大多数情况（推荐）
- `>1.5`: 强惩罚，可能过度抑制

**长度惩罚参数说明**（beam search）：
- `0.0`: 简单平均（score / length）
- `0.6`: 常用值，偏向短序列（推荐）
- `1.0`: 不偏向任何长度

### 3. 句法信息利用

- **依存分析**: 使用spaCy构建依存树
- **邻接矩阵**: 预计算并缓存，加速训练
- **软切分**: 智能处理长句子，优先在连词和标点处切分
- **训练/推理分离**: 训练时使用完整句法信息，推理时只使用source端句法信息

### 4. 训练优化

- **预计算缓存**: 邻接矩阵预计算，避免运行时重复计算
- **DataLoader优化**: 多进程、pin_memory等加速
- **权重初始化**: Xavier/He初始化，确保训练起点合理
- **训练钩子**: 支持自定义回调，解耦训练和验证逻辑

### 5. 对齐工具

新增subword↔word对齐工具（`mt/data/align.py`）：

```python
from mt.data.align import word_to_subword_map, pool_subwords_to_words

# 构建词到子词映射
mapping = word_to_subword_map(text, tokenizer)

# 将子词级别特征聚合到词级别
word_states = pool_subwords_to_words(subword_states, mapping, mode='mean')
```

### 6. 评估系统

统一的SacreBLEU评估接口：

```python
from mt.eval.sacrebleu_eval import evaluate_sacrebleu, batch_evaluate_runs

# 单次评估
metrics = evaluate_sacrebleu(hypotheses, references)

# 批量评估runs目录
results = batch_evaluate_runs("runs/", "test.en.ref")
```

---

## ❓ 常见问题

### Q1: numpy版本冲突

**问题**: `numpy 2.0.0` 在Python 3.10上可能不兼容

**解决**:
```bash
pip install "numpy>=1.24.0,<2.0.0"
```

### Q2: spaCy模型下载失败

**解决**:
```bash
# 使用国内镜像
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple zh_core_web_sm en_core_web_sm
```

### Q3: Beam search输出相同

**问题**: 所有预测结果都一样

**解决**: 使用改进的解码器（已修复）

```python
from mt.decoding.beam import beam_search_decode
# 添加repetition_penalty参数
pred_text = beam_search_decode(
    model, src_ids, sp_src, sp_tgt, device,
    beam_size=5, length_penalty=0.6, repetition_penalty=1.2
)
```

如果仍然所有结果相同，可能是：
- 模型没有学习（检查训练loss）
- logits分布过于集中（启用debug查看）

**调试模式**：

```python
pred_text = beam_search_decode(
    model, src_ids[0].cpu(), sp_src, sp_tgt, device,
    max_len=64, pad_idx=0, beam_size=4,
    length_penalty=0.6, repetition_penalty=1.2,
    debug=True  # 启用调试
)
```

### Q4: CUDA不可用

**检查**:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

**解决**: 安装对应CUDA版本的PyTorch（见安装步骤）

### Q5: 内存不足

**解决**: 在`configs/base.yaml`或`configs/gcn_fusion.yaml`中减小：
- `training.batch_size`（默认32）
- `data.train_size`（默认50000）
- `data.max_src_len`和`data.max_tgt_len`（默认64）

### Q6: 数据集下载慢

**解决**: 设置HuggingFace镜像

```bash
# Linux/macOS
export HF_ENDPOINT=https://hf-mirror.com

# Windows PowerShell
$env:HF_ENDPOINT="https://hf-mirror.com"
```

### Q7: 模块导入错误

**问题**: `ModuleNotFoundError: No module named 'mt'`

**解决**: 
- 确保在项目根目录运行脚本
- 检查是否有 `mt/__init__.py` 文件
- 确保虚拟环境已激活

### Q8: 配置文件找不到

**问题**: `FileNotFoundError: configs/gcn_fusion.yaml`

**解决**: 
- 确保配置文件存在于 `configs/` 目录
- 检查配置文件路径是否正确
- 可以使用绝对路径：`python train.py --config /path/to/config.yaml`

---

## 📊 性能优化

### 邻接矩阵缓存

项目支持预计算并缓存邻接矩阵，显著加速训练：

```bash
# 手动预计算（可选）
python precompute_cache.py --config configs/gcn_fusion.yaml

# 训练时会自动检查并计算缓存（如果不存在）
python train.py --config configs/gcn_fusion.yaml
```

缓存位置：`cache/train/` 和 `cache/valid/`

### DataLoader优化

- 多进程加载（`num_workers`，自动计算）
- Pin memory（GPU加速）
- Persistent workers（减少进程创建开销）

### 解码优化

- **禁用target端GCN**: 解码时自动禁用，只使用Transformer，速度提升约2-3倍
- **预计算source端GCN**: source端邻接矩阵只需计算一次

---

## 🧪 验证安装

运行验证脚本：

```bash
python test_installation.py
```

应该看到所有依赖都显示 ✓

---

## 📚 模块说明

### mt/models/
- **transformer.py**: Transformer核心组件（编码器、解码器、注意力）
- **gcn.py**: 语法GCN网络，处理依存树
- **fusion.py**: 融合Transformer和GCN输出
- **model.py**: 主模型类（TransformerGCN），支持禁用target端GCN
- **transformer_baseline.py**: 纯Transformer基线

### mt/data/
- **tokenizer.py**: SentencePiece分词器训练和编码/解码
- **dataset.py**: WMT数据集定义和批处理
- **dependency.py**: 使用spaCy构建依存树邻接矩阵
- **cache.py**: 邻接矩阵预计算和缓存
- **align.py**: subword↔word对齐工具

### mt/training/
- **trainer.py**: 训练器类，封装训练循环（纯粹化版本）
- **loss.py**: 标签平滑损失函数（支持EOS加权）
- **scheduler.py**: Noam学习率调度器
- **hooks.py**: 训练回调机制，解耦训练和验证逻辑

### mt/decoding/
- **beam.py**: 改进的Beam search解码器
  - ✅ 修正长度惩罚（NMT标准公式）
  - ✅ 重复惩罚
  - ✅ 禁用target端GCN
  - ✅ n-best输出
- **greedy.py**: 改进的贪心解码器
  - ✅ 重复惩罚
  - ✅ 禁用target端GCN

### mt/eval/
- **sacrebleu_eval.py**: 统一SacreBLEU评估（BLEU、chrF、TER）
- **dump_samples.py**: 保存样例翻译与注意力权重

### mt/utils/
- **masks.py**: Mask工具函数
- **config_loader.py**: YAML配置加载器（支持继承）
- **logging.py**: 日志工具
- **io.py**: IO工具（JSON、pickle、checkpoint）

---

## 🔄 项目状态

项目已完成重构，所有模块已迁移到`mt/`包：

- ✅ **已完成**: 所有模块已迁移到`mt/`包
- ✅ **已完成**: 统一使用YAML配置（已移除config.py）
- ✅ **已完成**: 解码模块改进（长度惩罚、重复惩罚、禁用target端GCN）
- ✅ **已完成**: 评估模块、脚本、测试

**导入路径**：所有新代码使用 `mt.*` 导入：

```python
from mt.models.model import TransformerGCN
from mt.data.dataset import WMTDataset
from mt.training.trainer import Trainer
from mt.decoding.beam import beam_search_decode
from mt.decoding.greedy import greedy_decode
from mt.eval.sacrebleu_eval import evaluate_sacrebleu
```

**注意**：旧的顶层包（`models/`, `data/`, `training/`, `utils/`）已被删除。**所有代码统一使用`mt/`包**。

---

## 📝 依赖列表

核心依赖：
- PyTorch >= 2.0.0
- datasets >= 2.0.0
- sentencepiece >= 0.1.99
- spacy >= 3.7.0
- numpy >= 1.24.0, < 2.0.0
- nltk >= 3.8
- tqdm >= 4.65.0
- pyyaml >= 6.0

完整列表见 `requirements.txt` 和 `pyproject.toml`

---

## 🎯 快速命令参考

```bash
# 安装
pip install -r requirements.txt
python -m spacy download zh_core_web_sm en_core_web_sm

# 数据准备（可选）
python precompute_cache.py --config configs/gcn_fusion.yaml

# 训练
python train.py                                    # 使用默认配置
python train.py --config configs/gcn_fusion.yaml  # 指定配置
python train_baseline.py                          # 纯Transformer基线
./scripts/train.sh configs/gcn_fusion.yaml my_experiment

# 解码
./scripts/decode.sh runs/my_exp/checkpoints/epoch_10.pt test.zh test.en.hyp

# 评估
python -c "from mt.eval.sacrebleu_eval import evaluate_from_files; print(evaluate_from_files('test.en.hyp', 'test.en.ref'))"

# 测试
pytest tests/

# 验证安装
python test_installation.py
```

---

## 📖 参考资源

- [PyTorch文档](https://pytorch.org/)
- [spaCy文档](https://spacy.io/)
- [HuggingFace Datasets](https://huggingface.co/docs/datasets/)
- [SentencePiece文档](https://github.com/google/sentencepiece)
- [SacreBLEU文档](https://github.com/mjpost/sacrebleu)

---

## 💡 提示

1. **首次运行**: 会下载数据集和模型，可能需要较长时间
2. **GPU加速**: 有GPU时训练速度显著提升
3. **缓存利用**: 邻接矩阵缓存后，后续训练会更快
4. **解码器选择**: 训练初期使用贪心解码，后期可尝试beam search
5. **重复惩罚**: 根据任务调整`repetition_penalty`参数（推荐1.1-1.3）
6. **长度惩罚**: beam search推荐使用`length_penalty=0.6`（NMT标准）
7. **配置管理**: 所有配置通过YAML文件管理，便于实验管理

---

## 🔧 开发指南

### 运行测试

```bash
# 运行所有测试
pytest tests/

# 运行特定测试
pytest tests/test_beam.py

# 带覆盖率
pytest tests/ --cov=mt
```

### 代码格式化

```bash
# 使用black格式化
black mt/ tests/

# 检查代码风格
flake8 mt/ tests/
```

### 添加新功能

1. 在对应的`mt/`子模块中添加代码
2. 更新`__init__.py`导出新功能
3. 添加单元测试
4. 更新README文档

### 创建新实验配置

1. 复制 `configs/gcn_fusion.yaml` 为新文件（如 `configs/my_exp.yaml`）
2. 修改需要覆盖的配置项
3. 运行：`python train.py --config configs/my_exp.yaml`

---

**如有问题，请提交Issue或查看项目文档。**
