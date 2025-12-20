# 机器翻译训练脚本

这是一个基于Transformer+GCN的机器翻译训练脚本，已重构为模块化结构，便于维护和扩展。

## 项目结构

```
.
├── train.py              # 主训练脚本
├── config.py             # 配置文件
├── models/               # 模型定义
│   ├── __init__.py
│   ├── transformer.py   # Transformer核心组件（编码器、解码器、注意力等）
│   ├── gcn.py           # 图卷积网络组件
│   ├── fusion.py        # 融合模块（Transformer和GCN的融合）
│   └── model.py         # 主模型（TransformerGCN）
├── data/                 # 数据处理
│   ├── __init__.py
│   ├── tokenizer.py     # SentencePiece分词器
│   ├── dataset.py       # 数据集定义
│   └── dependency.py    # 依存分析（构建邻接矩阵）
├── training/            # 训练相关
│   ├── __init__.py
│   ├── trainer.py       # 训练器
│   ├── loss.py          # 损失函数
│   ├── scheduler.py     # 学习率调度器
│   └── validator.py     # 验证器
└── utils/               # 工具函数
    ├── __init__.py
    ├── masks.py         # Mask工具函数
    └── decoder.py       # 解码器（beam search）
```

## 使用方法

直接运行主训练脚本：

```bash
python train.py
```

## 配置修改

所有配置都在 `config.py` 文件中，可以方便地修改：

- 模型参数（d_model, num_heads, num_layers等）
- 训练参数（batch_size, epochs等）
- 数据参数（max_len, vocab_size等）

## 模块说明

### models/
- **transformer.py**: 包含Transformer的所有核心组件
- **gcn.py**: 语法GCN网络，用于处理依存树信息
- **fusion.py**: 融合Transformer和GCN输出的模块
- **model.py**: 主模型类，整合所有组件

### data/
- **tokenizer.py**: SentencePiece分词器的训练和编码/解码
- **dataset.py**: WMT数据集的定义和批处理函数
- **dependency.py**: 使用spaCy构建依存树邻接矩阵

### training/
- **trainer.py**: 训练器类，封装训练循环
- **loss.py**: 标签平滑损失函数
- **scheduler.py**: Noam学习率调度器
- **validator.py**: 验证函数，计算损失并打印示例

### utils/
- **masks.py**: 生成各种mask（padding mask, causal mask）
- **decoder.py**: Beam search解码器

## 重构优势

1. **模块化**: 每个模块职责清晰，易于理解和维护
2. **可扩展**: 可以轻松添加新功能或修改现有功能
3. **可测试**: 每个模块可以独立测试
4. **可复用**: 模块可以在其他项目中复用

## 依赖

- PyTorch
- transformers / datasets
- sentencepiece
- spacy (需要下载zh_core_web_sm和en_core_web_sm模型)
- nltk
- tqdm

