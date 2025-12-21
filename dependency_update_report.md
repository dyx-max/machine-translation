# 依存句法分析模块更新报告

## ✅ 更新概述

已按要求重写依存句法分析模块（`data/dependency.py`），实现以下核心功能：

1.  **动态长度支持**：根据模型的 `max_len`，自动 `pad` 或截断邻接矩阵。
2.  **长句拆分**：当句子长度超过 `spacy_parse_limit`（默认64）时，自动切分子句进行解析。
3.  **结果拼接**：将子句的邻接矩阵拼接成一个完整的大矩阵。
4.  **保证对齐**：最终输出的邻接矩阵大小始终是 `[B, max_len, max_len]`，与 Transformer 输入保持一致。

## 🔧 实现细节

### `build_dep_adj` 函数重写

-   **输入**：`texts` (文本列表), `lang` (语言), `max_len` (目标长度), `spacy_parse_limit` (spaCy解析上限)
-   **输出**：`[B, max_len, max_len]` 的邻接矩阵

### 核心逻辑

1.  **句子长度判断**：
    -   如果 `len(tokens) <= spacy_parse_limit`，直接调用 `_parse_sub_sentence` 解析。
    -   如果 `len(tokens) > spacy_parse_limit`，进入长句处理逻辑。

2.  **长句拆分与拼接**：
    -   将长句按 `spacy_parse_limit` 切分成多个 `chunks`。
    -   对每个 `chunk` 单独调用 `_parse_sub_sentence` 生成小邻接矩阵。
    -   将小矩阵按偏移量拼接到一个大的 `full_adj` 矩阵中。

3.  **Pad 或截断**：
    -   创建一个 `[max_len, max_len]` 的零矩阵 `final_adj`。
    -   将 `full_adj` 的有效部分（`min(num_tokens, max_len)`）复制到 `final_adj` 中。
    -   这样，短句会自动 `pad`，长句会自动截断。

4.  **行归一化**：
    -   对最终的 `final_adj` 进行行归一化，保证矩阵的有效性。

### 新增辅助函数

-   `_parse_sub_sentence(sub_tokens, parser)`：用于解析单个子句并生成邻接矩阵。

## ✨ 优势

1.  **鲁棒性**：能处理任意长度的句子，避免 spaCy 解析超长句子时可能出现的错误。
2.  **灵活性**：`max_len` 和 `spacy_parse_limit` 可配置，适应不同模型和需求。
3.  **一致性**：保证输出的邻接矩阵大小始终与模型输入对齐，避免维度不匹配问题。
4.  **兼容性**：接口保持不变，对现有代码无影响。

## 🧪 验证

-   ✅ 代码已通过语法检查 (`py_compile`)。
-   ✅ 代码已通过 Linter 检查。
-   ✅ 逻辑符合设计要求。

## 💡 使用建议

-   在 `config.py` 中可以添加 `spacy_parse_limit` 配置项，方便调整。
-   建议 `spacy_parse_limit` 保持在 64-128 之间，以获得最佳性能和稳定性。

## 📝 总结

本次更新成功实现了对长句的自动拆分和处理，并保证了邻接矩阵与模型输入的动态对齐，提升了数据预处理的鲁棒性和灵活性。

