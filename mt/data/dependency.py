"""
依存分析相关功能（V7版：支持边列表 + GCN内部归一化）
"""
import torch
import spacy
from typing import List

# --- 全局加载spaCy模型 ---
_nlp_zh = None
_nlp_en = None

# --- 软切分配置 ---
SOFT_SPLIT_CONJUNCTIONS = {
    "zh": {"和", "并且", "但是", "因为", "所以", "如果", "虽然", "同时", "以及", "或", "或者"},
    "en": {"and", "but", "because", "although", "while", "which", "that", "or", "if", "so", "when"},
}


def _get_nlp(lang):
    """获取spaCy模型（延迟加载，并确保有sentencizer）"""
    global _nlp_zh, _nlp_en
    if lang == "zh":
        if _nlp_zh is None:
            _nlp_zh = spacy.load("zh_core_web_sm")
            if not _nlp_zh.has_pipe("sentencizer"):
                _nlp_zh.add_pipe("sentencizer")
        return _nlp_zh
    else:
        if _nlp_en is None:
            _nlp_en = spacy.load("en_core_web_sm")
            if not _nlp_en.has_pipe("sentencizer"):
                _nlp_en.add_pipe("sentencizer")
        return _nlp_en


# ----------------------
#   句子软切分辅助函数
# ----------------------

def _find_soft_split_point(tokens: List[spacy.tokens.Token], split_point: int, lang: str, window: int = 5) -> int:
    """在切分点附近寻找最佳的软切分位置（优先连词，其次标点）"""
    start = max(0, split_point - window)
    end = min(len(tokens), split_point + window)
    search_window = tokens[start:end]
    conjunctions = SOFT_SPLIT_CONJUNCTIONS.get(lang, set())

    # 优先查找连词
    for i, token in enumerate(search_window):
        if token.text.lower() in conjunctions:
            return start + i + 1

    # 其次查找标点符号
    for i, token in enumerate(search_window):
        if token.is_punct and i > 0:  # 避免在开头切分
            return start + i + 1

    # 如果都找不到，返回-1表示失败
    return -1


def _split_long_sentence(sent: spacy.tokens.span.Span, lang: str, limit: int) -> List[spacy.tokens.span.Span]:
    """对超过长度限制的单个句子进行智能软切分（修复了死循环问题）"""
    tokens = list(sent)
    if len(tokens) <= limit:
        return [sent]

    chunks = []
    offset = 0

    while offset < len(tokens):
        # 如果剩余部分不足以构成一个完整的chunk，直接作为一个chunk处理
        if offset + limit >= len(tokens):
            chunks.append(sent[offset:])
            break

        # 1. 尝试在[limit-window, limit]的范围内寻找软切分点
        soft_split_point = _find_soft_split_point(tokens, offset + limit, lang)

        # 2. 决定切分点
        if soft_split_point > offset:
            # 找到了有效的软切分点
            actual_split = soft_split_point
        else:
            # 未找到软切分点，执行硬切分
            actual_split = offset + limit

        # 3. 确保切分点有实质性推进，防止死循环
        if actual_split <= offset:
            # 这是一个安全保障，理论上不会发生，但可以防止无限循环
            actual_split = offset + limit

        chunks.append(sent[offset:actual_split])
        offset = actual_split

    return chunks


# ----------------------
#   依存边构建：边列表版本
# ----------------------

def build_dep_edges(texts, sp=None, lang: str = "zh", max_len: int = 64, spacy_parse_limit: int = 64, nlp=None):
    """构建依存树边列表（不在此处做归一化）。

    Args:
        texts: 文本列表
        sp:    保留旧接口参数（未使用）
        lang:  语言（"zh" 或 "en"）
        max_len: 对齐到的最大长度（用于裁剪 token 索引）
        spacy_parse_limit: 单句解析长度上限（结合软切分）
        nlp: 可选的 spaCy nlp 对象（用于多进程下的共享）

    Returns:
        List[Tensor]: 长度为 len(texts) 的列表，每个元素为 [num_edges, 2] 的 long tensor。
                      其中 num_edges 可能为 0（例如空句）。
    """
    parser = nlp if nlp is not None else _get_nlp(lang)
    batch_edges: List[torch.Tensor] = []

    for text in texts:
        doc = parser(text)
        num_tokens = len(doc)

        # 收集 (i, j) 边对
        edge_list = []

        for sent in doc.sents:
            if len(sent) > spacy_parse_limit:
                chunks = _split_long_sentence(sent, lang, spacy_parse_limit)
                for chunk in chunks:
                    chunk_text = chunk.text
                    chunk_doc = parser(chunk_text)
                    for token in chunk_doc:
                        if token.sent == token.head.sent:
                            chunk_start = chunk.start
                            i = chunk_start + token.i
                            j = chunk_start + token.head.i
                            if i < num_tokens and j < num_tokens:
                                edge_list.append((i, j))
                                edge_list.append((j, i))
            else:
                for token in sent:
                    if token.sent == token.head.sent:
                        i, j = token.i, token.head.i
                        edge_list.append((i, j))
                        edge_list.append((j, i))

        # 加自环：这里先只生成边，是否添加自环可以在后续转换为邻接矩阵时处理
        # 但为了与旧逻辑一致，我们可以在此处为有效 token 添加自环
        effective_len = min(num_tokens, max_len)
        for i in range(effective_len):
            edge_list.append((i, i))

        if edge_list:
            edges = torch.tensor(edge_list, dtype=torch.long)
        else:
            edges = torch.empty(0, 2, dtype=torch.long)

        # 裁剪到 max_len 以内（主要是安全保护，正常 edge 已在上面控制）
        if edges.numel() > 0:
            edges = edges.clamp(0, max_len - 1)

        batch_edges.append(edges)

    return batch_edges


# ----------------------
#   旧接口：仍保留 build_dep_adj 以兼容旧代码
# ----------------------

def build_dep_adj(texts, sp=None, lang="zh", max_len=64, spacy_parse_limit=64, nlp=None):
    """兼容旧接口的邻接矩阵构建函数。

    现在内部通过 `build_dep_edges` 先生成边列表，再转换为邻接矩阵，
    但保持了原先在预处理阶段做度归一化的行为，以兼容老缓存/老流程。

    后续新 pipeline 建议直接使用 `build_dep_edges` + 运行时归一化。
    """
    from mt.data.cache import edges_to_adjacency  # 延迟导入避免循环

    batch_edges = build_dep_edges(
        texts,
        sp=sp,
        lang=lang,
        max_len=max_len,
        spacy_parse_limit=spacy_parse_limit,
        nlp=nlp,
    )

    batch_adj = []
    for edges in batch_edges:
        adj = edges_to_adjacency(edges, max_len=max_len, normalized=True)
        batch_adj.append(adj)

    return torch.stack(batch_adj)
