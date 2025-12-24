"""
依存分析相关功能（V6版：修复长句切分死循环）
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
    "en": {"and", "but", "because", "although", "while", "which", "that", "or", "if", "so", "when"}
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


def build_dep_adj(texts, sp=None, lang="zh", max_len=64, spacy_parse_limit=64, nlp=None):
    """
    构建依存树邻接矩阵（优先软切分 + 过滤跨句依存边）
    """
    parser = nlp if nlp is not None else _get_nlp(lang)
    batch_adj = []

    for text in texts:
        doc = parser(text)
        num_tokens = len(doc)
        adj = torch.zeros(num_tokens, num_tokens, dtype=torch.float32)

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
                                adj[i, j] = 1.0
                                adj[j, i] = 1.0
            else:
                for token in sent:
                    if token.sent == token.head.sent:
                        i, j = token.i, token.head.i
                        adj[i, j] = 1.0
                        adj[j, i] = 1.0

        I = torch.eye(num_tokens, dtype=torch.float32)
        adj = adj + I

        final_adj = torch.zeros(max_len, max_len, dtype=torch.float32)
        effective_len = min(num_tokens, max_len)
        final_adj[:effective_len, :effective_len] = adj[:effective_len, :effective_len]

        deg = final_adj.sum(dim=-1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        final_adj = D_inv_sqrt @ final_adj @ D_inv_sqrt

        batch_adj.append(final_adj)

    return torch.stack(batch_adj)
