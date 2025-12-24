"""
依存分析相关功能（V5版：优先软切分 + 过滤跨句依存边）
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
        if token.text in conjunctions:
            return start + i + 1
    
    # 其次查找标点符号
    for i, token in enumerate(search_window):
        if token.is_punct and i > 0:  # 避免在开头切分
            return start + i + 1
    
    # 如果都找不到，返回原始切分点（硬切分）
    return split_point


def _split_long_sentence(sent: spacy.tokens.span.Span, lang: str, limit: int) -> List[spacy.tokens.span.Span]:
    """对超过长度限制的单个句子进行智能软切分（优先软切分）"""
    tokens = list(sent)
    if len(tokens) <= limit:
        return [sent]
    
    chunks = []
    offset = 0
    
    while offset < len(tokens):
        remaining = len(tokens) - offset
        if remaining <= limit:
            # 剩余部分可以直接作为一个chunk
            chunks.append(sent[offset:])
            break
        
        # 计算硬切分点
        hard_split_point = offset + limit
        
        # 优先使用软切分
        soft_split_point = _find_soft_split_point(tokens, hard_split_point, lang)
        
        # 确保切分点有效（不能小于offset，也不能超过剩余长度）
        actual_split = min(max(offset + 1, soft_split_point), len(tokens))
        
        # 创建chunk
        chunks.append(sent[offset:actual_split])
        offset = actual_split
    
    return chunks


def build_dep_adj(texts, sp=None, lang="zh", max_len=64, spacy_parse_limit=64, nlp=None):
    """
    构建依存树邻接矩阵（优先软切分 + 过滤跨句依存边）
    
    Args:
        texts: 原始句子字符串列表
        lang: 语言 ("zh" 或 "en")
        max_len: 目标长度（用于pad或截断）
        spacy_parse_limit: spaCy单次解析的token上限（默认64）
    
    Returns:
        [B, max_len, max_len] 邻接矩阵
    """
    # 使用传入的nlp对象，如果没有则获取默认的
    parser = nlp if nlp is not None else _get_nlp(lang)
    batch_adj = []

    for text in texts:
        # 1. 使用spaCy进行分词和句子切分
        doc = parser(text)
        num_tokens = len(doc)
        adj = torch.zeros(num_tokens, num_tokens, dtype=torch.float32)

        # 2. 对每个自然句子进行处理，长句子优先使用软切分
        for sent in doc.sents:
            # 如果句子超过限制，进行软切分
            if len(sent) > spacy_parse_limit:
                chunks = _split_long_sentence(sent, lang, spacy_parse_limit)
                # 对每个chunk单独解析以获得完整的依存树
                for chunk in chunks:
                    chunk_text = chunk.text
                    chunk_doc = parser(chunk_text)
                    # 在全局邻接矩阵中填充依存关系（过滤跨句依存边）
                    for token in chunk_doc:
                        if token.sent == token.head.sent:  # 确保在同一个句子内
                            # 计算在原始doc中的索引
                            chunk_start = chunk.start
                            i = chunk_start + token.i
                            j = chunk_start + token.head.i
                            if i < num_tokens and j < num_tokens:
                                adj[i, j] = 1.0
                                adj[j, i] = 1.0
            else:
                # 短句子直接处理
                for token in sent:
                    # 过滤跨句依存边
                    if token.sent == token.head.sent:
                        i, j = token.i, token.head.i
                        adj[i, j] = 1.0
                        adj[j, i] = 1.0

        # 3. 添加自环
        I = torch.eye(num_tokens, dtype=torch.float32)
        adj = adj + I

        # 4. Pad或截断到max_len
        final_adj = torch.zeros(max_len, max_len, dtype=torch.float32)
        effective_len = min(num_tokens, max_len)
        final_adj[:effective_len, :effective_len] = adj[:effective_len, :effective_len]

        # 5. 对称归一化 (D^-1/2 * A * D^-1/2)
        deg = final_adj.sum(dim=-1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        final_adj = D_inv_sqrt @ final_adj @ D_inv_sqrt
        
        batch_adj.append(final_adj)

    return torch.stack(batch_adj)  # [B, max_len, max_len]

