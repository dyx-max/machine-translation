"""
依存分析相关功能
"""
import torch
import spacy


# 全局加载spaCy模型（延迟加载）
_nlp_zh = None
_nlp_en = None


def _get_nlp(lang):
    """获取spaCy模型（延迟加载）"""
    global _nlp_zh, _nlp_en
    if lang == "zh":
        if _nlp_zh is None:
            _nlp_zh = spacy.load("zh_core_web_sm")
        return _nlp_zh
    else:
        if _nlp_en is None:
            _nlp_en = spacy.load("en_core_web_sm")
        return _nlp_en


def build_dep_adj(texts, sp, lang="zh", pad_idx=0, max_len=64):
    """
    构建依存树邻接矩阵
    Args:
        texts: 原始句子字符串列表
        sp: 对应的SentencePiece分词器
        lang: "zh" 或 "en"
        pad_idx: padding索引
        max_len: 最大长度
    Returns:
        [B, L, L] 邻接矩阵
    """
    parser = _get_nlp(lang)
    batch_adj = []
    
    for text in texts:
        # 可选：提前截断文本，避免spaCy解析过长
        tokens = text.strip().split()
        text = " ".join(tokens[:max_len])

        doc = parser(text)
        L = max_len
        adj = torch.zeros(L, L)

        # 自环（只处理前max_len个token）
        for i in range(min(len(doc), L)):
            adj[i, i] = 1.0

        # 添加依存边（限制在合法范围内）
        for token in doc:
            i, j = token.i, token.head.i
            if i < L and j < L:
                adj[i, j] = 1.0
                adj[j, i] = 1.0

        # 行归一化
        deg = adj.sum(dim=-1, keepdim=True) + 1e-8
        adj = adj / deg
        batch_adj.append(adj)

    return torch.stack(batch_adj)  # [B, L, L]

