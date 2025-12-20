"""
SentencePiece分词器相关功能
"""
import os
import sentencepiece as spm


def train_or_load_spm(corpus_path, model_prefix, vocab_size=8000):
    """
    训练或加载SentencePiece模型
    Args:
        corpus_path: 语料文件路径
        model_prefix: 模型前缀
        vocab_size: 词汇表大小
    Returns:
        SentencePiece处理器
    """
    model_file = f"{model_prefix}.model"
    if os.path.exists(model_file):
        sp = spm.SentencePieceProcessor()
        sp.load(model_file)
        return sp
    spm.SentencePieceTrainer.Train(
        input=corpus_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=0.9995,
        model_type="bpe",
        bos_id=1, eos_id=2, pad_id=0, unk_id=3
    )
    sp = spm.SentencePieceProcessor()
    sp.load(model_file)
    return sp


def encode_sp(sp, text, max_len):
    """
    使用SentencePiece编码文本
    Args:
        sp: SentencePiece处理器
        text: 文本字符串
        max_len: 最大长度
    Returns:
        编码后的ID列表
    """
    ids = sp.encode(text, out_type=int)
    ids = [1] + ids + [2]  # 添加BOS和EOS
    return ids[:max_len] + [0]*(max_len-len(ids)) if len(ids)<max_len else ids[:max_len]


def decode_sp(sp, ids):
    """
    使用SentencePiece解码ID列表
    Args:
        sp: SentencePiece处理器
        ids: ID列表
    Returns:
        解码后的文本字符串
    """
    if 2 in ids: 
        ids = ids[:ids.index(2)]  # 截取到EOS
    ids = [i for i in ids if i not in (0,1,2)]  # 移除PAD、BOS、EOS
    return sp.decode(ids)

