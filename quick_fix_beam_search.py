"""
快速修复：在现有代码中使用新的beam search解码器
使用方法：在training/validator.py中导入此模块
"""
import sys
import os

# 添加mt包到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mt.decoding.beam import beam_search_decode as new_beam_search
from mt.decoding.greedy import greedy_decode as new_greedy_decode

# 为了兼容现有代码，创建包装函数
def beam_search_decode_fixed(model, src_ids, sp_src, sp_tgt, device, max_len=64, pad_idx=0, beam_size=4, length_penalty=0.6, debug=False):
    """
    修复后的beam search解码器（添加重复惩罚）
    """
    return new_beam_search(
        model, src_ids, sp_src, sp_tgt, device,
        max_len=max_len, pad_idx=pad_idx, beam_size=beam_size,
        length_penalty=length_penalty, repetition_penalty=1.2,  # 添加重复惩罚
        n_best=1, early_stop=True, debug=debug
    )


def greedy_decode_fixed(model, src_ids, sp_src, sp_tgt, device, max_len=64, pad_idx=0):
    """
    修复后的贪心解码器（添加重复惩罚）
    """
    return new_greedy_decode(
        model, src_ids, sp_src, sp_tgt, device,
        max_len=max_len, pad_idx=pad_idx, repetition_penalty=1.1  # 轻微重复惩罚
    )


# 说明：在training/validator.py中，可以这样使用：
# from quick_fix_beam_search import beam_search_decode_fixed as beam_search_decode
# 这样就能使用修复后的版本，无需修改其他代码

