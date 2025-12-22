"""
解码模块
"""
from mt.decoding.greedy import greedy_decode
from mt.decoding.beam import beam_search_decode

__all__ = ['greedy_decode', 'beam_search_decode']

