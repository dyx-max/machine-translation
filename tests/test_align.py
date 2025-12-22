"""
测试对齐工具
"""
import unittest
import sentencepiece as spm
import torch
from mt.data.align import word_to_subword_map, pool_subwords_to_words, expand_words_to_subwords


class TestAlign(unittest.TestCase):
    """对齐工具测试"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建简单的tokenizer（用于测试）
        # 注意：实际使用时需要真实的SentencePiece模型
        self.text = "hello world"
        self.words = self.text.split()
    
    def test_word_to_subword_map(self):
        """测试词到子词映射"""
        # 这里需要实际的tokenizer，暂时跳过
        # 实际测试时需要提供真实的tokenizer
        pass
    
    def test_pool_subwords_to_words(self):
        """测试子词聚合到词"""
        # 创建模拟数据
        B, L_subword, d_model = 2, 10, 8
        hidden_states = torch.randn(B, L_subword, d_model)
        
        # 创建简单映射
        mapping = {
            0: [0, 1, 2],  # 第一个词对应子词0,1,2
            1: [3, 4],     # 第二个词对应子词3,4
            2: [5, 6, 7, 8, 9]  # 第三个词对应子词5-9
        }
        
        # 测试first模式
        word_states_first = pool_subwords_to_words(hidden_states, mapping, mode='first')
        self.assertEqual(word_states_first.shape, (B, 3, d_model))
        
        # 测试mean模式
        word_states_mean = pool_subwords_to_words(hidden_states, mapping, mode='mean')
        self.assertEqual(word_states_mean.shape, (B, 3, d_model))
        
        # 测试max模式
        word_states_max = pool_subwords_to_words(hidden_states, mapping, mode='max')
        self.assertEqual(word_states_max.shape, (B, 3, d_model))
    
    def test_expand_words_to_subwords(self):
        """测试词扩展到子词"""
        # 创建模拟数据
        B, L_word, d_model = 2, 3, 8
        word_states = torch.randn(B, L_word, d_model)
        
        # 创建映射
        mapping = {
            0: [0, 1, 2],
            1: [3, 4],
            2: [5, 6, 7, 8, 9]
        }
        
        # 扩展
        subword_states = expand_words_to_subwords(word_states, mapping)
        self.assertEqual(subword_states.shape[0], B)
        self.assertEqual(subword_states.shape[2], d_model)
        # 检查子词长度
        max_subword_idx = max(max(indices) for indices in mapping.values())
        self.assertEqual(subword_states.shape[1], max_subword_idx + 1)


if __name__ == '__main__':
    unittest.main()

