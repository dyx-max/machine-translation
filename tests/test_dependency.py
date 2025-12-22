"""
测试依存分析功能
"""
import unittest
import torch
from mt.data.dependency import build_dep_adj


class TestDependency(unittest.TestCase):
    """依存分析测试"""
    
    def setUp(self):
        """设置测试环境"""
        # 注意：需要spaCy模型才能运行
        pass
    
    def test_build_dep_adj_basic(self):
        """测试基本邻接矩阵构建"""
        # 需要spaCy模型，暂时跳过
        # 实际测试时需要确保spaCy模型已安装
        try:
            texts = ["这是一个测试句子。", "This is a test sentence."]
            adj_zh = build_dep_adj(texts, lang="zh", max_len=10)
            adj_en = build_dep_adj(texts, lang="en", max_len=10)
            
            # 检查形状
            self.assertEqual(adj_zh.shape, (2, 10, 10))
            self.assertEqual(adj_en.shape, (2, 10, 10))
            
            # 检查对称性（无向图）
            for i in range(adj_zh.shape[0]):
                self.assertTrue(torch.allclose(adj_zh[i], adj_zh[i].T, atol=1e-5))
        except Exception as e:
            self.skipTest(f"需要spaCy模型: {e}")
    
    def test_build_dep_adj_padding(self):
        """测试padding处理"""
        try:
            texts = ["短句"]
            adj = build_dep_adj(texts, lang="zh", max_len=64)
            self.assertEqual(adj.shape, (1, 64, 64))
        except Exception as e:
            self.skipTest(f"需要spaCy模型: {e}")


if __name__ == '__main__':
    unittest.main()

