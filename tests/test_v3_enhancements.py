"""V3 增强：多头注意力 + 严格 DP 组合定理 + (未来) 更丰富兴趣编码。"""

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import dp_composition_bounds, laplace_epsilon
from mlp import MLP, MultiHeadAttention


class TestMultiHeadAttention(unittest.TestCase):
    def test_forward_shape(self):
        att = MultiHeadAttention(latent_dim=32, num_heads=4, max_history_len=32)
        query = torch.randn(5, 1, 32)       # [B, 1, D]
        keys = torch.randn(5, 20, 32)       # [B, L, D]
        out = att(query, keys)
        self.assertEqual(tuple(out.shape), (5, 32))

    def test_divisibility_assertion(self):
        with self.assertRaises(AssertionError):
            MultiHeadAttention(latent_dim=32, num_heads=5)  # 5 不整除 32

    def test_mlp_uses_multihead_when_configured(self):
        cfg = {
            'num_users': 10, 'num_items': 20, 'latent_dim': 32,
            'layers': [96, 32, 16, 8], 'use_cuda': False,
            'use_attention': True, 'attention_type': 'multihead',
            'num_heads': 4, 'max_history_len': 32,
        }
        model = MLP(cfg)
        self.assertIsInstance(model.attention_layer, MultiHeadAttention)
        # 前向应能正常运行
        model.eval()
        items = torch.randint(0, 20, (3,), dtype=torch.long)
        hist = torch.randint(0, 20, (3, 10), dtype=torch.long)
        with torch.no_grad():
            out = model(items, hist)
        self.assertEqual(tuple(out.shape), (3, 1))

    def test_mlp_defaults_to_single_attention(self):
        from mlp import LightweightAttention
        cfg = {
            'num_users': 10, 'num_items': 20, 'latent_dim': 32,
            'layers': [96, 32, 16, 8], 'use_cuda': False,
            'use_attention': True,  # attention_type 缺省 = 'single'
        }
        model = MLP(cfg)
        self.assertIsInstance(model.attention_layer, LightweightAttention)


class TestDpComposition(unittest.TestCase):
    def test_naive_is_T_times_eps(self):
        # dp=0.01 → ε_per_round = 100
        per_round = laplace_epsilon(0.01)  # = 100
        T = 25
        res = dp_composition_bounds(per_round, T, delta=1e-5)
        self.assertAlmostEqual(res['naive'], T * per_round, places=4)

    def test_advanced_tighter_than_naive_for_large_T_small_eps(self):
        # 小单轮 ε + 很多轮 → advanced 应该比 naive 紧
        # (实际 ε=100 太大，advanced 反而差；用小 ε 测)
        per_round = 0.1
        T = 100
        res = dp_composition_bounds(per_round, T, delta=1e-5)
        self.assertLess(res['advanced'], res['naive'])
        self.assertEqual(res['tightest'], min(res['naive'], res['advanced'], res['rdp']))

    def test_zero_noise_returns_infinity(self):
        res = dp_composition_bounds(float('inf'), 10, delta=1e-5)
        self.assertEqual(res['naive'], float('inf'))

    def test_rdp_returns_finite(self):
        res = dp_composition_bounds(0.5, 50, delta=1e-5)
        self.assertTrue(res['rdp'] > 0)
        self.assertTrue(res['rdp'] < float('inf'))


if __name__ == '__main__':
    unittest.main()
