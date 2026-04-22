"""冒烟测试：验证 MLP 模型的 forward 形状符合预期。

不依赖完整数据集，只构造一个小 config 实例化 MLP 并走一次前向。
这避免了 100k/ml-1m 数据集可能不在 CI 环境的问题。
"""

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestMLPForward(unittest.TestCase):
    def test_forward_shape(self):
        from mlp import MLP

        config = {
            'num_users': 10,
            'num_items': 20,
            'latent_dim': 4,
            'layers': [12, 8, 4],  # latent_dim * 3 = 12（带 attention 的输入维度）
            'use_cuda': False,
            'use_attention': True,
        }
        model = MLP(config)
        model.eval()

        batch_size, seq_len = 5, 3
        items = torch.randint(0, config['num_items'], (batch_size,), dtype=torch.long)
        history = torch.randint(0, config['num_items'], (batch_size, seq_len), dtype=torch.long)

        with torch.no_grad():
            out = model(items, history)
        self.assertEqual(tuple(out.shape), (batch_size, 1))
        # sigmoid 输出应在 [0, 1]
        self.assertTrue(torch.all(out >= 0) and torch.all(out <= 1))


if __name__ == '__main__':
    unittest.main()
