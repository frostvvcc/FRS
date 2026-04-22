"""测试图构建与消息传递工具函数。"""

import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import (
    construct_user_relation_graph_via_item,
    select_topk_neighboehood,
    MP_on_graph,
)


def _make_params(num_users=3, num_items=2, latent_dim=2):
    """构造 3 个用户的 item embedding：user 0/1 接近，user 2 差异大。"""
    base = np.array([[1.0, 0.0], [0.0, 1.0]], dtype='float32')
    params = {
        0: {'embedding_item.weight': torch.from_numpy(base.copy())},
        1: {'embedding_item.weight': torch.from_numpy(base.copy() + 0.01)},
        2: {'embedding_item.weight': torch.from_numpy(-base.copy())},
    }
    return params, num_users, num_items, latent_dim


class TestGraph(unittest.TestCase):
    def test_relation_graph_shape(self):
        params, n_u, n_i, d = _make_params()
        adj = construct_user_relation_graph_via_item(params, n_i, d, 'cosine')
        self.assertEqual(adj.shape, (n_u, n_u))

    def test_topk_rows_sum_to_one(self):
        params, n_u, n_i, d = _make_params()
        adj = construct_user_relation_graph_via_item(params, n_i, d, 'cosine')
        topk = select_topk_neighboehood(adj, None, neighborhood_size=2, neighborhood_threshold=1.0)
        # 每行有 k 个 1/k，和应为 1
        np.testing.assert_allclose(topk.sum(axis=1), np.ones(n_u), atol=1e-6)

    def test_mp_output_shape_and_global(self):
        params, n_u, n_i, d = _make_params()
        adj = construct_user_relation_graph_via_item(params, n_i, d, 'cosine')
        topk = select_topk_neighboehood(adj, None, neighborhood_size=2, neighborhood_threshold=1.0)
        out = MP_on_graph(params, n_i, d, topk, layers=1)
        self.assertIn('global', out)
        for uid in [0, 1, 2]:
            self.assertEqual(tuple(out[uid].shape), (n_i, d))
        self.assertEqual(tuple(out['global'].shape), (n_i, d))


if __name__ == '__main__':
    unittest.main()
