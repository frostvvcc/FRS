"""测试 cosine 语义修复 + 新 fusion 模式。

关键验证：
1. semantic='similarity' 下，相似向量的用户彼此是 Top-K 邻居；语义反转则取到最不相似者
2. product 模式不会引入"单图独有"的高分邻居
3. rank_intersection 能在双图都 rank 高的用户上产生邻居
"""

import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import (
    construct_user_relation_graph_via_item,
    construct_user_relation_graph_via_interest,
    select_topk_neighboehood,
)


def _make_round_params():
    """4 个用户：
      user 0 和 user 1 在 item 和 interest 上都相似
      user 2 只在 item 上和 user 0 相似
      user 3 正交 / 不相似
    """
    def _dict(item_vec, int_vec):
        return {
            'embedding_item.weight': torch.tensor(item_vec, dtype=torch.float32),
            'interest_params': torch.tensor(int_vec, dtype=torch.float32).unsqueeze(0),
        }
    return {
        0: _dict([[1.0, 0.0], [0.0, 0.0]], [1.0, 0.0]),
        1: _dict([[0.98, 0.02], [0.0, 0.0]], [0.99, 0.01]),
        2: _dict([[0.95, 0.05], [0.0, 0.0]], [0.0, 1.0]),  # item 相似但 interest 正交
        3: _dict([[0.0, 0.0], [1.0, 0.0]], [0.0, 1.0]),
    }


class TestSimilaritySemantic(unittest.TestCase):
    def test_similarity_returns_correct_direction(self):
        params = _make_round_params()
        g = construct_user_relation_graph_via_item(params, 2, 2, 'cosine',
                                                   semantic='similarity')
        # user 0 与 user 1 相似度应 > user 0 与 user 3
        self.assertGreater(g[0, 1], g[0, 3])
        # 对角元素=1（完全相似）
        self.assertAlmostEqual(float(g[0, 0]), 1.0, places=4)

    def test_distance_legacy_inverts_direction(self):
        params = _make_round_params()
        g = construct_user_relation_graph_via_item(params, 2, 2, 'cosine',
                                                   semantic='distance')
        # 旧 bug：数值越大越不相似
        self.assertLess(g[0, 1], g[0, 3])
        # 对角元素=0（自距离）
        self.assertAlmostEqual(float(g[0, 0]), 0.0, places=4)


class TestProductFusion(unittest.TestCase):
    def test_product_prefers_dual_confirmed(self):
        params = _make_round_params()
        item_g = construct_user_relation_graph_via_item(params, 2, 2, 'cosine',
                                                       semantic='similarity')
        int_g = construct_user_relation_graph_via_interest(params, 'cosine',
                                                           semantic='similarity')
        adj, stats = select_topk_neighboehood(
            item_graph=item_g, mlp_graph=int_g,
            neighborhood_size=1, neighborhood_threshold=0.0,
            fusion='product', return_stats=True,
        )
        # user 0 的邻居应偏向 user 1（双图都相似），而非 user 2（仅 item 相似）
        # 在 top-K=1 下必定选到 1
        self.assertEqual(int(adj[0].argmax()), 1)


class TestRankIntersection(unittest.TestCase):
    def test_rank_intersection_produces_neighbors(self):
        params = _make_round_params()
        item_g = construct_user_relation_graph_via_item(params, 2, 2, 'cosine',
                                                       semantic='similarity')
        int_g = construct_user_relation_graph_via_interest(params, 'cosine',
                                                           semantic='similarity')
        adj, stats = select_topk_neighboehood(
            item_graph=item_g, mlp_graph=int_g,
            neighborhood_size=1, neighborhood_threshold=0.0,
            fusion='rank_intersection', return_stats=True,
        )
        # 应避开自环
        self.assertEqual(float(adj[0, 0]), 0.0)
        # user 0 双图都 rank user 1 高，所以应选 1
        self.assertEqual(int(adj[0].argmax()), 1)


if __name__ == '__main__':
    unittest.main()
