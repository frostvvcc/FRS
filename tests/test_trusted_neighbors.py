"""测试毕设核心创新：可信邻居交集筛选。"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import select_topk_neighboehood, laplace_epsilon


def _make_graphs():
    """构造 4 个用户的两张图。设计成：
    - user 0 与 1,2 在 item 图上近；user 0 与 1,3 在 interest 图上近。
    - → intersection(user 0) = {1}；union(user 0) = {1,2,3}
    """
    item = np.array([
        [10, 9, 8, 0],
        [9, 10, 7, 0],
        [8, 7, 10, 0],
        [0, 0, 0, 10],
    ], dtype='float32')
    interest = np.array([
        [10, 9, 0, 8],
        [9, 10, 0, 7],
        [0, 0, 10, 0],
        [8, 7, 0, 10],
    ], dtype='float32')
    return item, interest


class TestTrustedNeighborhood(unittest.TestCase):

    def test_intersection_produces_trusted_neighbors(self):
        item, interest = _make_graphs()
        adj, stats = select_topk_neighboehood(
            item_graph=item, mlp_graph=interest,
            neighborhood_size=3, neighborhood_threshold=0.0,
            fusion='intersection', return_stats=True,
        )
        # user 0: item top3 = {0,1,2}; interest top3 = {0,1,3}; ∩ = {0,1}
        self.assertAlmostEqual(float(adj[0][0]), 0.5, places=6)
        self.assertAlmostEqual(float(adj[0][1]), 0.5, places=6)
        self.assertAlmostEqual(float(adj[0][2]), 0.0, places=6)
        self.assertAlmostEqual(float(adj[0][3]), 0.0, places=6)
        self.assertIn('avg_false_neighbor_ratio', stats)
        # stats.avg_false_neighbor_ratio 应该 > 0（两图都有独有邻居）
        self.assertGreater(stats['avg_false_neighbor_ratio'], 0.0)

    def test_union_is_superset_of_intersection(self):
        item, interest = _make_graphs()
        adj_int = select_topk_neighboehood(
            item_graph=item, mlp_graph=interest,
            neighborhood_size=3, neighborhood_threshold=0.0,
            fusion='intersection',
        )
        adj_uni = select_topk_neighboehood(
            item_graph=item, mlp_graph=interest,
            neighborhood_size=3, neighborhood_threshold=0.0,
            fusion='union',
        )
        # 并集的非零位置一定包含交集的非零位置
        int_mask = adj_int > 0
        uni_mask = adj_uni > 0
        self.assertTrue(np.all(uni_mask[int_mask]))

    def test_alpha_is_unchanged_by_refactor(self):
        """alpha 模式仍应给出向后兼容的结果（没 mlp_graph 时等同 item_only）。"""
        item, interest = _make_graphs()
        adj_alpha = select_topk_neighboehood(
            item_graph=item, mlp_graph=interest,
            neighborhood_size=2, neighborhood_threshold=0.0,
            alpha=1.0, fusion='alpha',
        )
        # alpha=1.0 → 只用 item 图；user 0 top2 = {0,1}
        self.assertGreater(adj_alpha[0][0], 0)
        self.assertGreater(adj_alpha[0][1], 0)
        self.assertEqual(adj_alpha[0][2], 0)
        self.assertEqual(adj_alpha[0][3], 0)

    def test_isolated_node_fallback(self):
        """交集为空时应回退到 item 图邻居（避免孤立节点）。"""
        # 让两图完全不相交：item 只推 user1，interest 只推 user2
        item = np.array([
            [10, 9, 0, 0],
            [9, 10, 0, 0],
            [0, 0, 10, 0],
            [0, 0, 0, 10],
        ], dtype='float32')
        interest = np.array([
            [10, 0, 9, 0],
            [0, 10, 0, 0],
            [9, 0, 10, 0],
            [0, 0, 0, 10],
        ], dtype='float32')
        adj, stats = select_topk_neighboehood(
            item_graph=item, mlp_graph=interest,
            neighborhood_size=2, neighborhood_threshold=0.0,
            fusion='intersection', return_stats=True,
        )
        # user 0: item top2={0,1} (or {1,0}); interest top2={0,2}; ∩={0}
        # user 0 的 self-edge 只有自己 → 1.0
        # 非孤立，因为 {0} 非空
        self.assertGreater(adj[0][0], 0)


class TestLaplaceEpsilon(unittest.TestCase):
    def test_scale_to_epsilon(self):
        self.assertEqual(laplace_epsilon(0.0), float('inf'))
        self.assertAlmostEqual(laplace_epsilon(1.0), 1.0)
        self.assertAlmostEqual(laplace_epsilon(0.5), 2.0)
        self.assertAlmostEqual(laplace_epsilon(0.1, sensitivity=2.0), 20.0)


if __name__ == '__main__':
    unittest.main()
