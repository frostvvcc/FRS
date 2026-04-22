"""测试 MetronAtK 的 HR@K 与 NDCG@K 计算。

构造一个 2 用户 × (1 正 + 2 负) 的小数据：
- user 0: test_item=10，得分 0.9；neg_items=[20, 30]，得分 [0.8, 0.1] → 正样本排第 1 (命中)
- user 1: test_item=11，得分 0.2；neg_items=[21, 31]，得分 [0.9, 0.5] → 正样本排第 3 (未命中 @2)
"""

import os
import sys
import math
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from metrics import MetronAtK


class TestMetronAtK(unittest.TestCase):
    def _build(self):
        metron = MetronAtK(top_k=2)
        metron.subjects = [
            [0, 1],                       # test_users
            [10, 11],                     # test_items
            [0.9, 0.2],                   # test_scores
            [0, 0, 1, 1],                 # neg_users
            [20, 30, 21, 31],             # neg_items
            [0.8, 0.1, 0.9, 0.5],         # neg_scores
        ]
        return metron

    def test_hit_ratio_at_2(self):
        metron = self._build()
        hr = metron.cal_hit_ratio()
        self.assertAlmostEqual(hr, 0.5, places=6)

    def test_ndcg_at_2(self):
        metron = self._build()
        ndcg = metron.cal_ndcg()
        # user0 命中 rank=1 -> log(2)/log(2) = 1；user1 未命中 -> 0；均值=0.5
        self.assertAlmostEqual(ndcg, 0.5, places=6)

    def test_hit_ratio_at_3_all_hit(self):
        metron = self._build()
        metron.top_k = 3
        hr = metron.cal_hit_ratio()
        self.assertAlmostEqual(hr, 1.0, places=6)

    def test_ndcg_at_3(self):
        metron = self._build()
        metron.top_k = 3
        ndcg = metron.cal_ndcg()
        # user0: rank=1 -> 1；user1: rank=3 -> log(2)/log(4) = 0.5；均值 0.75
        expected = (1.0 + math.log(2) / math.log(4)) / 2
        self.assertAlmostEqual(ndcg, expected, places=6)


if __name__ == '__main__':
    unittest.main()
