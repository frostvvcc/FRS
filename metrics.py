# metrics.py

import math
import pandas as pd

class MetronAtK(object):
    """
    评估指标类：用于计算 Hit Ratio @ K 和 NDCG @ K
    用法：
        1. 创建实例：metron = MetronAtK(top_k=10)
        2. 设置 metron.subjects = [test_users, test_items, test_scores, neg_users, neg_items, neg_scores]
        3. 调用 metron.cal_hit_ratio() 和 metron.cal_ndcg()
    """

    def __init__(self, top_k):
        """
        初始化 MetronAtK

        参数:
            top_k: int，指定 Top-K 阈值
        功能:
            保存 top_k，初始化 subjects 为 None
        输出:
            无
        """
        self._top_k = top_k
        self._subjects = None  # 存储完整的评估 DataFrame

    @property
    def top_k(self):
        """返回当前 Top-K 值"""
        return self._top_k

    @top_k.setter
    def top_k(self, top_k):
        """
        设置 Top-K 值

        参数:
            top_k: int，新的 Top-K 阈值
        """
        self._top_k = top_k

    @property
    def subjects(self):
        """返回当前的评估 DataFrame (含排序、打分信息)"""
        return self._subjects

    @subjects.setter
    def subjects(self, subjects):
        """
        设置评估所需数据，并构建评分、排名 DataFrame

        参数:
            subjects: list，包含 6 个元素：
                - test_users: list 或 Series，长度 = num_users，每个元素为 userId 的正样本用户列表
                - test_items: list 或 Series，长度 = num_users，每个元素为 userId 的正样本 item 列表
                - test_scores: list 或 Series，长度 = num_users，正样本的预测得分
                - neg_users: list 或 Series，长度 = num_users * num_neg_per_user，负样本用户列表
                - neg_items: list 或 Series，长度 = num_users * num_neg_per_user，负样本 item 列表
                - neg_scores: list 或 Series，长度 = num_users * num_neg_per_user，负样本得分列表
        功能:
            1. 构建 DataFrame 'test'：包含列 ['user', 'test_item', 'test_score']
            2. 构建 DataFrame 'full'：包含所有正负样本，列 ['user', 'item', 'score']
            3. 将 full 与 test 按 ['user'] 左连接，得到额外列 test_item, test_score
            4. 对于每个用户，按 score 降序对 full DataFrame 排名（rank 从 1 开始）
            5. 将排序后的 DataFrame 存入 self._subjects，后续计算时直接使用
        输出:
            无
        """
        assert isinstance(subjects, list) and len(subjects) == 6
        test_users, test_items, test_scores = subjects[0], subjects[1], subjects[2]
        neg_users, neg_items, neg_scores = subjects[3], subjects[4], subjects[5]

        # 构建正样本 DataFrame
        test = pd.DataFrame({'user': test_users,
                             'test_item': test_items,
                             'test_score': test_scores})
        # 构建包含正负样本的 DataFrame
        full = pd.DataFrame({'user': neg_users + test_users,
                             'item': neg_items + test_items,
                             'score': neg_scores + test_scores})
        # 将 full 与 test 合并，为 full 中的正样本行添加 test_item 信息
        full = pd.merge(full, test, on=['user'], how='left')
        # 对每个用户 group，按 score 降序排名（rank 从 1 开始）
        full['rank'] = full.groupby('user')['score'].rank(method='first', ascending=False)
        full.sort_values(['user', 'rank'], inplace=True)
        self._subjects = full

    def cal_hit_ratio(self):
        """
        计算 Hit Ratio @ K

        功能:
            1. 从 self._subjects 中筛选 rank <= top_k 的行（Top-K 候选）
            2. 判断正样本（test_item）是否位于该用户的 Top-K 候选中
            3. 计算 Hit Ratio = 命中用户数 / 用户总数
        输出:
            hit_ratio: float，Hit Ratio @ K
        """
        full = self._subjects
        top_k = full[full['rank'] <= self._top_k]
        # 找到 top_k 中 test_item == item 的行，表示该用户被命中
        test_in_top_k = top_k[top_k['test_item'] == top_k['item']]
        # 用户总数 = 唯一 user 数
        num_users = full['user'].nunique()
        return len(test_in_top_k) * 1.0 / num_users

    def cal_ndcg(self):
        """
        计算 NDCG @ K

        功能:
            1. 从 self._subjects 中筛选 rank <= top_k 的行（Top-K 候选）
            2. 找到正样本所在的 rank（若不存在则不会被计入）
            3. 对每个命中的正样本，计算 DCG = log(2) / log(1 + rank)
            4. 将所有用户的 DCG 求和并除以用户总数，得到平均 NDCG
        输出:
            ndcg: float，NDCG @ K
        """
        full = self._subjects
        top_k = full[full['rank'] <= self._top_k]
        test_in_top_k = top_k[top_k['test_item'] == top_k['item']]
        # 计算每个命中用户的 DCG 分量
        test_in_top_k['ndcg'] = test_in_top_k['rank'].apply(lambda x: math.log(2) / math.log(1 + x))
        num_users = full['user'].nunique()
        return test_in_top_k['ndcg'].sum() * 1.0 / num_users
