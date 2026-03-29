import torch
import random
import pandas as pd
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset

# 设置随机种子以保证结果可复现
random.seed(0)

class UserItemRatingDataset(Dataset):
    """将<用户, 物品, 评分>张量封装为 PyTorch Dataset，用于 DataLoader"""

    def __init__(self, user_tensor, item_tensor, target_tensor):
        """
        初始化 Dataset

        参数:
            user_tensor: torch.Tensor，存储用户索引（LongTensor）
            item_tensor: torch.Tensor，存储物品索引（LongTensor）
            target_tensor: torch.Tensor，存储对应用户-物品对的评分（FloatTensor 或 LongTensor）
        功能:
            将传入的用户、物品、评分张量保存为成员变量，供 DataLoader 按批次索引调用
        输出:
            无（构造函数）
        """
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        """
        根据索引获取单个样本

        参数:
            index: int，要获取的样本索引
        功能:
            返回对应 index 的 (user, item, rating) 三元组
        输出:
            (user_tensor[index], item_tensor[index], target_tensor[index])
        """
        return self.user_tensor[index], self.item_tensor[index], self.target_tensor[index]

    def __len__(self):
        """
        获取数据集中样本数量

        参数:
            无
        功能:
            返回数据集中样本总数，即 user_tensor 的长度
        输出:
            int，样本数量
        """
        return self.user_tensor.size(0)

class SampleGenerator(object):
    """为神经协同过滤（NCF）模型构造训练、验证和测试数据集"""

    def __init__(self, ratings):
        """
        初始化 SampleGenerator

        参数:
            ratings: pandas.DataFrame，必须包含列 ['userId', 'itemId', 'rating', 'timestamp']
        功能:
            1. 检查输入 DataFrame 是否包含所需列
            2. 通过 _binarize 将评分二值化（implicit feedback）
            3. 生成用户集合和物品集合
            4. 对每个用户构造负采样列表（negatives）
            5. 进行 leave-one-out 划分，得到 train、val、test 三个子集
        输出:
            无（构造函数）
        """
        assert 'userId' in ratings.columns
        assert 'itemId' in ratings.columns
        assert 'rating' in ratings.columns

        self.ratings = ratings
        # 将评分二值化：rating > 0 → 1.0，否则 0（隐式反馈）
        self.preprocess_ratings = self._binarize(ratings)
        # 用户 ID 集合
        self.user_pool = set(self.ratings['userId'].unique())
        # 物品 ID 集合
        self.item_pool = set(self.ratings['itemId'].unique())
        # 针对每个用户生成负样本集合，以及从中采样的负样本列表
        self.negatives = self._sample_negative(ratings)
        # 对预处理后的 ratings 进行 Leave-One-Out 划分: train、val、test
        self.train_ratings, self.val_ratings, self.test_ratings = self._split_loo(self.preprocess_ratings)

    def _normalize(self, ratings):
        """
        将显式反馈评分归一化到 [0, 1] 区间（本系统使用隐式反馈，暂未调用）

        参数:
            ratings: pandas.DataFrame，原始评分表
        功能:
            1. 深度拷贝 ratings，避免在原 DataFrame 上修改
            2. 将 rating 列除以最大评分值，映射到 [0, 1]
        输出:
            pandas.DataFrame，归一化后的评分
        """
        ratings = deepcopy(ratings)
        max_rating = ratings.rating.max()
        ratings['rating'] = ratings.rating * 1.0 / max_rating
        return ratings

    def _binarize(self, ratings):
        """
        将评分二值化（隐式反馈处理）

        参数:
            ratings: pandas.DataFrame，原始评分表
        功能:
            1. 深度拷贝 ratings
            2. 对于 rating > 0 的条目，将 rating 设为 1.0；否则为 0（未显示在此处，但可做扩展）
        输出:
            pandas.DataFrame，二值化后的评分表
        """
        ratings = deepcopy(ratings)
        # 将所有大于 0 的评分都当作 1.0
        ratings['rating'][ratings['rating'] > 0] = 1.0
        return ratings

    def _split_loo(self, ratings):
        """
        采用 Leave-One-Out 方式划分 train/val/test 数据集

        参数:
            ratings: pandas.DataFrame，已预处理（通常是二值化）的评分表
        功能:

        
            1. 按照 userId 分组，按 timestamp 降序排序后，为每个用户的评分打上排名（rank_latest）
               - 排名 1：最新，根据该条作为测试集
               - 排名 2：次新，根据该条作为验证集
               - 其余排名 > 2：放在训练集
            2. 通过筛选 rank_latest，得到 train、val、test 三个 DataFrame，只保留 ['userId','itemId','rating'] 三列
            3. 确保所有用户在 train/val/test 中都有出现，且总条目数相等于原始 ratings
        输出:
            train: pandas.DataFrame，保留列 ['userId','itemId','rating']
            val: pandas.DataFrame，同上
            test: pandas.DataFrame，同上
        """
        # 为每个用户的 rating 打上按时间倒序的排名
        ratings['rank_latest'] = ratings.groupby(['userId'])['timestamp'].rank(method='first', ascending=False)
        # 最新一条作为测试集
        test = ratings[ratings['rank_latest'] == 1]
        # 第二新的一条作为验证集
        val = ratings[ratings['rank_latest'] == 2]
        # 其余放在训练集
        train = ratings[ratings['rank_latest'] > 2]
        # 检查：train/val/test 中用户数量一致，且总长度与原 ratings 一致
        assert train['userId'].nunique() == test['userId'].nunique() == val['userId'].nunique()
        assert len(train) + len(test) + len(val) == len(ratings)
        # 返回三组数据，仅保留 userId, itemId, rating 三列
        return train[['userId', 'itemId', 'rating']], val[['userId', 'itemId', 'rating']], test[['userId', 'itemId', 'rating']]

    def _sample_negative(self, ratings):
        """
        为每个用户生成负样本列表

        参数:
            ratings: pandas.DataFrame，原始或预处理后的评分表
        功能:
            1. 根据 userId 分组，获取每个用户交互过的正样本集合 interacted_items
            2. 计算负样本集合 negative_items = 全部物品集合 item_pool - 交互过的正样本集合
            3. 从 negative_items 中随机抽取 198 个负样本，作为 negative_samples
        输出:
            pandas.DataFrame，包含列 ['userId','negative_items','negative_samples']：
              - userId: 用户 ID
              - negative_items: 该用户所有未交互过的物品集合（Python set）
              - negative_samples: 从 negative_items 中随机抽取的 198 个负样本（Python list）
        """
        # groupby 之后 reset_index，并重命名列
        interact_status = ratings.groupby('userId')['itemId'].apply(set).reset_index().rename(
            columns={'itemId': 'interacted_items'})
        # 计算负样本全集
        interact_status['negative_items'] = interact_status['interacted_items'].apply(lambda x: self.item_pool - x)
        # 从负样本全集中随机取 198 个负样本
        interact_status['negative_samples'] = interact_status['negative_items'].apply(lambda x: random.sample(x, 198))
        return interact_status[['userId', 'negative_items', 'negative_samples']]

    def store_all_train_data(self, num_negatives):
        """
        构建完整的训练数据（包含正负样本）

        参数:
            num_negatives: 每个正样本对应要采样的负样本数量
        功能:
            1. 将训练集 train_ratings 与每个用户的负样本信息合并
            2. 对于 train_ratings 中的每个条目：
               - 将其正样本 (userId, itemId, rating=1.0) 加入列表
               - 随机从负样本集中抽取 num_negatives 个作为对应的负样本 (userId, itemId, rating=0)
            3. 结果按用户分组，每个用户对应一个子列表，包含该用户所有正负样本的 (user, item, rating)
        输出:
            返回一个长度为 num_users 的三元组列表 [users, items, ratings]：
              - users: List[List[int]]，外层长度为用户总数，内层列表为该用户所有样本的 userId
              - items: List[List[int]]，对应 users 中同位置的 itemId
              - ratings: List[List[float]]，对应 users/items 中同位置的 rating (1.0 或 0.0)
            保证:
              len(users) == len(items) == len(ratings) == 用户总数
        """
        users, items, ratings = [], [], []
        # 将 train_ratings 与负样本表合并，得到每条训练样本对应的负样本全集
        train_ratings = pd.merge(self.train_ratings, self.negatives[['userId', 'negative_items']], on='userId')
        # 对每个用户的训练条目生成负样本列表，包含 num_negatives 个
        train_ratings['negatives'] = train_ratings['negative_items'].apply(lambda x: random.sample(x, num_negatives))
        single_user = []
        user_item = []
        user_rating = []
        # 按用户分组遍历
        grouped_train_ratings = train_ratings.groupby('userId')
        train_users = []
        for userId, user_train_ratings in grouped_train_ratings:
            train_users.append(userId)
            user_length = len(user_train_ratings)
            # 对每个正样本条目，添加正样本和对应的负样本
            for row in user_train_ratings.itertuples():
                # 正样本
                single_user.append(int(row.userId))
                user_item.append(int(row.itemId))
                user_rating.append(float(row.rating))
                # num_negatives 个负样本
                for i in range(num_negatives):
                    single_user.append(int(row.userId))
                    user_item.append(int(row.negatives[i]))
                    user_rating.append(float(0))  # 负样本的 rating 设置为 0
            # 验证子列表长度
            assert len(single_user) == len(user_item) == len(user_rating)
            assert (1 + num_negatives) * user_length == len(single_user)
            # 将该用户所有样本列表添加到总列表
            users.append(single_user)
            items.append(user_item)
            ratings.append(user_rating)
            # 清空临时列表，准备下一个用户
            single_user = []
            user_item = []
            user_rating = []
        # 确保总列表长度与用户总数一致，且 train_users 已排序
        assert len(users) == len(items) == len(ratings) == len(self.user_pool)
        assert train_users == sorted(train_users)
        return [users, items, ratings]

    @property
    def validate_data(self):
        """
        构建验证集数据

        功能:
            1. 将验证集 val_ratings 与负样本表合并，获取每个验证条目的负样本列表
            2. 对于每个验证条目 (userId, itemId)，保留该正样本对，同时将负样本列表中的前一半抽取为验证负样本
            3. 返回四个张量列表：
               - val_users: torch.LongTensor，验证集正样本的用户 ID 列表
               - val_items: torch.LongTensor，验证集正样本的物品 ID 列表
               - negative_users: torch.LongTensor，与 val_users 等长，均为同一用户 ID（对应负样本）
               - negative_items: torch.LongTensor，负样本的物品 ID 列表
        输出:
            [val_users, val_items, negative_users, negative_items]，四个 LongTensor
        """
        # 合并验证集条目与负样本表
        val_ratings = pd.merge(self.val_ratings, self.negatives[['userId', 'negative_samples']], on='userId')
        val_users, val_items, negative_users, negative_items = [], [], [], []
        # 遍历每条验证集条目
        for row in val_ratings.itertuples():
            # 添加正样本对
            val_users.append(int(row.userId))
            val_items.append(int(row.itemId))
            # 负样本取负样本列表的一半（假设负样本列表长度为偶数）
            for i in range(int(len(row.negative_samples) / 2)):
                negative_users.append(int(row.userId))
                negative_items.append(int(row.negative_samples[i]))
        # 验证数据长度关系
        assert len(val_users) == len(val_items)
        assert len(negative_users) == len(negative_items)
        # 通常每个验证样本对应 99 个负样本（如果负样本总数为 198）
        assert 99 * len(val_users) == len(negative_users)
        assert val_users == sorted(val_users)
        # 将列表转为 LongTensor 并返回
        return [
            torch.LongTensor(val_users),
            torch.LongTensor(val_items),
            torch.LongTensor(negative_users),
            torch.LongTensor(negative_items)
        ]

    @property
    def test_data(self):
        """
        构建测试集数据

        功能:
            1. 将测试集 test_ratings 与负样本表合并，获取每个测试条目的负样本列表
            2. 对于每个测试条目 (userId, itemId)，保留该正样本对，同时将负样本列表的后一半抽取为测试负样本
            3. 返回四个张量列表：
               - test_users: torch.LongTensor，测试集正样本的用户 ID 列表
               - test_items: torch.LongTensor，测试集正样本的物品 ID 列表
               - negative_users: torch.LongTensor，与 test_users 等长，均为同一用户 ID（对应负样本）
               - negative_items: torch.LongTensor，负样本的物品 ID 列表
        输出:
            [test_users, test_items, negative_users, negative_items]，四个 LongTensor
        """
        test_ratings = pd.merge(self.test_ratings, self.negatives[['userId', 'negative_samples']], on='userId')
        test_users, test_items, negative_users, negative_items = [], [], [], []
        # 遍历每条测试集条目
        for row in test_ratings.itertuples():
            # 添加正样本对
            test_users.append(int(row.userId))
            test_items.append(int(row.itemId))
            # 负样本取负样本列表的后一半（例如负样本列表长度为 198，则后一半为最后 99 个）
            for i in range(int(len(row.negative_samples) / 2), len(row.negative_samples)):
                negative_users.append(int(row.userId))
                negative_items.append(int(row.negative_samples[i]))
        # 验证数据长度关系
        assert len(test_users) == len(test_items)
        assert len(negative_users) == len(negative_items)
        # 通常每个测试样本对应 99 个负样本
        assert 99 * len(test_users) == len(negative_users)
        assert test_users == sorted(test_users)
        # 将列表转为 LongTensor 并返回
        return [
            torch.LongTensor(test_users),
            torch.LongTensor(test_items),
            torch.LongTensor(negative_users),
            torch.LongTensor(negative_items)
        ]
