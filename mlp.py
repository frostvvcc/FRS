import torch
from engine import Engine
from utils import use_cuda, resume_checkpoint
import torch.nn.functional as F

class LightweightAttention(torch.nn.Module):
    def __init__(self, latent_dim):
        super(LightweightAttention, self).__init__()
        self.scale = latent_dim ** 0.5

    def forward(self, query, keys):
        scores = torch.bmm(query, keys.transpose(1, 2)) / self.scale
        weights = F.softmax(scores, dim=-1)
        attended_value = torch.bmm(weights, keys)
        return attended_value.squeeze(1)

class MLP(torch.nn.Module):
    """多层感知机（MLP）模型，用于预测给定用户-物品对的评分概率（隐式反馈）
        当这个用户，遇到这部具体的电影时，他有多大的概率会喜欢它？
        不管系统喂给它什么电影，它最后都会吐出一个 0 到 1 之间的数字（比如 0.89 代表极其推荐，0.12 代表千万别推）。
        这就是前面提到的 Sigmoid 激活函数的功劳。
第一步（拿画像）： 它先去字典里查出“用户的性格向量”（User Embedding）和“电影的属性向量”（Item Embedding）。

第二步（硬凑对）： 它把这两个向量简单粗暴地拼在一起（拼接操作 torch.cat）。缺乏序列概念，用户是静态的画像。

第三步（看反应）： 拼在一起后，它把这串数据扔进多层感知机（那几层 fc_layers 神经网络）里反复揉搓、过滤。这就像是在测试：这个人的性格和这部电影的属性之间，能不能产生强烈的“化学反应”？如果有，分数就高。
    """

    def __init__(self, config):
        """
        初始化 MLP

        参数:
            config: dict，包含模型结构和超参数，例如：
                - 'num_users': 总用户数
                - 'num_items': 总物品数
                - 'latent_dim': embedding 维度
                - 'layers': list, MLP 隐藏层维度列表，如 [64, 32, 16, 8]
                - 'use_cuda': 是否使用 CUDA（True/False）
                - 'device_id': CUDA 设备 ID
        功能:
            1. 保存 config
            2. 创建用户 embedding 层：Embedding(1, latent_dim)，在 MLP 设计中，每次只使用索引 0 代表“当前用户”
            3. 创建物品 embedding 层：Embedding(num_items, latent_dim)
            4. 构建 MLP 隐藏层：ModuleList，包括 len(layers)-1 个 Linear(in_size, out_size)
            5. 构建输出层 affine_output: Linear(last_layer_dim, 1)
            6. 定义 sigmoid 激活 logistic，用于将评分映射到 [0,1]
        输出:
            无
        """
        super(MLP, self).__init__()
        self.config = config
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim']

        # 用户 embedding：由于联邦设置下，每个 client 只对应一个“当前用户”，因此 embedding 大小为 (1, latent_dim)
        self.embedding_user = torch.nn.Embedding(num_embeddings=1, embedding_dim=self.latent_dim)
        # 物品 embedding：大小为 (num_items, latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        # 🌟 必须加这一句！把注意力雷达装上去！
        self.attention_layer = LightweightAttention(self.latent_dim)
        # MLP 隐藏层
        self.fc_layers = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(config['layers'][:-1], config['layers'][1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))

        # 输出层，将 MLP 最后的向量映射到标量预测值
        self.affine_output = torch.nn.Linear(in_features=config['layers'][-1], out_features=1)
        # Sigmoid 激活，用于将输出映射到 [0,1]
        self.logistic = torch.nn.Sigmoid()

    def forward(self, item_indices, history_indices):
        """
        前向推理

        参数:
            item_indices: LongTensor, shape=[batch_size]，当前批次中所有用户-物品对中的物品索引列表
                          注意：用户索引在联邦设置下默认使用 0（占位符），通过外部训练循环决定不同客户端
        功能:
            1. 构造 user_embedding：创建一个长度等于 batch_size 的 LongTensor，所有元素均为 0，代表当前“这批次都是同一个用户”
            2. 通过 embedding_user(0) 获取 user_embedding，shape=[batch_size, latent_dim]
            3. 通过 embedding_item(item_indices) 获取 item_embedding，shape=[batch_size, latent_dim]
            4. 将 user_embedding 和 item_embedding 拼接，得到 shape=[batch_size, 2*latent_dim]
            5. 依次通过 MLP 隐藏层（ReLU 激活）进行计算
            6. 通过输出层 affine_output，将向量映射到标量 logits
            7. 最后通过 sigmoid，将 logits 映射到 [0,1] 范围的评分概率
        输出:
            rating: Tensor, shape=[batch_size, 1]，预测评分概率
        """
        # 1. 构造 batch_size 长度的全 0 用户索引张量，将其移动到 GPU（如果使用 CUDA）
        device = 'cuda' if self.config['use_cuda'] else 'cpu'
        batch_size = len(item_indices)
        user_indices = torch.LongTensor([0 for _ in range(batch_size)]).to(device)

        # 2. 获取 user_embedding （长期兴趣）和 item_embedding（候选物品）
        user_embedding = self.embedding_user(user_indices)    # shape=[batch_size, latent_dim]
        item_embedding = self.embedding_item(item_indices)    # shape=[batch_size, latent_dim]
        # a. 从字典获取历史序列的特征 (Keys/Values)
        history_embedding = self.embedding_item(history_indices)  # shape=[batch_size, seq_len, latent_dim]

        # b. 给候选物品加一个维度，变成 Query，以满足注意力矩阵乘法要求
        query = item_embedding.unsqueeze(1)  # shape=[batch_size, 1, latent_dim]

        # c. 送入注意力层，算出带有权重的短期兴趣！
        # 注意：前提是你在 __init__ 里已经实例化了 self.attention_layer
        short_term_interest = self.attention_layer(query, history_embedding)  # shape=[batch_size, latent_dim]

        # 3. 拼接 user_embedding 和 item_embedding
        # 现在我们要拼 3 个：长期兴趣 + 短期兴趣 + 候选物品
        if self.config.get('use_attention', True):
            vector = torch.cat([user_embedding, short_term_interest, item_embedding], dim=-1)
        else:
            vector = torch.cat([user_embedding, item_embedding], dim=-1)  # shape=[batch_size, 2*latent_dim]

        # 4. 依次通过每一层 MLP 隐藏层（ReLU 激活）
        for idx, _ in enumerate(self.fc_layers):
            vector = self.fc_layers[idx](vector)
            vector = torch.nn.ReLU()(vector)

        # 5. 输出层映射
        logits = self.affine_output(vector)   # shape=[batch_size, 1]
        # 6. Sigmoid 激活，映射到 [0,1]
        # 模型把（用户特征 + 物品特征）算完之后，通过Sigmoid函数，吐出了一个
        # 0到1之间的小数（比如0.85、0.12）。
        rating = self.logistic(logits)       # shape=[batch_size, 1]

        return rating

    def init_weight(self):
        """
        可选：初始化模型权重（此处留空，若需自定义权重初始化可在此实现）
        """
        pass


class MLPEngine(Engine):
    """继承 Engine 的 MLP 专用训练和评估引擎"""

    def __init__(self, config):
        """
        初始化 MLPEngine

        参数:
            config: dict，包含模型和训练配置（同 MLP 与 Engine）
        功能:
            1. 实例化 MLP 模型，并根据 config['use_cuda']、config['device_id'] 决定是否将模型移动到 GPU
            2. 调用父类 Engine 的构造函数，完成损失函数等初始化
            3. 打印模型结构，供检查
        输出:
            无
        """
        # 1. 构造 MLP 模型
        self.model = MLP(config)
        # 2. 若使用 CUDA，则调用 utils.use_cuda 将模型移动到 GPU
        if config['use_cuda'] is True:
            self.model.cuda()
        # 3. 调用父类构造函数，初始化 Engine 相关属性
        super(MLPEngine, self).__init__(config)
        # 4. 打印模型结构
        print(self.model)
