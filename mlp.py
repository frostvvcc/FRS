import torch
from engine import Engine
from utils import use_cuda, resume_checkpoint


class MLP(torch.nn.Module):
    """多层感知机（MLP）模型，用于预测给定用户-物品对的评分概率（隐式反馈）"""

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

        # MLP 隐藏层
        self.fc_layers = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(config['layers'][:-1], config['layers'][1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))

        # 输出层，将 MLP 最后的向量映射到标量预测值
        self.affine_output = torch.nn.Linear(in_features=config['layers'][-1], out_features=1)
        # Sigmoid 激活，用于将输出映射到 [0,1]
        self.logistic = torch.nn.Sigmoid()

    def forward(self, item_indices):
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

        # 2. 获取 user_embedding 和 item_embedding
        user_embedding = self.embedding_user(user_indices)    # shape=[batch_size, latent_dim]
        item_embedding = self.embedding_item(item_indices)    # shape=[batch_size, latent_dim]

        # 3. 拼接 user_embedding 和 item_embedding
        vector = torch.cat([user_embedding, item_embedding], dim=-1)  # shape=[batch_size, 2*latent_dim]

        # 4. 依次通过每一层 MLP 隐藏层（ReLU 激活）
        for idx, _ in enumerate(self.fc_layers):
            vector = self.fc_layers[idx](vector)
            vector = torch.nn.ReLU()(vector)

        # 5. 输出层映射
        logits = self.affine_output(vector)   # shape=[batch_size, 1]
        # 6. Sigmoid 激活，映射到 [0,1]
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
            use_cuda(True, config['device_id'])
            self.model.cuda()
        # 3. 调用父类构造函数，初始化 Engine 相关属性
        super(MLPEngine, self).__init__(config)
        # 4. 打印模型结构
        print(self.model)
