import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from utils import *
from metrics import MetronAtK
import random
import copy
from data import UserItemRatingDataset
from torch.utils.data import DataLoader
from torch.distributions.laplace import Laplace


class Engine(object):
    """Meta Engine，用于训练和评估 NCF（Neural Collaborative Filtering）模型

    注意：子类需实现并赋值 self.model
    """

    def __init__(self, config):
        """
        初始化 Engine

        参数:
            config: dict，包含模型和训练配置，例如：
                - 'batch_size': 每个用户本地训练时的批次大小
                - 'use_cuda': 是否使用 CUDA（True/False）
                - 'lr': 基础学习率
                - 'clients_sample_ratio': 每轮采样用户的比例
                - 'local_epoch': 每个客户端本地训练的轮数
                - 'reg': 正则化系数
                - 'dp': 差分隐私噪声大小
                - 'num_users', 'num_items', 'latent_dim' 等其他超参数
                - 'mp_layers': 图消息传递（Message Passing）层数
                - 'neighborhood_size', 'neighborhood_threshold', 'similarity_metric'：用于构建用户关系图
                - 'model_dir': 模型保存路径模板
                - 'device_id': CUDA 设备 ID（如果使用 CUDA）
        功能:
            1. 保存配置 config 为实例属性
            2. 初始化 MetronAtK 评估对象，用于计算 Hit Ratio 和 NDCG
            3. 初始化存储 server 端和 client 端模型参数的字典
            4. 定义损失函数 crit（当前使用二分类 BCE Loss）
            5. 定义 top_k（用于评估时返回前 k 个候选）
        输出:
            无
        """
        self.config = config
        self._metron = MetronAtK(top_k=10)
        # 如果需要使用 TensorBoard，可取消下面两行注释
        # self._writer = SummaryWriter(log_dir='runs/{}'.format(config['alias']))
        # self._writer.add_text('config', str(config), 0)
        self.server_model_param = {}  # 存储服务端聚合后的模型参数（如 item embedding）
        self.client_model_params = {}  # 存储每个客户端本地更新后的模型参数（CPU 上的副本）
        # 显式反馈时可使用 MSELoss，该处使用隐式反馈的 BCE Loss
        # self.crit = torch.nn.MSELoss()
        self.crit = torch.nn.BCELoss()
        self.top_k = 10

    def instance_user_train_loader(self, user_train_data):
        """
        为单个用户实例化 PyTorch DataLoader

        参数:
            user_train_data: List[List], 三个列表分别为该用户的 [users, items, ratings]
                            - users: list of int，长度 = 该用户训练样本总数（正负样本）
                            - items: list of int，对应物品索引列表
                            - ratings: list of float，对应每个 (user, item) 的评分/标签（1 或 0）
        功能:
            1. 将三个 Python 列表转换为 LongTensor/FloatTensor
            2. 构造 UserItemRatingDataset 对象
            3. 包装为 DataLoader，shuffle=True，batch_size = config['batch_size']
        输出:
            DataLoader，供本地训练循环使用
        """
        dataset = UserItemRatingDataset(
            user_tensor=torch.LongTensor(user_train_data[0]),
            item_tensor=torch.LongTensor(user_train_data[1]),
            target_tensor=torch.FloatTensor(user_train_data[2])
        )
        return DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True)

    def fed_train_single_batch(self, model_client, batch_data, optimizers, user):
        """
        客户端本地训练单个批次

        参数:
            model_client: torch.nn.Module，客户端本地模型
            batch_data: Tuple[Tensor, Tensor, Tensor]，分别为 users, items, ratings 批次张量
            optimizers: List[torch.optim.Optimizer]，长度 3：
                - optimizer: 更新 MLP 中的全连接层权重和输出层
                - optimizer_u: 更新用户 embedding
                - optimizer_i: 更新物品 embedding
            user: int，该客户端对应的用户 ID，用于获取对应的 item embedding 正则化项
        功能:
            1. 从 batch_data 解包 users, items, ratings，并将 ratings 转为 float
            2. 深拷贝服务端存储的 item embedding 对应用户的权重，作为正则化项
            3. 如果使用 CUDA，则将数据和正则化 embedding 移动到 GPU
            4. 清零 optimizer, optimizer_u, optimizer_i 的梯度
            5. 前向计算 model_client(items)，得到预测评分 ratings_pred
            6. 计算 BCE 损失 loss = crit(ratings_pred, ratings)
            7. 计算正则化项 regularization_term = compute_regularization(model_client, reg_item_embedding)
            8. 总 loss 加上 reg * regularization_term
            9. 反向传播并分别调用 optimizer.step(), optimizer_u.step(), optimizer_i.step()
        输出:
            返回 tuple:
             - model_client: 更新后带梯度更新的本地模型
             - loss.item(): 本批次的标量损失值（float）
        """
        users, items, ratings = batch_data[0], batch_data[1], batch_data[2]
        ratings = ratings.float()
        # 从服务端参数中获取该用户对应的 item embedding，用于正则化
        reg_item_embedding = copy.deepcopy(self.server_model_param['embedding_item.weight'][user].data)

        optimizer, optimizer_u, optimizer_i = optimizers

        if self.config['use_cuda'] is True:
            users = users.cuda()
            items = items.cuda()
            ratings = ratings.cuda()
            reg_item_embedding = reg_item_embedding.cuda()

        # 清零梯度
        optimizer.zero_grad()
        optimizer_u.zero_grad()
        optimizer_i.zero_grad()

        # 前向计算：仅传入物品索引，让模型内部补上用户 embedding
        ratings_pred = model_client(items)
        # 计算 BCE Loss
        loss = self.crit(ratings_pred.view(-1), ratings)
        # 计算正则化项（通常是 L2 范数）
        regularization_term = compute_regularization(model_client, reg_item_embedding)
        # 加入正则化
        loss += self.config['reg'] * regularization_term

        # 反向传播
        loss.backward()
        # 三个优化器依次更新参数
        optimizer.step()
        optimizer_u.step()
        optimizer_i.step()

        return model_client, loss.item()

    def aggregate_clients_params(self, round_user_params):
        """
        服务端聚合当前轮参与用户上传的 embedding_item 参数

        参数:
            round_user_params: dict, key 为 userID，value 为 dict {'embedding_item.weight': Tensor}
                                包含所有参与用户上传的噪声添加后的 item embedding
        功能:
            1. 构建用户关系图：调用 construct_user_relation_graph_via_item(round_user_params, num_items, latent_dim, similarity_metric)
            2. 基于用户关系图，选取每个用户的 Top-K 邻居：select_topk_neighboehood(...)
            3. 使用图消息传递（Message Passing）更新 item embedding：MP_on_graph(...)
            4. 将更新后的 item embedding 存入 self.server_model_param['embedding_item.weight']
        输出:
            无，直接更新 self.server_model_param['embedding_item.weight']
        """
        # 1. 构建用户关系图
        user_relation_graph = construct_user_relation_graph_via_item(
            round_user_params,
            self.config['num_items'],
            self.config['latent_dim'],
            self.config['similarity_metric']
        )
        # 2. 选取 Top-K 邻居
        topk_user_relation_graph = select_topk_neighboehood(
            user_relation_graph,
            self.config['neighborhood_size'],
            self.config['neighborhood_threshold']
        )
        # 3. 图消息传递，更新全局 item embedding
        updated_item_embedding = MP_on_graph(
            round_user_params,
            self.config['num_items'],
            self.config['latent_dim'],
            topk_user_relation_graph,
            self.config['mp_layers']
        )
        # 4. 存储更新后的 item embedding
        self.server_model_param['embedding_item.weight'] = copy.deepcopy(updated_item_embedding)

    def fed_train_a_round(self, all_train_data, round_id):
        """
        执行联邦学习中的一轮训练

        参数:
            all_train_data: List[List], 长度为 3 的列表，元素为 [users_per_user, items_per_user, ratings_per_user]：
                - users_per_user[u]: 该用户 u 本地所有正负样本中的 userID 列表（重复 u 若干次）
                - items_per_user[u]: 该用户 u 本地所有正负样本中的 itemID 列表
                - ratings_per_user[u]: 该用户 u 本地所有正负样本中的标签列表（1/0）
            round_id: int，当前轮次 ID（从 0 开始）
        功能:
            1. 随机采样本轮参与训练的用户列表 participants，数量 = num_users * clients_sample_ratio
            2. 若 round_id == 0，初始化 self.server_model_param['embedding_item.weight']：
               - 对于每个参与用户，复制服务端模型初始的 embedding_item.weight 参数（CPU 上）
               - 额外存储一个 'global' 键，用于全局 item embedding 初始化
            3. 遍历每个参与用户 user：
               a. 深拷贝 self.model 得到 model_client
               b. 如果 round_id != 0，则将本地模型参数加载为上一次保留的 client_model_params[user]（如果存在），
                  并将 embedding_item.weight 替换为全局 server_model_param['embedding_item.weight']['global']
               c. 定义三个优化器：
                  - optimizer: 更新 MLP 里的全连接层和输出层
                  - optimizer_u: 更新用户 embedding；学习率缩放为 lr/clients_sample_ratio * lr_eta - lr
                  - optimizer_i: 更新物品 embedding；学习率缩放为 lr * num_items * lr_eta - lr
               d. 利用 instance_user_train_loader 构造 DataLoader，对 model_client 进行 local_epoch 轮本地训练，
                  每轮遍历所有批次，调用 fed_train_single_batch 更新 model_client
               e. 训练结束后，保存该用户本地模型参数到 self.client_model_params[user]（全部复制到 CPU）
               f. 将该用户上传的 item embedding（client_model_params[user]['embedding_item.weight']）拷贝到 round_participant_params[user]，
                  并对其添加拉普拉斯噪声（差分隐私），噪声尺度 = dp_value
            4. 调用 aggregate_clients_params(round_participant_params) 聚合更新 item embedding
        输出:
            返回 participants 列表，表示本轮参与的用户 ID
        """
        # 1. 随机采样参与用户
        num_participants = int(self.config['num_users'] * self.config['clients_sample_ratio'])
        participants = random.sample(range(self.config['num_users']), num_participants)
        # 存储本轮用户上传的 item embedding 参数
        round_participant_params = {}

        # 2. 若第一轮，初始化服务端 item embedding
        if round_id == 0:
            self.server_model_param['embedding_item.weight'] = {}
            # 对每个参与用户，拷贝一份模型初始化的 item embedding（CPU 张量）
            for user in participants:
                self.server_model_param['embedding_item.weight'][user] = copy.deepcopy(
                    self.model.state_dict()['embedding_item.weight'].data.cpu()
                )
            # "global" 存储一份全局初始化值
            self.server_model_param['embedding_item.weight']['global'] = copy.deepcopy(
                self.model.state_dict()['embedding_item.weight'].data.cpu()
            )

        # 3. 遍历每个参与用户，进行本地更新
        for user in participants:
            # a. 深拷贝模型架构
            model_client = copy.deepcopy(self.model)

            # b. 加载本地更新或全局 item embedding
            if round_id != 0:
                # 先复制一份原始 state_dict
                user_param_dict = copy.deepcopy(self.model.state_dict())
                # 若该用户存在本地更新，则覆盖对应参数
                if user in self.client_model_params.keys():
                    for key in self.client_model_params[user].keys():
                        user_param_dict[key] = copy.deepcopy(self.client_model_params[user][key].data).cuda()
                # 替换 embedding_item.weight 为全局聚合后的 embedding
                user_param_dict['embedding_item.weight'] = copy.deepcopy(
                    self.server_model_param['embedding_item.weight']['global'].data
                ).cuda()
                model_client.load_state_dict(user_param_dict)

            # c. 定义三个优化器
            optimizer = torch.optim.SGD(
                [
                    {"params": model_client.fc_layers.parameters()},
                    {"params": model_client.affine_output.parameters()}
                ],
                lr=self.config['lr']
            )  # MLP 优化器

            optimizer_u = torch.optim.SGD(
                model_client.embedding_user.parameters(),
                lr=self.config['lr'] / self.config['clients_sample_ratio'] * self.config['lr_eta'] - self.config['lr']
            )  # 用户 embedding 优化器

            optimizer_i = torch.optim.SGD(
                model_client.embedding_item.parameters(),
                lr=self.config['lr'] * self.config['num_items'] * self.config['lr_eta'] - self.config['lr']
            )  # 物品 embedding 优化器

            optimizers = [optimizer, optimizer_u, optimizer_i]

            # d. 加载该用户训练数据并实例化 DataLoader
            user_train_data = [
                all_train_data[0][user],  # users 列表
                all_train_data[1][user],  # items 列表
                all_train_data[2][user]  # ratings 列表
            ]
            user_dataloader = self.instance_user_train_loader(user_train_data)

            model_client.train()
            # 本地训练 local_epoch 轮
            for epoch in range(self.config['local_epoch']):
                for batch_id, batch in enumerate(user_dataloader):
                    assert isinstance(batch[0], torch.LongTensor)
                    model_client, loss = self.fed_train_single_batch(model_client, batch, optimizers, user)

            # e. 本地训练结束后，保存该用户更新后的全部模型参数（移到 CPU）
            client_param = model_client.state_dict()
            self.client_model_params[user] = copy.deepcopy(client_param)
            for key in self.client_model_params[user].keys():
                self.client_model_params[user][key] = self.client_model_params[user][key].data.cpu()

            # f. 将该用户上传的 item embedding 加入 round_participant_params，并添加差分隐私噪声
            round_participant_params[user] = {}
            round_participant_params[user]['embedding_item.weight'] = copy.deepcopy(
                self.client_model_params[user]['embedding_item.weight']
            )

            # 确保 dp_value > 0
            dp_value = max(self.config['dp'], 1e-6)
            # 添加拉普拉斯噪声
            noise = Laplace(1e-6, dp_value).expand(
                round_participant_params[user]['embedding_item.weight'].shape
            ).sample()
            round_participant_params[user]['embedding_item.weight'] += noise

        # 4. 服务端聚合本轮所有用户上传的 item embedding
        self.aggregate_clients_params(round_participant_params)

        return participants

    def fed_evaluate(self, evaluate_data):
        """
        在联邦设置下，评估所有客户端模型在测试集上的性能

        参数:
            evaluate_data: List[Tensor], 包含：
                - test_users: LongTensor, 每个用户正样本测试项的用户 ID
                - test_items: LongTensor, 每个用户正样本测试项的 item ID
                - negative_users: LongTensor, 每个负样本对应的用户 ID（重复）
                - negative_items: LongTensor, 每个负样本对应的 item ID（重复）
                注：每个用户对应 1 个正样本 + 99 个负样本
        功能:
            1. 如果使用 CUDA，则将 evaluate_data 和临时标签 ratings 移动到 GPU
            2. 遍历每个用户 user：
               a. 深拷贝 self.model 得到 user_model
               b. 准备 user_model 的 state_dict：若该用户有本地更新，则覆盖参数
               c. 加载用户参数并设置为 eval 模式
               d. 使用 test_item 和 negative_items 分别进行前向预测，得到 test_score（1 个）和 negative_score（99 个）
               e. 将预测结果拼接，并计算 BCE Loss
               f. 保存该用户的 loss 到 all_loss[user]
               g. 同时将 test_score 和 negative_score 拼接到 test_scores、negative_scores
            3. 将所有 test_users、test_items、test_scores、negative_users、negative_items、negative_scores 传入 MetronAtK
               并计算 Hit Ratio 与 NDCG
        输出:
            tuple (hit_ratio, ndcg, all_loss):
                - hit_ratio: float，所有用户的平均 Hit Ratio
                - ndcg: float，所有用户的平均 NDCG
                - all_loss: dict, key=userID, value=user 在测试集上的 BCE Loss
        """
        test_users, test_items = evaluate_data[0], evaluate_data[1]
        negative_users, negative_items = evaluate_data[2], evaluate_data[3]

        # 构造一个长度 100 的标签 tensor: [1, 0, 0, ..., 0]，用于计算本地 loss
        temp = [0] * 100
        temp[0] = 1
        ratings = torch.FloatTensor(temp)

        if self.config['use_cuda'] is True:
            test_users = test_users.cuda()
            test_items = test_items.cuda()
            negative_users = negative_users.cuda()
            negative_items = negative_items.cuda()
            ratings = ratings.cuda()

        test_scores = None
        negative_scores = None
        all_loss = {}

        # 遍历所有用户（按 user ID 顺序）
        for user in range(self.config['num_users']):
            # a. 深拷贝模型
            user_model = copy.deepcopy(self.model)
            user_param_dict = copy.deepcopy(self.model.state_dict())
            # 如果该用户有本地更新，则覆盖对应参数
            if user in self.client_model_params.keys():
                for key in self.client_model_params[user].keys():
                    user_param_dict[key] = copy.deepcopy(self.client_model_params[user][key].data).cuda()
            # user_param_dict['embedding_item.weight'] = copy.deepcopy(
            #     self.server_model_param['embedding_item.weight']['global'].data).cuda()
            user_model.load_state_dict(user_param_dict)
            user_model.eval()

            with torch.no_grad():
                # b. 准备该用户的正样本（1 条）和负样本（99 条）
                test_user = test_users[user: user + 1]  # LongTensor, shape=[1]
                test_item = test_items[user: user + 1]  # LongTensor, shape=[1]
                negative_user = negative_users[user * 99: (user + 1) * 99]  # LongTensor, shape=[99]
                negative_item = negative_items[user * 99: (user + 1) * 99]  # LongTensor, shape=[99]

                # c. 前向预测
                test_score = user_model(test_item)  # Tensor, shape=[1, 1]
                negative_score = user_model(negative_item)  # Tensor, shape=[99, 1]

                # 汇总所有用户得分
                if user == 0:
                    test_scores = test_score
                    negative_scores = negative_score
                else:
                    test_scores = torch.cat((test_scores, test_score))
                    negative_scores = torch.cat((negative_scores, negative_score))

                # d. 拼接正负样本分数，计算该用户的 BCE Loss
                ratings_pred = torch.cat((test_score, negative_score))
                loss = self.crit(ratings_pred.view(-1), ratings)
                all_loss[user] = loss.item()

        # 如果使用 CUDA，则将张量移回 CPU
        if self.config['use_cuda'] is True:
            test_users = test_users.cpu()
            test_items = test_items.cpu()
            test_scores = test_scores.cpu()
            negative_users = negative_users.cpu()
            negative_items = negative_items.cpu()
            negative_scores = negative_scores.cpu()

        # e. 将测试结果送入 MetronAtK 计算评估指标
        self._metron.subjects = [
            test_users.data.view(-1).tolist(),
            test_items.data.view(-1).tolist(),
            test_scores.data.view(-1).tolist(),
            negative_users.data.view(-1).tolist(),
            negative_items.data.view(-1).tolist(),
            negative_scores.data.view(-1).tolist()
        ]
        hit_ratio = self._metron.cal_hit_ratio()
        ndcg = self._metron.cal_ndcg()

        return hit_ratio, ndcg, all_loss

    def save(self, alias, epoch_id, hit_ratio, ndcg):
        """
        保存当前模型检查点

        参数:
            alias: str，用于区分实验或模型名称
            epoch_id: int，本次保存对应的训练轮次
            hit_ratio: float，本轮评估的 Hit Ratio
            ndcg: float，本轮评估的 NDCG
        功能:
            1. 确保 self.model 已定义
            2. 根据 config['model_dir'] 模板格式化路径（包括 alias、epoch_id、hit_ratio、ndcg 等信息）
            3. 调用 save_checkpoint 工具函数保存模型
        输出:
            无
        """
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        model_dir = self.config['model_dir'].format(alias, epoch_id, hit_ratio, ndcg)
        save_checkpoint(self.model, model_dir)
