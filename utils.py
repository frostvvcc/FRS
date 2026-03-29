# utils.py

import torch
import numpy as np
import copy
from sklearn.metrics import pairwise_distances
import logging
import math

# -----------------------------
# 保存与恢复模型检查点
# -----------------------------
def save_checkpoint(model, model_dir):
    """
    保存模型状态字典到指定路径

    参数:
        model: torch.nn.Module，需要保存的模型
        model_dir: str，保存文件路径（包含文件名）
    功能:
        使用 torch.save 将 model.state_dict() 存盘
    输出:
        无
    """
    torch.save(model.state_dict(), model_dir)

def resume_checkpoint(model, model_dir, device_id):
    """
    从已有检查点加载模型参数

    参数:
        model: torch.nn.Module，待加载参数的模型
        model_dir: str，检查点路径
        device_id: int，目标 GPU 设备 ID，用于 map_location
    功能:
        1. 使用 torch.load 加载参数到 GPU（map_location 设置）
        2. 将加载的 state_dict 赋值给 model
    输出:
        无
    """
    state_dict = torch.load(model_dir,
                            map_location=lambda storage, loc: storage.cuda(device=device_id))
    model.load_state_dict(state_dict)

# -----------------------------
# CUDA 与优化器辅助函数
# -----------------------------
def use_cuda(enabled, device_id=0):
    """
    配置是否使用 CUDA

    参数:
        enabled: bool，是否启用 CUDA
        device_id: int，CUDA 设备编号
    功能:
        1. 如果 enabled 为 True，检查 torch.cuda.is_available()
        2. 设置当前 CUDA 设备为 device_id
    输出:
        无
    """
    if enabled:
        assert torch.cuda.is_available(), 'CUDA is not available'
        torch.cuda.set_device(device_id)

def use_optimizer(network, params):
    """
    根据字符串参数构造对应的优化器

    参数:
        network: torch.nn.Module，需要优化的网络
        params: dict，包含优化器相关超参数，例如：
            - 'optimizer': 'sgd' / 'adam' / 'rmsprop'
            - 'sgd_lr', 'sgd_momentum', 'l2_regularization'
            - 'lr'（用于 Adam）
            - 'rmsprop_lr', 'rmsprop_alpha', 'rmsprop_momentum'
    功能:
        根据 params['optimizer'] 返回对应的 torch.optim 优化器对象
    输出:
        optimizer: torch.optim.Optimizer
    """
    if params['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(
            network.parameters(),
            lr=params['sgd_lr'],
            momentum=params['sgd_momentum'],
            weight_decay=params['l2_regularization']
        )
    elif params['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(
            network.parameters(),
            lr=params['lr'],
            weight_decay=params['l2_regularization']
        )
    elif params['optimizer'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(
            network.parameters(),
            lr=params['rmsprop_lr'],
            alpha=params['rmsprop_alpha'],
            momentum=params['rmsprop_momentum']
        )
    else:
        raise ValueError(f"Unknown optimizer type: {params['optimizer']}")
    return optimizer

# -----------------------------
# 构建用户关系图与图消息传递
# -----------------------------
def construct_user_relation_graph_via_item(round_user_params, item_num, latent_dim, similarity_metric):
    """
    构建用户关系图（基于 item embedding 的用户相似度矩阵）

    参数:
        round_user_params: dict, key 为 userID，value 为 dict 包含 'embedding_item.weight' Tensor
        item_num: int，总物品数
        latent_dim: int，embedding 维度
        similarity_metric: str，相似度度量方式（如 'cosine'）
    功能:
        1. 将每个用户的 item embedding 拉平为 shape = (item_num * latent_dim) 的向量
        2. 计算用户之间的距离矩阵（pairwise_distances）
        3. 若相似度度量为 'cosine'，返回距离矩阵；否则返回 -距离矩阵
    输出:
        adj: np.ndarray, shape = (num_users, num_users)，用户之间的相似度（或负距离）
    """
    num_users = len(round_user_params)
    # 初始化存储所有用户拉平后的 item embedding
    item_embedding = np.zeros((num_users, item_num * latent_dim), dtype='float32')
    for user in round_user_params.keys():
        # 将用户的 item embedding 拉平为一维向量
        item_embedding[user] = round_user_params[user]['embedding_item.weight'].numpy().flatten()
    # 使用 sklearn 的 pairwise_distances 计算用户间距离或相似度
    adj = pairwise_distances(item_embedding, metric=similarity_metric)
    if similarity_metric == 'cosine':
        return adj
    else:
        # 对于其他度量方式，返回负值以便后续取高相似度
        return -adj

def select_topk_neighboehood(user_relation_graph, neighborhood_size, neighborhood_threshold):
    """
    从用户关系图中筛选 Top-K 邻居或基于阈值选择邻居

    参数:
        user_relation_graph: np.ndarray, shape=(num_users, num_users)，用户相似度矩阵
        neighborhood_size: int，若 >0，则对每个用户选取值最大的 top-k 邻居
        neighborhood_threshold: float，当 neighborhood_size=0 时使用阈值方式：
            阈值 = user_relation_graph 平均值 * neighborhood_threshold
    功能:
        若 neighborhood_size > 0:
            对每个用户，找到相似度最高的 neighborhood_size 个用户，将对应位置置为 1/k，其余置 0
        否则:
            计算相似度阈值，对于每个用户，
                若某相似度 > 阈值，则将对应位置置为 1/高相似度个数
                若所有相似度均 <= 阈值，则将自身位置置为 1（孤立结点）
    输出:
        topk_user_relation_graph: np.ndarray, shape=(num_users, num_users)，稀疏邻接矩阵
    """
    num_users = user_relation_graph.shape[0]
    topk_user_relation_graph = np.zeros(user_relation_graph.shape, dtype='float32')

    if neighborhood_size > 0:
        # 对每个用户选取 Top-K 邻居
        for user in range(num_users):
            user_neighborhood = user_relation_graph[user]
            # argsort 取最大值对应的索引，[::-1] 反向获得降序
            topk_indexes = user_neighborhood.argsort()[-neighborhood_size:][::-1]
            for idx in topk_indexes:
                topk_user_relation_graph[user][idx] = 1.0 / neighborhood_size
    else:
        # 基于阈值选邻居
        similarity_threshold = np.mean(user_relation_graph) * neighborhood_threshold
        for i in range(num_users):
            high_idxs = np.where(user_relation_graph[i] > similarity_threshold)[0]
            if len(high_idxs) > 0:
                # 将所有相似度高于阈值的位置置为 1/高于阈值的用户数
                for j in high_idxs:
                    topk_user_relation_graph[i][j] = 1.0 / len(high_idxs)
            else:
                # 若没有高于阈值的邻居，则将自身置为 1
                topk_user_relation_graph[i][i] = 1.0

    return topk_user_relation_graph

def MP_on_graph(round_user_params, item_num, latent_dim, topk_user_relation_graph, layers):
    """
    在用户关系图上进行图消息传递，更新全局 item embedding

    参数:
        round_user_params: dict, key=userID，value={'embedding_item.weight': Tensor}
        item_num: int，总物品数
        latent_dim: int，embedding 维度
        topk_user_relation_graph: np.ndarray, shape=(num_users, num_users)，稀疏邻接矩阵
        layers: int，消息传递层数
    功能:
        1. 将每个用户的 item embedding 拉平为 (item_num * latent_dim) 向量，组成矩阵 (num_users, item_num*latent_dim)
        2. 多轮图消息传递： aggregated = A @ item_embedding
           重复 layers 次（第一次得到一次聚合结果，之后再聚合）
        3. 将聚合后的向量重构为 (item_num, latent_dim)，保存到字典 item_embedding_dict
        4. 计算 'global' embedding：为所有用户聚合后的 embedding 的均值
    输出:
        item_embedding_dict: dict, key=userID 或 'global', value=Tensor shape=(item_num, latent_dim)
    """
    num_users = len(round_user_params)
    # 1. 准备 item_embedding 矩阵
    item_embedding = np.zeros((num_users, item_num * latent_dim), dtype='float32')
    for user in round_user_params.keys():
        item_embedding[user] = round_user_params[user]['embedding_item.weight'].numpy().flatten()

    # 2. 多轮消息传递
    aggregated = np.matmul(topk_user_relation_graph, item_embedding)
    for _ in range(layers - 1):
        aggregated = np.matmul(topk_user_relation_graph, aggregated)

    # 3. 重构回 item embedding 并保存到字典
    item_embedding_dict = {}
    for user in round_user_params.keys():
        # 将用户对应行恢复为 (item_num, latent_dim)
        item_embedding_dict[user] = torch.from_numpy(
            aggregated[user].reshape(item_num, latent_dim)
        )
    # 4. 计算全局 embedding = 所有用户 embedding 的均值
    all_embeddings = list(item_embedding_dict.values())
    global_embedding = sum(all_embeddings) / num_users
    item_embedding_dict['global'] = global_embedding

    return item_embedding_dict

# -----------------------------
# 日志工具函数
# -----------------------------
def initLogging(logFilename):
    """
    初始化日志系统，将日志输出到文件和控制台

    参数:
        logFilename: str，日志文件路径
    功能:
        1. 使用 logging.basicConfig 设置日志等级和格式
        2. 同时创建一个 StreamHandler，将 INFO 及以上日志输出到控制台
    输出:
        无
    """
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s-%(levelname)s-%(message)s',
        datefmt='%y-%m-%d %H:%M',
        filename=logFilename,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

# -----------------------------
# 正则化辅助函数
# -----------------------------
def compute_regularization(model, parameter_label):
    """
    计算对 item embedding 的正则化损失（MSE）

    参数:
        model: torch.nn.Module，当前客户端模型
        parameter_label: Tensor，来自服务端的全局 item embedding 参考 Tensor
    功能:
        遍历 model 的所有参数，当 name == 'embedding_item.weight' 时，计算 MSE(parameter, parameter_label)
    输出:
        reg_loss: Tensor 标量，或 None（如果没有在模型中找到 'embedding_item.weight'）
    """
    reg_fn = torch.nn.MSELoss(reduction='mean')
    for name, param in model.named_parameters():
        if name == 'embedding_item.weight':
            reg_loss = reg_fn(param, parameter_label)
            return reg_loss
    return None
