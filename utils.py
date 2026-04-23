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
    for idx, user in enumerate(round_user_params.keys()):
        # 将用户的 item embedding 拉平为一维向量，用 idx 作为数组行号
        item_embedding[idx] = round_user_params[user]['embedding_item.weight'].numpy().flatten()
    # 使用 sklearn 的 pairwise_distances 计算用户间距离或相似度
    adj = pairwise_distances(item_embedding, metric=similarity_metric)
    if similarity_metric == 'cosine':
        return adj
    else:
        # 对于其他度量方式，返回负值以便后续取高相似度
        return -adj
def construct_user_relation_graph_via_interest(round_user_params, similarity_metric):
    """
    🌟 严格对齐论文：构建兴趣语义图（基于客户端上传的兴趣特征向量）
    """
    num_users = len(round_user_params)
    interest_embeddings = []

    for idx, user in enumerate(round_user_params.keys()):
        # 直接拉平提取到的兴趣向量
        user_interest_vector = round_user_params[user]['interest_params'].numpy().flatten()
        interest_embeddings.append(user_interest_vector)

    interest_embeddings = np.array(interest_embeddings, dtype='float32')

    # 使用 sklearn 计算用户兴趣特征的相似度
    adj = pairwise_distances(interest_embeddings, metric=similarity_metric)
    if similarity_metric == 'cosine':
        return adj
    else:
        return -adj



def _single_graph_neighbor_sets(graph, neighborhood_size, neighborhood_threshold):
    """对单张相似度图，按 Top-K 或阈值方式，返回每个用户的邻居索引集合列表。

    返回 List[Set[int]]，长度 = num_users。
    """
    n = graph.shape[0]
    neighbors: list[set] = []
    if neighborhood_size > 0:
        for u in range(n):
            topk = graph[u].argsort()[-neighborhood_size:][::-1]
            neighbors.append(set(int(i) for i in topk))
    else:
        threshold = np.mean(graph) * neighborhood_threshold
        for u in range(n):
            idxs = np.where(graph[u] > threshold)[0]
            neighbors.append(set(int(i) for i in idxs))
    return neighbors


def select_topk_neighboehood(item_graph, mlp_graph, neighborhood_size, neighborhood_threshold,
                             alpha=1.0, fusion='alpha', return_stats=False):
    """双图邻居筛选，支持三种融合模式。

    参数:
        item_graph:   (n, n) ndarray，基于 item embedding 的用户相似度（"行为关联图"）
        mlp_graph:    (n, n) ndarray 或 None，基于兴趣向量的用户相似度（"兴趣语义图"）
        neighborhood_size:      Top-K 大小；0 → 阈值方式
        neighborhood_threshold: 阈值方式下的相似度阈值乘数
        alpha:        fusion='alpha' 时的双图融合权重，1.0=仅 item，0.0=仅 interest
        fusion:       'alpha'        — 旧实现：alpha*item + (1-alpha)*mlp 后筛选
                      'intersection' — 毕设创新：两图独立筛选后取交集（可信邻居）
                      'union'        — 两图独立筛选后取并集（对照）
        return_stats: True 时额外返回 dict（交集去掉的假邻居率等统计）

    返回:
        topk_user_relation_graph: (n, n) ndarray，行归一化的邻居权重
        （若 return_stats=True，额外返回 stats dict）
    """
    stats = {
        'avg_item_neighbors': 0.0,
        'avg_interest_neighbors': 0.0,
        'avg_trusted_neighbors': 0.0,
        'avg_union_neighbors': 0.0,
        'avg_false_neighbor_ratio': 0.0,   # (|item ∪ interest| - |交集|) / |并集|
        'isolated_nodes': 0,               # 交集为空的用户数（触发回退）
    }

    # 兼容老路径：没有 mlp_graph 时，只能用 alpha 融合（即 item-only）
    if mlp_graph is None:
        fused = item_graph
        num_users = fused.shape[0]
        out = np.zeros(fused.shape, dtype='float32')
        if neighborhood_size > 0:
            for u in range(num_users):
                top = fused[u].argsort()[-neighborhood_size:][::-1]
                for i in top:
                    out[u][i] = 1.0 / neighborhood_size
        else:
            th = np.mean(fused) * neighborhood_threshold
            for u in range(num_users):
                idx = np.where(fused[u] > th)[0]
                if len(idx) > 0:
                    for j in idx:
                        out[u][j] = 1.0 / len(idx)
                else:
                    out[u][u] = 1.0
        return (out, stats) if return_stats else out

    num_users = item_graph.shape[0]
    out = np.zeros(item_graph.shape, dtype='float32')

    if fusion == 'alpha':
        # 旧实现（向后兼容），当作对照组保留
        fused = alpha * item_graph + (1.0 - alpha) * mlp_graph
        if neighborhood_size > 0:
            for u in range(num_users):
                top = fused[u].argsort()[-neighborhood_size:][::-1]
                for i in top:
                    out[u][i] = 1.0 / neighborhood_size
        else:
            th = np.mean(fused) * neighborhood_threshold
            for u in range(num_users):
                idx = np.where(fused[u] > th)[0]
                if len(idx) > 0:
                    for j in idx:
                        out[u][j] = 1.0 / len(idx)
                else:
                    out[u][u] = 1.0
        return (out, stats) if return_stats else out

    # 毕设核心创新：分图筛选后做"可信邻居"筛选
    item_neighbors = _single_graph_neighbor_sets(item_graph, neighborhood_size, neighborhood_threshold)
    int_neighbors = _single_graph_neighbor_sets(mlp_graph, neighborhood_size, neighborhood_threshold)

    sum_item = sum(len(s) for s in item_neighbors)
    sum_int = sum(len(s) for s in int_neighbors)
    sum_trusted = 0
    sum_union = 0
    sum_false = 0
    isolated = 0

    # 软交集：alpha 在此复用为 trust_weight β∈[0,1]
    #   - 可信边（两图交集）权重 β
    #   - 单图确认边权重 (1-β)
    #   - β=1.0 等同 intersection；β=0 等同仅保留单图独有；β≈0.5 等同 union（均匀权重）
    trust_weight = float(alpha) if fusion == 'soft_intersection' else 1.0

    for u in range(num_users):
        si, sj = item_neighbors[u], int_neighbors[u]
        inter = si & sj
        union = si | sj
        only_item = si - sj
        only_int = sj - si
        sum_trusted += len(inter)
        sum_union += len(union)
        if len(union) > 0:
            sum_false += (len(union) - len(inter))

        if fusion == 'intersection':
            chosen = inter
            if not chosen:
                chosen = si if si else {u}
                isolated += 1
            w = 1.0 / len(chosen)
            for v in chosen:
                out[u][v] = w

        elif fusion == 'union':
            chosen = union if union else {u}
            w = 1.0 / len(chosen)
            for v in chosen:
                out[u][v] = w

        elif fusion == 'soft_intersection':
            # 两类边 + 归一化
            unconfirmed = only_item | only_int  # 只有一张图确认的边（"弱"邻居）
            w_trust = trust_weight
            w_other = 1.0 - trust_weight
            # 处理端点：若 β=1 且交集空 → 回退 item 图
            if not inter and w_trust >= 1.0:
                chosen = si if si else {u}
                isolated += 1
                w = 1.0 / len(chosen)
                for v in chosen:
                    out[u][v] = w
                continue
            # 累加未归一化的权重
            raw = {}
            for v in inter:
                raw[v] = raw.get(v, 0.0) + w_trust
            for v in unconfirmed:
                raw[v] = raw.get(v, 0.0) + w_other
            if not raw:
                raw = {u: 1.0}
                isolated += 1
            tot = sum(raw.values())
            for v, w in raw.items():
                out[u][v] = w / tot

        else:
            raise ValueError(f"unknown fusion mode: {fusion}")

    if num_users > 0:
        stats['avg_item_neighbors'] = sum_item / num_users
        stats['avg_interest_neighbors'] = sum_int / num_users
        stats['avg_trusted_neighbors'] = sum_trusted / num_users
        stats['avg_union_neighbors'] = sum_union / num_users
        stats['avg_false_neighbor_ratio'] = sum_false / max(sum_union, 1)
        stats['isolated_nodes'] = isolated

    return (out, stats) if return_stats else out


def laplace_epsilon(noise_scale, sensitivity=1.0):
    """把 Laplace 噪声 scale b 换算为 (ε)-DP 预算：ε = sensitivity / b。

    假设单次查询 sensitivity=1（归一化后的 embedding 上传）。
    组合到 T 轮基础顺序组合：ε_total ≈ T × ε_per_round。
    精确 RDP/moments accountant 需要额外实现，这里给出朴素上界。
    """
    if noise_scale <= 0:
        return float('inf')
    return float(sensitivity) / float(noise_scale)


def MP_on_graph(round_user_params, item_num, latent_dim, topk_user_relation_graph, layers):
    """
    在用户关系图上进行图消息传递，更新全局 item embedding
    """
    num_users = len(round_user_params)
    # 1. 准备 item_embedding 矩阵
    item_embedding = np.zeros((num_users, item_num * latent_dim), dtype='float32')

    # 🌟 修改点 1：用 enumerate 获取索引 idx
    for idx, user in enumerate(round_user_params.keys()):
        item_embedding[idx] = round_user_params[user]['embedding_item.weight'].numpy().flatten()

    # 2. 多轮消息传递
    aggregated = np.matmul(topk_user_relation_graph, item_embedding)
    for _ in range(layers - 1):
        aggregated = np.matmul(topk_user_relation_graph, aggregated)

    # 3. 重构回 item embedding 并保存到字典
    item_embedding_dict = {}

    # 🌟 修改点 2：依然用 idx，确保从 aggregated 里按顺序取回正确的数据
    for idx, user in enumerate(round_user_params.keys()):
        # 将用户对应行恢复为 (item_num, latent_dim)
        item_embedding_dict[user] = torch.from_numpy(
            aggregated[idx].reshape(item_num, latent_dim)
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
