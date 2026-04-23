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
def _distances_to_similarity(adj: np.ndarray, metric: str) -> np.ndarray:
    """将 pairwise_distances 的输出归一化为"越大越相似"的相似度矩阵。

    历史 bug：`cosine` 分支原先直接返回距离矩阵，导致下游 `argmax` 选的是
    "距离最大 = 最不相似" 的用户作为邻居。本函数统一转为相似度。
    """
    if metric == 'cosine':
        # cosine distance ∈ [0, 2] → similarity = 1 - distance ∈ [-1, 1]
        return 1.0 - adj
    # 其他指标：以 -distance 表达相似度（数值越大越相似）
    return -adj


def construct_user_relation_graph_via_item(round_user_params, item_num, latent_dim,
                                           similarity_metric, semantic='similarity'):
    """构建基于 item embedding 的用户关系图。

    参数:
        semantic: 'similarity' — 返回相似度矩阵（越大越相似，推荐）🌟 新默认
                  'distance'   — 返回原始距离矩阵（旧 bug 行为，仅用于复现）
    """
    num_users = len(round_user_params)
    item_embedding = np.zeros((num_users, item_num * latent_dim), dtype='float32')
    for idx, user in enumerate(round_user_params.keys()):
        item_embedding[idx] = round_user_params[user]['embedding_item.weight'].numpy().flatten()
    adj = pairwise_distances(item_embedding, metric=similarity_metric)

    if semantic == 'distance':
        # 旧 bug 行为：cosine 返回距离，其他指标返回 -距离
        return adj if similarity_metric == 'cosine' else -adj
    return _distances_to_similarity(adj, similarity_metric)


def construct_user_relation_graph_via_interest(round_user_params, similarity_metric,
                                               semantic='similarity'):
    """构建基于兴趣向量的用户关系图（语义同上）。"""
    num_users = len(round_user_params)
    interest_embeddings = []
    for idx, user in enumerate(round_user_params.keys()):
        interest_embeddings.append(round_user_params[user]['interest_params'].numpy().flatten())
    interest_embeddings = np.array(interest_embeddings, dtype='float32')
    adj = pairwise_distances(interest_embeddings, metric=similarity_metric)

    if semantic == 'distance':
        return adj if similarity_metric == 'cosine' else -adj
    return _distances_to_similarity(adj, similarity_metric)



def _single_graph_neighbor_sets(graph, neighborhood_size, neighborhood_threshold,
                                exclude_self: bool = True):
    """对单张图（相似度或距离），按 Top-K 或阈值方式返回每个用户的邻居索引集合。

    参数:
        exclude_self: 自环排除。相似度图下 self-similarity=1 是最大值，若不排除
                      argmax 会永远选到自己；距离图下 self-distance=0 是最小值，
                      argmax 自然不会选到自己。默认排除，同时兼容两种语义。

    返回 List[Set[int]]，长度 = num_users。
    """
    n = graph.shape[0]
    g = graph.copy() if exclude_self else graph
    if exclude_self:
        np.fill_diagonal(g, -np.inf)

    neighbors: list[set] = []
    if neighborhood_size > 0:
        for u in range(n):
            topk = g[u].argsort()[-neighborhood_size:][::-1]
            neighbors.append(set(int(i) for i in topk))
    else:
        # 阈值模式：相对均值的缩放因子；对相似度图等价于"高于均值的才算邻居"
        # 注意：exclude_self=True 时对角线为 -inf，不影响 np.mean（numpy 会忽略 inf）
        # 为稳健起见显式用非对角元素计算均值
        mask = ~np.eye(n, dtype=bool)
        threshold = float(np.mean(g[mask])) * float(neighborhood_threshold)
        for u in range(n):
            idxs = np.where(g[u] > threshold)[0]
            # 排除自己（如果还在）
            if exclude_self:
                idxs = idxs[idxs != u]
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
    # 这些累计量被多个 fusion 分支共享，必须提前初始化
    isolated = 0

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

    # === 毕设核心创新（升级版）===
    # 'product':           两图相似度逐元素相乘（软 AND），然后 Top-K / 阈值选择
    #                      —— 不丢弃边，但强烈抑制双图分歧的邻居
    # 'rank_intersection': 分别算各图 rank，取"rank_item + rank_interest" 最小的 Top-K
    #                      —— 比阈值交集更稳健，可控邻居数
    if fusion == 'product':
        # 把两图相似度归一到 [0, 1] 防止负值相乘翻转方向
        def _normalize01(a: np.ndarray) -> np.ndarray:
            mn, mx = float(a.min()), float(a.max())
            return (a - mn) / max(mx - mn, 1e-9)

        item_sim = _normalize01(item_graph)
        int_sim = _normalize01(mlp_graph)
        product_graph = item_sim * int_sim  # 元素积：双图都高才高，一侧低则整体低
        neighbors = _single_graph_neighbor_sets(product_graph, neighborhood_size,
                                                neighborhood_threshold)
        # 记录统计（用原双图算假邻居率供对比）
        item_neighbors = _single_graph_neighbor_sets(item_graph, neighborhood_size,
                                                    neighborhood_threshold)
        int_neighbors = _single_graph_neighbor_sets(mlp_graph, neighborhood_size,
                                                    neighborhood_threshold)
        for u in range(num_users):
            si, sj = item_neighbors[u], int_neighbors[u]
            inter = si & sj
            union = si | sj
            stats['avg_trusted_neighbors'] = stats.get('avg_trusted_neighbors', 0.0) + len(inter)
            stats['avg_union_neighbors'] = stats.get('avg_union_neighbors', 0.0) + len(union)
            if len(union) > 0:
                stats['avg_false_neighbor_ratio'] = stats.get('avg_false_neighbor_ratio', 0.0) + \
                    (len(union) - len(inter)) / len(union)

            chosen = neighbors[u] if neighbors[u] else {u}
            if not neighbors[u]:
                isolated += 1
            w = 1.0 / len(chosen)
            for v in chosen:
                out[u][v] = w
        if num_users > 0:
            stats['avg_trusted_neighbors'] /= num_users
            stats['avg_union_neighbors'] /= num_users
            stats['avg_false_neighbor_ratio'] /= num_users
        stats['avg_item_neighbors'] = sum(len(s) for s in item_neighbors) / max(num_users, 1)
        stats['avg_interest_neighbors'] = sum(len(s) for s in int_neighbors) / max(num_users, 1)
        stats['isolated_nodes'] = isolated
        return (out, stats) if return_stats else out

    if fusion == 'rank_intersection':
        # 每张图的 rank（越小 = 越相似）
        def _rank_matrix(g: np.ndarray) -> np.ndarray:
            tmp = g.copy()
            np.fill_diagonal(tmp, -np.inf)
            # argsort desc → 排名（大值对应小 rank）
            order = tmp.argsort(axis=1)[:, ::-1]
            ranks = np.empty_like(order)
            idx = np.arange(g.shape[1])
            for r in range(g.shape[0]):
                ranks[r, order[r]] = idx
            return ranks  # 越小越相似

        r_item = _rank_matrix(item_graph)
        r_int = _rank_matrix(mlp_graph)
        combined_rank = r_item + r_int  # 两图 rank 和越小 → 双图都认为相似
        # 取 Top-K（combined_rank 越小越好）
        K = neighborhood_size if neighborhood_size > 0 else 20
        item_neighbors = _single_graph_neighbor_sets(item_graph, neighborhood_size,
                                                    neighborhood_threshold)
        int_neighbors = _single_graph_neighbor_sets(mlp_graph, neighborhood_size,
                                                    neighborhood_threshold)
        for u in range(num_users):
            si, sj = item_neighbors[u], int_neighbors[u]
            inter = si & sj
            union = si | sj
            stats['avg_trusted_neighbors'] = stats.get('avg_trusted_neighbors', 0.0) + len(inter)
            stats['avg_union_neighbors'] = stats.get('avg_union_neighbors', 0.0) + len(union)
            if len(union) > 0:
                stats['avg_false_neighbor_ratio'] = stats.get('avg_false_neighbor_ratio', 0.0) + \
                    (len(union) - len(inter)) / len(union)
            chosen = combined_rank[u].argsort()[:K].tolist()
            chosen = [int(v) for v in chosen if v != u][:K]
            if not chosen:
                chosen = [u]
                isolated += 1
            w = 1.0 / len(chosen)
            for v in chosen:
                out[u][v] = w
        if num_users > 0:
            stats['avg_trusted_neighbors'] /= num_users
            stats['avg_union_neighbors'] /= num_users
            stats['avg_false_neighbor_ratio'] /= num_users
        stats['avg_item_neighbors'] = sum(len(s) for s in item_neighbors) / max(num_users, 1)
        stats['avg_interest_neighbors'] = sum(len(s) for s in int_neighbors) / max(num_users, 1)
        stats['isolated_nodes'] = isolated
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
