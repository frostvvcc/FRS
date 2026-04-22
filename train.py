# train.py

import pandas as pd
import numpy as np
import datetime
import os
import argparse
from mlp import MLPEngine
from data import SampleGenerator
from utils import *

# -----------------------------
# 解析命令行参数
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--alias', type=str, default='fedgraph',
                    help="实验别名，用于日志和模型文件命名")
parser.add_argument('--clients_sample_ratio', type=float, default=1,
                    help="每轮采样参与训练的客户端比例（0～1）")
parser.add_argument('--clients_sample_num', type=int, default=0,
                    help="每轮采样固定数量的客户端（若为0，则使用比例采样）")
parser.add_argument('--num_round', type=int, default=100,
                    help="联邦学习总轮数")
parser.add_argument('--local_epoch', type=int, default=1,
                    help="每个客户端本地训练轮数")
parser.add_argument('--neighborhood_size', type=int, default=0,
                    help="用户关系图中每个节点的邻居数（若为0，则使用阈值方式）")
parser.add_argument('--neighborhood_threshold', type=float, default=1.0,
                    help="构建邻居时的相似度阈值系数（均值 * 阈值）")
parser.add_argument('--mp_layers', type=int, default=1,
                    help="图消息传递层数")
parser.add_argument('--similarity_metric', type=str, default='cosine',
                    help="构建用户相似度时使用的度量方式（如 'cosine'）")
parser.add_argument('--reg', type=float, default=1.0,
                    help="正则化系数（作用于 item embedding）")
parser.add_argument('--lr_eta', type=int, default=80,
                    help="学习率缩放因子（用于用户/物品 embedding 优化器）")
parser.add_argument('--batch_size', type=int, default=256,
                    help="每个客户端本地训练时的批次大小")
parser.add_argument('--optimizer', type=str, default='sgd',
                    help="优化器类型（当前未在本脚本直接使用）")
parser.add_argument('--lr', type=float, default=0.1,
                    help="基础学习率")
parser.add_argument('--dataset', type=str, default='ml-1m',
                    help="使用的数据集名称（'ml-1m', '100k', 'lastfm-2k', 'amazon'）")
parser.add_argument('--num_users', type=int,
                    help="总用户数（由 dataset 决定）")
parser.add_argument('--num_items', type=int,
                    help="总物品数（由 dataset 决定）")
parser.add_argument('--latent_dim', type=int, default=32,
                    help="Embedding 维度")
parser.add_argument('--num_negative', type=int, default=12,
                    help="每个正样本对应的负样本数量")
parser.add_argument('--layers', type=str, default='96, 32, 16, 8',
                    help="MLP 隐藏层维度列表，用逗号分隔")
parser.add_argument('--l2_regularization', type=float, default=0.0,
                    help="L2 正则化系数（未直接使用）")
parser.add_argument('--dp', type=float, default=0.0,
                    help="差分隐私噪声尺度")
parser.add_argument('--use_cuda', type=bool, default=True,
                    help="是否使用 GPU 训练")
parser.add_argument('--device_id', type=int, default=0,
                    help="CUDA 设备编号")
parser.add_argument('--model_dir', type=str,
                    default='checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model',
                    help="模型保存路径模板，包含别名、轮次、HR 和 NDCG")
parser.add_argument('--alpha', type=float, default=0.5,
                    help="双图融合权重，0=只用interest图，1=只用item图，0.5=两图各半")
parser.add_argument('--no_attention', action='store_true', default=False,
                    help="加上这个参数就关掉注意力机制")
parser.set_defaults(use_attention=True)
args = parser.parse_args()

# 将参数转换为字典，便于后续处理
config = vars(args)
config['use_cuda'] = False  # 强行把全局开关焊死在 False 上！
config['use_attention'] = not config['no_attention']

# 将层列表字符串转为 int 列表
if isinstance(config['layers'], str) and ',' in config['layers']:
    config['layers'] = [int(item) for item in config['layers'].split(',')]
else:
    config['layers'] = [int(config['layers'])]

# 根据 dataset 名称设置 num_users 和 num_items
if config['dataset'] == 'ml-1m':
    config['num_users'] = 6040
    config['num_items'] = 3706
elif config['dataset'] == '100k':
    config['num_users'] = 943
    config['num_items'] = 1682
elif config['dataset'] == 'lastfm-2k':
    config['num_users'] = 1600
    config['num_items'] = 12454
elif config['dataset'] == 'amazon':
    config['num_users'] = 8072
    config['num_items'] = 11830
else:
    # 如果出现未知 dataset，可自行添加或保持默认
    pass

# 实例化 MLPEngine，负责联邦训练与评估
engine = MLPEngine(config)

# -----------------------------
# 日志初始化
# -----------------------------
path = 'log/'
os.makedirs(path, exist_ok=True)  # 确保 log 目录存在
os.makedirs('sh_result', exist_ok=True)  # 结果摘要目录
os.makedirs('checkpoints', exist_ok=True)  # 模型检查点目录
current_time = datetime.datetime.now().strftime('%Y-%m-%d %H_%M_%S')
logname = os.path.join(path, current_time + '.txt')
initLogging(logname)  # 使用 utils.py 中的 initLogging
logging.info(f"alias: {config['alias']}")

# -----------------------------
# 加载并预处理数据
# -----------------------------
dataset_dir = "data/" + config['dataset'] + "/" + "ratings.dat"
if config['dataset'] == "ml-1m":
    # ml-1m 数据集以 "::" 分隔
    rating = pd.read_csv(dataset_dir, sep='::', header=None,
                         names=['uid', 'mid', 'rating', 'timestamp'], engine='python')
elif config['dataset'] == "100k":
    # 100k 数据集以 "," 分隔
    rating = pd.read_csv(dataset_dir, sep=",", header=None,
                         names=['uid', 'mid', 'rating', 'timestamp'], engine='python')
elif config['dataset'] == "lastfm-2k":
    rating = pd.read_csv(dataset_dir, sep=",", header=None,
                         names=['uid', 'mid', 'rating', 'timestamp'], engine='python')
elif config['dataset'] == "amazon":
    rating = pd.read_csv(dataset_dir, sep=",", header=None,
                         names=['uid', 'mid', 'rating', 'timestamp'], engine='python')
    rating = rating.sort_values(by='uid', ascending=True)
else:
    # 如果 dataset 不在上述选项中，可自行扩展
    pass

# -----------------------------
# 对用户和物品进行重新编号（重新索引）
# -----------------------------
# 构建用户 ID 映射表，将原始 uid 映射为连续的 userId（0 到 num_users-1）
user_id = rating[['uid']].drop_duplicates().reset_index(drop=True)
user_id['userId'] = np.arange(len(user_id))
rating = pd.merge(rating, user_id, on=['uid'], how='left')

# 构建物品 ID 映射表，将原始 mid 映射为连续的 itemId（0 到 num_items-1）
item_id = rating[['mid']].drop_duplicates().reset_index(drop=True)
item_id['itemId'] = np.arange(len(item_id))
rating = pd.merge(rating, item_id, on=['mid'], how='left')

# 保留最终字段：userId、itemId、rating、timestamp
rating = rating[['userId', 'itemId', 'rating', 'timestamp']]

logging.info('Range of userId is [{}, {}]'.format(rating.userId.min(), rating.userId.max()))
logging.info('Range of itemId is [{}, {}]'.format(rating.itemId.min(), rating.itemId.max()))

# -----------------------------
# 构造 SampleGenerator，用于生成训练、验证、测试所需数据
# -----------------------------
sample_generator = SampleGenerator(ratings=rating)
# 获取验证集张量：[val_users, val_items, negative_users, negative_items]
validate_data = sample_generator.validate_data
# 获取测试集张量：[test_users, test_items, negative_users, negative_items]
test_data = sample_generator.test_data

# -----------------------------
# 联邦训练与评估主循环
# -----------------------------
hit_ratio_list = []        # 每轮测试集 HR
ndcg_list = []             # 每轮测试集 NDCG
val_hr_list = []           # 每轮验证集 HR
val_ndcg_list = []         # 每轮验证集 NDCG
train_loss_list = []       # 每轮输出的训练损失（这里是返回参与者列表，通常只需记录参与用户）
test_loss_list = []        # 每轮测试集各用户 loss 字典
val_loss_list = []         # 每轮验证集各用户 loss 字典
best_val_hr = 0            # 用于记录验证集上的最佳 HR
final_test_round = 0       # 最佳验证轮次对应的测试结果轮次

for rnd in range(config['num_round']):
    """
    参数:
        rnd: 当前联邦学习轮次（从 0 开始）
    功能:
        1. 构建本轮所有用户的训练数据 all_train_data = [users_per_user, items_per_user, ratings_per_user]
        2. 调用 engine.fed_train_a_round 进行一轮联邦训练
        3. 在测试集上调用 engine.fed_evaluate，记录 HR、NDCG 和各用户 loss
        4. 在验证集上调用 engine.fed_evaluate，记录验证集指标，用于早停或模型挑选
    输出:
        无（结果保存在 hit_ratio_list, ndcg_list, 等列表中）
    """
    logging.info('-' * 80)
    logging.info('Round {} starts!'.format(rnd))

    # 1. 生成本轮所有用户正负样本数据
    all_train_data = sample_generator.store_all_train_data(config['num_negative'])
    logging.info('-' * 80)
    logging.info('Training phase!')

    # 2. 一轮联邦训练，返回参与用户列表（或其他指标）
    participants = engine.fed_train_a_round(all_train_data, round_id=rnd)
    # 将参与用户数或列表长度加入 train_loss_list 以供查看训练时进度
    train_loss_list.append(len(participants))

    logging.info('-' * 80)
    logging.info('Testing phase!')
    # 3. 在测试集上评估，返回 HR, NDCG, 以及每个用户的 loss 字典
    hit_ratio, ndcg, te_loss = engine.fed_evaluate(test_data)
    test_loss_list.append(te_loss)
    logging.info('[Testing Round {}] HR = {:.4f}, NDCG = {:.4f}'.format(rnd, hit_ratio, ndcg))
    hit_ratio_list.append(hit_ratio)
    ndcg_list.append(ndcg)

    logging.info('-' * 80)
    logging.info('Validating phase!')
    # 4. 在验证集上评估
    val_hit_ratio, val_ndcg, v_loss = engine.fed_evaluate(validate_data)
    val_loss_list.append(v_loss)
    logging.info('[Validating Round {}] HR = {:.4f}, NDCG = {:.4f}'.format(rnd, val_hit_ratio, val_ndcg))
    val_hr_list.append(val_hit_ratio)
    val_ndcg_list.append(val_ndcg)

    # 更新最佳验证指标
    if val_hit_ratio >= best_val_hr:
        best_val_hr = val_hit_ratio
        final_test_round = rnd

# -----------------------------
# 记录最终结果到文件
# -----------------------------
current_time = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
result_str = (
    f"[{config['alias']}] {current_time} - layers: {config['layers']} - lr: {config['lr']} - "
    f"clients_sample_ratio: {config['clients_sample_ratio']} - num_round: {config['num_round']} - "
    f"neighborhood_size: {config['neighborhood_size']} - mp_layers: {config['mp_layers']} - "
    f"negatives: {config['num_negative']} - lr_eta: {config['lr_eta']} - "
    f"batch_size: {config['batch_size']} - hr: {hit_ratio_list[final_test_round]:.4f} - "
    f"ndcg: {ndcg_list[final_test_round]:.4f} - best_round: {final_test_round} - "
    f"similarity_metric: {config['similarity_metric']} - "
    f"neighborhood_threshold: {config['neighborhood_threshold']} - reg: {config['reg']}"
)
file_name = f"sh_result/{config['dataset']}.txt"
with open(file_name, 'a') as file:
    file.write(result_str + '\n')

logging.info('fedgraph Training Complete')
logging.info(
    f"clients_sample_ratio: {config['clients_sample_ratio']}, lr_eta: {config['lr_eta']}, "
    f"batch_size: {config['batch_size']}, lr: {config['lr']}, dataset: {config['dataset']}, "
    f"layers: {config['layers']}, negatives: {config['num_negative']}, "
    f"neighborhood_size: {config['neighborhood_size']}, "
    f"neighborhood_threshold: {config['neighborhood_threshold']}, "
    f"mp_layers: {config['mp_layers']}, similarity_metric: {config['similarity_metric']}"
)
logging.info(f"hit_ratio_list: {hit_ratio_list}")
logging.info(f"ndcg_list: {ndcg_list}")
logging.info(
    f"Best test HR: {hit_ratio_list[final_test_round]:.4f}, "
    f"NDCG: {ndcg_list[final_test_round]:.4f} at round {final_test_round}"
)
