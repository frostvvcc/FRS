# FPRecommendation — 基于图增强的联邦推荐

基于 Neural Collaborative Filtering (NCF/MLP) 的联邦学习推荐系统，在服务端通过**用户关系图 + 图消息传递 (Message Passing)** 聚合各客户端上传的 item embedding，并支持**拉普拉斯差分隐私**噪声注入。

## 核心思路

1. 每个客户端（用户）本地训练自己的 MLP，保留私有的用户 embedding 与 MLP 权重。
2. 客户端仅上传加噪后的 item embedding。
3. 服务端基于用户上传的 item embedding 计算用户相似度，构建 Top-K 邻接图。
4. 在该图上做多层消息传递，得到新的 global item embedding 并下发。

## 目录结构

```
.
├── train.py                 # 训练入口
├── engine.py                # 联邦训练/评估引擎（含图聚合调度）
├── mlp.py                   # MLP 模型 + MLPEngine
├── data.py                  # 数据加载与负采样
├── utils.py                 # 图构建、消息传递、日志、正则化等工具
├── metrics.py               # HR@K / NDCG@K
├── modules.py               # 辅助模块
├── plot_result.py           # 结果可视化
├── visualize_results.py     # 结果可视化（进阶）
├── run_all_experiments.sh   # 批量实验脚本
├── tests/                   # 冒烟测试
└── data/                    # 数据集（ml-1m / 100k 等）
```

## 环境准备

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 快速开始

```bash
# 使用 MovieLens 100k，1 轮联邦训练（冒烟）
python train.py --dataset 100k --num_round 1 --use_cuda False

# 使用 MovieLens 1M，100 轮
python train.py --dataset ml-1m --num_round 100
```

### 常用参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--dataset` | `ml-1m` | 可选 `ml-1m` / `100k` / `lastfm-2k` / `amazon` |
| `--num_round` | 100 | 联邦学习总轮数 |
| `--clients_sample_ratio` | 1.0 | 每轮采样参与训练的客户端比例 |
| `--local_epoch` | 1 | 每个客户端本地训练轮数 |
| `--latent_dim` | 32 | embedding 维度 |
| `--layers` | `64,32,16,8` | MLP 隐藏层维度 |
| `--num_negative` | 12 | 每个正样本对应的负样本数量 |
| `--neighborhood_size` | 0 | 邻居图 Top-K（为 0 时用阈值） |
| `--neighborhood_threshold` | 1.0 | 阈值模式下的相似度系数（均值 × 系数） |
| `--mp_layers` | 1 | 图消息传递层数 |
| `--similarity_metric` | `cosine` | 用户相似度度量 |
| `--dp` | 0.0 | 拉普拉斯差分隐私噪声尺度 |
| `--reg` | 1.0 | item embedding 正则化系数 |
| `--lr` | 0.1 | 基础学习率 |
| `--lr_eta` | 80 | 用户/物品 embedding 学习率缩放因子 |
| `--optimizer` | `sgd` | 可选 `sgd` / `adam` / `adamw` |
| `--lr_u` | - | 直接指定用户 embedding 学习率（覆盖 `lr_eta` 公式） |
| `--lr_i` | - | 直接指定物品 embedding 学习率（覆盖 `lr_eta` 公式） |
| `--no_graph` | off | FedAvg 基线：聚合阶段直接平均 item embedding |
| `--alpha` | 0.5 | 双图融合权重（0=仅 interest 图，1=仅 item 图）|
| `--no_attention` | off | 关闭历史序列注意力 |
| `--seed` | 42 | 随机种子（python/numpy/torch）|
| `--early_stop_patience` | 0 | 验证集 HR 连续 N 轮不提升则停止（0=关闭）|
| `--metrics_json` | - | 结束时把完整指标写入该 JSON 路径 |
| `--use_cuda` | True | 是否使用 GPU |
| `--device_id` | 0 | CUDA 设备编号 |

## 输出

- `log/` — 每次训练的运行日志（按时间戳命名，运行时自动创建）
- `sh_result/<dataset>.txt` — 每次训练的最终结果单行摘要
- `checkpoints/` — 模型检查点（按需启用，运行时自动创建）

## 实验矩阵（第 7-8 周）

`experiments/run_week7_8.py` 驱动 23 组对照实验，完整结果与分析见 [WEEK7_8_REPORT.md](WEEK7_8_REPORT.md)。

```bash
# 跑全矩阵
python experiments/run_week7_8.py --dataset 100k --num_round 20 --early_stop 5

# 只跑某组（A=双图 / B=邻居 / C=优化器 / D=注意力 / E=MP层数 / F=DP）
python experiments/run_week7_8.py --only A

# 分析结果
python experiments/analyze.py --dir results/week7_8
```

**当前最佳配置（100k）**：`--mp_layers 2 --dp 0.01 --lr_eta 10`，HR@10=0.4380 / NDCG@10=0.2355。

## 测试

详见 [TEST.md](TEST.md)。

## 任务历程与经验

详见 [PROGRESS.md](PROGRESS.md)。
