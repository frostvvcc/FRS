# FPRecommendation — 基于双图"可信邻居"的联邦推荐

> 毕设题目：基于**客户端轻量注意力** + **服务端双图对比可信邻居筛选** + **差分隐私聚合**的联邦推荐系统。

## 三大创新点

1. **客户端侧**：`LightweightAttention` 模块从用户最近 `history_len` 条历史序列中提取短期兴趣
2. **服务端侧**：基于 item embedding 构建"行为关联图"，基于兴趣向量构建"兴趣语义图"，**取两图邻居交集**作为可信邻居（`--graph_fusion intersection`）
3. **隐私聚合**：在可信邻居聚合前注入 Laplace 噪声，`--dp` 控制尺度，对应 ε = 1/dp 的朴素上界

## 目录结构

```
.
├── train.py                         # 联邦训练入口
├── centralized_train.py             # 中心化 NCF 基线（上界对照）
├── engine.py                        # 联邦引擎（含 6 种聚合模式）
├── mlp.py                           # NCF + 注意力模型
├── data.py                          # 数据加载；SampleGenerator(history_len=...)
├── utils.py                         # 图构建、可信邻居选择、消息传递、ε 换算
├── metrics.py                       # HR@K / NDCG@K
├── experiments/
│   ├── run_thesis.py                # 毕设主矩阵 17 组
│   ├── run_thesis_soft.py           # 软交集 β sweep 4 组
│   ├── run_week7_8.py               # 第 7-8 周报告的旧矩阵（保留作为对照）
│   └── analyze.py                   # 按组排名 + 极差统计
├── tests/                           # 16 个单元测试
├── data/                            # MovieLens 100k / 1m
├── results/
│   ├── thesis/                      # 毕设矩阵所有 JSON + CSV + 学习曲线
│   └── week7_8/                     # 第 7-8 周旧矩阵
├── THESIS_REPORT.md                 # 毕设实验报告（主交付物）
├── WEEK7_8_REPORT.md                # 第 7-8 周分析（bug 修复 + 超参对比）
├── README.md / TEST.md / PROGRESS.md
```

## 环境准备

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 快速开始

```bash
# 严格双图交集（毕设默认）
python train.py --dataset 100k --num_round 25 \
    --graph_fusion intersection \
    --seed 42 --metrics_json results/demo.json

# 软交集 β=0.7（推荐在论文里展示参数化 trade-off）
python train.py --dataset 100k --num_round 25 \
    --graph_fusion soft_intersection --alpha 0.7 \
    --seed 42 --metrics_json results/soft_0.7.json

# 带差分隐私的联邦推荐
python train.py --dataset 100k --num_round 25 \
    --graph_fusion intersection --dp 0.005 \
    --seed 42 --metrics_json results/dp.json

# 中心化 NCF 作为上界
python centralized_train.py --dataset 100k --num_epoch 25 \
    --metrics_json results/centralized.json
```

## 核心 CLI 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--dataset` | `ml-1m` | 可选 `ml-1m` / `100k` / `lastfm-2k` / `amazon` |
| `--num_round` | 100 | 联邦学习总轮数 |
| `--graph_fusion` | `alpha` | 🌟 `alpha` / **`intersection`** / **`soft_intersection`** / `union` / `no_graph` / `item_only` / `interest_only` |
| `--alpha` | 0.5 | alpha 模式下的融合权重；soft_intersection 模式下作为 trust 权重 β |
| `--history_len` | 5 | 🌟 历史序列长度（注意力 keys 长度） |
| `--no_attention` | off | 关闭历史序列注意力 |
| `--dp` | 0.0 | Laplace 噪声 scale（ε = 1/dp） |
| `--optimizer` | `sgd` | 可选 `sgd` / `adam` / `adamw` |
| `--lr_u` / `--lr_i` | - | 直接指定用户/物品 embedding 学习率 |
| `--lr_eta` | 80 | 旧的学习率缩放公式（见 WEEK7_8_REPORT） |
| `--reg` | 1.0 | item embedding 正则化系数 |
| `--neighborhood_size` | 0 | Top-K（0=阈值方式） |
| `--mp_layers` | 1 | 图消息传递层数 |
| `--seed` | 42 | 随机种子 |
| `--early_stop_patience` | 0 | 验证集 HR 连续 N 轮不升则停止 |
| `--metrics_json` | - | 结束时把完整指标写入 JSON |

## 实验矩阵与报告

**毕设主交付**：[THESIS_REPORT.md](THESIS_REPORT.md) — 双图可信邻居的完整评估（21 组联邦 + 1 组中心化）。

```bash
# 毕设主矩阵（17 组，≈100 min CPU）
python experiments/run_thesis.py --dataset 100k --num_round 25 --early_stop 5

# 软交集 β sweep（4 组，≈25 min CPU）
python experiments/run_thesis_soft.py --dataset 100k --num_round 25 --early_stop 5

# 第 7-8 周报告的旧矩阵（bug 修复 + 超参对比）
python experiments/run_week7_8.py --dataset 100k --num_round 20 --early_stop 5
```

## 毕设关键发现摘要

| 方法 | HR@10 | vs 中心化 |
|------|-------|-----------|
| 中心化 NCF（上界） | 0.6872 | 100% |
| 双图并集（最佳联邦） | 0.4677 | 68% |
| 软交集 β=0.5（=并集均匀权） | 0.4677 | 68% |
| FedAvg 无图 | 0.4655 | 68% |
| **严格交集**（毕设原始设计） | 0.3998 | 58% |
| 严格交集 + DP (ε=200) | 0.4242 | 62% |

核心结论见 [THESIS_REPORT.md 第 10 节](THESIS_REPORT.md#10-毕设最终论点)。

## 测试

```bash
python -m unittest discover -s tests   # 16 个单测
```

详见 [TEST.md](TEST.md)。

## 任务历程

详见 [PROGRESS.md](PROGRESS.md)。
