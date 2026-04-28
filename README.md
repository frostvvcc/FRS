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
│   ├── run_exp2_dual_graph.py                # 实验二：双图融合矩阵 17 组
│   ├── run_exp2_dual_graph_soft.py           # 软交集 β sweep 4 组
│   ├── run_exp1_baseline_hparams.py               # 实验一：基础超参对照 (23 组)
│   └── analyze.py                   # 按组排名 + 极差统计
├── tests/                           # 16 个单元测试
├── data/                            # MovieLens 100k / 1m
├── results/
│   ├── thesis/                      # 实验二：双图融合所有 JSON + CSV + 学习曲线
│   └── exp1_baseline_hparams/       # 实验一：基础超参矩阵
├── EXP2_REPORT.md                 # 毕设实验报告（主交付物）
├── EXP1_REPORT.md                # 第 7-8 周分析（bug 修复 + 超参对比）
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
# 🚀 V4 最佳配置（毕设最终版，反超中心化 NCF）
python train.py --dataset 100k --num_round 30 \
    --graph_fusion intersection --graph_semantic similarity \
    --attention_type multihead --num_heads 2 --history_len 5 \
    --max_history_len 20 --interest_type both \
    --dp 0.01 \
    --reg 0.01 --mp_layers 2 --local_epoch 2 \
    --seed 42 --early_stop_patience 6 \
    --metrics_json results/best_v4.json
# 预期：HR@10 ≈ 0.775, NDCG@10 ≈ 0.487
# 相对中心化（HR=0.6872, NDCG=0.4110）：113% HR, 118% NDCG

# V3 最佳配置（多头注意力 + both 兴趣编码 + 严格交集）—— 保留作为对照
python train.py --dataset 100k --num_round 25 \
    --graph_fusion intersection --graph_semantic similarity \
    --attention_type multihead --num_heads 2 --history_len 5 \
    --max_history_len 20 --interest_type both \
    --dp 0.01 \
    --seed 42 --early_stop_patience 5 \
    --metrics_json results/best_v3.json
# 预期：HR@10 ≈ 0.517, NDCG@10 ≈ 0.325

# 严格双图交集（V1 基线）
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
| `--graph_semantic` | `similarity` | 🌟 V2：`similarity`=新默认（值越大越相似）；`distance`=旧 bug 行为（复现用） |
| `--attention_type` | `single` | 🌟 V3：`single`=旧单头；`multihead`=多头 + 可学习位置编码 |
| `--num_heads` | 2 | 多头注意力头数（仅 `--attention_type multihead` 生效） |
| `--max_history_len` | 20 | 位置编码最大长度（应 ≥ `--history_len`） |
| `--interest_type` | `user_emb` | 🌟 V3：`user_emb`（32 维）/`fc_layer`（~3072 维）/`both`（~3104 维） |
| `--dp` | 0.0 | Laplace 噪声 scale（ε_naive = 1/dp；严格 ε 在 metrics_json 中） |
| `--optimizer` | `sgd` | 可选 `sgd` / `adam` / `adamw` |
| `--lr_u` / `--lr_i` | - | 直接指定用户/物品 embedding 学习率 |
| `--lr_eta` | 80 | 旧的学习率缩放公式（见 EXP1_REPORT） |
| `--reg` | 1.0 | item embedding 正则化系数 |
| `--neighborhood_size` | 0 | Top-K（0=阈值方式） |
| `--mp_layers` | 1 | 图消息传递层数 |
| `--seed` | 42 | 随机种子 |
| `--early_stop_patience` | 0 | 验证集 HR 连续 N 轮不升则停止 |
| `--metrics_json` | - | 结束时把完整指标写入 JSON |

🌟 **V2 新增融合模式**（`--graph_fusion`）：
- `intersection` — 严格交集（毕设核心），修复 cosine 语义后 **HR=0.5080 登顶**
- `product` — 两图相似度逐元素积后筛选（软 AND）
- `rank_intersection` — 按两图 rank 和取 Top-K
- `soft_intersection` — 以 alpha 作为 trust 权重 β（V1 遗留）

## 实验矩阵与报告

**毕设主交付**：[EXP2_REPORT.md](EXP2_REPORT.md) — V2 突破性结果（cosine 语义修复 + 可信邻居登顶）。

```bash
# 🌟 V2 矩阵（16 组，≈100 min CPU）—— 毕设最终结果
python experiments/run_exp3_cosine_fix.py --dataset 100k --num_round 25 --early_stop 5

# V1 旧矩阵保留作为对照（17+4 组）
python experiments/run_exp2_dual_graph.py --dataset 100k --num_round 25 --early_stop 5
python experiments/run_exp2_dual_graph_soft.py --dataset 100k --num_round 25 --early_stop 5

# 第 7-8 周报告的旧矩阵（bug 修复 + 超参对比）
python experiments/run_exp1_baseline_hparams.py --dataset 100k --num_round 20 --early_stop 5
```

## 毕设关键发现摘要（V2）

| 方法 | HR@10 | vs 中心化 |
|------|-------|-----------|
| 🌟 **修复 + 严格交集** (V1_sim_intersection) | **0.5080** | **74%** |
| 修复 + 交集 + DP ε=100 | 0.5037 | 73% |
| 修复 + 仅 item 图 | 0.5016 | 72% |
| 修复 + product 融合 | 0.4889 | 71% |
| FedAvg 无图 | 0.4655 | 68% |
| 修复 + 并集 | 0.4592 | 67% |
| 旧 bug + intersection | 0.4189 | 61% |

详见 [EXP2_REPORT.md 第 0 节](EXP2_REPORT.md#0-v2-突破总结必读)。

## V1 结果（保留作为对照）

| 方法 | HR@10 | vs 中心化 |
|------|-------|-----------|
| 中心化 NCF（上界） | 0.6872 | 100% |
| 双图并集（最佳联邦） | 0.4677 | 68% |
| 软交集 β=0.5（=并集均匀权） | 0.4677 | 68% |
| FedAvg 无图 | 0.4655 | 68% |
| **严格交集**（毕设原始设计） | 0.3998 | 58% |
| 严格交集 + DP (ε=200) | 0.4242 | 62% |

核心结论见 [EXP2_REPORT.md 第 10 节](EXP2_REPORT.md#10-毕设最终论点)。

## 测试

```bash
python -m unittest discover -s tests   # 16 个单测
```

详见 [TEST.md](TEST.md)。

## 任务历程

详见 [PROGRESS.md](PROGRESS.md)。
