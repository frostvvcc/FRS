# 第 7-8 周实验报告

> 第 7 周：开展对比实验，验证双图对比策略对协同效率的提升，调整超参数以优化个性化推荐准确性
> 第 8 周：对设计进行测试和验证，收集详细的实验数据和结果

_数据由 `experiments/run_week7_8.py` 在 100k 上执行 23 组实验生成（总耗时 106.6 min，CPU）。全部实验固定 `seed=42, num_round=20, early_stop_patience=5`。所有原始 JSON 在 `results/week7_8/`。_

---

## 0. 一页总结

| 阶段 | 代表配置 | HR@10 | NDCG@10 | 相对上阶段 |
|------|----------|-------|---------|------------|
| 用户原始实验（第 5-6 周）| A1 "完整模型" default | 0.1424 | 0.0693 | — |
| 修复 `engine.py` 缩进 bug 后 baseline（= C1 默认）| SGD lr=0.1 lr_eta=80 | **0.4189** | 0.2181 | **+194% HR** |
| 本报告最佳（= E2）| mp_layers=2，其余 default | **0.4380** | 0.2355 | **+207% HR 总累计** |

**关键发现**（按量级排序）：

1. **最大提升 来自修复 engine bug**，而非超参调整。原代码 `fed_train_a_round` 中上传 item embedding + 加差分隐私噪声的整段代码被错误嵌套在 `for key in state_dict.keys()` 循环内（`engine.py:301-318` 的缩进问题），每用户空转 6 次，且 `round_participant_params[user]` 在内层不断被覆盖重置，最终只有最后一次键循环的结果生效。修复后 HR 提升 2.94×。
2. **双图 alpha 融合权重（Group A）对 100k 结果影响微弱**（0.4168 → 0.4316 只差 3.6%）。"仅 interest 图"略优于"仅 item 图"，但均衡融合（alpha=0.5）也并未最优。
3. **Top-K 邻居选择策略在本配置下严重反向**：Top-K=5/10/20 HR 停在 0.12-0.15，而阈值方式（threshold=1.0）HR=0.4189，差 2.3×。
4. **图消息传递层数 2 是全局最佳**（E2=0.4380），单层和 3 层都偏弱。
5. **小幅 DP 噪声起正则化作用**：dp=0.01（F2=0.4369）反而优于 dp=0 的 baseline（0.4189）。
6. **"看起来离谱的高 lr 公式"实际是生效的**：纯小 lr SGD（C3=0.0997）和 Adam（C4=0.2015 / C5=0.1029）都显著劣化；降 lr_eta 到 10（C2=0.4327）只带来微升。

---

## 1. 实验设置

- **数据集**：MovieLens 100k（943 用户 × 1682 物品）
- **任务**：Top-K 推荐，Leave-One-Out + 99 负样本
- **评估指标**：HR@10 / NDCG@10
- **联邦配置**：每轮全员参与（`clients_sample_ratio=1.0`），本地 1 epoch，`batch_size=256`，`num_negative=12`
- **模型**：NCF/MLP + 历史序列注意力；embedding dim=32；MLP 层 `[96, 32, 16, 8]`（注意力 ON 时匹配）或 `[64, 32, 16, 8]`（OFF 时）
- **训练预算**：每实验最多 20 轮，`early_stop_patience=5`
- **随机性**：统一 `seed=42` 给 `random` / `numpy` / `torch`，保证同配置可复现

### 本报告涉及的代码改动（相对第 5-6 周）

| 文件 | 变更 | 目的 |
|------|------|------|
| `engine.py` | 修复 `fed_train_a_round` 缩进 bug | **Bug fix — 直接引发 C1 相对用户原 A1 的 2.94× 提升** |
| `engine.py` | 新增 `--optimizer {sgd,adam,adamw}` 分支 | 支持 Adam 对比 |
| `engine.py` | 新增 `lr_u` / `lr_i` 覆写，旁路 `lr_eta` 公式 | 允许直接指定学习率 |
| `engine.py` | `aggregate_clients_params` 加 `no_graph` 退化分支 | 引入 FedAvg 基线 |
| `engine.py` | `fed_evaluate` 去掉 per-user `deepcopy(self.model)` | 性能优化，语义等价 |
| `train.py` | 新增 `--seed --early_stop_patience --metrics_json --result_tag` | 可复现 + 结构化结果 |
| `train.py` | 设置 `random/numpy/torch` 三处种子 | 可复现 |
| `experiments/run_week7_8.py` | 新增，驱动 23 组实验 | 矩阵自动化 |
| `experiments/analyze.py` | 新增，按组汇总 + 极差统计 | 分析输出 |
| `tests/test_engine_features.py` | 新增，3 个测试覆盖 no_graph / Adam / lr_u lr_i 覆写 | 回归保障 |

---

## 2. 全矩阵排名（按 HR@10 降序）

| 排名 | 标签 | HR@10 | NDCG@10 | 最佳轮次 | 实际轮次 |
|------|------|-------|---------|----------|----------|
| 1 | `E2_mp_2` | **0.4380** | 0.2355 | 18 | 20 |
| 2 | `F2_dp_001` | **0.4369** | 0.2437 | 19 | 20 |
| 3 | `C2_sgd_eta10` | **0.4327** | 0.2322 | 19 | 20 |
| 3 | `E3_mp_3` | 0.4327 | 0.2285 | 19 | 20 |
| 5 | `A0_no_graph` | 0.4316 | 0.2418 | 19 | 20 |
| 5 | `A2_alpha_0.7` | 0.4316 | 0.2322 | 19 | 20 |
| 7 | `A5_alpha_0.0` | 0.4295 | 0.2249 | 19 | 20 |
| 8 | `A4_alpha_0.3` | 0.4284 | 0.2268 | 19 | 20 |
| 9 | `D1_no_attention` | 0.4242 | 0.2308 | 18 | 20 |
| 10 | `A3_alpha_0.5` / `B4_threshold_1.0` / `C1_sgd_default` / `E1_mp_1` / `F1_dp_0` | 0.4189 | 0.2181 | 17 | 20 |
| 15 | `A1_alpha_1.0` | 0.4168 | 0.2236 | 17 | 20 |
| 16 | `B5_threshold_1.2` | 0.3860 | 0.2035 | 19 | 20 |
| 17 | `F3_dp_005` | 0.3659 | 0.1956 | 18 | 20 |
| 18 | `C4_adam_001` | 0.2015 | 0.1065 | 19 | 20 |
| 19 | `B3_topk_20` | 0.1485 | 0.0679 | 9 | 15 |
| 20 | `B1_topk_5` | 0.1336 | 0.0627 | 12 | 18 |
| 21 | `B2_topk_10` | 0.1273 | 0.0584 | 7 | 13 |
| 22 | `C5_adam_all_001` | 0.1029 | 0.0464 | 12 | 18 |
| 23 | `C3_sgd_override` | 0.0997 | 0.0444 | 8 | 14 |

**全矩阵极差**：`E2_mp_2` 0.4380 vs `C3_sgd_override` 0.0997 —— 4.39× 差距，说明超参数空间里确有深坑。

> 说明：`A3/B4/C1/E1/F1` 全是同配置（默认），结果严格相同（0.4189 / 0.2181 / round 17），验证 `seed=42` 的可复现性。

完整分组明细见 `results/week7_8/analysis.md`，学习曲线见 `results/week7_8/curves.png`。

---

## 3. Group A — 双图对比策略（第 7 周核心）

| 标签 | alpha | HR@10 | NDCG@10 |
|------|-------|-------|---------|
| `A0_no_graph` | —（无图，FedAvg） | **0.4316** | 0.2418 |
| `A1_alpha_1.0` | 1.0（仅 item）| 0.4168 | 0.2236 |
| `A2_alpha_0.7` | 0.7 | 0.4316 | 0.2322 |
| `A3_alpha_0.5` | 0.5（默认）| 0.4189 | 0.2181 |
| `A4_alpha_0.3` | 0.3 | 0.4284 | 0.2268 |
| `A5_alpha_0.0` | 0.0（仅 interest） | 0.4295 | 0.2249 |

**发现**：
- 组内极差 3.6%，远小于其他维度的敏感度
- **FedAvg (A0)、仅 interest 图 (A5)、alpha=0.7 (A2)** 并列第一梯队
- **仅 item 图 (A1)** 最差 —— 这印证了 interest 特征（来自 `embedding_user`）对用户关系建模比 item embedding 更具判别力
- 双图均衡融合（A3 alpha=0.5）并非最优，略偏 item 侧（A2 alpha=0.7）或偏 interest 侧（A4 alpha=0.3）都略好
- 双图机制的**增量收益在 100k 上不显著**，但它**没有"显著伤害"** —— 关键是 bug 修复让它从 0.14 跃升到 0.42

**建议**：
1. ml-1m 等更大数据集上重复该组，因为更多用户时图邻居结构差异可能被放大
2. 如果 100k 这一结论也在 ml-1m 成立，可以考虑将 `--no_graph` 作为更简洁的默认（减少一次 MP 计算）
3. 若要保留双图，建议 alpha=0.7（偏 item）或 0.3（偏 interest），而非 0.5

---

## 4. Group B — 邻居选择策略

| 标签 | 策略 | HR@10 | NDCG@10 |
|------|------|-------|---------|
| `B4_threshold_1.0` | 阈值 1.0（默认）| **0.4189** | 0.2181 |
| `B5_threshold_1.2` | 阈值 1.2（更严）| 0.3860 | 0.2035 |
| `B3_topk_20` | Top-K=20 | 0.1485 | 0.0679 |
| `B1_topk_5` | Top-K=5 | 0.1336 | 0.0627 |
| `B2_topk_10` | Top-K=10 | 0.1273 | 0.0584 |

**发现**：
- **Top-K 全盘败给阈值**（差 2.3× 以上）
- 收紧阈值（B5 threshold=1.2）会丢失 ~8% HR，说明邻居覆盖范围对聚合质量重要
- Top-K 模式下，每个用户拿到固定数量（5/10/20）的邻居；而阈值模式下，冷门用户可能自引用（`topk_user_relation_graph[i][i] = 1.0`），热门用户拿到很多邻居 —— 这种"按相关度自适应"的邻居规模比均匀 Top-K 更符合现实

**建议**：生产 / 论文中**保留阈值方式**，Top-K 仅用于消融对照。

---

## 5. Group C — 优化器 & 学习率（lr 公式的深坑）

| 标签 | 配置 | HR@10 | NDCG@10 | 备注 |
|------|------|-------|---------|------|
| `C2_sgd_eta10` | SGD lr=0.1 lr_eta=10 | **0.4327** | 0.2322 | 降 lr_eta 微升 +3.3% |
| `C1_sgd_default` | SGD lr=0.1 lr_eta=80 | 0.4189 | 0.2181 | 原始 baseline |
| `C4_adam_001` | Adam lr=0.001 lr_u/i=0.01 | 0.2015 | 0.1065 | Adam 也救不回来 |
| `C5_adam_all_001` | Adam 统一 lr=0.001 | 0.1029 | 0.0464 | Adam lr 太小 |
| `C3_sgd_override` | SGD lr_u=0.01 lr_i=0.1 | 0.0997 | 0.0444 | 小 lr 的 SGD 爬不动 |

**发现**：
- 原来的 `lr_i = lr * num_items * lr_eta - lr` 公式虽然表观上产出数千量级的学习率（100k: ≈ 13 456；ml-1m: ≈ 29 648），但在 *SGD + 稀疏梯度 + 强 reg=1.0* 组合下**反而是"正确刻度"**
- `lr_eta=10` 会让公式输出降到 1/8（≈1682），此时结果微升 —— 说明**原 80 偏高但未崩**
- 把 lr 直接覆写成"人类认为正常"的 0.01 ~ 0.1（C3/C4/C5），HR 立刻从 0.42 掉到 0.1-0.2

**背后的直觉**：每个客户端本地有 k 条样本（~100），item embedding 是 1682 × 32 维稀疏矩阵，每步只有 k 个 item 被更新。表观 `lr * num_items` 实际等价于 "见到一个 item 时的每次更新乘以 num_items/k" 的放大，**把稀疏梯度等效地"密集化"**。这是一种（无文档的）实现技巧。

**建议**：
- 保留默认 `lr_eta=80`
- 如果未来换到更大的数据集（ml-1m 的 num_items=3706），可以测 `lr_eta=40` 看是否有小幅提升
- 若要迁移到 Adam 等自适应优化器，需要把 lr 配成 0.01-0.05（不能 0.001），并同时调小 reg

---

## 6. Group D — 注意力机制

| 标签 | 配置 | HR@10 | NDCG@10 |
|------|------|-------|---------|
| `D1_no_attention` | `--no_attention --layers 64,32,16,8` | 0.4242 | 0.2308 |
| `A3_alpha_0.5`（对照，有注意力） | default | 0.4189 | 0.2181 |

**发现**：无/有注意力基本打平（差 1.3% HR）。100k 的历史序列长度为 5（硬编码），再加上 `LightweightAttention` 是最朴素的 scaled-dot-product，在这个小规模序列上没什么提炼空间。

**建议**：本数据集可以关掉注意力以省算力；或者把 history 长度从 5 增加到 10-20，并用多头注意力再测。

---

## 7. Group E — 图消息传递层数

| 标签 | mp_layers | HR@10 | NDCG@10 |
|------|-----------|-------|---------|
| `E2_mp_2` | 2 | **0.4380** | 0.2355 |
| `E3_mp_3` | 3 | 0.4327 | 0.2285 |
| `E1_mp_1` | 1（默认） | 0.4189 | 0.2181 |

**发现**：
- 2 层 MP 是最佳，优于 1 层 +4.6% HR，也是**本报告全矩阵最优**
- 3 层比 2 层低 0.053 —— 标准的"过度平滑"症状，节点表示趋同

**建议**：把 `--mp_layers 2` 设为新的默认。

---

## 8. Group F — 差分隐私

| 标签 | dp | HR@10 | NDCG@10 |
|------|-----|-------|---------|
| `F2_dp_001` | 0.01 | **0.4369** | 0.2437 |
| `F1_dp_0` | 0.0（默认） | 0.4189 | 0.2181 |
| `F3_dp_005` | 0.05 | 0.3659 | 0.1956 |

**发现**：
- dp=0.01 反而比无噪声高 +4.3% —— 小幅 Laplace 噪声**起到了正则化作用**，缓解过拟合
- dp=0.05 明显劣化（-12.6%），信号被覆盖

**建议**：
- 默认可以改为 `--dp 0.005` ~ `0.01`，同时获得"一点隐私"和"一点正则化"
- 差分隐私预算 ε 的计算需要再补充（当前 `dp` 只是 Laplace 的 scale，需要结合 sensitivity 换算）

---

## 9. 结论

**对第 7 周任务（验证双图对比策略 + 调超参）**：

1. **双图对比策略本身在 100k 上未带来显著增益**（各 alpha 结果挤在 0.4168-0.4316，差 3.6%），但它**也未伤害模型**。这部分需要在 ml-1m / amazon 等更大数据集上复验。
2. **超参数调整的主要胜者**：`mp_layers=2`（+4.6%）、`dp=0.01`（+4.3%）、`lr_eta=10`（+3.3%）。三者可以叠加使用。
3. **超参数调整的主要教训**：Top-K 邻居严重拖垮结果；试图"把 lr 调回正常值"会摧毁模型；注意力机制在短历史 (5) 下没有用。

**对第 8 周任务（测试 + 验证 + 收集数据）**：

1. **测试**：新增 `tests/test_engine_features.py`（3 个用例），总计 11 个单元测试全绿
2. **验证**：一项关键 bug 通过对照实验暴露 —— 用户原 A1 "完整模型" HR=0.1424 vs 本报告同配置 C1=0.4189，2.94× 差距全部归因于 `engine.py:301-318` 的缩进 bug
3. **数据**：23 组实验完整 JSON + CSV + 分组 markdown + 学习曲线 PNG，全部在 `results/week7_8/`

**推荐的新默认配置**：

```bash
python train.py --dataset ml-1m --num_round 30 \
    --mp_layers 2 --dp 0.01 --lr_eta 10 \
    --seed 42 --early_stop_patience 5 \
    --metrics_json results/best.json
```

（相对旧 default 的叠加：`mp_layers=2` + `dp=0.01` + `lr_eta=10`。）

---

## 10. 复现指南

```bash
# 一次跑全部矩阵
python experiments/run_week7_8.py --dataset 100k --num_round 20 --early_stop 5

# 只跑某一组
python experiments/run_week7_8.py --only A              # 双图 6 组
python experiments/run_week7_8.py --only B              # 邻居 5 组
python experiments/run_week7_8.py --only "A,E"          # 组合过滤

# dry-run 查看命令
python experiments/run_week7_8.py --dry_run

# 生成分析 markdown
python experiments/analyze.py --dir results/week7_8 --out results/week7_8/analysis.md
```

输出产物：
- `results/week7_8/<tag>.json` — 每实验全部 HR/NDCG 轨迹
- `results/week7_8/summary.csv` / `summary.md` — 简表
- `results/week7_8/analysis.md` — 分组排名 + 极差
- `results/week7_8/curves.png` — 分组学习曲线
- `results/week7_8/runner.log` — 每实验墙钟耗时

## 附录：关键文件位置

| 文件 | 作用 |
|------|------|
| `engine.py:301-323` | 修复前的缩进 bug；修复后 f 区块与 e 同级 |
| `engine.py:145-165` | `aggregate_clients_params` 中的 `no_graph` 退化分支 |
| `engine.py:265-295` | 三个优化器构造，支持 SGD/Adam/AdamW + lr 覆写 |
| `train.py:78-92` | 种子、CLI 旋钮、`--metrics_json` |
| `experiments/run_week7_8.py:22-78` | 实验矩阵定义 |
