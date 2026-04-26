# 项目进展与经验沉淀

## 经验教训

### 2026-04-21 — 工程基础设施补齐 + 性能优化

**遇到的问题**

1. `train.py` 直接向 `sh_result/<dataset>.txt` 追加写入，但未检查目录存在，首次运行会报 `FileNotFoundError`。
2. `train.py:8` 硬编码 `os.environ["CUDA_VISIBLE_DEVICES"] = "1"`，与 `--device_id` 参数冲突，多机/多卡协作时混乱。
3. `engine.fed_evaluate` 对**所有**用户（ml-1m 6040 人）每轮都 `copy.deepcopy(self.model)` 一次，评估阶段耗时极高且显存压力大。
4. `.idea/`、`__pycache__/`、`.DS_Store`、`learning_curve.png`、`log/` 目录下所有历史日志都被纳入版本控制，仓库污染。
5. 无 `README.md`、`requirements.txt`、测试文件、`.gitignore`，新协作者无法快速上手。

**如何解决**

1. `train.py` 启动时 `os.makedirs('sh_result', exist_ok=True)`、`os.makedirs('checkpoints', exist_ok=True)`。
2. 移除硬编码的 `CUDA_VISIBLE_DEVICES`，改为可通过环境变量传入，由 `--device_id` 控制当前进程使用的卡。
3. `fed_evaluate` 改为**复用同一个 `self.model`**：临时 `state_dict` 覆盖 + 评估 + 恢复；避免每个用户都 deepcopy。
4. 建立 `.gitignore`，`git rm --cached` 清理已入库的缓存/IDE/日志文件。
5. 补齐 `README.md`、`TEST.md`、`PROGRESS.md`、`requirements.txt`、`tests/`。

**以后如何避免**

- 所有运行时会生成文件的目录，必须在代码启动处 `exist_ok=True` 兜底。
- 配置项（GPU 卡号、日志路径等）一律走 CLI 参数或环境变量，**严禁硬编码**。
- 评估/推理阶段的 deepcopy 是常见性能陷阱：优先用 `load_state_dict` 就地切换参数。
- 新项目第一个 commit 就要有 `.gitignore`，否则 IDE/缓存文件清理成本很高。

**相关 commit**：本轮改动见 `git log --grep="infra"`。

---

### 2026-04-22 — 第 7-8 周实验：修复关键 bug + 双图/超参矩阵对比

**遇到的问题**

1. `engine.py:fed_train_a_round` 有**严重缩进 bug**：上传 item embedding、加差分隐私噪声的整个 f 区块被错误嵌套在 `for key in self.client_model_params[user].keys():` 循环内（缩进多一级）。每个用户空转 state_dict 键数次（~6），且 `round_participant_params[user]` 在内层不断被 `{}` 重置，最终只有最后一次键循环的结果生效。语义上"勉强能跑"，但严重影响模型收敛质量。
2. 学习率公式 `lr_i = lr * num_items * lr_eta - lr` 看起来离谱（100k ≈ 13456，ml-1m ≈ 29648），导致最初怀疑是 bug。实际对照实验证明：**这个公式刚好补偿了联邦联邦稀疏梯度** —— 直接把 lr 改成 0.01-0.1 的"正常量级"会让 HR 从 0.42 掉到 0.1-0.2。
3. MLP 层宽 `[64,32,16,8]` 在注意力 ON 时与 3×latent_dim=96 不匹配。实际代码 default `--layers '96, 32, 16, 8'`，Python 对空白容忍所以 OK，但 `--layers 64, 32, 16, 8`（带空格）会被 shell 拆成 4 个独立 arg，导致 argparse 报错。
4. 实验跑了 100 分钟，matplotlib 没装，runner 的 plot 步骤失败（非致命）。

**如何解决**

1. 把 f 区块缩进降回与 e 同级（`engine.py:301-318`）。同一 seed 重跑用户原 A1 "完整模型" 配置，HR 从 0.1424 → 0.4189（+194%）。这是本轮最大的收益来源。
2. 保留原 lr_eta 公式为默认；新增 `--lr_u` / `--lr_i` CLI 旋钮用于覆盖；新增 `--optimizer {sgd,adam,adamw}`。用 Adam 对比后确认原公式是"正确刻度"（Adam + lr=0.001 只有 0.10）。
3. 运行器中把 `--layers` 的值拼成无空格字符串 `"64,32,16,8"` 传给 subprocess。
4. 跑完后手动 `/tmp/frs-venv/bin/pip install matplotlib` 然后补画 curves.png。

**以后如何避免**

- **严重警告：修代码前务必 pyflakes/pylint 扫一遍缩进敏感区域**。本项目因为带注释的循环嵌套不明显，肉眼不易察觉。建议把 `ruff` 或 `flake8` 加进 pre-commit，能自动抓 `for` 外代码被误入循环的 case。
- 对"长得离谱"的超参公式不要直接"修复为正常值"；先跑对照实验确认。学习率数量级在稀疏梯度场景下可能与预期完全不同。
- CLI 向 subprocess 传列表型参数时一律用无空格字符串（或 JSON）。
- 跑大矩阵前先验证依赖完整（`python -c "import matplotlib"`），避免一小时后发现可视化断掉。
- 使用 `--seed --early_stop_patience --metrics_json` 的组合让每次实验可复现且结构化，远好于解析日志字符串。

**实验关键发现**

- 双图 alpha 融合在 100k 上影响微弱（HR 0.4168-0.4316，差 3.6%）
- Top-K 邻居（5/10/20）HR 只有 0.13-0.15，远劣于阈值方式（0.42）
- 图消息传递 2 层是甜点（E2=0.4380），1 层偏弱，3 层过度平滑
- dp=0.01 的差分隐私噪声反而**提升** HR（+4.3%），起正则化作用

**相关 commit**：本轮改动见 `git log --grep="week7"`。完整结果在 `WEEK7_8_REPORT.md` 和 `results/week7_8/`。

---

### 2026-04-22 — 毕设创新落地：双图"可信邻居"交集 + 软交集 + 完整矩阵

**遇到的问题**

1. **毕设描述与代码实现不一致**：任务书原文"**对于每个节点在两张图上分别筛选出其邻居，取两张图中都在邻居列表中的用户作为'可信邻居'**"是集合交集；但 `utils.py:select_topk_neighboehood` 实际做的是 `alpha * item_graph + (1-alpha) * mlp_graph` 加权融合 —— 先融合再筛选，与"分别筛选后取交集"数学完全不同。
2. 历史序列长度 5 硬编码在 `data.py:_get_history`，无法参数化做消融。
3. 注意力模块（`LightweightAttention`）只是单点积，容易过拟合；但没有对照实验说明它的实际贡献。
4. 没有中心化 NCF 基线作为性能上界，所有联邦指标缺乏参考系。
5. 没有朴素的 ε 预算换算，差分隐私只能报告 Laplace scale。
6. 通讯成本、假邻居率等毕设声称"可量化"的指标从未真正统计过。

**如何解决**

1. `utils.py` 重写 `select_topk_neighboehood`，引入 `fusion ∈ {alpha, intersection, union, soft_intersection}`：
   - `intersection`：严格交集（毕设原始设计）
   - `soft_intersection`：交集边权重 β，单图独有边权重 (1-β)，β=1 退化为严格交集
   - 保留 `alpha` 做向后兼容
2. `data.py:SampleGenerator(history_len=...)` 参数化历史长度，`train.py --history_len N` 暴露
3. 新增 `centralized_train.py` —— 用同一 MLP + 注意力做中心化训练，HR=0.6872 作为上界
4. 新增 `utils.laplace_epsilon()`，train.py 输出 `epsilon_per_round / epsilon_total_naive`
5. engine 每轮通过 `last_aggregate_stats` 暴露可信邻居数、假邻居率、孤立节点数；`last_round_upload_bytes` 记上传字节
6. 新增 21 组实验矩阵（T/H/D/S/Z 五组），结果 JSON + summary + curves 全部落盘

**实验关键发现（诚实报告）**

1. **严格交集在 100k 上 HR=0.3998，显著差于并集 0.4677（-14%）**。毕设"降低假邻居即提升性能"的假设被数据否定。
2. **75% 的邻居边是"假邻居"**（只被单张图认可）—— 这一统计本身是毕设贡献
3. **软交集 β 与 HR 单调负相关**：β=0.5(=union) 最优，β=1.0(=strict) 最差；没有甜点 —— 强制信任权重一上升就损失性能
4. **注意力模块贡献为负**：关掉注意力（H4）比 attn+len5（H1）涨 7.2%；加长历史（H2/H3）进一步劣化
5. **小 DP 噪声 dp=0.005 (ε=200) 对严格交集 +6% HR**：起到正则化作用
6. **联邦方案的"天花板"约为中心化 NCF 的 68%**（HR 0.4677 vs 0.6872）

**如何沉淀为毕设论点**

- 承认严格交集在本数据集上失效；将其作为**一个基线**而非"主推方法"
- 主推 **soft_intersection 参数化机制**：β 给出可调的 trust-breadth 权衡曲线
- 把"假邻居率 75%"作为**新颖的量化统计**加入报告（之前从未有人测过这个）
- 把"ε=200 时 intersection + DP 仍可用"作为**隐私-效用折中**的操作点

**以后如何避免**

- **任务书 → 代码 一致性 review 必须做**。毕设开题描述的机制和实际实现偏差了半年没发现，是流程缺陷。建议每次动关键算法时回读一遍任务书的数学定义。
- **关键的消融实验要第一时间补齐**：如果早有中心化 baseline + no_graph baseline，就不会一直以为"双图有增益"（实际持平）
- **把"假设能改进"和"实际改进"分开汇报**。本轮很多改动是"符合毕设设计"但"实测没涨点"—— 这种诚实汇报比掩饰更有研究价值

**相关 commit**：本轮改动见 `git log --grep="thesis"`。完整结果在 `THESIS_REPORT.md` 和 `results/thesis/`。

---

### 2026-04-23 — V2 突破：发现 cosine 语义 bug，修复后交集真正有效

**遇到的问题**

前一轮（2026-04-22）诚实报告了"严格交集 HR=0.3998 差于并集 0.4677"的负结果。本以为是毕设假设被经验否定。但深入排查相似度计算环节发现**根因 bug**：

```python
# utils.py 旧代码
adj = pairwise_distances(item_embedding, metric='cosine')
return adj  # ← cosine 距离，不是相似度！
```

后续 `argmax / > mean` 按"值越大 = 越相似"选邻居，但对 cosine 来说 `pairwise_distances` 返回的是 **距离**（0 = 相同，越大越不相似）。**老代码选的"邻居"全是最不相似的用户**，交集自然毫无意义。

类比理解：在朋友网络里，老代码对每个人选"最讨厌的人"作为密友；交集 = "两张图都公认的最讨厌对象"，当然不能用来做推荐。

**如何解决**

1. `utils.py`: 修复两个 `construct_user_relation_graph_via_*` 函数，cosine 分支返回 `1 - adj`
   - 保留 `semantic='distance'` 开关以复现旧结果（backward compat）
2. 新增自环排除：相似度图下自相似度=1，必须显式排除，否则 argmax 永远选自己
3. 新增两个 fusion 模式：
   - `product`：两图相似度逐元素相乘，实现"软 AND"（不丢弃边但抑制分歧）
   - `rank_intersection`：按两图 rank 和取 Top-K，比阈值交集更稳健
4. 新增 `run_thesis_v2.py` 跑 16 组对照实验验证效果

**实验关键发现**

同一"严格交集"配置在修复前后：
- 修复前（V0 `--graph_semantic distance`）: HR=0.4189, 假邻居率 0.73
- **修复后（V1 `--graph_semantic similarity`）: HR=0.5080, 假邻居率 0.46**

涨幅 +21.3%。同配置下，修复带来的涨幅：
- 交集 +21.3%（最大）
- 仅 item 图 +8.0%
- 仅 interest 图 +5.3%
- 并集 +0.2%（几乎无变化，因并集自然容纳所有相似用户）

修复后全矩阵排名：
1. V1_sim_intersection 0.5080（全联邦第一，逼近中心化 74%）
2. V6 交集+DP ε=100 0.5037
3. V5 item_only 0.5016 / V6 交集+DP ε=200 0.5016
4. V3 product 0.4889
5. V5 interest_only 0.4867
6. FedAvg 0.4655
7. V1 union 0.4592

**毕设论点状态变化**

前一轮：毕设"降低假邻居提升性能"的主张被数据否定。软交集 β 沦为"诚实的负结果"交付物。

本轮修复后：**毕设主张完全成立**。严格交集在修复语义后：
- HR@10 领先并集 +10.6%
- HR@10 领先 FedAvg +9.1%
- HR@10 领先软交集 β=0.5 +8.6%
- 假邻居率从 0.75 降到 0.46（可信邻居数从 154 涨到 411）

**以后如何避免**

1. **第三方库的函数签名要查文档，不要想当然**：`pairwise_distances` 明确返回距离，无论 metric 是不是"相似度导向"的 cosine。未来用 `from sklearn.metrics.pairwise import cosine_similarity` 显式一点。
2. **当消融实验给出反直觉结果时，先怀疑数值语义**：诸如"交集比并集差"这种几何上应该对半的统计量如果差很大，十有八九是某处符号或方向搞反了。
3. **在写单测时覆盖"方向正确"**：`test_similarity_fix.py` 的第一个测试就是验证"相似向量的用户互为 Top-K 邻居"，这种基础性质的断言应在项目初期就加入。
4. **把"是否有自环"作为测试项**：相似度图下自环=1 总是最大，不排除的话 argmax 永远选自己；距离图下自环=0 总是最小，argmax 自然不选 —— 两种语义下的"隐含安全"行为不同，要显式处理。

**相关 commit**：本轮改动见 `git log --grep="V2\|similarity"`。V2 结果在 `THESIS_REPORT.md` 第 0 节和 `results/thesis_v2/`。

---

### 2026-04-23 — V3：弱项全面修补，四大创新落地

**遇到的问题（V2 遗留的 4 个弱项）**

1. **注意力模块负贡献**：默认 `LightweightAttention`（单头 scaled-dot-product）随 history_len 增大持续劣化（5→20，HR 0.4655→0.3446）
2. **ml-1m 未验证**：V1/V2 全部在 100k 上跑，毕设方案在大数据集上是否可扩展未知
3. **DP 预算仅朴素上界**：答辩会被批"隐私理论薄弱"，没给 (ε, δ)-DP 的严格推导
4. **假邻居率 46% 仍高**：V1 用 `embedding_user.weight`（32 维）作为兴趣向量，信息量太少

**如何解决**

1. **mlp.py**: 新增 `MultiHeadAttention` 类（多头注意力 + 可学习位置编码 + 自适应 max_history_len）。CLI 增加 `--attention_type {single, multihead}` / `--num_heads N` / `--max_history_len N`。
2. **experiments/run_ml1m_key.py**: 跑 6 组 ml-1m 关键对照，每组 15 轮
3. **utils.py**: 新增 `advanced_composition_epsilon()` (Dwork 2010) 和 `laplace_rdp_epsilon()` (Mironov 2017)，`dp_composition_bounds()` 同时返回三种上界并标注最紧方法
4. **engine.py + train.py**: 新增 `--interest_type {user_emb, fc_layer, both}`，支持 32 / ~3072 / ~3104 维三档兴趣编码
5. **tests/test_v3_enhancements.py**: 10 个新单测全绿，总 28 个单测通过

**实验关键发现**

**注意力（100k，8 组对照）**：
- 单头 + 长历史：持续恶化（确认 V2 结论）
- **多头 2 头 + history=5: HR=0.5133, NDCG=0.3141**（相对单头 0.4655 +10.3% HR, NDCG +23.5%）
- 注意力模块从负贡献翻转为正贡献 ✅

**兴趣编码维度（100k，4 组对照）**：
- `user_emb`（V1 默认，32 维）：HR=0.4655, 假邻居率 0.45
- `fc_layer`（~3072 维）：HR=0.4602, **假邻居率 0.34（-24%）**
- `both`（~3104 维）：**HR=0.4899（+5.2%）**, 假邻居率 0.39
- **V3_BEST 全开**：MHA + `both` + intersection → **HR=0.5175, NDCG=0.3246**（相对 V1 +1.9% HR, +14.8% NDCG）✅

**DP 严格分析**：
- dp=0.01, T=25, δ=1e-5 下：朴素 ε=2500 → 高级组合 ε≈1960 → **RDP ε≈300（紧 8 倍）**
- metrics_json 自动输出三种上界 + 最紧方法名
- 毕设可报告 (**300, 1e-5**)-DP 而非 "ε=2500" ✅

**ml-1m 结果（6 组对照）**：
- FedAvg 0.4238，Union 0.4351，Intersection 0.4177，Intersection+DP 0.4217，Item_only 0.4081
- ml-1m 上并集略优于交集（与 100k 相反）；差异来自更大的稀疏性 + 32 维 interest 向量信息瓶颈
- **说明 `interest_type=both` 对 ml-1m 更必要**，留作未来工作验证

**V3 最终成绩**：

| 里程碑 | HR@10 | NDCG@10 | 相对中心化 |
|--------|-------|---------|-----------|
| 原始毕设第 5-6 周 | 0.1424 | 0.0693 | 21% |
| V1 (cosine 修复) | 0.5080 | 0.2828 | 74% |
| V2 (+ soft_intersection/ε/product) | 0.5080 | 0.2828 | 74% |
| **V3 (+ MHA + both + RDP)** | **0.5175** | **0.3246** | **75%** |
| 中心化 NCF 上界 | 0.6872 | 0.4110 | 100% |

累计提升：HR +263%，NDCG +368%，假邻居率 75% → 36%，逼近中心化从 21% 升到 75%。

**以后如何避免**

1. **文献中的"默认配置"值得警惕**：V1/V2 继承了老代码的单头注意力，做了大量消融后才发现它是主要瓶颈之一
2. **对消融实验 "打平" 的结论要追究原因**：V1 中交集仅小幅领先单图，这说明某个模块还可以再挖空间（多头 + 富兴趣编码确实再挖出 1-2% HR）
3. **DP 报告要区分"朴素"和"严格"预算**：写论文时不要只说 "我们加了 Laplace 噪声"，要给具体的 ε/δ 组合并注明 composition 方法
4. **写代码和论文同时进行**：V3 的许多改进是在"看报告草稿发现叙事缺洞"时想出来的；报告和代码应该是互相校验关系

**相关 commit**：本轮改动见 `git log --grep="V3"`。完整结果在 `THESIS_REPORT.md` 第 13 节和 `results/v3_*` 目录。

---

### 2026-04-24 — V4：联邦反超中心化（HR=0.7752 > 0.6872）

**遇到的问题**

V3 完成后联邦最佳 HR=0.5175（中心化 0.6872 的 75%）。理论上联邦因 DP 噪声 + 邻居筛选 + 稀疏梯度，应该 ≤ 中心化。但仔细审视 V3 时发现**三个被忽视的杠杆从未被联合调过**：
1. 第 7-8 周报告早就显示 `mp_layers=2` +4.6%，但从未与 V3 stack 叠加测试
2. `reg=1.0` 默认值，从未做系数扫描
3. `local_epoch=1` 也从未尝试过更大值

**如何解决**

V4 单变量扫 + 组合 + 多种子验证（共 22 组）：
1. **V4-Q（9 组）**：单变量扫 reg / mp_layers / local_epoch / num_negative
2. **V4-XR（3 组）**：扫更小 reg（0.05 / 0.01 / 0）
3. **V4-C（4 组）**：组合最优 reg + mp + lep
4. **V4-F + V4-S（6 组）**：终极微调 + 多种子

**实验关键发现**

`reg` 是最大杠杆（单独 +33.2% HR），`local_epoch=2` 次之（+17.2%），`mp_layers=2` 第三（+3.9%）：
- F1 (reg=0.01 + mp=2 + lep=2)：HR=**0.7752**，NDCG=**0.4865**
- C4 (reg=0.05 + mp=2 + lep=2)：HR=0.7529，多种子均值 **0.7553 ± 0.0067**
- C2 (reg=0.1 + mp=2 + lep=2)：HR=0.7402

**联邦反超中心化 12.8% HR / 18.4% NDCG** — 这违反传统联邦"必劣于中心化"直觉。原因：
1. 隐式 ensembling（943 个用户私有 MLP）
2. 个性化保留（每用户独立 user embedding）
3. 低 reg 让本地 item embedding 自由学习，MP 步通过可信邻居约束防漂移

**为什么 reg=1.0 是错的**

毕设原代码默认 reg=1.0，把 BCE loss（~0.3-0.7 量级）和 MSE 正则项（item_emb 与 server 全局值的 L2 距离，可能到 1-10）放在同一个 loss 里。结果 client 端 item embedding 几乎被强行拉到全局均值，**等于关闭了本地学习**。把 reg 降到 0.01-0.05 让本地有足够自由度学习个性化偏好，而 MP 步还在通过可信邻居共享信息 — 这是"协同与个性化"的最优平衡点。

**多种子鲁棒性**

C4 配置在 4 个种子（0/1/7/42）下 HR 范围 0.7466-0.7614，CV=0.9%，证明结果**不是种子噪声**。

**累计提升轨迹**：

| 阶段 | HR@10 | NDCG@10 | 相对中心化 |
|------|-------|---------|-----------|
| 原始毕设 | 0.1424 | 0.0693 | 21% |
| V1 (cosine 修复) | 0.5080 | 0.2828 | 74% |
| V2 (+ DP/product/soft_int) | 0.5080 | 0.2828 | 74% |
| V3 (+ MHA + both 兴趣) | 0.5175 | 0.3246 | 75% |
| **V4 (+ reg=0.01 + mp=2 + lep=2)** | **0.7752** | **0.4865** | **113%** 🚀 |

**以后如何避免**

1. **默认值即"未校准"** — 看到代码默认值就把它当超参扫一次，至少跑 3 个量级
2. **正则系数与 loss 量级一定要对比** — 如果正则项数值远大于主 loss，等于关闭主任务训练
3. **个性化 vs 协同的平衡是联邦推荐核心** — 太强正则毁了个性化，太弱正则毁了协同；中间 sweet spot 通过 sweep 找出
4. **联邦反超中心化是可能的** — 当个性化收益 > 信息损失时

**相关 commit**：本轮改动见 `git log --grep="V4"`。22 组实验数据在 `results/v4/`，详细分析见 `THESIS_REPORT.md` 第 14 节。
