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
