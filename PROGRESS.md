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
