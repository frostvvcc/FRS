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
