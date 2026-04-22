# 测试指南

本项目的测试分为两层：**单元测试**（验证关键函数）与**冒烟测试**（验证端到端 pipeline 可运行）。

## 运行

```bash
# 所有测试
python -m unittest discover -s tests -v

# 单个测试文件
python -m unittest tests/test_graph.py -v

# 冒烟测试（端到端 1 轮联邦训练）
python tests/test_smoke.py
```

## 测试文件说明

| 文件 | 覆盖范围 |
|------|----------|
| `tests/test_graph.py` | `construct_user_relation_graph_via_item` / `select_topk_neighboehood` / `MP_on_graph` |
| `tests/test_metrics.py` | `MetronAtK` 的 HR@K / NDCG@K 计算 |
| `tests/test_smoke.py` | `train.py` 在 100k 数据集 + 1 轮的端到端运行 |

## 开发约定

- **改代码前**：`python -m unittest discover -s tests` 确认基线全绿
- **改代码后**：再跑一遍确认无回归
- **新增功能**：同步在 `tests/` 增加对应测试用例并更新本文件
- **修 bug**：先写能复现 bug 的测试（红），修复后测试应变绿
