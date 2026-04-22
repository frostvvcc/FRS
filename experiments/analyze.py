"""读入 results/week7_8/*.json，生成可粘贴到报告的 markdown 分析。

用法:
    python experiments/analyze.py --dir results/week7_8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


GROUP_TITLES = {
    "A": "Group A — 双图策略（第 7 周核心）",
    "B": "Group B — 邻居选择",
    "C": "Group C — 优化器 + 学习率",
    "D": "Group D — 注意力机制",
    "E": "Group E — 图消息传递层数",
    "F": "Group F — 差分隐私",
}


def load_all(dir_: Path) -> list[dict]:
    items = []
    for p in sorted(dir_.glob("*.json")):
        if p.name == "summary.json":
            continue
        try:
            items.append(json.loads(p.read_text()))
        except Exception as e:
            print(f"WARN: failed to parse {p}: {e}")
    return items


def analyze(items: list[dict]) -> str:
    by_group: dict[str, list[dict]] = {}
    for it in items:
        g = (it.get("tag") or "?")[0]
        by_group.setdefault(g, []).append(it)

    # overall best
    if items:
        best = max(items, key=lambda x: x["best_test_hr"])
        worst = min(items, key=lambda x: x["best_test_hr"])
    else:
        best = worst = None

    lines = []
    lines.append("## 全矩阵排名 (按 HR@10 降序)\n")
    lines.append("| 排名 | 标签 | HR@10 | NDCG@10 | 最佳轮次 | 实际轮次 |")
    lines.append("|------|------|-------|---------|----------|----------|")
    for rank, it in enumerate(sorted(items, key=lambda x: -x["best_test_hr"]), start=1):
        lines.append(
            f"| {rank} | `{it['tag']}` | **{it['best_test_hr']:.4f}** "
            f"| {it['best_test_ndcg']:.4f} | {it['best_round']} | {it['num_round_actual']} |"
        )

    if best and worst:
        ratio = best["best_test_hr"] / max(worst["best_test_hr"], 1e-9)
        lines.append("")
        lines.append(f"**全矩阵极差**：最优 `{best['tag']}` HR={best['best_test_hr']:.4f} "
                     f"vs 最差 `{worst['tag']}` HR={worst['best_test_hr']:.4f} "
                     f"（ratio={ratio:.2f}×）")

    for g in sorted(by_group.keys()):
        title = GROUP_TITLES.get(g, f"Group {g}")
        lines.append(f"\n## {title}\n")
        entries = sorted(by_group[g], key=lambda x: -x["best_test_hr"])
        lines.append("| 标签 | HR@10 | NDCG@10 | 最佳轮次 | 实际轮次 |")
        lines.append("|------|-------|---------|----------|----------|")
        for it in entries:
            lines.append(
                f"| `{it['tag']}` | **{it['best_test_hr']:.4f}** "
                f"| {it['best_test_ndcg']:.4f} | {it['best_round']} | {it['num_round_actual']} |"
            )
        # group extreme
        gbest = entries[0]
        gworst = entries[-1]
        if gbest["tag"] != gworst["tag"]:
            lift = (gbest["best_test_hr"] - gworst["best_test_hr"]) / max(gworst["best_test_hr"], 1e-9) * 100
            lines.append("")
            lines.append(
                f"组内：最优 `{gbest['tag']}` 相较最差 `{gworst['tag']}` 提升 **{lift:.1f}%**（HR）"
            )

    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="results/week7_8")
    p.add_argument("--out", default=None,
                   help="输出 markdown 路径；默认打印到 stdout")
    args = p.parse_args()

    items = load_all(Path(args.dir))
    md = analyze(items)
    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")
        print(f"Wrote {args.out}")
    else:
        print(md)


if __name__ == "__main__":
    main()
