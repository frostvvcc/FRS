"""第 7-8 周实验驱动脚本。

用法：
    python experiments/run_exp1_baseline_hparams.py --dataset 100k --num_round 25 \
        --out_dir results/exp1_baseline_hparams [--only GROUP] [--dry_run]

输出：
    results/exp1_baseline_hparams/
      ├── <tag>.json          每个实验的完整 metrics（hr_list, ndcg_list 等）
      ├── summary.csv         所有实验的关键指标汇总
      ├── summary.md          Markdown 表格
      └── curves.png          HR@10 学习曲线对比图
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def build_matrix():
    """返回 [(tag, group, description, extra_cli_args)]。

    extra_cli_args 是要追加给 train.py 的参数列表。
    """
    return [
        # =========== Group A: 双图策略对比（核心创新） ===========
        ("A0_no_graph", "A", "FedAvg 基线：跳过图构建，直接平均 item embedding",
         ["--no_graph"]),
        ("A1_alpha_1.0", "A", "仅 item graph (alpha=1.0)",
         ["--alpha", "1.0"]),
        ("A2_alpha_0.7", "A", "双图融合 alpha=0.7（偏 item）",
         ["--alpha", "0.7"]),
        ("A3_alpha_0.5", "A", "双图融合 alpha=0.5（均衡，默认）",
         ["--alpha", "0.5"]),
        ("A4_alpha_0.3", "A", "双图融合 alpha=0.3（偏 interest）",
         ["--alpha", "0.3"]),
        ("A5_alpha_0.0", "A", "仅 interest graph (alpha=0.0)",
         ["--alpha", "0.0"]),

        # =========== Group B: 邻居选择策略 ===========
        ("B1_topk_5",  "B", "Top-K=5",     ["--neighborhood_size", "5"]),
        ("B2_topk_10", "B", "Top-K=10",    ["--neighborhood_size", "10"]),
        ("B3_topk_20", "B", "Top-K=20",    ["--neighborhood_size", "20"]),
        ("B4_threshold_1.0", "B", "阈值方式 threshold=1.0（默认）",
         ["--neighborhood_size", "0", "--neighborhood_threshold", "1.0"]),
        ("B5_threshold_1.2", "B", "阈值方式 threshold=1.2（更严）",
         ["--neighborhood_size", "0", "--neighborhood_threshold", "1.2"]),

        # =========== Group C: 优化器 & 学习率（修复 lr 爆炸） ===========
        ("C1_sgd_default", "C", "SGD baseline: lr=0.1 lr_eta=80（原始）", []),
        ("C2_sgd_eta10",   "C", "SGD lr=0.1 lr_eta=10（降低 eta）",
         ["--lr_eta", "10"]),
        ("C3_sgd_override","C", "SGD 直接指定 lr_u=0.01 lr_i=0.1",
         ["--lr_u", "0.01", "--lr_i", "0.1"]),
        ("C4_adam_001",    "C", "Adam lr=0.001 lr_u=0.01 lr_i=0.01",
         ["--optimizer", "adam", "--lr", "0.001", "--lr_u", "0.01", "--lr_i", "0.01"]),
        ("C5_adam_all_001","C", "Adam 统一 lr=0.001",
         ["--optimizer", "adam", "--lr", "0.001", "--lr_u", "0.001", "--lr_i", "0.001"]),

        # =========== Group D: 注意力机制 ===========
        ("D1_no_attention", "D", "关闭注意力（layers=[64,32,16,8]）",
         ["--no_attention", "--layers", "64,32,16,8"]),

        # =========== Group E: 图消息传递层数 ===========
        ("E1_mp_1", "E", "MP 层数=1（默认）", ["--mp_layers", "1"]),
        ("E2_mp_2", "E", "MP 层数=2",        ["--mp_layers", "2"]),
        ("E3_mp_3", "E", "MP 层数=3",        ["--mp_layers", "3"]),

        # =========== Group F: 差分隐私 ===========
        ("F1_dp_0",    "F", "DP 噪声=0（默认）", ["--dp", "0.0"]),
        ("F2_dp_001",  "F", "DP 噪声=0.01",     ["--dp", "0.01"]),
        ("F3_dp_005",  "F", "DP 噪声=0.05",     ["--dp", "0.05"]),
    ]


def run_one(tag: str, extra_args: list[str], dataset: str, num_round: int,
            out_dir: Path, early_stop: int, seed: int, dry_run: bool) -> dict | None:
    json_path = out_dir / f"{tag}.json"
    if json_path.exists():
        print(f"  [SKIP] {tag} 已存在 → {json_path}")
        with json_path.open() as f:
            return json.load(f)

    cmd = [
        sys.executable, str(ROOT / "train.py"),
        "--dataset", dataset,
        "--num_round", str(num_round),
        "--alias", tag,
        "--result_tag", tag,
        "--seed", str(seed),
        "--early_stop_patience", str(early_stop),
        "--metrics_json", str(json_path),
    ] + extra_args

    print(f"  [RUN]  {tag}  args={extra_args}")
    if dry_run:
        print(f"       CMD: {' '.join(cmd)}")
        return None

    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    dt = time.time() - t0
    if r.returncode != 0:
        print(f"  [FAIL] {tag}  exit={r.returncode}  ({dt:.1f}s)")
        print("  --- stderr tail ---")
        print(r.stderr[-800:])
        return None

    print(f"  [DONE] {tag}  ({dt:.1f}s)")
    with json_path.open() as f:
        return json.load(f)


def write_summary(results: list[dict], out_dir: Path):
    csv_path = out_dir / "summary.csv"
    md_path = out_dir / "summary.md"

    fieldnames = [
        "tag", "group", "description", "best_test_hr", "best_test_ndcg",
        "best_round", "num_round_actual",
    ]

    rows = []
    for r in results:
        if r is None:
            continue
        tag = r.get("tag")
        rows.append({
            "tag": tag,
            "group": tag[0] if tag else "",
            "description": GROUP_DESC.get(tag, ""),
            "best_test_hr": f"{r['best_test_hr']:.4f}",
            "best_test_ndcg": f"{r['best_test_ndcg']:.4f}",
            "best_round": r["best_round"],
            "num_round_actual": r["num_round_actual"],
        })

    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  Wrote {csv_path}")

    lines = [
        "# 第 7-8 周实验汇总",
        "",
        f"Dataset: 见各实验 JSON。汇总 {len(rows)} 条实验。",
        "",
        "| 标签 | 组 | 说明 | 最佳 HR@10 | 最佳 NDCG@10 | 最佳轮次 | 总轮次 |",
        "|------|----|------|-----------|--------------|----------|--------|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['tag']}` | {row['group']} | {row['description']} "
            f"| **{row['best_test_hr']}** | **{row['best_test_ndcg']}** "
            f"| {row['best_round']} | {row['num_round_actual']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  Wrote {md_path}")


GROUP_DESC: dict[str, str] = {}


def plot_curves(results: list[dict], out_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  [WARN] matplotlib not available: {e}; skip plot")
        return

    # 按 group 分图
    groups: dict[str, list[dict]] = {}
    for r in results:
        if r is None:
            continue
        tag = r.get("tag") or ""
        g = tag[0] if tag else "?"
        groups.setdefault(g, []).append(r)

    fig, axes = plt.subplots(nrows=len(groups), ncols=1,
                             figsize=(10, 3 * len(groups)), sharex=False)
    if len(groups) == 1:
        axes = [axes]
    for ax, (g, items) in zip(axes, sorted(groups.items())):
        for r in items:
            ax.plot(r["hr_list"], label=r["tag"], linewidth=1.4)
        ax.set_title(f"Group {g} — HR@10 per round")
        ax.set_xlabel("Round")
        ax.set_ylabel("HR@10")
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    path = out_dir / "curves.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  Wrote {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="100k")
    p.add_argument("--num_round", type=int, default=25)
    p.add_argument("--early_stop", type=int, default=0,
                   help="early_stop_patience for all runs (0=off)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="results/exp1_baseline_hparams")
    p.add_argument("--only", default=None,
                   help="只跑指定 group 前缀（如 A / A,C / A3_alpha_0.5）")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix = build_matrix()
    for tag, g, desc, _ in matrix:
        GROUP_DESC[tag] = desc

    if args.only:
        filters = [s.strip() for s in args.only.split(",") if s.strip()]
        def keep(tag: str) -> bool:
            return any(tag.startswith(f) or tag == f for f in filters)
        matrix = [m for m in matrix if keep(m[0])]

    print(f"Planned {len(matrix)} experiments in {args.out_dir}")
    print(f"Dataset={args.dataset}  num_round={args.num_round}  seed={args.seed}")

    t_start = time.time()
    results = []
    for tag, group, desc, extra in matrix:
        r = run_one(tag, extra, args.dataset, args.num_round, out_dir,
                    args.early_stop, args.seed, args.dry_run)
        results.append(r)

    elapsed = time.time() - t_start
    print(f"\nAll runs finished in {elapsed/60:.1f} min.")
    if not args.dry_run:
        write_summary(results, out_dir)
        plot_curves(results, out_dir)


if __name__ == "__main__":
    main()
