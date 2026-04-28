"""毕设实验矩阵驱动 —— 验证双图"可信邻居"创新点。

运行分三大组：
  T: 融合策略对比（no_graph / item_only / interest_only / alpha_blend / union / intersection）
  H: 历史序列长度 × 注意力 开/关
  D: 差分隐私预算 ε sweep
  加 Z：中心化上界（一次性）

用法:
    python experiments/run_exp2_dual_graph.py --dataset 100k --num_round 25 --out_dir results/exp2_dual_graph
    python experiments/run_exp2_dual_graph.py --only T         # 只跑 T 组
    python experiments/run_exp2_dual_graph.py --dry_run
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def build_matrix():
    """(tag, group, description, mode, extra_cli_args)"""
    return [
        # ============ T: 融合策略对比（毕设核心） ============
        ("T0_no_graph",      "T", "FedAvg 基线：无图聚合", "fed",
         ["--graph_fusion", "no_graph"]),
        ("T1_item_only",     "T", "仅 item 图（行为关联）", "fed",
         ["--graph_fusion", "item_only"]),
        ("T2_interest_only", "T", "仅 interest 图（兴趣语义）", "fed",
         ["--graph_fusion", "interest_only"]),
        ("T3_alpha_0.5",     "T", "旧实现：alpha=0.5 加权融合", "fed",
         ["--graph_fusion", "alpha", "--alpha", "0.5"]),
        ("T4_union",         "T", "两图并集（消融）", "fed",
         ["--graph_fusion", "union"]),
        ("T5_intersection",  "T", "🌟 可信邻居交集（毕设创新）", "fed",
         ["--graph_fusion", "intersection"]),

        # ============ H: 历史长度 × 注意力 ============
        ("H1_attn_len5",     "H", "有注意力 history_len=5（默认）", "fed",
         ["--history_len", "5", "--graph_fusion", "intersection"]),
        ("H2_attn_len10",    "H", "有注意力 history_len=10", "fed",
         ["--history_len", "10", "--graph_fusion", "intersection"]),
        ("H3_attn_len20",    "H", "有注意力 history_len=20", "fed",
         ["--history_len", "20", "--graph_fusion", "intersection"]),
        ("H4_noattn_len5",   "H", "无注意力 history_len=5", "fed",
         ["--no_attention", "--layers", "64,32,16,8", "--history_len", "5",
          "--graph_fusion", "intersection"]),
        ("H5_noattn_len20",  "H", "无注意力 history_len=20", "fed",
         ["--no_attention", "--layers", "64,32,16,8", "--history_len", "20",
          "--graph_fusion", "intersection"]),

        # ============ D: 差分隐私预算 ε sweep（intersection 基线） ============
        ("D1_dp_0",          "D", "ε=∞ (dp=0, 无隐私保护)", "fed",
         ["--graph_fusion", "intersection", "--dp", "0.0"]),
        ("D2_dp_0.005",      "D", "ε=200 (dp=0.005, 强隐私)", "fed",
         ["--graph_fusion", "intersection", "--dp", "0.005"]),
        ("D3_dp_0.01",       "D", "ε=100 (dp=0.01)", "fed",
         ["--graph_fusion", "intersection", "--dp", "0.01"]),
        ("D4_dp_0.05",       "D", "ε=20 (dp=0.05, 中隐私)", "fed",
         ["--graph_fusion", "intersection", "--dp", "0.05"]),
        ("D5_dp_0.1",        "D", "ε=10 (dp=0.1, 弱隐私)", "fed",
         ["--graph_fusion", "intersection", "--dp", "0.1"]),

        # ============ Z: 中心化 NCF 上界对照 ============
        ("Z_centralized",    "Z", "中心化 NCF（上界参考）", "centralized", []),
    ]


def run_one(tag: str, mode: str, extra_args: list[str], dataset: str, num_round: int,
            out_dir: Path, early_stop: int, seed: int, dry_run: bool) -> dict | None:
    json_path = out_dir / f"{tag}.json"
    if json_path.exists():
        print(f"  [SKIP] {tag} already exists → {json_path}")
        with json_path.open() as f:
            return json.load(f)

    if mode == 'fed':
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
    elif mode == 'centralized':
        cmd = [
            sys.executable, str(ROOT / "centralized_train.py"),
            "--dataset", dataset,
            "--num_epoch", str(num_round),
            "--result_tag", tag,
            "--seed", str(seed),
            "--early_stop_patience", str(max(early_stop, 3)),
            "--metrics_json", str(json_path),
        ] + extra_args
    else:
        raise ValueError(mode)

    print(f"  [RUN]  {tag}  mode={mode}  args={extra_args}")
    if dry_run:
        print(f"       CMD: {' '.join(cmd)}")
        return None

    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    dt = time.time() - t0
    if r.returncode != 0:
        print(f"  [FAIL] {tag}  exit={r.returncode}  ({dt:.1f}s)")
        print(r.stderr[-800:])
        return None
    print(f"  [DONE] {tag}  ({dt:.1f}s)")
    with json_path.open() as f:
        return json.load(f)


def write_summary(results: list[dict], matrix_info: list[tuple], out_dir: Path):
    desc_by_tag = {m[0]: m[2] for m in matrix_info}

    csv_path = out_dir / "summary.csv"
    md_path = out_dir / "summary.md"
    fieldnames = ["tag", "group", "description", "method", "best_test_hr", "best_test_ndcg",
                  "best_round", "num_round_actual", "total_upload_MB", "epsilon_total",
                  "final_false_neighbor_ratio", "final_avg_trusted_neighbors"]

    rows = []
    for r in results:
        if r is None:
            continue
        tag = r.get("tag", "")
        upload_mb = r.get("total_upload_bytes", 0) / (1024 * 1024)
        eps = r.get("epsilon_total_naive")
        rows.append({
            "tag": tag,
            "group": tag[0] if tag else "",
            "description": desc_by_tag.get(tag, ""),
            "method": r.get("method", "federated"),
            "best_test_hr": f"{r['best_test_hr']:.4f}",
            "best_test_ndcg": f"{r['best_test_ndcg']:.4f}",
            "best_round": r["best_round"],
            "num_round_actual": r["num_round_actual"],
            "total_upload_MB": f"{upload_mb:.2f}",
            "epsilon_total": (f"{eps:.2f}" if isinstance(eps, (int, float)) else "∞"),
            "final_false_neighbor_ratio": f"{r.get('final_false_neighbor_ratio', 0.0):.3f}",
            "final_avg_trusted_neighbors": f"{r.get('final_avg_trusted_neighbors', 0.0):.1f}",
        })

    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  Wrote {csv_path}")

    lines = [
        "# 毕设实验汇总",
        "",
        f"总共 {len(rows)} 个实验。",
        "",
        "| 标签 | 组 | 说明 | 方法 | HR@10 | NDCG@10 | 最佳轮 | 实际轮 | 上行 (MB) | ε (朴素上界) | 假邻居率 | 可信邻居数 |",
        "|------|----|------|------|-------|---------|--------|--------|-----------|--------------|----------|------------|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['tag']}` | {row['group']} | {row['description']} | {row['method']} "
            f"| **{row['best_test_hr']}** | **{row['best_test_ndcg']}** "
            f"| {row['best_round']} | {row['num_round_actual']} "
            f"| {row['total_upload_MB']} | {row['epsilon_total']} "
            f"| {row['final_false_neighbor_ratio']} | {row['final_avg_trusted_neighbors']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  Wrote {md_path}")


def plot_curves(results: list[dict], out_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  [WARN] matplotlib missing: {e}")
        return

    groups: dict[str, list[dict]] = {}
    for r in results:
        if r is None:
            continue
        groups.setdefault((r.get('tag') or '?')[0], []).append(r)

    fig, axes = plt.subplots(nrows=len(groups), ncols=1,
                             figsize=(10, 3 * len(groups)), sharex=False)
    if len(groups) == 1:
        axes = [axes]
    for ax, (g, items) in zip(axes, sorted(groups.items())):
        for r in items:
            ax.plot(r["hr_list"], label=r["tag"], linewidth=1.5)
        ax.set_title(f"Group {g} — HR@10 per round")
        ax.set_xlabel("Round / Epoch")
        ax.set_ylabel("HR@10")
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "curves.png", dpi=140)
    plt.close(fig)
    print(f"  Wrote {out_dir/'curves.png'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="100k")
    p.add_argument("--num_round", type=int, default=25)
    p.add_argument("--early_stop", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="results/exp2_dual_graph")
    p.add_argument("--only", default=None,
                   help="逗号分隔，如 'T' 或 'T,D' 或 'T5_intersection'")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    mat = build_matrix()
    if args.only:
        filters = [s.strip() for s in args.only.split(",") if s.strip()]
        mat = [m for m in mat if any(m[0].startswith(f) or m[0] == f for f in filters)]

    print(f"Planned {len(mat)} experiments in {args.out_dir}")
    print(f"Dataset={args.dataset}  num_round={args.num_round}  seed={args.seed}")

    t0 = time.time()
    results = []
    for tag, grp, desc, mode, extra in mat:
        r = run_one(tag, mode, extra, args.dataset, args.num_round, out,
                    args.early_stop, args.seed, args.dry_run)
        results.append(r)

    dt = time.time() - t0
    print(f"\nAll runs finished in {dt/60:.1f} min.")
    if not args.dry_run:
        write_summary(results, mat, out)
        plot_curves(results, out)


if __name__ == "__main__":
    main()
