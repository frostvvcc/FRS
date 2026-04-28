"""毕设 V2 实验矩阵：cosine 语义修复 + 新融合模式 + 更紧的 Top-K。

目标：验证"修复假邻居方向 + 合适邻居策略" → 可信邻居真正优于并集/FedAvg。
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
        # === V0：旧 bug 行为基线（复现原 T5 严格交集）===
        ("V0_legacy_intersection", "V", "旧 bug（distance）+ intersection 对比基线", "fed",
         ["--graph_fusion", "intersection", "--graph_semantic", "distance"]),
        ("V0_legacy_union", "V", "旧 bug + union", "fed",
         ["--graph_fusion", "union", "--graph_semantic", "distance"]),

        # === V1：修复语义后，相同配置再对比 ===
        ("V1_sim_intersection", "V", "修复语义 + 严格交集（Top-K 阈值）", "fed",
         ["--graph_fusion", "intersection", "--graph_semantic", "similarity"]),
        ("V1_sim_union", "V", "修复语义 + 并集", "fed",
         ["--graph_fusion", "union", "--graph_semantic", "similarity"]),
        ("V1_sim_no_graph", "V", "FedAvg baseline（对照）", "fed",
         ["--graph_fusion", "no_graph", "--graph_semantic", "similarity"]),

        # === V2：紧 Top-K 交集（更高质量可信邻居）===
        ("V2_topk10_intersection", "V", "Top-K=10 严格交集（修复语义）", "fed",
         ["--graph_fusion", "intersection", "--graph_semantic", "similarity",
          "--neighborhood_size", "10"]),
        ("V2_topk20_intersection", "V", "Top-K=20 严格交集", "fed",
         ["--graph_fusion", "intersection", "--graph_semantic", "similarity",
          "--neighborhood_size", "20"]),
        ("V2_topk50_intersection", "V", "Top-K=50 严格交集", "fed",
         ["--graph_fusion", "intersection", "--graph_semantic", "similarity",
          "--neighborhood_size", "50"]),

        # === V3：product 模式（双图相似度逐元素乘）===
        ("V3_product_thresh", "V", "product 模式（阈值筛选）", "fed",
         ["--graph_fusion", "product", "--graph_semantic", "similarity"]),
        ("V3_product_topk20", "V", "product 模式 + Top-K=20", "fed",
         ["--graph_fusion", "product", "--graph_semantic", "similarity",
          "--neighborhood_size", "20"]),

        # === V4：rank_intersection（基于 rank 和的 Top-K）===
        ("V4_rank_topk20", "V", "rank_intersection Top-K=20（按两图 rank 和）", "fed",
         ["--graph_fusion", "rank_intersection", "--graph_semantic", "similarity",
          "--neighborhood_size", "20"]),
        ("V4_rank_topk50", "V", "rank_intersection Top-K=50", "fed",
         ["--graph_fusion", "rank_intersection", "--graph_semantic", "similarity",
          "--neighborhood_size", "50"]),

        # === V5：对照 - 修复语义下 item_only / interest_only / alpha ===
        ("V5_item_only_sim", "V", "修复语义 + 仅 item 图（对照）", "fed",
         ["--graph_fusion", "item_only", "--graph_semantic", "similarity"]),
        ("V5_interest_only_sim", "V", "修复语义 + 仅 interest 图（对照）", "fed",
         ["--graph_fusion", "interest_only", "--graph_semantic", "similarity"]),

        # === V6：DP 在修复后的 intersection 上的效果 ===
        ("V6_int_dp0.005", "V", "修复语义 + intersection + dp=0.005", "fed",
         ["--graph_fusion", "intersection", "--graph_semantic", "similarity", "--dp", "0.005"]),
        ("V6_int_dp0.01", "V", "修复语义 + intersection + dp=0.01", "fed",
         ["--graph_fusion", "intersection", "--graph_semantic", "similarity", "--dp", "0.01"]),
    ]


def run_one(tag, mode, extra, dataset, num_round, out_dir, early_stop, seed, dry):
    json_path = out_dir / f"{tag}.json"
    if json_path.exists():
        with json_path.open() as f:
            return json.load(f)
    cmd = [sys.executable, str(ROOT / "train.py"),
           "--dataset", dataset, "--num_round", str(num_round),
           "--alias", tag, "--result_tag", tag, "--seed", str(seed),
           "--early_stop_patience", str(early_stop),
           "--metrics_json", str(json_path)] + extra
    print(f"[RUN] {tag}")
    if dry:
        print("  CMD:", " ".join(cmd))
        return None
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    dt = time.time() - t0
    if r.returncode != 0:
        print(f"[FAIL] {tag} ({dt:.1f}s)\n{r.stderr[-500:]}")
        return None
    print(f"[DONE] {tag} ({dt:.1f}s)")
    with json_path.open() as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="100k")
    p.add_argument("--num_round", type=int, default=25)
    p.add_argument("--early_stop", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="results/exp3_cosine_fix")
    p.add_argument("--only", default=None)
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mat = build_matrix()
    if args.only:
        fs = [s.strip() for s in args.only.split(",")]
        mat = [m for m in mat if any(m[0].startswith(f) for f in fs)]

    print(f"Planned {len(mat)} experiments")
    t0 = time.time()
    results = []
    for tag, g, desc, mode, extra in mat:
        results.append(run_one(tag, mode, extra, args.dataset, args.num_round,
                               out, args.early_stop, args.seed, args.dry_run))
    print(f"\nTotal {(time.time()-t0)/60:.1f} min")

    if not args.dry_run:
        # 写 summary
        rows = []
        for r in results:
            if r is None: continue
            tag = r.get('tag', '')
            fr = r.get('final_false_neighbor_ratio', 0.0)
            trusted = r.get('final_avg_trusted_neighbors', 0.0)
            rows.append({
                'tag': tag, 'hr': f"{r['best_test_hr']:.4f}",
                'ndcg': f"{r['best_test_ndcg']:.4f}",
                'best_round': r['best_round'], 'actual': r['num_round_actual'],
                'false_ratio': f"{fr:.3f}", 'trusted_n': f"{trusted:.1f}",
            })
        with (out / 'summary.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['tag','hr','ndcg','best_round','actual','false_ratio','trusted_n'])
            w.writeheader(); w.writerows(rows)
        print(f"Wrote {out/'summary.csv'}")


if __name__ == "__main__":
    main()
