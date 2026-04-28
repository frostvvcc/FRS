"""ml-1m 关键对照：证明方法可扩展到更大数据集。

只跑最重要的 6 个配置，num_round=15 + 早停。预计 CPU 3-4 小时。
"""
from __future__ import annotations
import argparse, json, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num_round", type=int, default=15)
    p.add_argument("--early_stop", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="results/exp6_ml1m")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    matrix = [
        ("M1_sim_intersection", "fed",
         ["--dataset", "ml-1m", "--graph_fusion", "intersection",
          "--graph_semantic", "similarity"]),
        ("M1_sim_no_graph", "fed",
         ["--dataset", "ml-1m", "--graph_fusion", "no_graph",
          "--graph_semantic", "similarity"]),
        ("M1_sim_union", "fed",
         ["--dataset", "ml-1m", "--graph_fusion", "union",
          "--graph_semantic", "similarity"]),
        ("M1_item_only", "fed",
         ["--dataset", "ml-1m", "--graph_fusion", "item_only",
          "--graph_semantic", "similarity"]),
        ("M1_int_dp_0.01", "fed",
         ["--dataset", "ml-1m", "--graph_fusion", "intersection",
          "--graph_semantic", "similarity", "--dp", "0.01"]),
        ("M_centralized", "centralized",
         ["--dataset", "ml-1m"]),
    ]

    t0 = time.time()
    for tag, mode, extra in matrix:
        json_path = out / f"{tag}.json"
        if json_path.exists():
            print(f"[SKIP] {tag}")
            continue
        script = "train.py" if mode == "fed" else "centralized_train.py"
        rounds_flag = "--num_round" if mode == "fed" else "--num_epoch"
        cmd = [sys.executable, str(ROOT / script),
               rounds_flag, str(args.num_round),
               "--seed", str(args.seed),
               "--result_tag", tag, "--alias", tag,
               "--metrics_json", str(json_path)] + extra
        if mode == "fed":
            cmd.extend(["--early_stop_patience", str(args.early_stop)])
        print(f"[RUN] {tag}  {extra}")
        if args.dry_run:
            print("  ", " ".join(cmd))
            continue
        st = time.time()
        r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        dt = time.time() - st
        if r.returncode != 0:
            print(f"[FAIL] {tag} ({dt:.1f}s)\n{r.stderr[-500:]}")
        else:
            print(f"[DONE] {tag} ({dt:.1f}s)")

    print(f"\nTotal {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
