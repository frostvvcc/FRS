"""软交集 β sweep 实验 —— 在主矩阵完成后追加。

β=1.0 等同严格交集；β 减小保留更多单图邻居，逼近 union。
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def build_matrix():
    return [
        ("S1_soft_beta_0.5",  "S", "软交集 β=0.5（≈均匀 union）",
         ["--graph_fusion", "soft_intersection", "--alpha", "0.5"]),
        ("S2_soft_beta_0.7",  "S", "软交集 β=0.7（中度信任）",
         ["--graph_fusion", "soft_intersection", "--alpha", "0.7"]),
        ("S3_soft_beta_0.85", "S", "软交集 β=0.85（高信任）",
         ["--graph_fusion", "soft_intersection", "--alpha", "0.85"]),
        ("S4_soft_beta_0.95", "S", "软交集 β=0.95（近严格）",
         ["--graph_fusion", "soft_intersection", "--alpha", "0.95"]),
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="100k")
    p.add_argument("--num_round", type=int, default=25)
    p.add_argument("--early_stop", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="results/thesis")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for tag, g, desc, extra in build_matrix():
        json_path = out / f"{tag}.json"
        if json_path.exists():
            print(f"[SKIP] {tag}")
            continue
        cmd = [sys.executable, str(ROOT / "train.py"),
               "--dataset", args.dataset, "--num_round", str(args.num_round),
               "--alias", tag, "--result_tag", tag, "--seed", str(args.seed),
               "--early_stop_patience", str(args.early_stop),
               "--metrics_json", str(json_path)] + extra
        print(f"[RUN] {tag}  {extra}")
        if args.dry_run:
            print("  CMD:", " ".join(cmd))
            continue
        st = time.time()
        r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        dt = time.time() - st
        if r.returncode != 0:
            print(f"[FAIL] {tag} ({dt:.1f}s)\n{r.stderr[-400:]}")
        else:
            print(f"[DONE] {tag} ({dt:.1f}s)")

    print(f"\nTotal {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
