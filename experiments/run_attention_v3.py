"""V3 注意力对照：multi-head + positional encoding 能否让注意力转为正贡献。

基线：V1_sim_intersection = 0.5080（无注意力的 H4 = 0.4284，即弱项 1）。
目标：multi-head + longer history 应至少打平单点积，最好超越。
"""
from __future__ import annotations
import argparse, json, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def build_matrix():
    base = ["--graph_fusion", "intersection", "--graph_semantic", "similarity"]
    return [
        # 基线：单头 attention（V1 默认）
        ("A1_single_len5",  base + ["--history_len", "5", "--attention_type", "single"]),
        ("A1_single_len10", base + ["--history_len", "10", "--attention_type", "single"]),
        ("A1_single_len20", base + ["--history_len", "20", "--attention_type", "single"]),
        # 多头 attention（V3 升级）
        ("A2_mha2_len5",    base + ["--history_len", "5", "--attention_type", "multihead",
                                    "--num_heads", "2", "--max_history_len", "20"]),
        ("A2_mha4_len10",   base + ["--history_len", "10", "--attention_type", "multihead",
                                    "--num_heads", "4", "--max_history_len", "20"]),
        ("A2_mha4_len20",   base + ["--history_len", "20", "--attention_type", "multihead",
                                    "--num_heads", "4", "--max_history_len", "32"]),
        ("A2_mha8_len20",   base + ["--history_len", "20", "--attention_type", "multihead",
                                    "--num_heads", "8", "--max_history_len", "32"]),
        # 无注意力基线
        ("A0_no_attn_len5", base + ["--history_len", "5", "--no_attention",
                                    "--layers", "64,32,16,8"]),
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="100k")
    p.add_argument("--num_round", type=int, default=20)
    p.add_argument("--early_stop", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="results/v3_attention")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    for tag, extra in build_matrix():
        json_path = out / f"{tag}.json"
        if json_path.exists():
            print(f"[SKIP] {tag}"); continue
        cmd = [sys.executable, str(ROOT / "train.py"),
               "--dataset", args.dataset, "--num_round", str(args.num_round),
               "--seed", str(args.seed), "--alias", tag, "--result_tag", tag,
               "--early_stop_patience", str(args.early_stop),
               "--metrics_json", str(json_path)] + extra
        print(f"[RUN] {tag}  {extra}")
        if args.dry_run:
            print(" ", " ".join(cmd)); continue
        st = time.time()
        r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        dt = time.time() - st
        if r.returncode != 0:
            print(f"[FAIL] {tag} ({dt:.0f}s)\n{r.stderr[-400:]}")
        else:
            print(f"[DONE] {tag} ({dt:.0f}s)")

    print(f"\nTotal {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
