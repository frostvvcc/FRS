"""V4-COMBO：组合 V4-Q 的胜利者 + 更小 reg + 多种子验证。"""
from __future__ import annotations
import argparse, json, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV_PY = "/tmp/frs-venv2/bin/python"

V3_BASE = [
    "--graph_fusion", "intersection",
    "--graph_semantic", "similarity",
    "--attention_type", "multihead",
    "--num_heads", "2",
    "--history_len", "5",
    "--max_history_len", "20",
    "--interest_type", "both",
    "--dp", "0.01",
]


def matrix():
    return [
        # === XR: 更小 reg 扫 ===
        ("XR1_reg_0.05", "V3 + reg=0.05", V3_BASE + ["--reg", "0.05"]),
        ("XR2_reg_0.01", "V3 + reg=0.01", V3_BASE + ["--reg", "0.01"]),
        ("XR3_reg_0.0",  "V3 + reg=0",    V3_BASE + ["--reg", "0.0"]),

        # === C: 组合最优 reg + mp + lep ===
        ("C1_r0.1_mp2",       "reg=0.1 + mp=2",                V3_BASE + ["--reg", "0.1", "--mp_layers", "2"]),
        ("C2_r0.1_mp2_lep2",  "reg=0.1 + mp=2 + lep=2",        V3_BASE + ["--reg", "0.1", "--mp_layers", "2", "--local_epoch", "2"]),
        ("C3_r0.05_mp2",      "reg=0.05 + mp=2",                V3_BASE + ["--reg", "0.05", "--mp_layers", "2"]),
        ("C4_r0.05_mp2_lep2", "reg=0.05 + mp=2 + lep=2",       V3_BASE + ["--reg", "0.05", "--mp_layers", "2", "--local_epoch", "2"]),

        # === S: 多种子验证（在 C 跑完后挑最佳）===
        # 后续手动运行
    ]


def run_one(tag, desc, extra, dataset, num_round, out_dir, early_stop, seed, dry):
    json_path = out_dir / f"{tag}.json"
    if json_path.exists():
        print(f"[SKIP] {tag}")
        return
    py = VENV_PY if Path(VENV_PY).exists() else sys.executable
    cmd = [py, str(ROOT / "train.py"),
           "--dataset", dataset, "--num_round", str(num_round),
           "--alias", tag, "--result_tag", tag, "--seed", str(seed),
           "--early_stop_patience", str(early_stop),
           "--metrics_json", str(json_path)] + extra
    print(f"[RUN] {tag}  ({desc})")
    if dry:
        print(" CMD:", " ".join(cmd)); return
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    dt = time.time() - t0
    if r.returncode != 0:
        print(f"[FAIL] {tag} ({dt:.1f}s)\n{r.stderr[-400:]}")
    else:
        print(f"[DONE] {tag} ({dt:.1f}s)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="100k")
    p.add_argument("--num_round", type=int, default=30)
    p.add_argument("--early_stop", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="results/exp7_tuning")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for tag, desc, extra in matrix():
        run_one(tag, desc, extra, args.dataset, args.num_round, out,
                args.early_stop, args.seed, args.dry_run)
    print(f"\nTotal {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
