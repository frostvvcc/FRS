"""V4 实验矩阵：基于 V3_BEST 进一步压榨性能。

V3_BEST 配置：
  --graph_fusion intersection --graph_semantic similarity
  --attention_type multihead --num_heads 2 --history_len 5
  --max_history_len 20 --interest_type both --dp 0.01

V4 在此基础上扫 mp_layers / reg / local_epoch / num_round。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

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
        # === Q: 已知胜利组合的叠加 ===
        ("Q1_v3_mp2",  "V3 + mp_layers=2",  V3_BASE + ["--mp_layers", "2"]),
        ("Q2_v3_mp3",  "V3 + mp_layers=3",  V3_BASE + ["--mp_layers", "3"]),

        # === R: reg 系数扫 ===
        ("R1_reg_0.1", "V3 + reg=0.1", V3_BASE + ["--reg", "0.1"]),
        ("R2_reg_0.3", "V3 + reg=0.3", V3_BASE + ["--reg", "0.3"]),
        ("R3_reg_0.5", "V3 + reg=0.5", V3_BASE + ["--reg", "0.5"]),

        # === L: 本地 epoch 扫 ===
        ("L1_lep_2",   "V3 + local_epoch=2", V3_BASE + ["--local_epoch", "2"]),
        ("L2_lep_3",   "V3 + local_epoch=3", V3_BASE + ["--local_epoch", "3"]),

        # === N: 负样本数扫 ===
        ("N1_neg_8",   "V3 + num_negative=8",  V3_BASE + ["--num_negative", "8"]),
        ("N2_neg_20",  "V3 + num_negative=20", V3_BASE + ["--num_negative", "20"]),

        # === C: 组合最佳（待 Q/R/L 出结果后再补）===
        # 见 run_v4_combo.py 第二阶段
    ]


VENV_PY = "/tmp/frs-venv2/bin/python"


def run_one(tag: str, desc: str, extra: list[str], dataset: str, num_round: int,
            out_dir: Path, early_stop: int, seed: int, dry: bool):
    json_path = out_dir / f"{tag}.json"
    if json_path.exists():
        print(f"[SKIP] {tag}")
        return None
    py = VENV_PY if Path(VENV_PY).exists() else sys.executable
    cmd = [py, str(ROOT / "train.py"),
           "--dataset", dataset, "--num_round", str(num_round),
           "--alias", tag, "--result_tag", tag, "--seed", str(seed),
           "--early_stop_patience", str(early_stop),
           "--metrics_json", str(json_path)] + extra
    print(f"[RUN] {tag}  ({desc})")
    if dry:
        print(" CMD:", " ".join(cmd))
        return None
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    dt = time.time() - t0
    if r.returncode != 0:
        print(f"[FAIL] {tag} ({dt:.1f}s)\n{r.stderr[-400:]}")
        return None
    print(f"[DONE] {tag} ({dt:.1f}s)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="100k")
    p.add_argument("--num_round", type=int, default=30)
    p.add_argument("--early_stop", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="results/v4")
    p.add_argument("--only", default=None)
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mat = matrix()
    if args.only:
        fs = [s.strip() for s in args.only.split(",") if s.strip()]
        mat = [m for m in mat if any(m[0].startswith(f) for f in fs)]
    print(f"Planned {len(mat)} experiments, num_round={args.num_round}")

    t0 = time.time()
    for tag, desc, extra in mat:
        run_one(tag, desc, extra, args.dataset, args.num_round, out,
                args.early_stop, args.seed, args.dry_run)
    print(f"\nTotal {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
