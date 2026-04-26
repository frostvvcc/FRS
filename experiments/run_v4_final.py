"""V4-FINAL：在 C4 周围微调 + 多种子验证 V4_BEST 的鲁棒性。"""
from __future__ import annotations
import argparse, subprocess, sys, time
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
    # 进一步扫 reg + lep 组合，以及种子鲁棒性
    base_C4 = V3_BASE + ["--reg", "0.05", "--mp_layers", "2", "--local_epoch", "2"]
    base_C5 = V3_BASE + ["--reg", "0.01", "--mp_layers", "2", "--local_epoch", "2"]
    base_C6 = V3_BASE + ["--reg", "0.05", "--mp_layers", "2", "--local_epoch", "3"]
    return [
        # 微调
        ("F1_r0.01_mp2_lep2", "reg=0.01 + mp=2 + lep=2", base_C5),
        ("F2_r0.05_mp2_lep3", "reg=0.05 + mp=2 + lep=3", base_C6),
        ("F3_r0.1_mp2_lep3",  "reg=0.1 + mp=2 + lep=3",
         V3_BASE + ["--reg", "0.1", "--mp_layers", "2", "--local_epoch", "3"]),

        # 多种子验证 C4 (V4_BEST 候选)
        ("S1_C4_seed0",  "C4 seed=0",  base_C4),
        ("S2_C4_seed1",  "C4 seed=1",  base_C4),
        ("S3_C4_seed7",  "C4 seed=7",  base_C4),
    ]


def run_one(tag, desc, extra, dataset, num_round, out_dir, early_stop, seed, dry):
    if tag.startswith("S"):
        # multi-seed
        seed_map = {"S1_C4_seed0": 0, "S2_C4_seed1": 1, "S3_C4_seed7": 7}
        seed = seed_map.get(tag, seed)
    json_path = out_dir / f"{tag}.json"
    if json_path.exists():
        print(f"[SKIP] {tag}"); return
    py = VENV_PY if Path(VENV_PY).exists() else sys.executable
    cmd = [py, str(ROOT / "train.py"),
           "--dataset", dataset, "--num_round", str(num_round),
           "--alias", tag, "--result_tag", tag, "--seed", str(seed),
           "--early_stop_patience", str(early_stop),
           "--metrics_json", str(json_path)] + extra
    print(f"[RUN] {tag}  ({desc}, seed={seed})")
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
    p.add_argument("--out_dir", default="results/v4")
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
