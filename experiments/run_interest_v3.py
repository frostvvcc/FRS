"""V3 interest_type 对照：用更丰富的兴趣编码降低假邻居率。"""
from __future__ import annotations
import argparse, json, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def build_matrix():
    base = ["--graph_fusion", "intersection", "--graph_semantic", "similarity"]
    return [
        ("I1_user_emb",   base + ["--interest_type", "user_emb"]),
        ("I2_fc_layer",   base + ["--interest_type", "fc_layer"]),
        ("I3_both",       base + ["--interest_type", "both"]),
        # 对照：并集下的 interest_type 对假邻居率的影响
        ("I4_fc_union",   ["--graph_fusion", "union", "--graph_semantic", "similarity",
                           "--interest_type", "fc_layer"]),
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="100k")
    p.add_argument("--num_round", type=int, default=20)
    p.add_argument("--early_stop", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="results/v3_interest")
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
