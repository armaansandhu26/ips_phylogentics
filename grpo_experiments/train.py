#!/usr/bin/env python3
"""
Train phylogenetic tree sampling policies: PhyloGFN (TB) vs GRPO.

Quick start (from repo root)
----------------------------
# GRPO — pure on-policy, group size G=64
python -m grpo_experiments.train --method grpo --on-policy-batch-size 64 --epochs 5 --steps-per-epoch 10

# PhyloGFN TB baseline (same model, Trajectory Balance loss)
python -m grpo_experiments.train --method phylgfn --on-policy-batch-size 64 --epochs 5 --steps-per-epoch 10

# GRPO with replay (G = 512 on-policy + 512 replay = 1024)
python -m grpo_experiments.train --method grpo --on-policy-batch-size 512 --replay-batch-size 512

# GRPO with policy importance sampling (same as former is_grpo)
python -m grpo_experiments.train --method grpo --enable-policy-is \
  --resample-rounds 5 --update-cycles 10 --buffer-size 1000

See grpo_experiments/config.py for full parameter documentation.
Runs are saved under grpo_experiments/runs/ by default.
"""

from grpo_experiments.config import build_arg_parser, config_from_args
from grpo_experiments.runner import run_experiment


def main() -> None:
    from src.utils.cpu_threads import apply_cpu_thread_limit

    apply_cpu_thread_limit()
    parser = build_arg_parser()
    exp_cfg = config_from_args(parser.parse_args())
    run_experiment(exp_cfg)


if __name__ == "__main__":
    main()
