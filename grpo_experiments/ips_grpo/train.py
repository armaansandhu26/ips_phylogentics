#!/usr/bin/env python3
"""
Train phylogenetic tree sampling with IPS-GRPO (inverse probability scaling).

Separate entry point from grpo_experiments.train (PhyloGFN / GRPO).

Quick start (repo root)
-----------------------
python -m grpo_experiments.ips_grpo.train --on-policy-batch-size 64 --epochs 5 --steps-per-epoch 10

Compare with:
  python -m grpo_experiments.train --method phylgfn ...
  python -m grpo_experiments.train --method grpo ...

Outcome-level comparison (topology vs signature p_hat):
  python -m grpo_experiments.ips_grpo.train --list-presets
  python -m grpo_experiments.ips_grpo.train --preset topology_sanity
  python -m grpo_experiments.ips_grpo.train --preset signature_sanity
"""

from grpo_experiments.ips_grpo.config import parse_experiment_config
from grpo_experiments.ips_grpo.runner import run_experiment


def main() -> None:
    from src.utils.cpu_threads import apply_cpu_thread_limit

    apply_cpu_thread_limit()
    exp_cfg = parse_experiment_config()
    run_experiment(exp_cfg)


if __name__ == "__main__":
    main()
