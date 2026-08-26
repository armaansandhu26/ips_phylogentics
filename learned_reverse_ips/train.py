#!/usr/bin/env python3
"""
Train phylogenetic tree sampling with learned-reverse IPS-GRPO.

Run from the repository root:

    python -m learned_reverse_ips.train

Compare with:

    python og_code/train.py <cfg> <dataset> <output>   # original PhyloGFN TB
    python -m grpo_experiments.ips_grpo.train          # count/exact IPS-GRPO
"""

from learned_reverse_ips.config import parse_config
from learned_reverse_ips.runner import run_experiment


def main() -> None:
    from src.utils.cpu_threads import apply_cpu_thread_limit

    apply_cpu_thread_limit()
    exp_cfg = parse_config()
    run_experiment(exp_cfg)


if __name__ == "__main__":
    main()
