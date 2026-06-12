#!/usr/bin/env python3
"""
Train hybrid IPS-GRPO:
  - sample once under current policy (pi_old),
  - mix in best-tree replay samples,
  - run multiple IPS + IS-weighted update cycles (PPO-style reuse).
"""

from grpo_experiments.hybrid_ips_grpo.config import build_arg_parser, config_from_args
from grpo_experiments.hybrid_ips_grpo.runner import run_experiment


def main() -> None:
    parser = build_arg_parser()
    exp_cfg = config_from_args(parser.parse_args())
    run_experiment(exp_cfg)


if __name__ == "__main__":
    main()
