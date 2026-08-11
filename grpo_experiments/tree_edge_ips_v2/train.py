#!/usr/bin/env python3
from __future__ import annotations

from grpo_experiments.tree_edge_ips_v2.config import build_arg_parser, config_from_args
from grpo_experiments.tree_edge_ips_v2.run_experiment import run_experiment


def main() -> None:
    parser = build_arg_parser()
    config = config_from_args(parser.parse_args())
    run_experiment(config)


if __name__ == "__main__":
    main()
