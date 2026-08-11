"""CLI entry point for marginal (backward-corrected) exact IPS-GRPO.

Example (mirrors the full-model 1M run, on-policy with replay via the data loader):

    .venv/bin/python -m grpo_experiments.marginal_ips_grpo.train \
        --cfg src/configs/benchmark_dna_cfgs/discrete_branch_lengths/cfg_0.001binsize_50bins_temperature_anneal_0.4.yaml \
        --dataset dataset/benchmark_datasets/DS1_reduced.pickle \
        --output grpo_experiments/full_model \
        --run-name marginal_g4096_1m_full_replay \
        --on-policy-batch-size 3277 --replay-batch-size 819 --replay-buffer-size 4096 \
        --epochs 10000 --steps-per-epoch 1 \
        --device cuda:0 --checkpoint-every 1000

Ablation (recovers plain exact IPS, i.e. no backward correction):

    ... --no-backward-correction

Defaults here differ from ips_grpo to match the recommended setup:
exact propensity, exp_linear reward, adaptive ESS target 0.5, entropy 0.01,
signature outcome level. Any flag still overrides these.
"""

from __future__ import annotations

import sys

from grpo_experiments.ips_grpo.config import build_arg_parser, config_from_args
from grpo_experiments.marginal_ips_grpo.config import MarginalIPSExperimentConfig
from grpo_experiments.marginal_ips_grpo.runner import run_experiment


def parse_config(argv: list[str] | None = None) -> MarginalIPSExperimentConfig:
    argv = sys.argv[1:] if argv is None else argv

    parser = build_arg_parser()
    # Recommended defaults for the marginal setup (still overridable on the CLI).
    parser.set_defaults(
        ips_propensity_mode="exact",
        advantage_reward_mode="exp_linear",
        ips_target_ess_fraction=0.5,
        grpo_entropy_coef=0.01,
        outcome_level="signature",
    )
    parser.add_argument(
        "--no-backward-correction",
        dest="backward_correction",
        action="store_false",
        help="Ablation: use plain exact IPS weight exp(-log P_F(tau)) (no log P_B term).",
    )
    parser.set_defaults(backward_correction=True)

    args = parser.parse_args(argv)
    backward_correction = bool(args.backward_correction)
    base = config_from_args(args)
    return MarginalIPSExperimentConfig.from_ips_config(
        base, backward_correction=backward_correction
    )


def main() -> None:
    exp_cfg = parse_config()
    run_experiment(exp_cfg)


if __name__ == "__main__":
    main()
