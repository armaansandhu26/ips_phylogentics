from __future__ import annotations

import argparse

from config import REWARD_PROFILES, TrainConfig


def add_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--n-leaves", type=int, default=5)
    parser.add_argument("--reward-profile", choices=tuple(REWARD_PROFILES), default="phylo_peaked")
    parser.add_argument("--beta", type=float, default=None, help="Override reward-profile beta (dynamic-range knob)")
    parser.add_argument("--reward-mode", choices=("linear", "exp", "log_score"), default=None, help="Override reward-profile mode")

    parser.add_argument("--num-updates", type=int, default=500)
    parser.add_argument("--group-size", type=int, default=256)
    parser.add_argument("--num-groups", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--train-epochs", type=int, default=1)
    parser.add_argument("--clip-ratio", type=float, default=0.4)
    parser.add_argument("--entropy-coef", type=float, default=0.01)

    parser.add_argument(
        "--propensity-mode",
        choices=("none", "exact", "marginal", "count"),
        default="exact",
        help="none=GRPO, exact=trajectory IPS (biased to R*m), marginal=backward-corrected (R), count=legacy",
    )
    parser.add_argument("--max-inverse-weight", type=float, default=-1.0, help="Cap on 1/P_F(tau); <=0 disables")
    parser.add_argument("--naive-expspace", action="store_true", help="Use unsafe exp-space reward*weight (overflows for large beta)")


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    max_w = None if args.max_inverse_weight is None or args.max_inverse_weight <= 0 else args.max_inverse_weight
    return TrainConfig(
        n_leaves=args.n_leaves,
        reward_profile=args.reward_profile,
        beta=args.beta,
        reward_mode=args.reward_mode,
        num_updates=args.num_updates,
        group_size=args.group_size,
        num_groups=args.num_groups,
        lr=args.lr,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        seed=args.seed,
        log_every=args.log_every,
        train_epochs=args.train_epochs,
        clip_ratio=args.clip_ratio,
        entropy_coef=args.entropy_coef,
        propensity_mode=args.propensity_mode,
        max_inverse_weight=max_w,
        naive_expspace=args.naive_expspace,
    )


def print_train_header(config: TrainConfig, catalog_summary: dict) -> None:
    profile = REWARD_PROFILES[config.reward_profile]
    print(
        f"Merge (phylo) toy — n_leaves={config.n_leaves}: "
        f"{catalog_summary['num_histories']} histories -> {catalog_summary['num_topologies']} topologies "
        f"(multiplicity {catalog_summary['multiplicity_min']}..{catalog_summary['multiplicity_max']})"
    )
    print(f"  reward_profile={config.reward_profile}: {profile['description']}")
    print(f"  reward_mode={config.resolved_mode()}  beta={config.resolved_beta()}")
    print(f"  propensity_mode={config.propensity_mode}  max_inverse_weight={config.max_inverse_weight}  naive_expspace={config.naive_expspace}")
    print(f"  group_size={config.group_size}  num_groups={config.num_groups}  clip={config.clip_ratio}  epochs={config.train_epochs}  entropy_coef={config.entropy_coef}")
