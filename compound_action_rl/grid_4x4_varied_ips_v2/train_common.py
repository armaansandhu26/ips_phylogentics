from __future__ import annotations

import argparse

from config import COLOR_PROFILES, TrainConfig
from grid_paths import make_env, num_trajectories


def add_train_args(parser: argparse.ArgumentParser) -> None:
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
        "--max-inverse-weight",
        type=float,
        default=2560.0,
        help="Cap on 1/p_theta(tau); set negative to disable",
    )
    parser.add_argument(
        "--propensity-mode",
        choices=("exact", "count"),
        default="exact",
        help="IPS weight: exact p_theta(tau) or legacy count-based",
    )
    parser.add_argument(
        "--detach-color-rep",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Detach state_rep before color policy loss",
    )
    parser.add_argument("--aux-pos-coef", type=float, default=0.1, help="Position aux loss weight")
    parser.add_argument(
        "--color-credit",
        choices=("trajectory", "counterfactual"),
        default="trajectory",
        help="Color advantage: shared IPS trajectory credit (matching) or per-step counterfactual (maximizing — collapses colorings)",
    )
    parser.add_argument(
        "--color-profile",
        choices=tuple(COLOR_PROFILES),
        default="default",
        help="Red/green spatial field preset",
    )
    parser.add_argument(
        "--trainable",
        choices=("both", "path_only", "color_only"),
        default="both",
    )


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    max_w = None if args.max_inverse_weight < 0 else args.max_inverse_weight
    return TrainConfig(
        color_profile=args.color_profile,
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
        max_inverse_weight=max_w,
        propensity_mode=args.propensity_mode,
        detach_color_rep=args.detach_color_rep,
        aux_pos_coef=args.aux_pos_coef,
        color_credit=args.color_credit,  # type: ignore[arg-type]
        trainable=args.trainable,  # type: ignore[arg-type]
    )


def print_train_header(config: TrainConfig, *, agent: str) -> None:
    profile = COLOR_PROFILES[config.color_profile]
    env = make_env(**config.profile_kwargs())
    n_traj = num_trajectories(env)
    print(f"{agent} — {config.grid_size}×{config.grid_size} grid, {n_traj} trajectories, profile={config.color_profile}")
    print(f"  {profile['description']}")
    print(f"  red_center={profile['red_center']}  green_center={profile['green_center']}  T={profile['temperature']}")
    print(f"  group_size={config.group_size}  num_groups={config.num_groups}")
    print(f"  propensity={config.propensity_mode}  clip={config.clip_ratio}  epochs={config.train_epochs}  entropy_coef={config.entropy_coef}")
    print(f"  detach_color_rep={config.detach_color_rep}  aux_pos_coef={config.aux_pos_coef}  color_credit={config.color_credit}")
