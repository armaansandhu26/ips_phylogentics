from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
from typing import Literal

from grpo_experiments.ips_grpo.config import (
    IPSExperimentConfig,
    build_arg_parser,
    config_from_args,
)

from learned_reverse_ips.checkpoint import METHOD

RewardTarget = Literal["likelihood", "shifted_linear"]
ReversePolicyType = Literal["mlp", "uniform"]
DEFAULT_REVERSE_POLICY_TYPE: ReversePolicyType = "mlp"


def _remove_parser_dests(parser: argparse.ArgumentParser, *dest_names: str) -> None:
    blocked = set(dest_names)
    parser._actions = [
        action for action in parser._actions if action.dest not in blocked
    ]


@dataclass
class LearnedReverseExperimentConfig(IPSExperimentConfig):
    """Configuration for learned-reverse IPS-GRPO training."""

    only_train_tree_model: bool = False
    reward_target: RewardTarget = "likelihood"
    reverse_policy_type: ReversePolicyType = DEFAULT_REVERSE_POLICY_TYPE
    reverse_lr: float = 1e-3
    reverse_train_epochs: int = 4
    reverse_grad_clip_norm: float = 1.0
    reverse_hidden_size: int = 128
    reverse_num_layers: int = 2
    advantage_normalization: str = "running"
    running_scale_decay: float = 0.99
    running_advantage_clip: float = 10.0
    running_log_ratio_clip: float = 20.0
    post_train: bool = True
    post_train_sample_size: int = 10_000
    post_train_sample_batch_size: int | None = None

    @property
    def method(self) -> str:
        return METHOD

    @classmethod
    def from_base(
        cls,
        base: IPSExperimentConfig,
        *,
        reward_target: RewardTarget,
        reverse_policy_type: ReversePolicyType,
        reverse_lr: float,
        reverse_train_epochs: int,
        reverse_grad_clip_norm: float,
        reverse_hidden_size: int,
        reverse_num_layers: int,
        advantage_normalization: str,
        running_scale_decay: float,
        running_advantage_clip: float,
        running_log_ratio_clip: float,
        post_train: bool,
        post_train_sample_size: int,
        post_train_sample_batch_size: int | None,
    ) -> "LearnedReverseExperimentConfig":
        kwargs = {field.name: getattr(base, field.name) for field in fields(IPSExperimentConfig)}
        kwargs.pop("only_train_tree_model", None)
        return cls(
            **kwargs,
            only_train_tree_model=False,
            reward_target=reward_target,
            reverse_policy_type=reverse_policy_type,
            reverse_lr=reverse_lr,
            reverse_train_epochs=reverse_train_epochs,
            reverse_grad_clip_norm=reverse_grad_clip_norm,
            reverse_hidden_size=reverse_hidden_size,
            reverse_num_layers=reverse_num_layers,
            advantage_normalization=advantage_normalization,
            running_scale_decay=running_scale_decay,
            running_advantage_clip=running_advantage_clip,
            running_log_ratio_clip=running_log_ratio_clip,
            post_train=post_train,
            post_train_sample_size=post_train_sample_size,
            post_train_sample_batch_size=post_train_sample_batch_size,
        )


def _prepare_ips_args(args: argparse.Namespace) -> None:
    """IPS config_from_args expects flags removed from this CLI."""
    if not hasattr(args, "full_model"):
        args.full_model = False
    if not hasattr(args, "tree_only"):
        args.tree_only = False


def parse_config(argv: list[str] | None = None) -> LearnedReverseExperimentConfig:
    parser = build_arg_parser()
    _remove_parser_dests(parser, "full_model", "tree_only")
    parser.description = (
        "Learned-reverse IPS-GRPO for phylogenetic tree sampling. "
        "weight(tau) = R(x) * q_phi(tau | x) / P_F(tau)."
    )
    parser.set_defaults(
        cfg_path=(
            "src/configs/benchmark_dna_cfgs/discrete_branch_lengths/"
            "cfg_0.001binsize_50bins_temperature_anneal_0.4.yaml"
        ),
        dataset_path="dataset/benchmark_datasets/DS1_reduced.pickle",
        output_root="learned_reverse_ips/experiments",
        run_name="learned_reverse_5taxa",
        epochs=4_000,
        steps_per_epoch=1,
        on_policy_batch_size=128,
        replay_batch_size=0,
        disable_replay=True,
        grpo_lr=1e-4,
        grpo_entropy_coef=0.0,
        grpo_num_iterations=1,
        outcome_level="topology",
        policy_loss_mode="ppo",
        print_every=25,
        checkpoint_every=500,
        rollout_chunk_size=128,
        post_train=True,
        post_train_sample_size=10_000,
    )
    group = parser.add_argument_group("learned reverse proposal")
    group.add_argument(
        "--reverse-policy",
        choices=("mlp", "uniform"),
        default=DEFAULT_REVERSE_POLICY_TYPE,
        help=(
            "mlp: fit q_phi by MLE after each PPO step; "
            "uniform: frozen GFlowNet-style uniform P_B(tau|x) from log_paths_pb."
        ),
    )
    group.add_argument(
        "--reward-target",
        choices=("likelihood", "shifted_linear"),
        default="likelihood",
        help=(
            "likelihood: q*(x) proportional to exp(log L(x)); "
            "shifted_linear: q*(x) proportional to the positive shifted log score."
        ),
    )
    group.add_argument("--reverse-lr", type=float, default=1e-3)
    group.add_argument("--reverse-train-epochs", type=int, default=4)
    group.add_argument("--reverse-grad-clip-norm", type=float, default=1.0)
    group.add_argument("--reverse-hidden-size", type=int, default=128)
    group.add_argument("--reverse-num-layers", type=int, default=2)
    group.add_argument(
        "--advantage-normalization",
        choices=("batch", "running"),
        default="running",
    )
    group.add_argument("--running-scale-decay", type=float, default=0.99)
    group.add_argument("--running-advantage-clip", type=float, default=10.0)
    group.add_argument("--running-log-ratio-clip", type=float, default=20.0)
    group = parser.add_argument_group("post-training pipeline")
    group.add_argument(
        "--skip-post-train",
        action="store_false",
        dest="post_train",
        help=(
            "Skip post-training evaluation (training curves, 10k sampling, "
            "probability-vs-reward plots)."
        ),
    )
    group.add_argument(
        "--post-train-sample-size",
        type=int,
        default=10_000,
        help="Terminal trees to sample after training (default: 10000).",
    )
    group.add_argument(
        "--post-train-sample-batch-size",
        type=int,
        default=None,
        help="Sampling batch size (default: on-policy batch size).",
    )
    args = parser.parse_args(argv)
    _prepare_ips_args(args)
    base = config_from_args(args)
    return LearnedReverseExperimentConfig.from_base(
        base,
        reward_target=args.reward_target,
        reverse_policy_type=args.reverse_policy,
        reverse_lr=args.reverse_lr,
        reverse_train_epochs=args.reverse_train_epochs,
        reverse_grad_clip_norm=args.reverse_grad_clip_norm,
        reverse_hidden_size=args.reverse_hidden_size,
        reverse_num_layers=args.reverse_num_layers,
        advantage_normalization=args.advantage_normalization,
        running_scale_decay=args.running_scale_decay,
        running_advantage_clip=args.running_advantage_clip,
        running_log_ratio_clip=args.running_log_ratio_clip,
        post_train=args.post_train,
        post_train_sample_size=args.post_train_sample_size,
        post_train_sample_batch_size=args.post_train_sample_batch_size,
    )
