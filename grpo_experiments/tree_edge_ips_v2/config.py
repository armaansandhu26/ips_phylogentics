from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Literal


PropensityMode = Literal["exact", "count"]
EdgeCreditMode = Literal["trajectory", "counterfactual"]


@dataclass
class TrainConfig:
    """Phylo-scale defaults for on-policy tree/edge IPS-GRPO v2."""

    cfg_path: str = (
        "src/configs/benchmark_dna_cfgs/discrete_branch_lengths/"
        "cfg_0.001binsize_50bins_temperature_anneal_0.4.yaml"
    )
    dataset_path: str = "dataset/benchmark_datasets/DS1_reduced.pickle"
    output_root: str = "grpo_experiments/tree_edge_ips_v2/runs"
    run_name: str | None = None
    seed: int = 0
    device: str | None = None

    num_updates: int = 10000
    num_groups: int = 1
    group_size: int = 4096
    train_epochs: int = 1
    rollout_chunk_size: int = 2048

    lr: float = 1e-4
    clip_eps: float = 0.2
    entropy_coef: float = 1e-3
    max_grad_norm: float = 1.0
    advantage_eps: float = 1e-8

    propensity_mode: PropensityMode = "exact"
    max_inverse_weight: float = 40960.0
    ips_weight_temperature: float = 1.0
    snips_truncate_ratio: float | None = None
    ips_target_ess_fraction: float | None = 0.5
    backward_correction: bool = True
    count_propensity_eps: float = 1e-8
    edge_credit: EdgeCreditMode = "trajectory"
    detach_edge_rep: bool = True
    only_train_tree_model: bool = False
    aux_loss_weight: float = 0.1

    eval_every: int = 500
    eval_episodes: int = 2048
    eval_batch_size: int = 128
    plot_episodes: int = 25000
    checkpoint_every: int = 1000
    print_every: int = 10
    cpu_threads: int = 0

    reward_c: float = 0.0
    reward_scale: float = 1.0
    reward_mode: Literal["exp_linear", "log_reward", "raw"] = "log_reward"

    def batch_size(self) -> int:
        return self.num_groups * self.group_size

    @property
    def total_batch_size(self) -> int:
        return self.batch_size()

    @property
    def epochs(self) -> int:
        return self.num_updates

    @property
    def steps_per_epoch(self) -> int:
        return 1

    @property
    def on_policy_batch_size(self) -> int:
        return self.batch_size()

    @property
    def effective_replay_batch_size(self) -> int:
        return 0

    @property
    def replay_buffer_size(self) -> int:
        return 0

    @property
    def mini_batch_splits(self) -> int:
        return 1

    @property
    def enable_policy_is(self) -> bool:
        return False

    @property
    def edge_rep_grad_alpha(self) -> float:
        return 0.0 if self.detach_edge_rep else 1.0

    def to_dict(self) -> dict:
        data = asdict(self)
        data["batch_size"] = self.batch_size()
        data["edge_rep_grad_alpha"] = self.edge_rep_grad_alpha
        return data

    def save_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train split tree/edge IPS-GRPO v2 (phylo-scale defaults).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--cfg", dest="cfg_path", default=TrainConfig.cfg_path)
    p.add_argument("--dataset", dest="dataset_path", default=TrainConfig.dataset_path)
    p.add_argument("--output", dest="output_root", default=TrainConfig.output_root)
    p.add_argument("--run-name", default=None)
    p.add_argument("--seed", type=int, default=TrainConfig.seed)
    p.add_argument("--device", default=None)

    p.add_argument("--num-updates", type=int, default=TrainConfig.num_updates)
    p.add_argument("--num-groups", type=int, default=TrainConfig.num_groups)
    p.add_argument("--group-size", type=int, default=TrainConfig.group_size)
    p.add_argument("--train-epochs", type=int, default=TrainConfig.train_epochs)
    p.add_argument("--rollout-chunk-size", type=int, default=TrainConfig.rollout_chunk_size)

    p.add_argument("--lr", type=float, default=TrainConfig.lr)
    p.add_argument("--clip-eps", type=float, default=TrainConfig.clip_eps)
    p.add_argument("--entropy-coef", type=float, default=TrainConfig.entropy_coef)
    p.add_argument("--max-grad-norm", type=float, default=TrainConfig.max_grad_norm)
    p.add_argument("--advantage-eps", type=float, default=TrainConfig.advantage_eps)

    p.add_argument("--propensity-mode", choices=["exact", "count"], default=TrainConfig.propensity_mode)
    p.add_argument("--max-inverse-weight", type=float, default=TrainConfig.max_inverse_weight)
    p.add_argument(
        "--ips-weight-temperature",
        type=float,
        default=TrainConfig.ips_weight_temperature,
        help=(
            "exact mode: beta in (0,1] on log inverse propensity. "
            "1.0 = pure exact IPS (usually inert/collapsing on phylo); "
            "~0.2-0.5 keeps ESS healthy. Overridden by --ips-target-ess-fraction."
        ),
    )
    p.add_argument(
        "--snips-truncate-ratio",
        type=float,
        default=None,
        help="Clip SNIPS weights (mean 1) to this multiple of the mean, then renormalize.",
    )
    p.add_argument(
        "--ips-target-ess-fraction",
        type=float,
        default=TrainConfig.ips_target_ess_fraction,
        help=(
            "Auto-solve temperature beta each group so SNIPS ESS stays at this "
            "fraction of group size (recommended: 0.5). Pass 0 or negative to disable."
        ),
    )
    p.add_argument(
        "--backward-correction",
        action=argparse.BooleanOptionalAction,
        default=TrainConfig.backward_correction,
        help=(
            "For exact IPS, weight by exp(-(log P_F - log P_B)) so the denominator "
            "tracks final-object marginal probability rather than a single merge history."
        ),
    )
    p.add_argument("--count-propensity-eps", type=float, default=TrainConfig.count_propensity_eps)
    p.add_argument("--edge-credit", choices=["trajectory", "counterfactual"], default=TrainConfig.edge_credit)
    p.add_argument("--detach-edge-rep", action=argparse.BooleanOptionalAction, default=TrainConfig.detach_edge_rep)
    p.add_argument(
        "--only-train-tree-model",
        action=argparse.BooleanOptionalAction,
        default=TrainConfig.only_train_tree_model,
        help="If true, freeze/use fixed edge lengths. Default false trains the full tree+edge model.",
    )
    p.add_argument("--aux-loss-weight", type=float, default=TrainConfig.aux_loss_weight)

    p.add_argument("--eval-every", type=int, default=TrainConfig.eval_every)
    p.add_argument("--eval-episodes", type=int, default=TrainConfig.eval_episodes)
    p.add_argument("--eval-batch-size", type=int, default=TrainConfig.eval_batch_size)
    p.add_argument("--plot-episodes", type=int, default=TrainConfig.plot_episodes)
    p.add_argument("--checkpoint-every", type=int, default=TrainConfig.checkpoint_every)
    p.add_argument("--print-every", type=int, default=TrainConfig.print_every)
    p.add_argument(
        "--cpu-threads",
        type=int,
        default=TrainConfig.cpu_threads,
        help=(
            "Cap PyTorch/BLAS CPU threads for this process. "
            "0 uses PHYLOGFN_CPU_THREADS/YAML/default resolution."
        ),
    )

    p.add_argument("--reward-c", type=float, default=TrainConfig.reward_c)
    p.add_argument("--reward-scale", type=float, default=TrainConfig.reward_scale)
    p.add_argument("--reward-mode", choices=["exp_linear", "log_reward", "raw"], default=TrainConfig.reward_mode)
    return p


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    fields = TrainConfig.__dataclass_fields__
    kwargs = {name: getattr(args, name) for name in fields if hasattr(args, name)}
    # argparse default for optional float may be the TrainConfig sentinel; normalize disable.
    target = kwargs.get("ips_target_ess_fraction")
    if target is not None and float(target) <= 0.0:
        kwargs["ips_target_ess_fraction"] = None
    return TrainConfig(**kwargs)
