"""
Configuration for hybrid GRPO:
  fresh pi_old samples + best-tree replay samples, then multi-cycle IS updates.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from typing import Literal, Optional

OutcomeLevel = Literal["signature", "topology"]


@dataclass
class HybridExperimentConfig:
    """Knobs for hybrid best-tree replay + policy-IS GRPO."""

    cfg_path: str = (
        "src/configs/benchmark_dna_cfgs/discrete_branch_lengths/"
        "cfg_0.001binsize_50bins_temperature_anneal_0.4.yaml"
    )
    dataset_path: str = "dataset/benchmark_datasets/DS1_reduced.pickle"
    output_root: str = "grpo_experiments/runs"
    run_name: Optional[str] = None
    seed: int = 0
    device: Optional[str] = None

    # Outer/inner loop schedule.
    epochs: int = 100
    steps_per_epoch: int = 20
    resample_rounds: Optional[int] = None
    update_cycles: Optional[int] = None

    # Hybrid sampling at each resample round.
    fresh_buffer_size: int = 512
    replay_sample_size: int = 512
    best_tree_buffer_size: int = 2048
    best_trees_topology_only: bool = False
    replay_warmstart_samples: int = 0

    # Replay annealing (Panel G): linear replay_start → replay_end over resample rounds.
    replay_anneal_start: Optional[int] = None
    replay_anneal_end: Optional[int] = None
    replay_anneal_total_batch: int = 512

    # Rollout/replay execution.
    rollout_chunk_size: int = 64

    # GRPO optimizer.
    grpo_lr: float = 1e-4
    grpo_max_grad_norm: float = 1.0
    grpo_advantage_eps: float = 1e-8
    grpo_clip_eps: float = 0.2
    grpo_clip_eps_high: float | None = None
    entropy_coef: float = 0.01

    # Tracking/logging.
    outcome_level: OutcomeLevel = "topology"
    print_every: int = 1
    checkpoint_every: int = 0
    log_trajectories: bool = True
    trajectory_flush_every: int = 20

    # Resume.
    resume_from: Optional[str] = None
    resume_checkpoint: Optional[str] = None

    extra: dict = field(default_factory=dict)

    @property
    def method(self) -> str:
        return "hybrid_grpo"

    @property
    def enable_policy_is(self) -> bool:
        # Used by shared cfg override helper.
        return True

    @property
    def effective_resample_rounds(self) -> int:
        return self.epochs if self.resample_rounds is None else self.resample_rounds

    @property
    def effective_update_cycles(self) -> int:
        return self.steps_per_epoch if self.update_cycles is None else self.update_cycles

    @property
    def total_batch_size(self) -> int:
        return self.fresh_buffer_size + self.replay_sample_size

    @property
    def effective_replay_batch_size(self) -> int:
        # We handle replay manually in this runner.
        return 0

    @property
    def replay_buffer_size(self) -> int:
        # Alias expected by shared cfg helpers.
        return self.best_tree_buffer_size

    @property
    def mini_batch_splits(self) -> int:
        # Alias expected by shared cfg helpers.
        return 1

    @property
    def on_policy_batch_size(self) -> int:
        # Alias expected by shared cfg helpers.
        return self.fresh_buffer_size

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["method"] = self.method
        payload["enable_policy_is"] = self.enable_policy_is
        payload["effective_resample_rounds"] = self.effective_resample_rounds
        payload["effective_update_cycles"] = self.effective_update_cycles
        payload["total_batch_size"] = self.total_batch_size
        return payload

    def save_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Hybrid GRPO: sample fresh trees under pi_old + replay best trees, "
            "then run multiple IS-weighted update cycles."
        )
    )
    p.add_argument("--cfg", dest="cfg_path", default=HybridExperimentConfig.cfg_path)
    p.add_argument("--dataset", dest="dataset_path", default=HybridExperimentConfig.dataset_path)
    p.add_argument("--output", dest="output_root", default=HybridExperimentConfig.output_root)
    p.add_argument("--run-name", dest="run_name", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)

    g = p.add_argument_group("schedule")
    g.add_argument("--epochs", type=int, default=100)
    g.add_argument("--steps-per-epoch", type=int, default=20)
    g.add_argument("--resample-rounds", type=int, default=None)
    g.add_argument("--update-cycles", type=int, default=None)

    g = p.add_argument_group("hybrid sampling")
    g.add_argument("--fresh-buffer-size", type=int, default=512)
    g.add_argument("--replay-sample-size", type=int, default=512)
    g.add_argument("--best-tree-buffer-size", type=int, default=2048)
    g.add_argument("--best-trees-topology-only", action="store_true")
    g.add_argument(
        "--replay-warmstart-samples",
        type=int,
        default=0,
        help="If >0, seed best-tree buffer from env.sample() before first round.",
    )
    g.add_argument(
        "--replay-anneal-start",
        type=int,
        default=None,
        help="If set with --replay-anneal-end, linearly anneal replay count over rounds.",
    )
    g.add_argument(
        "--replay-anneal-end",
        type=int,
        default=None,
        help="Final replay count when annealing (see --replay-anneal-start).",
    )
    g.add_argument(
        "--replay-anneal-total-batch",
        type=int,
        default=512,
        help="Fixed fresh+replay batch size during replay annealing.",
    )

    g = p.add_argument_group("optimizer / IS")
    g.add_argument("--grpo-lr", type=float, default=1e-4)
    g.add_argument("--grpo-max-grad-norm", type=float, default=1.0)
    g.add_argument("--grpo-advantage-eps", type=float, default=1e-8)
    g.add_argument(
        "--entropy-coef",
        type=float,
        default=0.01,
        help="Entropy bonus: L = L_policy - coef * H(pi). Set 0 to disable (TRL uses KL via beta instead).",
    )
    g.add_argument(
        "--grpo-clip-eps",
        type=float,
        default=0.2,
        help="PPO clip epsilon for GRPO surrogate (TRL default 0.2). 0 disables clipping.",
    )
    g.add_argument("--grpo-clip-eps-high", type=float, default=None)
    g.add_argument("--rollout-chunk-size", type=int, default=64)

    g = p.add_argument_group("tracking / logging")
    g.add_argument("--outcome-level", choices=["signature", "topology"], default="topology")
    g.add_argument("--print-every", type=int, default=1)
    g.add_argument("--checkpoint-every", type=int, default=0)
    g.add_argument(
        "--log-trajectories",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record every sampled tree signature/topology/score during training.",
    )
    g.add_argument(
        "--trajectory-flush-every",
        type=int,
        default=20,
        help="Flush trajectory JSONL buffers every N resample rounds.",
    )

    g = p.add_argument_group("resume")
    g.add_argument("--resume-from", default=None)
    g.add_argument("--resume-checkpoint", default=None)
    return p


def config_from_args(args: argparse.Namespace) -> HybridExperimentConfig:
    return HybridExperimentConfig(
        cfg_path=args.cfg_path,
        dataset_path=args.dataset_path,
        output_root=args.output_root,
        run_name=args.run_name,
        seed=args.seed,
        device=args.device,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        resample_rounds=args.resample_rounds,
        update_cycles=args.update_cycles,
        fresh_buffer_size=args.fresh_buffer_size,
        replay_sample_size=args.replay_sample_size,
        best_tree_buffer_size=args.best_tree_buffer_size,
        best_trees_topology_only=args.best_trees_topology_only,
        replay_warmstart_samples=args.replay_warmstart_samples,
        replay_anneal_start=args.replay_anneal_start,
        replay_anneal_end=args.replay_anneal_end,
        replay_anneal_total_batch=args.replay_anneal_total_batch,
        rollout_chunk_size=args.rollout_chunk_size,
        grpo_lr=args.grpo_lr,
        grpo_max_grad_norm=args.grpo_max_grad_norm,
        grpo_advantage_eps=args.grpo_advantage_eps,
        grpo_clip_eps=args.grpo_clip_eps,
        grpo_clip_eps_high=args.grpo_clip_eps_high,
        entropy_coef=args.entropy_coef,
        outcome_level=args.outcome_level,
        print_every=args.print_every,
        checkpoint_every=args.checkpoint_every,
        log_trajectories=args.log_trajectories,
        trajectory_flush_every=args.trajectory_flush_every,
        resume_from=args.resume_from,
        resume_checkpoint=args.resume_checkpoint,
    )

