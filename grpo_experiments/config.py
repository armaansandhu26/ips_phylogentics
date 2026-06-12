"""
Experiment configuration for PhyloGFN vs GRPO comparisons.

Terminology
-----------
PhyloGFN (baseline)
    Trajectory Balance (TB) GFlowNet trained to sample trees proportional to
    reward R(x) = exp(log_likelihood). See Zhou et al., arXiv:2310.08774.

GRPO (Group Relative Policy Optimization)
    Policy-gradient method: sample a *group* of trees, compute group-relative
    advantages, update forward policy log P_F(tau). See Shao et al. (2024).

GRPO group size (G)
    Number of terminal trees in one policy update. Advantages are computed as:
        A_i = (r_i - mean(r)) / (std(r) + eps)
    where r_i = log R(x_i) and the mean/std are taken over the group.

    In this codebase:
        G = on_policy_batch_size + replay_batch_size

    - on_policy_batch_size: fresh trees sampled from the current policy
    - replay_batch_size: trees replayed from a best-tree buffer (0 = pure on-policy)

    Larger G → lower-variance advantage estimates, but each step is more expensive.
    Typical values: 64–1024 for sanity runs, 512–4096 for longer training.

Replay buffer
    Keeps the top-scoring trees seen so far. Each step, replay_batch_size trees
    are sampled from this buffer and included in the batch (with backward
    trajectories regenerated). Helps GRPO revisit high-reward regions.

Outcome (for diversity metrics; IPS-GRPO will use these next)
    - signature: full tree = topology + discretized branch lengths + score
    - topology:  tree shape only (ignores branch lengths)

    Duplicate fraction measures mode collapse: if the policy keeps resampling
    the same trees, duplicate_fraction → 1.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from typing import Literal, Optional


TrainingMethod = Literal["phylgfn", "grpo"]
OutcomeLevel = Literal["signature", "topology"]


@dataclass
class ExperimentConfig:
    """All knobs for a single training run."""

    # --- run identity ---
    method: TrainingMethod = "grpo"
    cfg_path: str = (
        "src/configs/benchmark_dna_cfgs/discrete_branch_lengths/"
        "cfg_0.001binsize_50bins_temperature_anneal_0.4.yaml"
    )
    dataset_path: str = "dataset/benchmark_datasets/DS1_reduced.pickle"
    output_root: str = "grpo_experiments/runs"
    run_name: Optional[str] = None  # subfolder under output_root; timestamp added if None
    seed: int = 0
    device: Optional[str] = None  # auto: cuda:0 or cpu

    # --- training schedule ---
    epochs: int = 100
    steps_per_epoch: int = 20

    # --- batch / GRPO group ---
    on_policy_batch_size: int = 64
    """Fresh on-policy trees per step (= GFN_BATCH_SIZE in PhyloGFN config)."""

    replay_batch_size: int = 0
    """Best-tree replay trees per step (= BEST_STATE_BATCH_SIZE). 0 disables replay."""

    replay_buffer_size: int = 512
    """Capacity of the best-tree heap (= BEST_TREES_BUFFER_SIZE)."""

    mini_batch_splits: int = 1
    """Split each step into this many mini-updates (usually 1)."""

    disable_replay: bool = False
    """Force replay_batch_size = 0 regardless of replay_batch_size setting."""

    # --- GRPO optimizer ---
    grpo_lr: float = 1e-4
    grpo_max_grad_norm: float = 1.0
    grpo_advantage_eps: float = 1e-8
    grpo_clip_eps: float = 0.2
    """PPO-style epsilon for TRL GRPO surrogate: clip(r, 1-eps, 1+eps). 0 disables clipping."""

    grpo_clip_eps_high: float | None = None
    """Optional asymmetric upper clip; defaults to grpo_clip_eps."""

    grpo_entropy_coef: float = 0.0
    """Entropy bonus (replaces KL-to-reference in TRL). 0 = disabled."""

    grpo_num_iterations: int = 1
    """Reuse the same on-policy rollout for this many optimizer steps (TRL mu). Hybrid uses update_cycles."""

    enable_policy_is: bool = False
    """If True: sample buffer under behavior policy, replay with pi_new/pi_old weights."""

    resample_rounds: Optional[int] = None
    update_cycles: Optional[int] = None
    buffer_size: Optional[int] = None
    rollout_chunk_size: int = 64

    # --- diversity / outcome tracking ---
    outcome_level: OutcomeLevel = "topology"
    """Which tree ID to use for duplicate/outcome statistics."""

    # --- logging ---
    print_every: int = 1
    checkpoint_every: int = 0
    """Save checkpoint every N epochs. 0 = only save final checkpoint."""

    resume_from: Optional[str] = None
    """Existing run directory to continue (appends metrics, reuses best_trees.pt)."""

    resume_checkpoint: Optional[str] = None
    """Checkpoint file inside resume_from (default: latest/final)."""

    # --- derived (not set from CLI directly) ---
    extra: dict = field(default_factory=dict)

    @property
    def effective_resample_rounds(self) -> int:
        return self.epochs if self.resample_rounds is None else self.resample_rounds

    @property
    def effective_update_cycles(self) -> int:
        return self.steps_per_epoch if self.update_cycles is None else self.update_cycles

    @property
    def effective_buffer_size(self) -> int:
        return self.on_policy_batch_size if self.buffer_size is None else self.buffer_size

    @property
    def grpo_group_size(self) -> int:
        """Effective GRPO group size G for one update."""
        if self.enable_policy_is:
            return self.effective_buffer_size
        replay = 0 if self.disable_replay else self.replay_batch_size
        return self.on_policy_batch_size + replay

    @property
    def effective_replay_batch_size(self) -> int:
        return 0 if self.disable_replay else self.replay_batch_size

    def to_dict(self) -> dict:
        d = asdict(self)
        d["grpo_group_size"] = self.grpo_group_size
        d["effective_replay_batch_size"] = self.effective_replay_batch_size
        return d

    def save_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Train a phylogenetic tree sampling policy with PhyloGFN (TB) or GRPO.\n\n"
            "Compare methods:\n"
            "  phylgfn  — Trajectory Balance GFlowNet (Zhou et al. 2023)\n"
            "  grpo     — Group Relative Policy Optimization (Shao et al. 2024)\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument(
        "--method",
        choices=["phylgfn", "grpo"],
        default="grpo",
        help="Training algorithm. 'phylgfn' = TB baseline; 'grpo' = policy gradient.",
    )
    p.add_argument(
        "--cfg",
        dest="cfg_path",
        default=ExperimentConfig.cfg_path,
        help="PhyloGFN model/env YAML (architecture, reward, exploration).",
    )
    p.add_argument("--dataset", dest="dataset_path", default=ExperimentConfig.dataset_path)
    p.add_argument("--output", dest="output_root", default=ExperimentConfig.output_root)
    p.add_argument("--run-name", dest="run_name", default=None,
                   help="Optional label appended to the timestamped run folder.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--steps-per-epoch", type=int, default=20)

    g = p.add_argument_group("batch / GRPO group")
    g.add_argument(
        "--on-policy-batch-size",
        type=int,
        default=64,
        help="Fresh trees sampled per step. With replay disabled, this IS the GRPO group size G.",
    )
    g.add_argument(
        "--replay-batch-size",
        type=int,
        default=0,
        help="Trees replayed from best-tree buffer per step. G = on_policy + replay.",
    )
    g.add_argument("--replay-buffer-size", type=int, default=512)
    g.add_argument("--mini-batch-splits", type=int, default=1)
    g.add_argument("--disable-replay", action="store_true")

    g = p.add_argument_group("GRPO optimizer")
    g.add_argument("--grpo-lr", type=float, default=1e-4)
    g.add_argument("--grpo-max-grad-norm", type=float, default=1.0)
    g.add_argument("--grpo-advantage-eps", type=float, default=1e-8)
    g.add_argument("--grpo-clip-eps", type=float, default=0.2)
    g.add_argument("--grpo-clip-eps-high", type=float, default=None)
    g.add_argument(
        "--grpo-entropy-coef",
        type=float,
        default=0.0,
        help="Entropy regularization (not TRL KL). 0 disables.",
    )
    g.add_argument(
        "--grpo-num-iterations",
        type=int,
        default=1,
        help="Reuse each on-policy rollout for this many updates before resampling (TRL mu).",
    )

    g = p.add_argument_group("policy importance sampling (pi_new / pi_old)")
    g.add_argument(
        "--enable-policy-is",
        action="store_true",
        help="Sample a fixed buffer under the behavior policy, then GRPO updates "
             "weighted by pi_new/pi_old on stored trajectories.",
    )
    g.add_argument(
        "--resample-rounds",
        type=int,
        default=None,
        help="Outer loops when policy IS is on (default: --epochs).",
    )
    g.add_argument(
        "--update-cycles",
        type=int,
        default=None,
        help="Inner IS updates per buffer (default: --steps-per-epoch).",
    )
    g.add_argument(
        "--buffer-size",
        type=int,
        default=None,
        help="Trees per behavior rollout when policy IS is on (default: --on-policy-batch-size).",
    )
    g.add_argument(
        "--rollout-chunk-size",
        type=int,
        default=64,
        help="Rollout/replay chunk size when policy IS is on.",
    )

    g = p.add_argument_group("tracking")
    g.add_argument(
        "--outcome-level",
        choices=["signature", "topology"],
        default="topology",
        help="'topology' = tree shape only; 'signature' = topology + branch lengths.",
    )
    g.add_argument("--print-every", type=int, default=1)
    g.add_argument("--checkpoint-every", type=int, default=0)

    g = p.add_argument_group("resume")
    g.add_argument(
        "--resume-from",
        default=None,
        help="Continue training in an existing run directory (same output folder).",
    )
    g.add_argument(
        "--resume-checkpoint",
        default=None,
        help="Checkpoint filename inside --resume-from (default: final or latest epoch/round).",
    )

    return p


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        method=args.method,
        cfg_path=args.cfg_path,
        dataset_path=args.dataset_path,
        output_root=args.output_root,
        run_name=args.run_name,
        seed=args.seed,
        device=args.device,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        on_policy_batch_size=args.on_policy_batch_size,
        replay_batch_size=args.replay_batch_size,
        replay_buffer_size=args.replay_buffer_size,
        mini_batch_splits=args.mini_batch_splits,
        disable_replay=args.disable_replay,
        grpo_lr=args.grpo_lr,
        grpo_max_grad_norm=args.grpo_max_grad_norm,
        grpo_advantage_eps=args.grpo_advantage_eps,
        grpo_clip_eps=args.grpo_clip_eps,
        grpo_clip_eps_high=args.grpo_clip_eps_high,
        grpo_entropy_coef=args.grpo_entropy_coef,
        grpo_num_iterations=args.grpo_num_iterations,
        enable_policy_is=args.enable_policy_is,
        resample_rounds=args.resample_rounds,
        update_cycles=args.update_cycles,
        buffer_size=args.buffer_size,
        rollout_chunk_size=args.rollout_chunk_size,
        outcome_level=args.outcome_level,
        print_every=args.print_every,
        checkpoint_every=args.checkpoint_every,
        resume_from=args.resume_from,
        resume_checkpoint=args.resume_checkpoint,
    )
