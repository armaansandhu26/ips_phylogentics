"""Frozen-dataset terminal-count IPS for the direction/step DAG.

This runner has two strictly separated phases:

1. collect a fixed dataset from one frozen behavior policy (or load one);
2. train a fresh forward policy only from that dataset.

For a dataset of ``N`` trajectories, let ``n_D(x)`` be the number ending at
terminal ``x``.  Every stored trajectory receives the terminal IPS weight

    w_count(tau) = R(x(tau)) / (n_D(x(tau)) / N).

The weights are self-normalized once and then held fixed.  Training minimizes
the IPS-weighted complete-trajectory negative log likelihood.  There are no
new training rollouts, PPO ratios, or reverse policy updates after collection.

The dataset is saved as plain Python/Tensor data and can be passed to
``offline_learned_reverse_ips.py`` for a matched comparison.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from config import TrainConfig
from count_ips import CountIPSTrainer, Episode
from dag_env import (
    RIGHT,
    UP,
    State,
    Trajectory,
    reward_per_terminal_state,
)
from exact_probability_ips import _resolve_device
from run_count_ips import (  # noqa: E402
    _plot_final_counts,
    _plot_trajectory_diagnostics,
)


DATASET_FORMAT = "dag_offline_trajectories_v1"


@dataclass(frozen=True)
class OfflineTrajectory:
    """Compact trajectory record; observations are reconstructed deterministically."""

    trajectory: Trajectory
    terminal: State
    reward: float
    behavior_log_probability: float


@dataclass(frozen=True)
class OfflinePolicyBatch:
    observations: torch.Tensor
    direction_masks: torch.Tensor
    step_masks: torch.Tensor
    directions: torch.Tensor
    step_indices: torch.Tensor
    episode_indices: torch.Tensor
    num_episodes: int


class FrozenTrajectoryDataset:
    """An immutable collection of complete behavior-policy trajectories."""

    def __init__(
        self,
        trajectories: Sequence[OfflineTrajectory],
        *,
        budget: int,
        max_step: int,
        collection: dict[str, Any] | None = None,
    ) -> None:
        if not trajectories:
            raise ValueError("offline dataset must contain at least one trajectory")
        self.trajectories = tuple(trajectories)
        self.budget = int(budget)
        self.max_step = int(max_step)
        self.collection = dict(collection or {})
        self._validate()

    def _validate(self) -> None:
        for item in self.trajectories:
            if not item.trajectory:
                raise ValueError("offline trajectories must be non-empty")
            if item.terminal.depth != self.budget:
                raise ValueError("offline terminal does not match dataset budget")
            if not math.isfinite(item.reward) or item.reward <= 0.0:
                raise ValueError("offline rewards must be finite and positive")
            if (
                not math.isfinite(item.behavior_log_probability)
                or item.behavior_log_probability > 1e-7
            ):
                raise ValueError(
                    "behavior log-probabilities must be finite and <= 0"
                )
            x = 0
            y = 0
            for direction, length in item.trajectory:
                remaining = self.budget - x - y
                if direction not in (RIGHT, UP):
                    raise ValueError(f"invalid offline direction {direction}")
                if length < 1 or length > min(self.max_step, remaining):
                    raise ValueError(
                        f"invalid offline step {length} with remaining={remaining}"
                    )
                if direction == RIGHT:
                    x += length
                else:
                    y += length
            if State(x, y) != item.terminal:
                raise ValueError("offline trajectory does not reach its terminal")

    def __len__(self) -> int:
        return len(self.trajectories)

    @property
    def terminal_counts(self) -> Counter[State]:
        return Counter(item.terminal for item in self.trajectories)

    def summary(self) -> dict[str, Any]:
        counts = self.terminal_counts
        behavior_log_probabilities = np.asarray(
            [item.behavior_log_probability for item in self.trajectories],
            dtype=np.float64,
        )
        lengths = np.asarray(
            [len(item.trajectory) for item in self.trajectories],
            dtype=np.float64,
        )
        return {
            "size": len(self),
            "budget": self.budget,
            "max_step": self.max_step,
            "unique_terminals": len(counts),
            "terminal_coverage_fraction": len(counts) / (self.budget + 1),
            "min_observed_terminal_count": min(counts.values()),
            "max_observed_terminal_count": max(counts.values()),
            "terminal_counts": {
                State(x, self.budget - x).signature: int(
                    counts[State(x, self.budget - x)]
                )
                for x in range(self.budget + 1)
            },
            "mean_length": float(lengths.mean()),
            "min_length": int(lengths.min()),
            "max_length": int(lengths.max()),
            "behavior_log_probability_mean": float(
                behavior_log_probabilities.mean()
            ),
            "behavior_log_probability_min": float(
                behavior_log_probabilities.min()
            ),
            "behavior_log_probability_max": float(
                behavior_log_probabilities.max()
            ),
            "collection": self.collection,
        }

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": DATASET_FORMAT,
                "budget": self.budget,
                "max_step": self.max_step,
                "trajectories": [
                    tuple((int(direction), int(length)) for direction, length in item.trajectory)
                    for item in self.trajectories
                ],
                "terminal_x": torch.tensor(
                    [item.terminal.x for item in self.trajectories],
                    dtype=torch.int16
                    if self.budget < np.iinfo(np.int16).max
                    else torch.int32,
                ),
                "rewards": torch.tensor(
                    [item.reward for item in self.trajectories],
                    dtype=torch.float64,
                ),
                "behavior_log_probabilities": torch.tensor(
                    [
                        item.behavior_log_probability
                        for item in self.trajectories
                    ],
                    dtype=torch.float64,
                ),
                "collection": self.collection,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: Path | str) -> "FrozenTrajectoryDataset":
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        if payload.get("format") != DATASET_FORMAT:
            raise ValueError(f"unsupported offline dataset format in {path}")
        budget = int(payload["budget"])
        terminal_x = payload["terminal_x"].tolist()
        rewards = payload["rewards"].tolist()
        behavior_log_probabilities = payload[
            "behavior_log_probabilities"
        ].tolist()
        raw_trajectories = payload["trajectories"]
        size = len(raw_trajectories)
        if not (
            len(terminal_x)
            == len(rewards)
            == len(behavior_log_probabilities)
            == size
        ):
            raise ValueError("offline dataset fields have inconsistent lengths")
        trajectories = [
            OfflineTrajectory(
                trajectory=tuple(
                    (int(direction), int(length))
                    for direction, length in raw_trajectory
                ),
                terminal=State(int(x), budget - int(x)),
                reward=float(reward),
                behavior_log_probability=float(log_probability),
            )
            for raw_trajectory, x, reward, log_probability in zip(
                raw_trajectories,
                terminal_x,
                rewards,
                behavior_log_probabilities,
            )
        ]
        return cls(
            trajectories,
            budget=budget,
            max_step=int(payload["max_step"]),
            collection=payload.get("collection", {}),
        )


def collect_frozen_dataset(
    config: TrainConfig,
    *,
    size: int,
    batch_size: int,
    device: str,
    behavior_checkpoint: Path | None = None,
    behavior_seed: int = 0,
) -> FrozenTrajectoryDataset:
    """Collect once from a behavior policy whose parameters never change."""
    if size < 1 or batch_size < 1:
        raise ValueError("dataset size and collection batch size must be >= 1")
    behavior_config = TrainConfig(
        budget=config.budget,
        max_step=config.max_step,
        terminal_rewards=config.terminal_rewards,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        detach_step_rep=config.detach_step_rep,
        group_size=min(batch_size, size),
        num_groups=1,
        num_updates=1,
        lr=config.lr,
        entropy_coef=0.0,
        seed=behavior_seed,
    )
    behavior = CountIPSTrainer(behavior_config, device=device)
    checkpoint_description: str | None = None
    if behavior_checkpoint is not None:
        payload = torch.load(
            behavior_checkpoint, map_location=device, weights_only=False
        )
        checkpoint_config = payload.get("config")
        if checkpoint_config is not None:
            if (
                checkpoint_config.budget != config.budget
                or checkpoint_config.max_step != config.max_step
            ):
                raise ValueError(
                    "behavior checkpoint budget/max_step does not match the run"
                )
        behavior.direction_policy.load_state_dict(payload["direction_policy"])
        behavior.step_policy.load_state_dict(payload["step_policy"])
        checkpoint_description = str(behavior_checkpoint.resolve())
    behavior.direction_policy.eval()
    behavior.step_policy.eval()

    records: list[OfflineTrajectory] = []
    remaining = size
    while remaining:
        current_size = min(batch_size, remaining)
        episodes = behavior.rollout_batch(current_size, explore=False)
        records.extend(
            OfflineTrajectory(
                trajectory=episode.trajectory,
                terminal=episode.terminal,
                reward=episode.reward,
                behavior_log_probability=float(
                    sum(step.log_prob_joint for step in episode.steps)
                ),
            )
            for episode in episodes
        )
        remaining -= current_size
        if remaining == 0 or len(records) % max(batch_size * 10, 1) == 0:
            print(f"Collected offline trajectories: {len(records):,}/{size:,}")
    return FrozenTrajectoryDataset(
        records,
        budget=config.budget,
        max_step=config.max_step,
        collection={
            "behavior": (
                "checkpoint" if behavior_checkpoint is not None else "random_initial_policy"
            ),
            "behavior_checkpoint": checkpoint_description,
            "behavior_seed": behavior_seed,
            "device": str(device),
        },
    )


def materialize_policy_batch(
    trajectories: Sequence[OfflineTrajectory],
    config: TrainConfig,
    *,
    device: torch.device,
) -> OfflinePolicyBatch:
    """Reconstruct observations and masks for a compact trajectory minibatch."""
    if not trajectories:
        raise ValueError("offline policy minibatch must be non-empty")
    edge_count = sum(len(item.trajectory) for item in trajectories)
    width = config.budget + 1
    observations = np.zeros(
        (edge_count, 3 * width), dtype=np.float32
    )
    direction_masks = np.ones((edge_count, 2), dtype=bool)
    step_masks = np.zeros((edge_count, config.max_step), dtype=bool)
    directions = np.empty(edge_count, dtype=np.int64)
    step_indices = np.empty(edge_count, dtype=np.int64)
    episode_indices = np.empty(edge_count, dtype=np.int64)

    row = 0
    for episode_index, item in enumerate(trajectories):
        x = 0
        y = 0
        for direction, physical_step in item.trajectory:
            remaining = config.budget - x - y
            observations[row, x] = 1.0
            observations[row, width + y] = 1.0
            observations[row, 2 * width + remaining] = 1.0
            step_masks[row, : min(config.max_step, remaining)] = True
            directions[row] = direction
            step_indices[row] = physical_step - 1
            episode_indices[row] = episode_index
            if direction == RIGHT:
                x += physical_step
            else:
                y += physical_step
            row += 1
        if State(x, y) != item.terminal:
            raise ValueError("offline trajectory failed reconstruction")

    return OfflinePolicyBatch(
        observations=torch.as_tensor(
            observations, dtype=torch.float32, device=device
        ),
        direction_masks=torch.as_tensor(
            direction_masks, dtype=torch.bool, device=device
        ),
        step_masks=torch.as_tensor(
            step_masks, dtype=torch.bool, device=device
        ),
        directions=torch.as_tensor(
            directions, dtype=torch.long, device=device
        ),
        step_indices=torch.as_tensor(
            step_indices, dtype=torch.long, device=device
        ),
        episode_indices=torch.as_tensor(
            episode_indices, dtype=torch.long, device=device
        ),
        num_episodes=len(trajectories),
    )


class OfflineWeightedIPSTrainer(CountIPSTrainer):
    """Forward policy optimized by fixed IPS-weighted trajectory likelihood."""

    def path_log_probabilities(
        self, batch: OfflinePolicyBatch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        direction_dist, representation = self.direction_policy.dist_with_rep(
            batch.observations, batch.direction_masks
        )
        step_representation = (
            representation.detach()
            if self.config.detach_step_rep
            else representation
        )
        step_dist = self.step_policy.dist(
            step_representation, batch.directions, batch.step_masks
        )
        edge_log_probabilities = direction_dist.log_prob(
            batch.directions
        ) + step_dist.log_prob(batch.step_indices)
        path_log_probabilities = torch.zeros(
            batch.num_episodes,
            dtype=torch.float32,
            device=self.device,
        )
        path_log_probabilities.scatter_add_(
            0, batch.episode_indices, edge_log_probabilities
        )
        edge_entropy = direction_dist.entropy() + step_dist.entropy()
        return path_log_probabilities, edge_entropy

    def offline_update(
        self,
        trajectories: Sequence[OfflineTrajectory],
        weights: np.ndarray,
    ) -> dict[str, float]:
        if weights.shape != (len(trajectories),):
            raise ValueError("offline minibatch weights have the wrong shape")
        batch = materialize_policy_batch(
            trajectories, self.config, device=self.device
        )
        weight_tensor = torch.as_tensor(
            weights, dtype=torch.float32, device=self.device
        )
        parameters = list(self.direction_policy.parameters()) + list(
            self.step_policy.parameters()
        )
        self.optimizer.zero_grad(set_to_none=True)
        path_log_probabilities, edge_entropy = self.path_log_probabilities(batch)
        weighted_nll = -(weight_tensor.detach() * path_log_probabilities).mean()
        loss = weighted_nll
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            parameters, self.config.grad_clip_norm
        )
        self.optimizer.step()
        with torch.no_grad():
            squared_sum = float(weight_tensor.square().sum().item())
            ess = float(
                weight_tensor.sum().item() ** 2
                / max(squared_sum, self.config.advantage_eps)
            )
            return {
                "loss": float(loss.item()),
                "weighted_nll": float(weighted_nll.item()),
                "unweighted_nll": float(
                    -path_log_probabilities.mean().item()
                ),
                "path_log_probability_mean": float(
                    path_log_probabilities.mean().item()
                ),
                "entropy": float(edge_entropy.mean().item()),
                "grad_norm": float(grad_norm.item()),
                "param_norm": float(
                    sum(
                        parameter.detach().norm().item() ** 2
                        for parameter in parameters
                    )
                    ** 0.5
                ),
                "batch_weight_mean": float(weight_tensor.mean().item()),
                "batch_weight_min": float(weight_tensor.min().item()),
                "batch_weight_max": float(weight_tensor.max().item()),
                "batch_ips_ess": ess,
                "batch_ips_ess_fraction": ess / len(trajectories),
            }

    def save_offline(
        self,
        path: Path | str,
        *,
        update_step: int,
        algorithm: dict[str, Any],
    ) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": self.config,
                "direction_policy": self.direction_policy.state_dict(),
                "step_policy": self.step_policy.state_dict(),
                "update_step": update_step,
                "algorithm": algorithm,
            },
            path,
        )
        return path


def terminal_count_ips_weights(
    dataset: FrozenTrajectoryDataset,
    *,
    clip: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return fixed, mean-one ``R(x) / p_hat_D(x)`` weights."""
    counts = dataset.terminal_counts
    size = len(dataset)
    raw = np.asarray(
        [
            item.reward / (counts[item.terminal] / size)
            for item in dataset.trajectories
        ],
        dtype=np.float64,
    )
    weights, metrics = normalize_fixed_weights(
        raw,
        clip=clip,
        extra_metrics={
            "estimator": "terminal_count",
            "unique_terminals": len(counts),
            "terminal_coverage_fraction": len(counts) / (dataset.budget + 1),
        },
    )
    metrics.update(weighted_terminal_metrics(dataset, weights))
    return weights, metrics


def normalize_fixed_weights(
    raw_weights: np.ndarray,
    *,
    clip: float | None,
    extra_metrics: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if (
        raw_weights.ndim != 1
        or raw_weights.size == 0
        or np.any(~np.isfinite(raw_weights))
        or np.any(raw_weights <= 0.0)
    ):
        raise ValueError("raw IPS weights must be a finite positive vector")
    normalized = raw_weights / raw_weights.mean()
    preclip_max = float(normalized.max())
    clip_fraction = 0.0
    if clip is not None:
        if clip <= 0.0:
            raise ValueError("weight clip must be positive")
        clipped = np.minimum(normalized, clip)
        clip_fraction = float(np.mean(clipped != normalized))
        normalized = clipped / clipped.mean()
    squared_sum = float(np.square(normalized).sum())
    ess = float(normalized.sum() ** 2 / squared_sum)
    metrics: dict[str, Any] = {
        "raw_weight_mean": float(raw_weights.mean()),
        "raw_weight_min": float(raw_weights.min()),
        "raw_weight_max": float(raw_weights.max()),
        "normalized_weight_mean": float(normalized.mean()),
        "normalized_weight_min": float(normalized.min()),
        "normalized_weight_max": float(normalized.max()),
        "normalized_weight_max_before_clip": preclip_max,
        "weight_clip": clip,
        "weight_clip_fraction": clip_fraction,
        "ips_ess": ess,
        "ips_ess_fraction": ess / normalized.size,
    }
    metrics.update(extra_metrics or {})
    return normalized, metrics


def weighted_terminal_metrics(
    dataset: FrozenTrajectoryDataset,
    normalized_weights: np.ndarray,
) -> dict[str, Any]:
    """Describe the terminal target represented by the finite weighted dataset."""
    if normalized_weights.shape != (len(dataset),):
        raise ValueError("one normalized weight is required per trajectory")
    mass = Counter()
    rewards: dict[State, float] = {}
    for item, weight in zip(dataset.trajectories, normalized_weights):
        mass[item.terminal] += float(weight)
        rewards[item.terminal] = item.reward
    # Compare on observed support. Coverage is reported separately because a
    # finite offline dataset contains no reward record for an unseen terminal.
    terminals = sorted(rewards)
    empirical_total = float(sum(mass.values()))
    reward_total = float(sum(rewards[state] for state in terminals))
    empirical = np.asarray(
        [mass[state] / empirical_total for state in terminals],
        dtype=np.float64,
    )
    target = np.asarray(
        [rewards[state] / reward_total for state in terminals],
        dtype=np.float64,
    )
    return {
        "finite_dataset_observed_support_weighted_terminal_probs": {
            state.signature: float(empirical[index])
            for index, state in enumerate(terminals)
        },
        "finite_dataset_observed_support_target_terminal_probs": {
            state.signature: float(target[index])
            for index, state in enumerate(terminals)
        },
        "finite_dataset_observed_support_weighted_terminal_tv": float(
            0.5 * np.abs(empirical - target).sum()
        ),
        "finite_dataset_observed_support_weighted_terminal_max_error": float(
            np.abs(empirical - target).max()
        ),
    }


def train_offline_policy(
    trainer: OfflineWeightedIPSTrainer,
    dataset: FrozenTrajectoryDataset,
    weights: np.ndarray,
    *,
    num_updates: int,
    batch_size: int,
    eval_every: int | None,
    eval_episodes: int,
    checkpoint_every: int | None,
    checkpoint_dir: Path,
    algorithm: dict[str, Any],
) -> list[dict[str, Any]]:
    if weights.shape != (len(dataset),):
        raise ValueError("one fixed IPS weight is required per offline trajectory")
    rng = np.random.default_rng(trainer.config.seed)
    history: list[dict[str, Any]] = []
    for update_step in range(1, num_updates + 1):
        indices = rng.integers(0, len(dataset), size=batch_size)
        trajectories = [dataset.trajectories[int(index)] for index in indices]
        batch_weights = weights[indices]
        stats = trainer.offline_update(trajectories, batch_weights)
        batch_counts = Counter(item.terminal for item in trajectories)
        row: dict[str, Any] = {
            "step": update_step,
            "offline_only": 1.0,
            "dataset_size": len(dataset),
            "offline_batch_size": batch_size,
            "mean_reward": float(
                np.mean([item.reward for item in trajectories])
            ),
            "mean_length": float(
                np.mean([len(item.trajectory) for item in trajectories])
            ),
            "unique_terminals": len(batch_counts),
            **stats,
        }
        if eval_every and (update_step == 1 or update_step % eval_every == 0):
            row.update(trainer.evaluate(eval_episodes))
        history.append(row)
        if checkpoint_every and update_step % checkpoint_every == 0:
            checkpoint = trainer.save_offline(
                checkpoint_dir / f"checkpoint_update_{update_step:06d}.pt",
                update_step=update_step,
                algorithm=algorithm,
            )
            print(f"Checkpoint: {checkpoint}")
        if update_step == 1 or update_step % trainer.config.log_every == 0:
            print(
                f"update {update_step:5d}  "
                f"weighted_NLL={row['weighted_nll']:.3f}  "
                f"batch_ESS={row['batch_ips_ess_fraction']:.3f}  "
                f"grad={row['grad_norm']:.3f}"
                + (
                    f"  eval_TV={row['tv_reward_target']:.3f}"
                    if "tv_reward_target" in row
                    else ""
                )
            )
    return history


def plot_offline_training(
    history: list[dict[str, Any]],
    *,
    output: Path,
    title: str,
) -> None:
    steps = [row["step"] for row in history]
    eval_rows = [row for row in history if "tv_reward_target" in row]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    axes[0, 0].plot(
        steps, [row["weighted_nll"] for row in history], label="weighted"
    )
    axes[0, 0].plot(
        steps,
        [row["unweighted_nll"] for row in history],
        label="unweighted",
        alpha=0.7,
    )
    axes[0, 0].set_title("Offline trajectory NLL")
    axes[0, 0].legend()

    axes[0, 1].plot(
        steps, [row["batch_ips_ess_fraction"] for row in history]
    )
    axes[0, 1].set_ylim(-0.02, 1.02)
    axes[0, 1].set_title("Minibatch IPS ESS fraction")

    axes[1, 0].plot(
        steps, [row["grad_norm"] for row in history], label="gradient norm"
    )
    axes[1, 0].plot(
        steps, [row["entropy"] for row in history], label="edge entropy"
    )
    axes[1, 0].set_title("Optimization")
    axes[1, 0].legend()

    if eval_rows:
        eval_steps = [row["step"] for row in eval_rows]
        axes[1, 1].plot(
            eval_steps,
            [row["tv_reward_target"] for row in eval_rows],
            "o-",
            label="TV distance",
        )
        axes[1, 1].plot(
            eval_steps,
            [row["max_abs_prob_error"] for row in eval_rows],
            "s--",
            label="max error",
        )
    axes[1, 1].set_ylim(bottom=0.0)
    axes[1, 1].set_title("Distance from reward target")
    axes[1, 1].legend()
    for axis in axes.flat:
        axis.set_xlabel("Offline update")
        axis.grid(alpha=0.22)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--budget", type=int, default=128)
    parser.add_argument("--max-step", type=int, default=3)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--dataset-size", type=int, default=50_000)
    parser.add_argument("--collection-batch-size", type=int, default=4_096)
    parser.add_argument("--behavior-checkpoint", type=Path, default=None)
    parser.add_argument("--behavior-seed", type=int, default=0)
    parser.add_argument("--num-updates", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument(
        "--weight-clip",
        type=float,
        default=0.0,
        help="cap mean-one IPS weights; 0 disables clipping",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-episodes", type=int, default=2_000)
    parser.add_argument("--final-samples", type=int, default=10_000)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument(
        "--terminal-rewards", type=float, nargs="+", default=None
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument("--run-dir", type=Path, default=None)


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        budget=args.budget,
        max_step=args.max_step,
        terminal_rewards=(
            tuple(args.terminal_rewards)
            if args.terminal_rewards is not None
            else None
        ),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        group_size=args.batch_size,
        num_groups=1,
        num_updates=args.num_updates,
        train_epochs=1,
        lr=args.lr,
        entropy_coef=0.0,
        grad_clip_norm=args.grad_clip_norm,
        seed=args.seed,
        log_every=args.log_every,
    )


def load_or_collect_dataset(
    args: argparse.Namespace,
    config: TrainConfig,
    *,
    device: str,
    run_dir: Path,
) -> tuple[FrozenTrajectoryDataset, Path]:
    if args.dataset is not None:
        dataset = FrozenTrajectoryDataset.load(args.dataset)
        dataset_path = args.dataset.resolve()
    else:
        dataset = collect_frozen_dataset(
            config,
            size=args.dataset_size,
            batch_size=args.collection_batch_size,
            device=device,
            behavior_checkpoint=args.behavior_checkpoint,
            behavior_seed=args.behavior_seed,
        )
        dataset_path = dataset.save(run_dir / "dataset.pt")
    if dataset.budget != config.budget or dataset.max_step != config.max_step:
        raise ValueError("offline dataset environment does not match run config")
    expected_rewards = reward_per_terminal_state(
        config.budget, config.terminal_rewards
    )
    for item in dataset.trajectories:
        if not math.isclose(
            item.reward,
            expected_rewards[item.terminal],
            rel_tol=1e-10,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "offline dataset rewards do not match the requested run config"
            )
    return dataset, dataset_path


def finish_run(
    *,
    run_dir: Path,
    trainer: OfflineWeightedIPSTrainer,
    dataset: FrozenTrajectoryDataset,
    dataset_path: Path,
    history: list[dict[str, Any]],
    weight_metrics: dict[str, Any],
    algorithm: dict[str, Any],
    final_samples: int,
    extra_summary: dict[str, Any] | None = None,
) -> None:
    (run_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    trainer.save_offline(
        run_dir / "checkpoint.pt",
        update_step=trainer.config.num_updates,
        algorithm=algorithm,
    )
    training_plot = run_dir / "training_curves.png"
    plot_offline_training(
        history,
        output=training_plot,
        title=algorithm["display_name"],
    )
    evaluation = trainer.evaluate(final_samples)
    sampling = _plot_final_counts(
        trainer,
        evaluation,
        samples=final_samples,
        output=run_dir / "sampling_counts.png",
        suptitle=f"{algorithm['display_name']} vs ideal reward sampling",
    )
    trajectory_sampling = _plot_trajectory_diagnostics(
        history,
        trainer,
        evaluation,
        samples=final_samples,
        output=run_dir / "trajectory_sampling.png",
        subtitle="Forward policy trained only on the frozen offline dataset",
    )
    eval_rows = [row for row in history if "tv_reward_target" in row]
    summary = {
        "environment": trainer.environment_summary(),
        "algorithm": algorithm,
        "dataset_path": str(dataset_path),
        "dataset": dataset.summary(),
        "fixed_weight_diagnostics": weight_metrics,
        "final_sampling": sampling,
        "trajectory_sampling": trajectory_sampling,
        "best_eval": (
            min(eval_rows, key=lambda row: row["tv_reward_target"])
            if eval_rows
            else None
        ),
        "plots": {
            "training_curves": training_plot.name,
            "sampling_counts": "sampling_counts.png",
            "trajectory_sampling": "trajectory_sampling.png",
        },
    }
    summary.update(extra_summary or {})
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Final ideal-line R^2: {sampling['r2_reward_target']:.4f}")
    print(f"Final TV distance: {sampling['tv_reward_target']:.4f}")
    print(f"Artifacts: {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    args = parser.parse_args()
    config = config_from_args(args)
    device = _resolve_device(args.device)
    run_dir = args.run_dir or (
        Path(__file__).resolve().parent
        / "data"
        / "offline_count_ips_runs"
        / (
            f"{datetime.now():%Y%m%d_%H%M%S}_b{config.budget}"
            f"_n{args.dataset_size}_seed{config.seed}"
        )
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    dataset, dataset_path = load_or_collect_dataset(
        args, config, device=device, run_dir=run_dir
    )
    dataset_summary = dataset.summary()
    print(f"Run directory: {run_dir}")
    print(f"Device: {device}")
    print(
        "Frozen dataset: "
        f"{len(dataset):,} trajectories, "
        f"{dataset_summary['unique_terminals']}/{config.budget + 1} terminals"
    )
    if dataset_summary["unique_terminals"] < config.budget + 1:
        print(
            "WARNING: the frozen dataset has missing terminal support; "
            "offline IPS cannot recover unseen terminals."
        )

    clip = args.weight_clip if args.weight_clip > 0.0 else None
    weights, weight_metrics = terminal_count_ips_weights(
        dataset, clip=clip
    )
    algorithm = {
        "name": "offline_terminal_count_ips",
        "display_name": "Offline terminal-count IPS",
        "raw_weight": "R(x) / p_hat_D(x)",
        "forward_loss": "fixed_IPS_weighted_full_trajectory_NLL",
        "training_rollouts_after_collection": 0,
    }
    trainer = OfflineWeightedIPSTrainer(config, device=device)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "train_config": asdict(config),
                "device": str(device),
                "algorithm": algorithm,
                "dataset_path": str(dataset_path),
                "dataset_collection": {
                    "requested_size": args.dataset_size,
                    "collection_batch_size": args.collection_batch_size,
                    "behavior_checkpoint": (
                        str(args.behavior_checkpoint)
                        if args.behavior_checkpoint is not None
                        else None
                    ),
                    "behavior_seed": args.behavior_seed,
                },
                "weight_clip": clip,
                "checkpoint_every": args.checkpoint_every,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    history = train_offline_policy(
        trainer,
        dataset,
        weights,
        num_updates=config.num_updates,
        batch_size=args.batch_size,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        checkpoint_every=args.checkpoint_every or None,
        checkpoint_dir=run_dir / "checkpoints",
        algorithm=algorithm,
    )
    finish_run(
        run_dir=run_dir,
        trainer=trainer,
        dataset=dataset,
        dataset_path=dataset_path,
        history=history,
        weight_metrics=weight_metrics,
        algorithm=algorithm,
        final_samples=args.final_samples,
    )


if __name__ == "__main__":
    main()
