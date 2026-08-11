"""Known-multiplicity trajectory IPS experiment for the direction/step DAG.

The target distribution over complete action trajectories is

    p*(tau) = R(x(tau)) / (Z * m(x(tau)))

where m(x) is the exactly known number of trajectories reaching terminal x.
This preserves reward-proportional terminal sampling while making trajectories
conditional on each terminal uniform. There is no backward policy.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import torch  # noqa: E402

from config import TrainConfig
from count_ips import CountIPSTrainer, Episode, _pad_episode_values
from dag_env import RIGHT, UP, State, find_default_terminal_states
from run_count_ips import (
    _plot_final_counts,
    _plot_training_curves,
    _plot_trajectory_diagnostics,
)


def terminal_multiplicities(budget: int, max_step: int) -> dict[State, int]:
    """Count paths to each terminal with dynamic programming, not enumeration."""
    ways: dict[State, int] = {State(0, 0): 1}
    for depth in range(budget):
        for x in range(depth + 1):
            state = State(x, depth - x)
            state_ways = ways.get(state, 0)
            if not state_ways:
                continue
            remaining = budget - depth
            for direction in (RIGHT, UP):
                for length in range(1, min(max_step, remaining) + 1):
                    child = (
                        State(state.x + length, state.y)
                        if direction == RIGHT
                        else State(state.x, state.y + length)
                    )
                    ways[child] = ways.get(child, 0) + state_ways
    return {state: ways[state] for state in find_default_terminal_states(budget)}


def trajectory_ips_advantages(
    rewards: Sequence[float],
    outcome_ids: Sequence[object],
    trajectory_ids: Sequence[object],
    multiplicity: dict[object, int],
    *,
    eps: float = 1e-8,
) -> tuple[np.ndarray, dict[str, float]]:
    """Return normalized R(x) / (m(x) * batch_frequency(tau)) scores."""
    size = len(rewards)
    if size == 0 or len(outcome_ids) != size or len(trajectory_ids) != size:
        raise ValueError("reward, outcome, and trajectory sequences must match")
    if any(outcome not in multiplicity for outcome in outcome_ids):
        raise ValueError("multiplicity must be known for every sampled outcome")

    rewards_array = np.asarray(rewards, dtype=np.float64)
    trajectory_counts = Counter(trajectory_ids)
    outcome_counts = Counter(outcome_ids)
    p_hat_trajectory = np.asarray(
        [trajectory_counts[trajectory] / size for trajectory in trajectory_ids],
        dtype=np.float64,
    )
    multiplicities = np.asarray(
        [multiplicity[outcome] for outcome in outcome_ids], dtype=np.float64
    )
    scaled = rewards_array / (multiplicities * p_hat_trajectory)
    centered = scaled - scaled.mean()
    std = float(scaled.std())
    advantages = centered if std < eps else centered / (std + eps)

    inverse = 1.0 / p_hat_trajectory
    ess = float(inverse.sum() ** 2 / np.square(inverse).sum())
    return advantages, {
        "ips_prob_mean": float(p_hat_trajectory.mean()),
        "ips_prob_min": float(p_hat_trajectory.min()),
        "ips_prob_max": float(p_hat_trajectory.max()),
        "ips_unique_outcomes": float(len(outcome_counts)),
        "ips_unique_trajectories": float(len(trajectory_counts)),
        "ips_max_outcome_count": float(max(outcome_counts.values())),
        "ips_min_outcome_count": float(min(outcome_counts.values())),
        "ips_max_trajectory_count": float(max(trajectory_counts.values())),
        "ips_min_trajectory_count": float(min(trajectory_counts.values())),
        "ips_scaled_reward_mean": float(scaled.mean()),
        "ips_scaled_reward_std": std,
        "ips_ess": ess,
        "ips_ess_fraction": ess / size,
        "advantage_mean": float(advantages.mean()),
        "advantage_std": float(advantages.std()),
        "advantage_min": float(advantages.min()),
        "advantage_max": float(advantages.max()),
    }


class FullTrajectoryPPOTrainer(CountIPSTrainer):
    """Shared full-path PPO loss for trajectory-targeting experiments."""

    def _joint_policy_loss(
        self, episodes: list[Episode]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """PPO on summed complete-path log probabilities, not token averages."""
        steps = [step for episode in episodes for step in episode.steps]
        obs = torch.as_tensor(
            np.stack([step.obs for step in steps]),
            dtype=torch.float32,
            device=self.device,
        )
        direction_masks = torch.as_tensor(
            np.stack([step.direction_mask for step in steps]),
            dtype=torch.bool,
            device=self.device,
        )
        step_masks = torch.as_tensor(
            np.stack([step.step_mask for step in steps]),
            dtype=torch.bool,
            device=self.device,
        )
        directions = torch.tensor(
            [step.direction for step in steps], dtype=torch.long, device=self.device
        )
        step_indices = torch.tensor(
            [step.step_index for step in steps], dtype=torch.long, device=self.device
        )

        direction_dist, representation = self.direction_policy.dist_with_rep(
            obs, direction_masks
        )
        step_rep = (
            representation.detach()
            if self.config.detach_step_rep
            else representation
        )
        step_dist = self.step_policy.dist(step_rep, directions, step_masks)
        new_flat = direction_dist.log_prob(directions) + step_dist.log_prob(step_indices)
        old_flat = torch.tensor(
            [step.log_prob_joint for step in steps],
            dtype=torch.float32,
            device=self.device,
        )
        entropy_flat = direction_dist.entropy() + step_dist.entropy()

        lengths = [len(episode.steps) for episode in episodes]
        max_length = max(lengths)
        new_tokens = _pad_episode_values(new_flat, lengths, max_length)
        old_tokens = _pad_episode_values(old_flat, lengths, max_length)
        entropy_tokens = _pad_episode_values(entropy_flat, lengths, max_length)
        mask = torch.arange(max_length, device=self.device).unsqueeze(0) < torch.tensor(
            lengths, device=self.device
        ).unsqueeze(1)
        advantages = torch.tensor(
            [episode.steps[0].advantage for episode in episodes],
            dtype=torch.float32,
            device=self.device,
        )

        new_path_log_prob = (new_tokens * mask).sum(dim=-1)
        old_path_log_prob = (old_tokens * mask).sum(dim=-1).detach()
        ratio = torch.exp(new_path_log_prob - old_path_log_prob)
        clipped_ratio = torch.clamp(
            ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio
        )
        policy_loss = -torch.min(
            ratio * advantages, clipped_ratio * advantages
        ).mean()
        path_entropy = (entropy_tokens * mask).sum(dim=-1).mean()
        loss = policy_loss - self.config.entropy_coef * path_entropy

        with torch.no_grad():
            return loss, {
                "loss": float(loss.item()),
                "policy_loss": float(policy_loss.item()),
                "entropy": float(path_entropy.item()),
                "mean_importance_ratio": float(ratio.mean().item()),
                "max_importance_ratio": float(ratio.max().item()),
                "min_importance_ratio": float(ratio.min().item()),
                "clip_fraction": float((ratio != clipped_ratio).float().mean().item()),
            }


class KnownMultiplicityTrajectoryIPSTrainer(FullTrajectoryPPOTrainer):
    """Count IPS over exact trajectories with known terminal multiplicities."""

    def __init__(self, config: TrainConfig | None = None, *, device: str = "cpu") -> None:
        super().__init__(config, device=device)
        self.known_multiplicity = terminal_multiplicities(
            self.config.budget, self.config.max_step
        )

    def _group_advantages(self, episodes: list[Episode]) -> float:
        advantages, metrics = trajectory_ips_advantages(
            [episode.reward for episode in episodes],
            [episode.terminal for episode in episodes],
            [episode.trajectory for episode in episodes],
            self.known_multiplicity,
            eps=self.config.advantage_eps,
        )
        for episode, advantage in zip(episodes, advantages):
            for step in episode.steps:
                step.advantage = float(advantage)
        self._last_ips_metrics = metrics
        return metrics["ips_ess"]


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is not available")
    return requested


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=32)
    parser.add_argument("--max-step", type=int, default=3)
    parser.add_argument("--num-updates", type=int, default=500)
    parser.add_argument("--group-size", type=int, default=512)
    parser.add_argument("--num-groups", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=2_000)
    parser.add_argument("--final-samples", type=int, default=10_000)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=500,
        help="save an intermediate checkpoint every N updates",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--terminal-rewards",
        type=float,
        nargs="+",
        default=None,
        metavar="R",
        help="budget + 1 rewards in increasing terminal x order",
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    args = parser.parse_args()

    config = TrainConfig(
        budget=args.budget,
        max_step=args.max_step,
        terminal_rewards=(
            tuple(args.terminal_rewards) if args.terminal_rewards is not None else None
        ),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        group_size=args.group_size,
        num_groups=args.num_groups,
        num_updates=args.num_updates,
        lr=args.lr,
        clip_ratio=args.clip_ratio,
        entropy_coef=args.entropy_coef,
        seed=args.seed,
        log_every=args.log_every,
    )
    run_dir = args.run_dir or (
        Path(__file__).resolve().parent
        / "data"
        / "trajectory_ips_runs"
        / (
            f"{datetime.now():%Y%m%d_%H%M%S}_b{config.budget}"
            f"_gs{config.group_size}_seed{config.seed}"
        )
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    trainer = KnownMultiplicityTrajectoryIPSTrainer(
        config, device=_resolve_device(args.device)
    )
    print(f"Run directory: {run_dir}")
    print(f"Device: {trainer.device}")
    print(f"DAG: {trainer.environment_summary()}")
    print(f"Known multiplicities: {trainer.known_multiplicity}")
    print("Algorithm: R(x) / [m(x) * within-group trajectory frequency]")

    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "train_config": asdict(config),
                "device": str(trainer.device),
                "algorithm": "known_multiplicity_trajectory_ips",
                "score": "R(x) / (m(x) * p_hat(tau))",
                "checkpoint_every": args.checkpoint_every,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    history = trainer.train(
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        checkpoint_every=args.checkpoint_every,
        checkpoint_dir=run_dir / "checkpoints",
    )
    (run_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    trainer.save(run_dir / "checkpoint.pt", update_step=config.num_updates)

    training_plot = run_dir / "training_curves.png"
    _plot_training_curves(history, trainer, output=training_plot)
    final_evaluation = trainer.evaluate(args.final_samples)
    sampling = _plot_final_counts(
        trainer,
        final_evaluation,
        samples=args.final_samples,
        output=run_dir / "sampling_counts.png",
    )
    trajectory_plot = run_dir / "trajectory_sampling.png"
    trajectory_sampling = _plot_trajectory_diagnostics(
        history,
        trainer,
        final_evaluation,
        samples=args.final_samples,
        output=trajectory_plot,
        subtitle="Dashed lines = known-multiplicity uniform trajectory target",
    )
    eval_rows = [row for row in history if "tv_reward_target" in row]
    summary = {
        "environment": trainer.environment_summary(),
        "known_multiplicity": {
            state.signature: trainer.known_multiplicity[state]
            for state in trainer.terminals
        },
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
            "trajectory_sampling": trajectory_plot.name,
        },
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"Final counts: {sampling['actual_counts']}")
    print(f"Ideal counts: {sampling['ideal_counts']}")
    print(f"Final TV distance: {sampling['tv_reward_target']:.4f}")
    print(
        "Trajectory coverage: "
        f"{trajectory_sampling['coverage_by_terminal']} "
        f"({trajectory_sampling['unique_trajectories_hit']} unique sampled paths)"
    )
    print(f"Training curves: {training_plot}")
    print(f"Sampling plots: {run_dir / 'sampling_counts.png'}")
    print(f"Trajectory plots: {trajectory_plot}")
    print(f"Artifacts: {run_dir}")


if __name__ == "__main__":
    main()
