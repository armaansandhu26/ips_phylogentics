"""Trajectory-balancing RL without using terminal multiplicity in training.

Training combines the existing terminal count-IPS advantage with a centered
within-terminal trajectory-surprisal advantage:

    A = A_terminal + lambda * A_path
    A_terminal <- normalize(R(x) / p_hat(x))
    A_path     <- normalize_x(-log p_hat(tau | x) - baseline_x)

The known toy multiplicities are used only after rollout for evaluation plots.
There is no backward policy and no multiplicity input to the training rule.
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

from config import TrainConfig
from count_ips import Episode, count_ips_advantages
from run_count_ips import (
    _plot_final_counts,
    _plot_training_curves,
    _plot_trajectory_diagnostics,
)
from trajectory_ips import FullTrajectoryPPOTrainer, _resolve_device


def unknown_m_trajectory_advantages(
    rewards: Sequence[float],
    outcome_ids: Sequence[object],
    trajectory_ids: Sequence[object],
    *,
    path_coefficient: float,
    conditional_probability_override: Sequence[float] | None = None,
    eps: float = 1e-8,
) -> tuple[np.ndarray, dict[str, float]]:
    """Combine terminal IPS with centered conditional trajectory surprisal."""
    size = len(rewards)
    if size == 0 or len(outcome_ids) != size or len(trajectory_ids) != size:
        raise ValueError("reward, outcome, and trajectory sequences must match")
    if path_coefficient < 0:
        raise ValueError("path_coefficient must be non-negative")

    terminal_advantages, terminal_metrics = count_ips_advantages(
        rewards, outcome_ids, eps=eps
    )
    outcome_counts = Counter(outcome_ids)
    trajectory_counts = Counter(trajectory_ids)
    if conditional_probability_override is None:
        conditional_probabilities = np.asarray(
            [
                trajectory_counts[trajectory] / outcome_counts[outcome]
                for outcome, trajectory in zip(outcome_ids, trajectory_ids)
            ],
            dtype=np.float64,
        )
    else:
        conditional_probabilities = np.asarray(
            conditional_probability_override, dtype=np.float64
        )
        if conditional_probabilities.shape != (size,):
            raise ValueError("conditional probability override must match batch size")
        if np.any(conditional_probabilities <= 0):
            raise ValueError("conditional probabilities must be positive")
    surprisal = -np.log(conditional_probabilities + eps)

    path_centered = np.empty(size, dtype=np.float64)
    for outcome in outcome_counts:
        indices = np.asarray(
            [index for index, value in enumerate(outcome_ids) if value == outcome]
        )
        path_centered[indices] = surprisal[indices] - surprisal[indices].mean()
    path_std = float(path_centered.std())
    path_advantages = (
        path_centered
        if path_std < eps
        else path_centered / (path_std + eps)
    )

    combined = terminal_advantages + path_coefficient * path_advantages
    combined_std = float(combined.std())
    if combined_std >= eps:
        combined = (combined - combined.mean()) / (combined_std + eps)

    metrics = {
        **terminal_metrics,
        "path_coefficient": float(path_coefficient),
        "path_unique_trajectories": float(len(trajectory_counts)),
        "path_conditional_prob_mean": float(conditional_probabilities.mean()),
        "path_conditional_prob_min": float(conditional_probabilities.min()),
        "path_conditional_prob_max": float(conditional_probabilities.max()),
        "path_surprisal_mean": float(surprisal.mean()),
        "path_surprisal_std": float(surprisal.std()),
        "path_advantage_mean": float(path_advantages.mean()),
        "path_advantage_std": float(path_advantages.std()),
        "terminal_advantage_std": float(terminal_advantages.std()),
        "advantage_mean": float(combined.mean()),
        "advantage_std": float(combined.std()),
        "advantage_min": float(combined.min()),
        "advantage_max": float(combined.max()),
    }
    return combined, metrics


class UnknownMultiplicityTrajectoryIPSTrainer(FullTrajectoryPPOTrainer):
    """Terminal IPS plus conditional path entropy, with hidden multiplicity."""

    def __init__(
        self,
        config: TrainConfig | None = None,
        *,
        device: str = "cpu",
        path_coefficient: float = 1.0,
        path_warmup_updates: int = 100,
        path_ramp_updates: int = 100,
        path_count_decay: float = 0.95,
    ) -> None:
        if path_coefficient < 0:
            raise ValueError("path_coefficient must be non-negative")
        if path_warmup_updates < 0 or path_ramp_updates < 0:
            raise ValueError("path schedule lengths must be non-negative")
        if not 0 <= path_count_decay < 1:
            raise ValueError("path_count_decay must be in [0, 1)")
        self.target_path_coefficient = float(path_coefficient)
        self.path_warmup_updates = int(path_warmup_updates)
        self.path_ramp_updates = int(path_ramp_updates)
        self.path_count_decay = float(path_count_decay)
        self.current_path_coefficient = 0.0
        self._ema_trajectory_counts: dict[object, float] = {}
        self._ema_outcome_counts: dict[object, float] = {}
        super().__init__(config, device=device)

    def _on_update_start(self, update_step: int) -> None:
        progress = update_step - self.path_warmup_updates
        if progress <= 0:
            fraction = 0.0
        elif self.path_ramp_updates == 0:
            fraction = 1.0
        else:
            fraction = min(1.0, progress / self.path_ramp_updates)
        self.current_path_coefficient = self.target_path_coefficient * fraction

    def _group_advantages(self, episodes: list[Episode]) -> float:
        for key in self._ema_trajectory_counts:
            self._ema_trajectory_counts[key] *= self.path_count_decay
        for key in self._ema_outcome_counts:
            self._ema_outcome_counts[key] *= self.path_count_decay
        for episode in episodes:
            self._ema_trajectory_counts[episode.trajectory] = (
                self._ema_trajectory_counts.get(episode.trajectory, 0.0) + 1.0
            )
            self._ema_outcome_counts[episode.terminal] = (
                self._ema_outcome_counts.get(episode.terminal, 0.0) + 1.0
            )
        ema_conditional_probabilities = [
            self._ema_trajectory_counts[episode.trajectory]
            / self._ema_outcome_counts[episode.terminal]
            for episode in episodes
        ]
        advantages, metrics = unknown_m_trajectory_advantages(
            [episode.reward for episode in episodes],
            [episode.terminal for episode in episodes],
            [episode.trajectory for episode in episodes],
            path_coefficient=self.current_path_coefficient,
            conditional_probability_override=ema_conditional_probabilities,
            eps=self.config.advantage_eps,
        )
        metrics["path_count_decay"] = self.path_count_decay
        metrics["path_ema_trajectories"] = float(len(self._ema_trajectory_counts))
        for episode, advantage in zip(episodes, advantages):
            for step in episode.steps:
                step.advantage = float(advantage)
        self._last_ips_metrics = metrics
        return metrics["ips_ess"]


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
    parser.add_argument("--path-coef", type=float, default=1.0)
    parser.add_argument("--path-warmup-updates", type=int, default=100)
    parser.add_argument("--path-ramp-updates", type=int, default=100)
    parser.add_argument("--path-count-decay", type=float, default=0.95)
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
        / "unknown_m_trajectory_ips_runs"
        / (
            f"{datetime.now():%Y%m%d_%H%M%S}_b{config.budget}"
            f"_gs{config.group_size}_seed{config.seed}"
        )
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    trainer = UnknownMultiplicityTrajectoryIPSTrainer(
        config,
        device=_resolve_device(args.device),
        path_coefficient=args.path_coef,
        path_warmup_updates=args.path_warmup_updates,
        path_ramp_updates=args.path_ramp_updates,
        path_count_decay=args.path_count_decay,
    )
    print(f"Run directory: {run_dir}")
    print(f"Device: {trainer.device}")
    print(f"DAG: {trainer.environment_summary()}")
    print("Training multiplicity: hidden / unused")
    print(
        "Algorithm: terminal count-IPS + centered conditional trajectory surprisal; "
        f"lambda={args.path_coef:g}, warmup={args.path_warmup_updates}, "
        f"ramp={args.path_ramp_updates}, EMA decay={args.path_count_decay:g}"
    )

    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "train_config": asdict(config),
                "device": str(trainer.device),
                "algorithm": "unknown_m_terminal_ips_plus_conditional_path_entropy",
                "training_uses_multiplicity": False,
                "path_coefficient": args.path_coef,
                "path_warmup_updates": args.path_warmup_updates,
                "path_ramp_updates": args.path_ramp_updates,
                "path_count_decay": args.path_count_decay,
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
        subtitle="Dashed lines = hidden-multiplicity conditional-entropy target",
    )
    eval_rows = [row for row in history if "tv_reward_target" in row]
    summary = {
        "environment": trainer.environment_summary(),
        "training_uses_multiplicity": False,
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
