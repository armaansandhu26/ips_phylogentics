"""Backward-corrected exact trajectory IPS for the direction/step DAG.

For a trajectory ``tau`` sampled by the behavior policy and ending at ``x``:

    scaled(tau) = R(x) * P_B(tau | x) / P_F(tau)
    advantage(tau) = normalize(scaled(tau))

``P_F`` is the exact product of the rollout-time forward action probabilities.
``P_B`` is fixed and locally uniform over valid parent edges.  It is normalized
over all reverse paths from each terminal, so summing the target trajectory
mass for terminal x gives exactly R(x), without knowing path multiplicity.

This runner retains the token-level clipped PPO loss from ``count_ips.py`` so
the experiment changes the advantage scaling rule, not the optimizer.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from config import TrainConfig
from count_ips import Episode
from dag_env import uniform_backward_log_probability
from exact_probability_ips import ExactProbabilityIPSTrainer, _resolve_device
from run_count_ips import (
    _plot_final_counts,
    _plot_training_curves,
    _plot_trajectory_diagnostics,
)


def backward_corrected_ips_advantages(
    rewards: Sequence[float],
    outcome_ids: Sequence[object],
    trajectory_ids: Sequence[object],
    forward_log_probabilities: Sequence[float],
    backward_log_probabilities: Sequence[float],
    *,
    eps: float = 1e-8,
) -> tuple[np.ndarray, dict[str, float]]:
    """Normalize the raw IPS weights ``R * P_B / P_F``."""
    size = len(rewards)
    if (
        size == 0
        or len(outcome_ids) != size
        or len(trajectory_ids) != size
        or len(forward_log_probabilities) != size
        or len(backward_log_probabilities) != size
    ):
        raise ValueError(
            "all backward-corrected IPS inputs must have equal non-zero length"
        )

    reward_array = np.asarray(rewards, dtype=np.float64)
    log_p_f = np.asarray(forward_log_probabilities, dtype=np.float64)
    log_p_b = np.asarray(backward_log_probabilities, dtype=np.float64)
    if np.any(~np.isfinite(reward_array)) or np.any(reward_array <= 0.0):
        raise ValueError("rewards must be finite and strictly positive")
    if np.any(~np.isfinite(log_p_f)) or np.any(log_p_f > 1e-7):
        raise ValueError("forward log-probabilities must be finite and <= 0")
    if np.any(~np.isfinite(log_p_b)) or np.any(log_p_b > 1e-7):
        raise ValueError("backward log-probabilities must be finite and <= 0")

    p_f = np.exp(log_p_f)
    p_b = np.exp(log_p_b)
    if np.any(p_f == 0.0) or np.any(p_b == 0.0):
        raise ValueError("a forward or backward path probability underflowed to zero")

    log_weights = np.log(reward_array) + log_p_b - log_p_f
    scaled = np.exp(log_weights)
    if np.any(~np.isfinite(scaled)):
        raise ValueError("R * P_B / P_F overflowed; the IPS weight is too large")

    centered = scaled - scaled.mean()
    std = float(scaled.std())
    advantages = centered if std < eps else centered / (std + eps)

    squared_weight_sum = float(np.square(scaled).sum())
    ess = float(scaled.sum() ** 2 / max(squared_weight_sum, eps))
    outcome_counts = Counter(outcome_ids)
    trajectory_counts = Counter(trajectory_ids)
    return advantages, {
        # Compatibility keys consumed by CountIPSTrainer.train and its plots.
        "ips_prob_mean": float(p_f.mean()),
        "ips_prob_min": float(p_f.min()),
        "ips_prob_max": float(p_f.max()),
        "ips_unique_outcomes": float(len(outcome_counts)),
        "ips_max_outcome_count": float(max(outcome_counts.values())),
        "ips_min_outcome_count": float(min(outcome_counts.values())),
        "ips_scaled_reward_mean": float(scaled.mean()),
        "ips_scaled_reward_std": std,
        "ips_ess": ess,
        "ips_ess_fraction": ess / size,
        "advantage_mean": float(advantages.mean()),
        "advantage_std": float(advantages.std()),
        "advantage_min": float(advantages.min()),
        "advantage_max": float(advantages.max()),
        # Backward-corrected trajectory diagnostics.
        "ips_unique_trajectories": float(len(trajectory_counts)),
        "ips_max_trajectory_count": float(max(trajectory_counts.values())),
        "ips_min_trajectory_count": float(min(trajectory_counts.values())),
        "forward_log_probability_mean": float(log_p_f.mean()),
        "backward_log_probability_mean": float(log_p_b.mean()),
        "backward_probability_mean": float(p_b.mean()),
        "backward_probability_min": float(p_b.min()),
        "backward_probability_max": float(p_b.max()),
        "log_importance_weight_mean": float(log_weights.mean()),
        "importance_weight_min": float(scaled.min()),
        "importance_weight_max": float(scaled.max()),
    }


class BackwardCorrectedIPSTrainer(ExactProbabilityIPSTrainer):
    """Raw trajectory IPS with a normalized fixed backward reference."""

    def _group_advantages(self, episodes: list[Episode]) -> float:
        forward_log_probabilities = [
            sum(step.log_prob_joint for step in episode.steps)
            for episode in episodes
        ]
        backward_log_probabilities = [
            uniform_backward_log_probability(
                episode.trajectory,
                max_step=self.config.max_step,
            )
            for episode in episodes
        ]
        advantages, metrics = backward_corrected_ips_advantages(
            [episode.reward for episode in episodes],
            [episode.terminal for episode in episodes],
            [episode.trajectory for episode in episodes],
            forward_log_probabilities,
            backward_log_probabilities,
            eps=self.config.advantage_eps,
        )
        for episode, advantage in zip(episodes, advantages):
            for step in episode.steps:
                step.advantage = float(advantage)
        self._last_ips_metrics = metrics
        return metrics["ips_ess"]

    def save(self, path: Path | str, *, update_step: int = 0) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": self.config,
                "direction_policy": self.direction_policy.state_dict(),
                "step_policy": self.step_policy.state_dict(),
                "update_step": update_step,
                "algorithm": {
                    "name": "backward_corrected_ips",
                    "forward_propensity": (
                        "behavior_policy_complete_path_probability"
                    ),
                    "backward_policy": "uniform_over_valid_parents",
                    "raw_weight": "R(x) * P_B(tau|x) / P_F(tau)",
                },
            },
            path,
        )
        return path

    @classmethod
    def load(
        cls, path: Path | str, *, device: str = "cpu"
    ) -> "BackwardCorrectedIPSTrainer":
        payload = torch.load(Path(path), map_location=device, weights_only=False)
        algorithm = payload.get("algorithm", {})
        if algorithm.get("name") != "backward_corrected_ips":
            raise ValueError("checkpoint is not a backward-corrected IPS run")
        trainer = cls(payload["config"], device=device)
        trainer.direction_policy.load_state_dict(payload["direction_policy"])
        trainer.step_policy.load_state_dict(payload["step_policy"])
        trainer.direction_policy.eval()
        trainer.step_policy.eval()
        return trainer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=32)
    parser.add_argument("--max-step", type=int, default=3)
    parser.add_argument("--num-updates", type=int, default=500)
    parser.add_argument("--group-size", type=int, default=128)
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
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--terminal-rewards", type=float, nargs="+", default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
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
        entropy_coef=args.entropy_coef,
        clip_ratio=args.clip_ratio,
        seed=args.seed,
        log_every=args.log_every,
    )
    run_dir = args.run_dir or (
        Path(__file__).resolve().parent
        / "data"
        / "backward_corrected_ips_runs"
        / (
            f"{datetime.now():%Y%m%d_%H%M%S}_b{config.budget}"
            f"_gs{config.group_size}_seed{config.seed}"
        )
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    trainer = BackwardCorrectedIPSTrainer(
        config, device=_resolve_device(args.device)
    )
    print(f"Run directory: {run_dir}")
    print(f"Device: {trainer.device}")
    print(f"DAG: {trainer.environment_summary()}")
    print("Algorithm: R(x) * P_B(tau|x) / P_F(tau) -> PPO advantage")

    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "train_config": asdict(config),
                "algorithm": "backward_corrected_ips",
                "backward_policy": "uniform_over_valid_parents",
                "device": str(trainer.device),
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
    _plot_training_curves(
        history,
        trainer,
        output=training_plot,
        propensity_title="Exact forward path propensities",
        suptitle="Backward-corrected IPS training diagnostics",
    )
    evaluation = trainer.evaluate(args.final_samples)
    sampling = _plot_final_counts(
        trainer,
        evaluation,
        samples=args.final_samples,
        output=run_dir / "sampling_counts.png",
        suptitle="Backward-corrected IPS vs ideal reward sampling",
    )
    trajectories = _plot_trajectory_diagnostics(
        history,
        trainer,
        evaluation,
        samples=args.final_samples,
        output=run_dir / "trajectory_sampling.png",
    )
    eval_rows = [row for row in history if "tv_reward_target" in row]
    summary = {
        "environment": trainer.environment_summary(),
        "algorithm": "backward_corrected_ips",
        "backward_policy": "uniform_over_valid_parents",
        "target": "P(x) proportional to R(x)",
        "final_sampling": sampling,
        "trajectory_sampling": trajectories,
        "best_eval": (
            min(eval_rows, key=lambda row: row["tv_reward_target"])
            if eval_rows
            else None
        ),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Final ideal-line R^2: {sampling['r2_reward_target']:.4f}")
    print(f"Final TV distance: {sampling['tv_reward_target']:.4f}")
    print(f"Artifacts: {run_dir}")


if __name__ == "__main__":
    main()
