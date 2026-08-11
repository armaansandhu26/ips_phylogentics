"""Exact forward-path-probability IPS for the direction/step DAG.

For every trajectory sampled from the behavior policy, this variant uses the
probability recorded at rollout time instead of estimating a probability from
within-group counts:

    P_F(tau) = product_t pi(direction_t | state_t)
                        * pi(step_t | state_t, direction_t)
    scaled(tau) = R(x(tau)) / P_F(tau)
    advantage(tau) = normalize(scaled(tau))

This is intentionally a path-level variant of ``count_ips.py``.  It does not
correct for the number of paths reaching a terminal.  Consequently its
path-level target gives terminal x aggregate weight proportional to
``m(x) * R(x)`` when there are ``m(x)`` distinct paths to x, rather than the
terminal-level ``R(x)`` target of count-IPS.
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
from count_ips import CountIPSTrainer, Episode
from run_count_ips import (
    _plot_final_counts,
    _plot_training_curves,
    _plot_trajectory_diagnostics,
)


def exact_probability_ips_advantages(
    rewards: Sequence[float],
    outcome_ids: Sequence[object],
    trajectory_ids: Sequence[object],
    forward_log_probabilities: Sequence[float],
    *,
    eps: float = 1e-8,
) -> tuple[np.ndarray, dict[str, float]]:
    """Compute normalized ``reward / exact P_F(trajectory)`` advantages.

    Log-probabilities are accepted because the rollout already stores them and
    summing them is more stable than multiplying per-action probabilities.
    The denominator is still the exact behavior-policy path probability; no
    empirical count or frequency estimate is used.
    """
    size = len(rewards)
    if (
        size == 0
        or len(outcome_ids) != size
        or len(trajectory_ids) != size
        or len(forward_log_probabilities) != size
    ):
        raise ValueError(
            "all exact-probability IPS inputs must have equal non-zero length"
        )

    reward_array = np.asarray(rewards, dtype=np.float64)
    log_p_f = np.asarray(forward_log_probabilities, dtype=np.float64)
    if np.any(~np.isfinite(reward_array)) or np.any(reward_array <= 0.0):
        raise ValueError("rewards must be finite and strictly positive")
    if np.any(~np.isfinite(log_p_f)) or np.any(log_p_f > 1e-7):
        raise ValueError("forward log-probabilities must be finite and <= 0")

    p_f = np.exp(log_p_f)
    if np.any(p_f == 0.0):
        raise ValueError("a forward path probability underflowed to zero")
    scaled = reward_array / p_f
    if np.any(~np.isfinite(scaled)):
        raise ValueError("reward / P_F overflowed; path probability is too small")

    centered = scaled - scaled.mean()
    std = float(scaled.std())
    advantages = centered if std < eps else centered / (std + eps)

    inverse = 1.0 / p_f
    ess = float(inverse.sum() ** 2 / np.square(inverse).sum())
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
        # Exact-path-specific diagnostics.
        "ips_unique_trajectories": float(len(trajectory_counts)),
        "ips_max_trajectory_count": float(max(trajectory_counts.values())),
        "ips_min_trajectory_count": float(min(trajectory_counts.values())),
        "forward_log_probability_mean": float(log_p_f.mean()),
        "forward_log_probability_min": float(log_p_f.min()),
        "forward_log_probability_max": float(log_p_f.max()),
    }


class ExactProbabilityIPSTrainer(CountIPSTrainer):
    """Count-IPS PPO with exact behavior path probabilities as denominators."""

    probability_label = "p_f"

    def _group_advantages(self, episodes: list[Episode]) -> float:
        forward_log_probabilities = [
            sum(step.log_prob_joint for step in episode.steps)
            for episode in episodes
        ]
        advantages, metrics = exact_probability_ips_advantages(
            [episode.reward for episode in episodes],
            [episode.terminal for episode in episodes],
            [episode.trajectory for episode in episodes],
            forward_log_probabilities,
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
                    "name": "exact_probability_ips",
                    "propensity": "behavior_policy_complete_path_probability",
                },
            },
            path,
        )
        return path

    @classmethod
    def load(
        cls, path: Path | str, *, device: str = "cpu"
    ) -> "ExactProbabilityIPSTrainer":
        payload = torch.load(Path(path), map_location=device, weights_only=False)
        algorithm = payload.get("algorithm", {})
        if algorithm.get("name") != "exact_probability_ips":
            raise ValueError("checkpoint is not an exact-probability IPS run")
        trainer = cls(payload["config"], device=device)
        trainer.direction_policy.load_state_dict(payload["direction_policy"])
        trainer.step_policy.load_state_dict(payload["step_policy"])
        trainer.direction_policy.eval()
        trainer.step_policy.eval()
        return trainer


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
        / "exact_probability_ips_runs"
        / (
            f"{datetime.now():%Y%m%d_%H%M%S}_b{config.budget}"
            f"_gs{config.group_size}_seed{config.seed}"
        )
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    trainer = ExactProbabilityIPSTrainer(
        config, device=_resolve_device(args.device)
    )
    print(f"Run directory: {run_dir}")
    print(f"Device: {trainer.device}")
    print(f"DAG: {trainer.environment_summary()}")
    print("Algorithm: reward / exact behavior-policy path probability -> PPO advantage")

    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "train_config": asdict(config),
                "algorithm": "exact_probability_ips",
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
        propensity_title="Exact behavior path probabilities",
        suptitle="Exact path-probability IPS training diagnostics",
    )
    evaluation = trainer.evaluate(args.final_samples)
    sampling = _plot_final_counts(
        trainer,
        evaluation,
        samples=args.final_samples,
        output=run_dir / "sampling_counts.png",
        suptitle="Exact path-probability IPS vs reward-only terminal reference",
    )
    trajectories = _plot_trajectory_diagnostics(
        history,
        trainer,
        evaluation,
        samples=args.final_samples,
        output=run_dir / "trajectory_sampling.png",
    )
    summary = {
        "environment": trainer.environment_summary(),
        "algorithm": "exact_probability_ips",
        "target_note": (
            "Path weighting implies aggregate terminal weight m(x) * R(x); "
            "the final sampling plot retains the reward-only reference for comparison."
        ),
        "final_sampling": sampling,
        "trajectory_sampling": trajectories,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Final ideal-line R^2: {sampling['r2_reward_target']:.4f}")
    print(f"Final TV distance: {sampling['tv_reward_target']:.4f}")
    print(f"Artifacts: {run_dir}")


if __name__ == "__main__":
    main()
