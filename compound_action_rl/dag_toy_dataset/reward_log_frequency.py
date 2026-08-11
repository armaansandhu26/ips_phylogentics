"""PPO on the direction/step DAG with reward-log-frequency advantages.

For each independently sampled group of episodes, let ``freq(o)`` be the
number of times terminal outcome ``o`` appears in that group. Every episode
ending at ``o`` receives the raw, unnormalized advantage

    advantage(o) = reward(o) * log(reward(o) / freq(o))

The advantage is then used by the same token-level clipped PPO objective as
``count_ips.py``. ``log`` is the natural logarithm and ``freq`` is the raw
count, not the empirical probability ``freq / group_size``.
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
from count_ips import CountIPSTrainer, Episode
from run_count_ips import (
    _plot_final_counts,
    _plot_training_curves,
    _plot_trajectory_diagnostics,
)


def reward_log_frequency_advantages(
    rewards: Sequence[float],
    outcome_ids: Sequence[object],
) -> tuple[np.ndarray, dict[str, float]]:
    """Compute raw ``reward * log(reward / within-group count)`` advantages."""
    if len(rewards) == 0 or len(rewards) != len(outcome_ids):
        raise ValueError("rewards and outcome_ids must have the same non-zero length")

    reward_array = np.asarray(rewards, dtype=np.float64)
    if not np.all(np.isfinite(reward_array)):
        raise ValueError("rewards must be finite")
    if np.any(reward_array <= 0.0):
        raise ValueError(
            "rewards must be strictly positive because the advantage uses log(reward)"
        )

    counts = Counter(outcome_ids)
    group_size = len(outcome_ids)
    frequencies = np.asarray(
        [counts[outcome] for outcome in outcome_ids], dtype=np.float64
    )
    p_hat = frequencies / group_size
    log_reward_frequency_ratio = np.log(reward_array / frequencies)
    advantages = reward_array * log_reward_frequency_ratio

    # Keep the common metric names expected by CountIPSTrainer's logger and
    # plotting helpers, while also recording metrics specific to this formula.
    inverse_probability = 1.0 / p_hat
    ess = float(
        inverse_probability.sum() ** 2
        / np.square(inverse_probability).sum()
    )
    metrics = {
        "ips_prob_mean": float(p_hat.mean()),
        "ips_prob_min": float(p_hat.min()),
        "ips_prob_max": float(p_hat.max()),
        "ips_unique_outcomes": float(len(counts)),
        "ips_max_outcome_count": float(max(counts.values())),
        "ips_min_outcome_count": float(min(counts.values())),
        "ips_scaled_reward_mean": float(advantages.mean()),
        "ips_scaled_reward_std": float(advantages.std()),
        "ips_ess": ess,
        "ips_ess_fraction": ess / group_size,
        "frequency_mean": float(frequencies.mean()),
        "frequency_min": float(frequencies.min()),
        "frequency_max": float(frequencies.max()),
        "log_reward_frequency_ratio_mean": float(
            log_reward_frequency_ratio.mean()
        ),
        "advantage_mean": float(advantages.mean()),
        "advantage_std": float(advantages.std()),
        "advantage_min": float(advantages.min()),
        "advantage_max": float(advantages.max()),
    }
    return advantages, metrics


class RewardLogFrequencyTrainer(CountIPSTrainer):
    """Count-group trainer using ``R * log(R / freq)`` as its advantage."""

    probability_label = "freq/G"

    def _group_advantages(self, episodes: list[Episode]) -> float:
        advantages, metrics = reward_log_frequency_advantages(
            [episode.reward for episode in episodes],
            [episode.terminal for episode in episodes],
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
                "algorithm": {"name": "reward_log_frequency"},
            },
            path,
        )
        return path

    @classmethod
    def load(
        cls, path: Path | str, *, device: str = "cpu"
    ) -> "RewardLogFrequencyTrainer":
        payload = torch.load(Path(path), map_location=device, weights_only=False)
        algorithm = payload.get("algorithm", {})
        if algorithm.get("name") != "reward_log_frequency":
            raise ValueError("checkpoint is not a reward-log-frequency run")
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
    parser.add_argument(
        "--terminal-rewards",
        type=float,
        nargs="+",
        default=None,
        metavar="R",
        help="budget + 1 rewards in increasing terminal x order",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="auto uses CUDA when available, otherwise CPU",
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
        entropy_coef=args.entropy_coef,
        clip_ratio=args.clip_ratio,
        seed=args.seed,
        log_every=args.log_every,
    )
    run_dir = args.run_dir or (
        Path(__file__).resolve().parent
        / "data"
        / "reward_log_frequency_runs"
        / (
            f"{datetime.now():%Y%m%d_%H%M%S}_b{config.budget}"
            f"_gs{config.group_size}_seed{config.seed}"
        )
    )
    run_dir.mkdir(parents=True, exist_ok=False)

    trainer = RewardLogFrequencyTrainer(
        config, device=_resolve_device(args.device)
    )
    print(f"Run directory: {run_dir}")
    print(f"Device: {trainer.device}")
    print(f"DAG: {trainer.environment_summary()}")
    print("Advantage: reward(o) * ln(reward(o) / within-group count(o))")

    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "train_config": asdict(config),
                "advantage": "reward * ln(reward / raw_group_frequency)",
                "normalized_advantage": False,
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

    _plot_training_curves(
        history,
        trainer,
        output=run_dir / "training_curves.png",
        propensity_title="Within-group outcome frequencies",
        suptitle="Reward-log-frequency PPO diagnostics",
    )
    evaluation = trainer.evaluate(args.final_samples)
    sampling = _plot_final_counts(
        trainer,
        evaluation,
        samples=args.final_samples,
        output=run_dir / "sampling_counts.png",
        suptitle="Reward-log-frequency PPO vs ideal reward sampling",
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
        "advantage": "reward * ln(reward / raw_group_frequency)",
        "normalized_advantage": False,
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
