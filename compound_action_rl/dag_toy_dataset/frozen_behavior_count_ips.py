"""Count-IPS with a frozen-policy propensity pool and small optimizer batch.

For each policy update:

1. Keep the current behavior policy unchanged.
2. Collect ``estimation_size`` trajectories from that policy.
3. Estimate terminal probabilities from the complete frozen-policy pool.
4. Select ``optimization_batch_size`` trajectories from the same pool.
5. Compute normalized ``R(o) / p_hat_behavior(o)`` advantages for that small
   optimizer batch and perform one policy update.

This separates the number of samples used to estimate the current terminal
distribution from the number of trajectories used in the stochastic gradient.
It requires no exact terminal-probability computation and never mixes outcome
counts produced by different policy versions.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from config import TrainConfig
from count_ips import CountIPSTrainer, Episode
from dag_env import State
from run_count_ips import (
    _plot_final_counts,
    _plot_training_curves,
    _plot_trajectory_diagnostics,
)


@dataclass(frozen=True)
class FrozenBehaviorCountIPSConfig:
    estimation_size: int = 512
    optimization_batch_size: int = 16

    def validate(self) -> None:
        if self.estimation_size < 1 or self.optimization_batch_size < 1:
            raise ValueError("estimation and optimization sizes must be >= 1")
        if self.optimization_batch_size > self.estimation_size:
            raise ValueError(
                "optimization_batch_size must not exceed estimation_size"
            )


def frozen_behavior_count_ips_advantages(
    rewards: Sequence[float],
    outcome_ids: Sequence[object],
    propensity_counts: Mapping[object, int],
    *,
    estimation_size: int,
    eps: float = 1e-8,
) -> tuple[np.ndarray, dict[str, float]]:
    """Use frozen-policy pool frequencies for a small optimizer batch."""
    if len(rewards) == 0 or len(rewards) != len(outcome_ids):
        raise ValueError("rewards and outcome_ids must have the same non-zero length")
    if estimation_size < 1:
        raise ValueError("estimation_size must be >= 1")
    if sum(propensity_counts.values()) != estimation_size:
        raise ValueError("propensity counts must sum to estimation_size")

    reward_array = np.asarray(rewards, dtype=np.float64)
    p_hat = np.asarray(
        [
            propensity_counts.get(outcome, 0) / estimation_size
            for outcome in outcome_ids
        ],
        dtype=np.float64,
    )
    if np.any(p_hat <= 0.0):
        raise ValueError(
            "every optimizer outcome must occur in the propensity-estimation pool"
        )

    scaled = reward_array / p_hat
    centered = scaled - scaled.mean()
    std = float(scaled.std())
    advantages = centered if std < eps else centered / (std + eps)
    inverse = 1.0 / p_hat
    ess = float(inverse.sum() ** 2 / np.square(inverse).sum())
    optimizer_counts = Counter(outcome_ids)
    estimator_positive_counts = [
        count for count in propensity_counts.values() if count > 0
    ]
    return advantages, {
        "ips_prob_mean": float(p_hat.mean()),
        "ips_prob_min": float(p_hat.min()),
        "ips_prob_max": float(p_hat.max()),
        "ips_unique_outcomes": float(len(optimizer_counts)),
        "ips_max_outcome_count": float(max(optimizer_counts.values())),
        "ips_min_outcome_count": float(min(optimizer_counts.values())),
        "ips_scaled_reward_mean": float(scaled.mean()),
        "ips_scaled_reward_std": std,
        "ips_ess": ess,
        "ips_ess_fraction": ess / len(outcome_ids),
        "advantage_mean": float(advantages.mean()),
        "advantage_std": float(advantages.std()),
        "advantage_min": float(advantages.min()),
        "advantage_max": float(advantages.max()),
        "estimator_unique_outcomes": float(len(estimator_positive_counts)),
        "estimator_max_outcome_count": float(max(estimator_positive_counts)),
        "estimator_min_outcome_count": float(min(estimator_positive_counts)),
        "estimator_probability_mass": float(
            sum(propensity_counts.values()) / estimation_size
        ),
        "estimation_size": float(estimation_size),
        "optimization_batch_size": float(len(outcome_ids)),
    }


class FrozenBehaviorCountIPSTrainer(CountIPSTrainer):
    """Count-IPS with a large frozen-policy estimator and small gradient batch."""

    probability_label = "p_frozen"

    def __init__(
        self,
        config: TrainConfig | None = None,
        *,
        device: str = "cpu",
        frozen_ips: FrozenBehaviorCountIPSConfig | None = None,
    ) -> None:
        super().__init__(config, device=device)
        self.frozen_ips = frozen_ips or FrozenBehaviorCountIPSConfig(
            optimization_batch_size=self.config.group_size
        )
        self.frozen_ips.validate()

    def _assign_frozen_pool_advantages(
        self,
        episodes: list[Episode],
        propensity_counts: Mapping[State, int],
    ) -> float:
        advantages, metrics = frozen_behavior_count_ips_advantages(
            [episode.reward for episode in episodes],
            [episode.terminal for episode in episodes],
            propensity_counts,
            estimation_size=self.frozen_ips.estimation_size,
            eps=self.config.advantage_eps,
        )
        for episode, advantage in zip(episodes, advantages):
            for step in episode.steps:
                step.advantage = float(advantage)
        self._last_ips_metrics = metrics
        return metrics["ips_ess"]

    def train(
        self,
        *,
        eval_every: int | None = None,
        eval_episodes: int = 10_000,
        checkpoint_every: int | None = None,
        checkpoint_dir: Path | str | None = None,
    ) -> list[dict[str, Any]]:
        if checkpoint_every is not None:
            if checkpoint_every < 1:
                raise ValueError("checkpoint_every must be >= 1")
            if checkpoint_dir is None:
                raise ValueError(
                    "checkpoint_dir is required when checkpoint_every is set"
                )
            checkpoint_dir = Path(checkpoint_dir)

        history: list[dict[str, Any]] = []
        for update_step in range(1, self.config.num_updates + 1):
            self._on_update_start(update_step)

            # No optimizer step occurs during this rollout, so all episodes are
            # generated by exactly the same behavior-policy parameters.
            estimation_episodes = self.rollout_batch(
                self.frozen_ips.estimation_size,
                explore=True,
            )
            propensity_counts = Counter(
                episode.terminal for episode in estimation_episodes
            )
            optimization_episodes = estimation_episodes[
                : self.frozen_ips.optimization_batch_size
            ]
            self._assign_frozen_pool_advantages(
                optimization_episodes,
                propensity_counts,
            )
            stats = self.update(optimization_episodes)

            optimizer_counts = Counter(
                episode.terminal for episode in optimization_episodes
            )
            self._seen_terminals.update(propensity_counts)
            row: dict[str, Any] = {
                "step": update_step,
                "mean_reward": float(
                    np.mean([episode.reward for episode in optimization_episodes])
                ),
                "estimator_mean_reward": float(
                    np.mean([episode.reward for episode in estimation_episodes])
                ),
                "mean_length": float(
                    np.mean([len(episode.steps) for episode in optimization_episodes])
                ),
                "unique_terminals": len(optimizer_counts),
                "estimator_unique_terminals": len(propensity_counts),
                "global_unique_outcomes": float(len(self._seen_terminals)),
                "batch_outcome_counts": {
                    state.signature: int(optimizer_counts[state])
                    for state in self.terminals
                },
                "batch_outcome_probs": {
                    state.signature: float(
                        optimizer_counts[state]
                        / self.frozen_ips.optimization_batch_size
                    )
                    for state in self.terminals
                },
                "estimator_outcome_counts": {
                    state.signature: int(propensity_counts[state])
                    for state in self.terminals
                },
                "estimator_outcome_probs": {
                    state.signature: float(
                        propensity_counts[state] / self.frozen_ips.estimation_size
                    )
                    for state in self.terminals
                },
                **self._last_ips_metrics,
                **stats,
            }
            if eval_every and (update_step == 1 or update_step % eval_every == 0):
                row.update(self.evaluate(eval_episodes))
            history.append(row)

            if checkpoint_every and update_step % checkpoint_every == 0:
                assert checkpoint_dir is not None
                checkpoint_path = self.save(
                    checkpoint_dir / f"checkpoint_update_{update_step:06d}.pt",
                    update_step=update_step,
                )
                print(f"Checkpoint: {checkpoint_path}")
            if update_step == 1 or update_step % self.config.log_every == 0:
                print(
                    f"update {update_step:4d}  "
                    f"reward={row['mean_reward']:.3f}  "
                    f"optimizer_outcomes={row['unique_terminals']}  "
                    f"estimator_outcomes={row['estimator_unique_terminals']}  "
                    f"{self.probability_label}={row['ips_prob_mean']:.3f}  "
                    f"grad={row['grad_norm']:.3f}  entropy={row['entropy']:.3f}"
                    + (
                        f"  eval_TV={row['tv_reward_target']:.3f}"
                        if "tv_reward_target" in row
                        else ""
                    )
                )
        return history

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
                    "name": "frozen_behavior_count_ips",
                    "frozen_ips": asdict(self.frozen_ips),
                },
            },
            path,
        )
        return path

    @classmethod
    def load(
        cls, path: Path | str, *, device: str = "cpu"
    ) -> "FrozenBehaviorCountIPSTrainer":
        payload = torch.load(Path(path), map_location=device, weights_only=False)
        algorithm = payload.get("algorithm", {})
        if algorithm.get("name") != "frozen_behavior_count_ips":
            raise ValueError("checkpoint is not a frozen-behavior Count-IPS run")
        trainer = cls(
            payload["config"],
            device=device,
            frozen_ips=FrozenBehaviorCountIPSConfig(**algorithm["frozen_ips"]),
        )
        trainer.direction_policy.load_state_dict(payload["direction_policy"])
        trainer.step_policy.load_state_dict(payload["step_policy"])
        trainer.direction_policy.eval()
        trainer.step_policy.eval()
        return trainer


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    return requested


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=128)
    parser.add_argument("--max-step", type=int, default=3)
    parser.add_argument("--num-updates", type=int, default=500)
    parser.add_argument("--estimation-size", type=int, default=512)
    parser.add_argument("--optimization-batch-size", type=int, default=16)
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

    frozen_ips = FrozenBehaviorCountIPSConfig(
        estimation_size=args.estimation_size,
        optimization_batch_size=args.optimization_batch_size,
    )
    config = TrainConfig(
        budget=args.budget,
        max_step=args.max_step,
        terminal_rewards=(
            tuple(args.terminal_rewards) if args.terminal_rewards is not None else None
        ),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        group_size=args.optimization_batch_size,
        num_groups=1,
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
        / "frozen_behavior_count_ips_runs"
        / (
            f"{datetime.now():%Y%m%d_%H%M%S}_b{config.budget}_"
            f"est{frozen_ips.estimation_size}_"
            f"opt{frozen_ips.optimization_batch_size}_seed{config.seed}"
        )
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    trainer = FrozenBehaviorCountIPSTrainer(
        config,
        device=_resolve_device(args.device),
        frozen_ips=frozen_ips,
    )
    print(f"Run directory: {run_dir}")
    print(f"Device: {trainer.device}")
    print(f"DAG: {trainer.environment_summary()}")
    print(
        "Frozen behavior Count-IPS: "
        f"estimation_size={frozen_ips.estimation_size}, "
        f"optimization_batch_size={frozen_ips.optimization_batch_size}"
    )

    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "train_config": asdict(config),
                "frozen_ips_config": asdict(frozen_ips),
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
        propensity_title="Frozen-policy terminal propensities",
        suptitle="Frozen-behavior Count-IPS diagnostics",
    )
    evaluation = trainer.evaluate(args.final_samples)
    sampling = _plot_final_counts(
        trainer,
        evaluation,
        samples=args.final_samples,
        output=run_dir / "sampling_counts.png",
        suptitle="Frozen-behavior Count-IPS vs ideal reward sampling",
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
        "frozen_ips": asdict(frozen_ips),
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
