"""Annealed-rollout Count-IPS for small and very large output spaces.

This module provides two related trainers. Both keep the advantage group size
fixed, collect many groups early in training, anneal exploration, and reduce
the number of groups over time.

``ScheduledGroupLocalCountIPSTrainer`` computes both empirical propensities and
advantage normalization independently inside every group::

    p_hat_group(o) = count_group(o) / group_size

``ScheduledFrozenPoolCountIPSTrainer`` first collects every group under one
unchanged behavior policy, estimates propensities from that complete pool, and
then normalizes the resulting IPS scores independently inside each group::

    p_hat_pool(o) = count_pool(o) / pool_size

No optimizer step occurs until the complete update pool has been collected and
all group advantages have been assigned. Thus recorded rollout probabilities,
PPO ratios, and frozen-pool propensities refer to one behavior-policy version.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from config import TrainConfig  # noqa: E402
from count_ips import Episode  # noqa: E402
from epsilon_greedy_count_ips import (  # noqa: E402
    EpsilonGreedyCountIPSTrainer,
    ExplorationConfig,
    annealed_value,
    _plot_exploration,
)
from frozen_behavior_count_ips import (  # noqa: E402
    frozen_behavior_count_ips_advantages,
)
from run_count_ips import (  # noqa: E402
    _plot_final_counts,
    _plot_training_curves,
    _plot_trajectory_diagnostics,
)


ScheduleKind = Literal["linear", "cosine"]


@dataclass(frozen=True)
class RolloutScheduleConfig:
    """Schedule for the number of fixed-size groups collected per update."""

    groups_start: int = 64
    groups_end: int = 4
    anneal_updates: int | None = None
    schedule: ScheduleKind = "cosine"

    def validate(self) -> None:
        if self.groups_start < 1 or self.groups_end < 1:
            raise ValueError("groups_start and groups_end must be >= 1")
        if self.groups_end > self.groups_start:
            raise ValueError("groups_end must not exceed groups_start")
        if self.anneal_updates is not None and self.anneal_updates < 1:
            raise ValueError("anneal_updates must be >= 1")
        if self.schedule not in ("linear", "cosine"):
            raise ValueError("schedule must be 'linear' or 'cosine'")


def scheduled_group_count(
    groups_start: int,
    groups_end: int,
    *,
    update_step: int,
    anneal_updates: int,
    schedule: ScheduleKind,
) -> int:
    """Anneal an integer group count while exactly preserving its endpoints."""
    if groups_start < 1 or groups_end < 1:
        raise ValueError("group counts must be >= 1")
    value = annealed_value(
        float(groups_start),
        float(groups_end),
        update_step=update_step,
        anneal_updates=anneal_updates,
        schedule=schedule,
    )
    lower = min(groups_start, groups_end)
    upper = max(groups_start, groups_end)
    return min(upper, max(lower, int(math.floor(value + 0.5))))


class ScheduledGroupLocalCountIPSTrainer(EpsilonGreedyCountIPSTrainer):
    """Annealed exploration and rollout count with strictly local Count-IPS."""

    probability_label = "p_group"
    algorithm_name = "scheduled_group_local_count_ips"

    def __init__(
        self,
        config: TrainConfig | None = None,
        *,
        device: str = "cpu",
        exploration: ExplorationConfig | None = None,
        rollout_schedule: RolloutScheduleConfig | None = None,
    ) -> None:
        super().__init__(config, device=device, exploration=exploration)
        self.rollout_schedule = rollout_schedule or RolloutScheduleConfig(
            groups_start=self.config.num_groups,
            groups_end=self.config.num_groups,
        )
        self.rollout_schedule.validate()
        self.rollout_anneal_updates = (
            self.config.num_updates
            if self.rollout_schedule.anneal_updates is None
            else self.rollout_schedule.anneal_updates
        )
        self.current_rollout_groups = self.rollout_schedule.groups_start

    def _on_update_start(self, update_step: int) -> None:
        super()._on_update_start(update_step)
        self.current_rollout_groups = scheduled_group_count(
            self.rollout_schedule.groups_start,
            self.rollout_schedule.groups_end,
            update_step=update_step,
            anneal_updates=self.rollout_anneal_updates,
            schedule=self.rollout_schedule.schedule,
        )

    def _groups_for_update(self) -> int:
        return self.current_rollout_groups

    def _schedule_metrics(self) -> dict[str, float]:
        return {
            "scheduled_rollout_groups": float(self.current_rollout_groups),
            "scheduled_rollouts_per_update": float(
                self.current_rollout_groups * self.config.group_size
            ),
        }

    def _group_advantages(self, episodes: list[Episode]) -> float:
        ess = super()._group_advantages(episodes)
        counts = Counter(episode.terminal for episode in episodes)
        singleton_episodes = sum(
            count for count in counts.values() if count == 1
        )
        self._last_ips_metrics.update(
            {
                **self._schedule_metrics(),
                "group_singleton_fraction": singleton_episodes / len(episodes),
            }
        )
        return ess

    def _update_training_groups(
        self, groups: list[list[Episode]]
    ) -> dict[str, float]:
        """Accumulate equal-weight group gradients before each optimizer step.

        This produces the same mean episode loss as concatenating equal-sized
        groups, but the largest policy-loss tensor contains only ``group_size``
        episodes. The behavior policy remains unchanged until every group has
        contributed its gradient.
        """
        if not groups or any(
            len(group) != self.config.group_size for group in groups
        ):
            raise ValueError(
                "every optimizer group must have config.group_size episodes"
            )

        metric_names = (
            "loss",
            "policy_loss",
            "entropy",
            "mean_importance_ratio",
            "max_importance_ratio",
            "min_importance_ratio",
            "clip_fraction",
            "grad_norm",
            "param_norm",
        )
        totals = {name: 0.0 for name in metric_names}
        parameters = list(self.direction_policy.parameters()) + list(
            self.step_policy.parameters()
        )
        for _ in range(self.config.train_epochs):
            self.optimizer.zero_grad(set_to_none=True)
            group_stats: list[dict[str, float]] = []
            for group in groups:
                loss, stats = self._joint_policy_loss(group)
                (loss / len(groups)).backward()
                group_stats.append(stats)

            grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters, self.config.grad_clip_norm
            )
            self.optimizer.step()
            epoch_stats = {
                name: float(np.mean([stats[name] for stats in group_stats]))
                for name in (
                    "loss",
                    "policy_loss",
                    "entropy",
                    "mean_importance_ratio",
                    "clip_fraction",
                )
            }
            epoch_stats["max_importance_ratio"] = max(
                stats["max_importance_ratio"] for stats in group_stats
            )
            epoch_stats["min_importance_ratio"] = min(
                stats["min_importance_ratio"] for stats in group_stats
            )
            epoch_stats["grad_norm"] = float(grad_norm.item())
            epoch_stats["param_norm"] = float(
                sum(
                    parameter.detach().norm().item() ** 2
                    for parameter in parameters
                )
                ** 0.5
            )
            for name in metric_names:
                totals[name] += epoch_stats[name]

        return {
            **{
                name: value / self.config.train_epochs
                for name, value in totals.items()
            },
            "gradient_accumulation_groups": float(len(groups)),
        }

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
                    "name": self.algorithm_name,
                    "exploration": asdict(self.exploration),
                    "rollout_schedule": asdict(self.rollout_schedule),
                    "current_epsilon": self.current_epsilon,
                    "current_temperature": self.current_temperature,
                    "current_rollout_groups": self.current_rollout_groups,
                },
            },
            path,
        )
        return path

    @classmethod
    def load(
        cls, path: Path | str, *, device: str = "cpu"
    ) -> "ScheduledGroupLocalCountIPSTrainer":
        payload = torch.load(Path(path), map_location=device, weights_only=False)
        algorithm = payload.get("algorithm", {})
        if algorithm.get("name") != cls.algorithm_name:
            raise ValueError(
                f"checkpoint is not a {cls.algorithm_name} run"
            )
        trainer = cls(
            payload["config"],
            device=device,
            exploration=ExplorationConfig(**algorithm["exploration"]),
            rollout_schedule=RolloutScheduleConfig(
                **algorithm["rollout_schedule"]
            ),
        )
        trainer.current_epsilon = float(algorithm["current_epsilon"])
        trainer.current_temperature = float(algorithm["current_temperature"])
        trainer.current_rollout_groups = int(
            algorithm["current_rollout_groups"]
        )
        trainer.direction_policy.load_state_dict(payload["direction_policy"])
        trainer.step_policy.load_state_dict(payload["step_policy"])
        trainer.direction_policy.eval()
        trainer.step_policy.eval()
        return trainer


class ScheduledFrozenPoolCountIPSTrainer(ScheduledGroupLocalCountIPSTrainer):
    """Use a scheduled frozen rollout pool for propensity estimation.

    The pool contains ``current_rollout_groups * group_size`` episodes. Pool
    counts determine every sampled outcome's propensity, while score centering
    and normalization remain local to each group of ``group_size`` episodes.
    """

    probability_label = "p_frozen"
    algorithm_name = "scheduled_frozen_pool_count_ips"

    def _collect_training_groups(
        self,
    ) -> tuple[list[list[Episode]], list[dict[str, float]]]:
        group_size = self.config.group_size
        pool_size = self.current_rollout_groups * group_size
        pool = self.rollout_batch(pool_size, explore=True)
        propensity_counts = Counter(episode.terminal for episode in pool)
        groups = [
            pool[start : start + group_size]
            for start in range(0, pool_size, group_size)
        ]

        pool_singletons = sum(
            count for count in propensity_counts.values() if count == 1
        )
        group_metrics: list[dict[str, float]] = []
        for group in groups:
            advantages, metrics = frozen_behavior_count_ips_advantages(
                [episode.reward for episode in group],
                [episode.terminal for episode in group],
                propensity_counts,
                estimation_size=pool_size,
                eps=self.config.advantage_eps,
            )
            for episode, advantage in zip(group, advantages):
                for step in episode.steps:
                    step.advantage = float(advantage)

            group_counts = Counter(episode.terminal for episode in group)
            group_singletons = sum(
                count for count in group_counts.values() if count == 1
            )
            metrics.update(
                {
                    "exploration_epsilon": self.current_epsilon,
                    "exploration_temperature": self.current_temperature,
                    **self._schedule_metrics(),
                    "group_singleton_fraction": group_singletons / group_size,
                    "pool_singleton_fraction": pool_singletons / pool_size,
                    "propensity_pool_size": float(pool_size),
                    "propensity_pool_unique_outcomes": float(
                        len(propensity_counts)
                    ),
                }
            )
            group_metrics.append(metrics)

        self._last_ips_metrics = dict(group_metrics[-1])
        return groups, group_metrics


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is not available")
    return requested


def _plot_rollout_schedule(history: list[dict[str, Any]], output: Path) -> None:
    steps = [row["step"] for row in history]
    groups = [row["rollout_groups"] for row in history]
    rollouts = [row["rollouts_per_update"] for row in history]
    fig, group_axis = plt.subplots(figsize=(8, 4.5))
    rollout_axis = group_axis.twinx()
    group_axis.step(steps, groups, where="mid", color="#6c5ce7", label="groups")
    rollout_axis.plot(steps, rollouts, color="#00b894", label="rollouts")
    group_axis.set_xlabel("update")
    group_axis.set_ylabel("advantage groups", color="#6c5ce7")
    rollout_axis.set_ylabel("rollouts per update", color="#00b894")
    group_axis.grid(alpha=0.22)
    fig.suptitle("Annealed rollout budget")
    fig.tight_layout()
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--propensity-mode",
        choices=("group_local", "frozen_pool"),
        default="group_local",
    )
    parser.add_argument("--budget", type=int, default=32)
    parser.add_argument("--max-step", type=int, default=3)
    parser.add_argument("--num-updates", type=int, default=500)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--groups-start", type=int, default=64)
    parser.add_argument("--groups-end", type=int, default=4)
    parser.add_argument("--rollout-anneal-updates", type=int, default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--train-epochs", type=int, default=1)
    parser.add_argument("--epsilon-start", type=float, default=0.50)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--temperature-start", type=float, default=2.5)
    parser.add_argument("--temperature-end", type=float, default=1.0)
    parser.add_argument("--exploration-anneal-updates", type=int, default=None)
    parser.add_argument("--schedule", choices=("linear", "cosine"), default="cosine")
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
        num_groups=args.groups_start,
        num_updates=args.num_updates,
        train_epochs=args.train_epochs,
        lr=args.lr,
        entropy_coef=args.entropy_coef,
        clip_ratio=args.clip_ratio,
        seed=args.seed,
        log_every=args.log_every,
    )
    exploration = ExplorationConfig(
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        temperature_start=args.temperature_start,
        temperature_end=args.temperature_end,
        anneal_updates=args.exploration_anneal_updates,
        schedule=args.schedule,
    )
    rollout_schedule = RolloutScheduleConfig(
        groups_start=args.groups_start,
        groups_end=args.groups_end,
        anneal_updates=args.rollout_anneal_updates,
        schedule=args.schedule,
    )
    trainer_type = (
        ScheduledGroupLocalCountIPSTrainer
        if args.propensity_mode == "group_local"
        else ScheduledFrozenPoolCountIPSTrainer
    )
    run_dir = args.run_dir or (
        Path(__file__).resolve().parent
        / "data"
        / "scheduled_rollout_count_ips_runs"
        / (
            f"{datetime.now():%Y%m%d_%H%M%S}_{args.propensity_mode}_"
            f"b{config.budget}_g{config.group_size}_seed{config.seed}"
        )
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    trainer = trainer_type(
        config,
        device=_resolve_device(args.device),
        exploration=exploration,
        rollout_schedule=rollout_schedule,
    )
    print(f"Run directory: {run_dir}")
    print(f"Device: {trainer.device}")
    print(f"DAG: {trainer.environment_summary()}")
    print(
        f"Mode: {args.propensity_mode}; group_size={config.group_size}; "
        f"groups {rollout_schedule.groups_start}->{rollout_schedule.groups_end}; "
        f"epsilon {exploration.epsilon_start:g}->{exploration.epsilon_end:g}; "
        f"temperature {exploration.temperature_start:g}->{exploration.temperature_end:g}"
    )

    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "train_config": asdict(config),
                "exploration_config": asdict(exploration),
                "rollout_schedule_config": asdict(rollout_schedule),
                "propensity_mode": args.propensity_mode,
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
    _plot_training_curves(history, trainer, output=run_dir / "training_curves.png")
    _plot_exploration(history, run_dir / "exploration_schedule.png")
    _plot_rollout_schedule(history, run_dir / "rollout_schedule.png")

    evaluation = trainer.evaluate(args.final_samples)
    sampling = _plot_final_counts(
        trainer,
        evaluation,
        samples=args.final_samples,
        output=run_dir / "sampling_counts.png",
        suptitle=f"Scheduled {args.propensity_mode} Count-IPS vs reward target",
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
        "propensity_mode": args.propensity_mode,
        "exploration": asdict(exploration),
        "rollout_schedule": asdict(rollout_schedule),
        "total_training_rollouts": history[-1]["cumulative_rollouts"],
        "final_sampling": sampling,
        "trajectory_sampling": trajectories,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Total training rollouts: {history[-1]['cumulative_rollouts']:,}")
    print(f"Final ideal-line R^2: {sampling['r2_reward_target']:.4f}")
    print(f"Final TV distance: {sampling['tv_reward_target']:.4f}")
    print(f"Artifacts: {run_dir}")


if __name__ == "__main__":
    main()
