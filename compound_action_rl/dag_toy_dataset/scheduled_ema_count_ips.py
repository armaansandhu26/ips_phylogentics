"""Scheduled Count-IPS with exploration-history propensities.

Decouples the terminal probability estimate ``p̂`` from the on-policy gradient
group. Nothing requires ``p̂`` to come from the same ``G`` samples used for the
policy-gradient step.

Each update:

1. Anneal the number of fixed-size groups (``groups_start -> groups_end``).
2. Roll out every group under one frozen behavior policy.
3. Record every explored terminal into a lifetime histogram and update a global
   EMA of group outcome frequencies::

       f_t(o) = count_group(o) / G
       p_ema(o) <- (1 - alpha) * p_ema(o) + alpha * f_t(o)

   Optionally use the lifetime histogram itself as ``p̂(o) = n(o) / N``.
4. Scale rewards with the history-based ``p̂`` and z-normalize inside each
   on-policy group of size ``G`` (default 16).
5. Accumulate PPO gradients over those groups.

This raises the statistical sample size for ``p̂`` (EMA effective window
``~1/alpha`` groups, or the full exploration history) without enlarging the
per-group gradient batch — the regime that makes small-``G`` IPS viable. The
only cost is mild off-policy staleness in ``p̂``; terminal frequencies drift
slowly per step, so a modest ``alpha`` is fine.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Mapping, MutableMapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from config import TrainConfig  # noqa: E402
from count_ips import Episode  # noqa: E402
from dag_env import State  # noqa: E402
from ema_count_ips import (  # noqa: E402
    EMACountIPSConfig,
    ema_count_ips_advantages,
    update_ema_outcome_frequencies,
)
from epsilon_greedy_count_ips import (  # noqa: E402
    ExplorationConfig,
    _plot_exploration,
)
from run_count_ips import (  # noqa: E402
    _plot_final_counts,
    _plot_training_curves,
    _plot_trajectory_diagnostics,
)
from scheduled_rollout_count_ips import (  # noqa: E402
    RolloutScheduleConfig,
    ScheduledGroupLocalCountIPSTrainer,
    _plot_rollout_schedule,
)


PropensityMode = Literal["ema", "running"]


@dataclass(frozen=True)
class ExplorationHistoryIPSConfig:
    """History-based propensity estimator shared across gradient groups."""

    propensity_mode: PropensityMode = "ema"
    alpha: float = 0.1
    initialization: Literal["first_batch", "uniform"] = "first_batch"
    decay_absent_outcomes: bool = True
    tracker_eps: float = 1e-6
    ips_weight_mode: Literal["normalized", "raw"] = "normalized"
    # Update the EMA once from the full update pool (``pool``) or once per
    # advantage group (``group``). Both still accumulate across updates.
    ema_update_unit: Literal["pool", "group"] = "group"

    def validate(self) -> None:
        if self.propensity_mode not in ("ema", "running"):
            raise ValueError("propensity_mode must be 'ema' or 'running'")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if self.initialization not in ("first_batch", "uniform"):
            raise ValueError("initialization must be 'first_batch' or 'uniform'")
        if self.tracker_eps <= 0.0:
            raise ValueError("tracker_eps must be > 0")
        if self.ips_weight_mode not in ("normalized", "raw"):
            raise ValueError("ips_weight_mode must be 'normalized' or 'raw'")
        if self.ema_update_unit not in ("pool", "group"):
            raise ValueError("ema_update_unit must be 'pool' or 'group'")

    def as_ema_config(self) -> EMACountIPSConfig:
        return EMACountIPSConfig(
            alpha=self.alpha,
            initialization=self.initialization,
            decay_absent_outcomes=self.decay_absent_outcomes,
            tracker_eps=self.tracker_eps,
            ips_weight_mode=self.ips_weight_mode,
        )


def running_histogram_probabilities(
    outcome_ids: Sequence[object],
    lifetime_counts: Mapping[object, int],
    *,
    tracker_eps: float = 1e-6,
) -> np.ndarray:
    """Return ``n(o) / N`` for each outcome, floored at ``tracker_eps``."""
    total = float(sum(lifetime_counts.values()))
    if total <= 0.0:
        raise ValueError("lifetime_counts must contain at least one visit")
    raw = np.asarray(
        [float(lifetime_counts.get(outcome, 0)) / total for outcome in outcome_ids],
        dtype=np.float64,
    )
    return np.maximum(raw, tracker_eps)


def running_histogram_ips_advantages(
    rewards: Sequence[float],
    outcome_ids: Sequence[object],
    lifetime_counts: Mapping[object, int],
    *,
    tracker_eps: float = 1e-6,
    normalize: bool = True,
    eps: float = 1e-8,
) -> tuple[np.ndarray, dict[str, float]]:
    """Compute ``reward / running-histogram frequency`` episode weights."""
    if len(rewards) == 0 or len(rewards) != len(outcome_ids):
        raise ValueError("rewards and outcome_ids must have the same non-zero length")
    if tracker_eps <= 0.0:
        raise ValueError("tracker_eps must be > 0")

    reward_array = np.asarray(rewards, dtype=np.float64)
    pi_estimate = running_histogram_probabilities(
        outcome_ids, lifetime_counts, tracker_eps=tracker_eps
    )
    if np.any(pi_estimate <= 0.0):
        raise ValueError("every sampled outcome must have a positive frequency")

    scaled = reward_array / pi_estimate
    std = float(scaled.std())
    if normalize:
        centered = scaled - scaled.mean()
        advantages = centered if std < eps else centered / (std + eps)
    else:
        advantages = scaled.copy()

    batch_counts = Counter(outcome_ids)
    inverse = 1.0 / pi_estimate
    ess = float(inverse.sum() ** 2 / np.maximum(np.square(inverse).sum(), eps))
    total_visits = float(sum(lifetime_counts.values()))
    return advantages, {
        "ips_prob_mean": float(pi_estimate.mean()),
        "ips_prob_min": float(pi_estimate.min()),
        "ips_prob_max": float(pi_estimate.max()),
        "ips_unique_outcomes": float(len(batch_counts)),
        "ips_max_outcome_count": float(max(batch_counts.values())),
        "ips_min_outcome_count": float(min(batch_counts.values())),
        "ips_scaled_reward_mean": float(scaled.mean()),
        "ips_scaled_reward_std": std,
        "ips_ess": ess,
        "ips_ess_fraction": ess / len(outcome_ids),
        "advantage_mean": float(advantages.mean()),
        "advantage_std": float(advantages.std()),
        "advantage_min": float(advantages.min()),
        "advantage_max": float(advantages.max()),
        "histogram_total_visits": total_visits,
        "histogram_tracked_outcomes": float(
            sum(1 for count in lifetime_counts.values() if count > 0)
        ),
    }


def update_ema_from_groups(
    groups: Sequence[Sequence[object]],
    ema_frequencies: MutableMapping[object, float],
    *,
    alpha: float,
    decay_absent_outcomes: bool = True,
    update_unit: Literal["pool", "group"] = "group",
) -> dict[object, float]:
    """Update EMA frequencies from explored groups; return last batch freqs."""
    if not groups:
        raise ValueError("groups must be non-empty")
    if update_unit == "pool":
        pool = [outcome for group in groups for outcome in group]
        return update_ema_outcome_frequencies(
            pool,
            ema_frequencies,
            alpha=alpha,
            decay_absent_outcomes=decay_absent_outcomes,
        )

    batch_frequencies: dict[object, float] = {}
    for group in groups:
        batch_frequencies = update_ema_outcome_frequencies(
            group,
            ema_frequencies,
            alpha=alpha,
            decay_absent_outcomes=decay_absent_outcomes,
        )
    return batch_frequencies


class ScheduledEMACountIPSTrainer(ScheduledGroupLocalCountIPSTrainer):
    """Annealed rollout Count-IPS with EMA / lifetime-histogram propensities."""

    probability_label = "p_history"
    algorithm_name = "scheduled_ema_count_ips"

    def __init__(
        self,
        config: TrainConfig | None = None,
        *,
        device: str = "cpu",
        exploration: ExplorationConfig | None = None,
        rollout_schedule: RolloutScheduleConfig | None = None,
        history_ips: ExplorationHistoryIPSConfig | None = None,
    ) -> None:
        super().__init__(
            config,
            device=device,
            exploration=exploration,
            rollout_schedule=rollout_schedule,
        )
        self.history_ips = history_ips or ExplorationHistoryIPSConfig()
        self.history_ips.validate()
        if (
            self.history_ips.propensity_mode == "ema"
            and self.history_ips.initialization == "uniform"
        ):
            initial_probability = 1.0 / len(self.terminals)
            self._ema_terminal_frequencies: dict[State, float] = {
                state: initial_probability for state in self.terminals
            }
        else:
            self._ema_terminal_frequencies = {}
        self._lifetime_terminal_counts: Counter[State] = Counter()

    @property
    def ema_terminal_frequencies(self) -> Mapping[State, float]:
        return self._ema_terminal_frequencies

    @property
    def lifetime_terminal_counts(self) -> Mapping[State, int]:
        return self._lifetime_terminal_counts

    def _history_metrics(self) -> dict[str, float]:
        return {
            "history_propensity_mode": float(
                self.history_ips.propensity_mode == "ema"
            ),
            "ema_alpha": self.history_ips.alpha,
            "ema_effective_groups": 1.0 / self.history_ips.alpha,
            "ema_absent_decay": float(self.history_ips.decay_absent_outcomes),
            "ema_raw_ips_weights": float(
                self.history_ips.ips_weight_mode == "raw"
            ),
            "ema_probability_mass": float(
                sum(self._ema_terminal_frequencies.values())
            ),
            "ema_tracked_outcomes": float(len(self._ema_terminal_frequencies)),
            "histogram_total_visits": float(
                sum(self._lifetime_terminal_counts.values())
            ),
            "histogram_tracked_outcomes": float(
                sum(1 for count in self._lifetime_terminal_counts.values() if count > 0)
            ),
        }

    def _assign_group_advantages(
        self, episodes: list[Episode]
    ) -> dict[str, float]:
        outcomes = [episode.terminal for episode in episodes]
        normalize = self.history_ips.ips_weight_mode == "normalized"
        if self.history_ips.propensity_mode == "ema":
            advantages, metrics = ema_count_ips_advantages(
                [episode.reward for episode in episodes],
                outcomes,
                self._ema_terminal_frequencies,
                tracker_eps=self.history_ips.tracker_eps,
                normalize=normalize,
                eps=self.config.advantage_eps,
            )
        else:
            advantages, metrics = running_histogram_ips_advantages(
                [episode.reward for episode in episodes],
                outcomes,
                self._lifetime_terminal_counts,
                tracker_eps=self.history_ips.tracker_eps,
                normalize=normalize,
                eps=self.config.advantage_eps,
            )

        for episode, advantage in zip(episodes, advantages):
            for step in episode.steps:
                step.advantage = float(advantage)

        group_counts = Counter(outcomes)
        singleton_episodes = sum(
            count for count in group_counts.values() if count == 1
        )
        metrics.update(
            {
                "exploration_epsilon": self.current_epsilon,
                "exploration_temperature": self.current_temperature,
                **self._schedule_metrics(),
                **self._history_metrics(),
                "group_singleton_fraction": singleton_episodes / len(episodes),
                "advantage_group_size": float(len(episodes)),
            }
        )
        return metrics

    def _collect_training_groups(
        self,
    ) -> tuple[list[list[Episode]], list[dict[str, float]]]:
        """Roll out, update history ``p̂``, then assign per-group advantages.

        The tracker sees every explored terminal from the update pool. Advantage
        normalization remains local to each on-policy group of size ``G``.
        """
        group_size = self.config.group_size
        num_groups = self.current_rollout_groups
        pool_size = num_groups * group_size
        pool = self.rollout_batch(pool_size, explore=True)
        groups = [
            pool[start : start + group_size]
            for start in range(0, pool_size, group_size)
        ]
        if any(len(group) != group_size for group in groups):
            raise RuntimeError("scheduled EMA pool must partition into equal groups")

        group_outcomes = [[episode.terminal for episode in group] for group in groups]
        for outcomes in group_outcomes:
            self._lifetime_terminal_counts.update(outcomes)

        if self.history_ips.propensity_mode == "ema":
            update_ema_from_groups(
                group_outcomes,
                self._ema_terminal_frequencies,
                alpha=self.history_ips.alpha,
                decay_absent_outcomes=self.history_ips.decay_absent_outcomes,
                update_unit=self.history_ips.ema_update_unit,
            )

        group_metrics: list[dict[str, float]] = []
        for group in groups:
            metrics = self._assign_group_advantages(group)
            group_metrics.append(metrics)

        pool_counts = Counter(episode.terminal for episode in pool)
        pool_singletons = sum(
            count for count in pool_counts.values() if count == 1
        )
        for metrics in group_metrics:
            metrics.update(
                {
                    "propensity_pool_size": float(pool_size),
                    "propensity_pool_unique_outcomes": float(len(pool_counts)),
                    "pool_singleton_fraction": pool_singletons / pool_size,
                }
            )

        self._last_ips_metrics = dict(group_metrics[-1])
        return groups, group_metrics

    def _group_advantages(self, episodes: list[Episode]) -> float:
        """Single-group fallback: update history tracker, then scale with ``p̂``."""
        outcomes = [episode.terminal for episode in episodes]
        self._lifetime_terminal_counts.update(outcomes)
        if self.history_ips.propensity_mode == "ema":
            update_ema_outcome_frequencies(
                outcomes,
                self._ema_terminal_frequencies,
                alpha=self.history_ips.alpha,
                decay_absent_outcomes=self.history_ips.decay_absent_outcomes,
            )
        metrics = self._assign_group_advantages(episodes)
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
                    "name": self.algorithm_name,
                    "exploration": asdict(self.exploration),
                    "rollout_schedule": asdict(self.rollout_schedule),
                    "history_ips": asdict(self.history_ips),
                    "current_epsilon": self.current_epsilon,
                    "current_temperature": self.current_temperature,
                    "current_rollout_groups": self.current_rollout_groups,
                    "ema_terminal_frequencies": {
                        state.signature: probability
                        for state, probability in self._ema_terminal_frequencies.items()
                    },
                    "lifetime_terminal_counts": {
                        state.signature: count
                        for state, count in self._lifetime_terminal_counts.items()
                    },
                },
            },
            path,
        )
        return path

    @classmethod
    def load(
        cls, path: Path | str, *, device: str = "cpu"
    ) -> "ScheduledEMACountIPSTrainer":
        payload = torch.load(Path(path), map_location=device, weights_only=False)
        algorithm = payload.get("algorithm", {})
        if algorithm.get("name") != cls.algorithm_name:
            raise ValueError(f"checkpoint is not a {cls.algorithm_name} run")
        trainer = cls(
            payload["config"],
            device=device,
            exploration=ExplorationConfig(**algorithm["exploration"]),
            rollout_schedule=RolloutScheduleConfig(
                **algorithm["rollout_schedule"]
            ),
            history_ips=ExplorationHistoryIPSConfig(**algorithm["history_ips"]),
        )
        trainer.current_epsilon = float(algorithm["current_epsilon"])
        trainer.current_temperature = float(algorithm["current_temperature"])
        trainer.current_rollout_groups = int(algorithm["current_rollout_groups"])
        trainer.direction_policy.load_state_dict(payload["direction_policy"])
        trainer.step_policy.load_state_dict(payload["step_policy"])
        trainer._ema_terminal_frequencies.clear()

        def state_from_signature(signature: str) -> State:
            x_text, y_text = signature.strip("()").split(",")
            return State(int(x_text), int(y_text))

        for signature, probability in algorithm.get(
            "ema_terminal_frequencies", {}
        ).items():
            trainer._ema_terminal_frequencies[state_from_signature(signature)] = float(
                probability
            )
        for signature, count in algorithm.get("lifetime_terminal_counts", {}).items():
            state = state_from_signature(signature)
            trainer._lifetime_terminal_counts[state] = int(count)
            trainer._seen_terminals.add(state)
        trainer.direction_policy.eval()
        trainer.step_policy.eval()
        return trainer


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is not available")
    return requested


def _plot_history_tracker(history: list[dict[str, Any]], output: Path) -> None:
    steps = [row["step"] for row in history]
    tracked = [row.get("histogram_tracked_outcomes", 0.0) for row in history]
    visits = [row.get("histogram_total_visits", 0.0) for row in history]
    fig, track_axis = plt.subplots(figsize=(8, 4.5))
    visit_axis = track_axis.twinx()
    track_axis.plot(steps, tracked, color="#0984e3", label="tracked outcomes")
    visit_axis.plot(steps, visits, color="#e17055", label="lifetime visits")
    track_axis.set_xlabel("update")
    track_axis.set_ylabel("unique terminals in histogram", color="#0984e3")
    visit_axis.set_ylabel("lifetime exploration visits", color="#e17055")
    track_axis.grid(alpha=0.22)
    fig.suptitle("Exploration-history propensity tracker")
    fig.tight_layout()
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--propensity-mode",
        choices=("ema", "running"),
        default="ema",
        help="use EMA frequencies or the lifetime exploration histogram as p̂",
    )
    parser.add_argument("--budget", type=int, default=32)
    parser.add_argument("--max-step", type=int, default=3)
    parser.add_argument("--num-updates", type=int, default=500)
    parser.add_argument(
        "--group-size",
        type=int,
        default=16,
        help="on-policy advantage / gradient group size G",
    )
    parser.add_argument("--groups-start", type=int, default=64)
    parser.add_argument("--groups-end", type=int, default=4)
    parser.add_argument("--rollout-anneal-updates", type=int, default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--train-epochs", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--tracker-eps", type=float, default=1e-6)
    parser.add_argument(
        "--initialization",
        choices=("first_batch", "uniform"),
        default="first_batch",
    )
    parser.add_argument(
        "--keep-absent-stale",
        action="store_true",
        help="EMA: update only outcomes present in a batch",
    )
    parser.add_argument(
        "--ips-weight-mode",
        choices=("normalized", "raw"),
        default="normalized",
    )
    parser.add_argument(
        "--ema-update-unit",
        choices=("group", "pool"),
        default="group",
        help="EMA update once per advantage group or once from the full pool",
    )
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
    history_ips = ExplorationHistoryIPSConfig(
        propensity_mode=args.propensity_mode,
        alpha=args.alpha,
        initialization=args.initialization,
        decay_absent_outcomes=not args.keep_absent_stale,
        tracker_eps=args.tracker_eps,
        ips_weight_mode=args.ips_weight_mode,
        ema_update_unit=args.ema_update_unit,
    )
    update_mode = (
        "decay_absent" if history_ips.decay_absent_outcomes else "stale_absent"
    )
    run_dir = args.run_dir or (
        Path(__file__).resolve().parent
        / "data"
        / "scheduled_ema_count_ips_runs"
        / (
            f"{datetime.now():%Y%m%d_%H%M%S}_{args.propensity_mode}_"
            f"b{config.budget}_g{config.group_size}_"
            f"gs{rollout_schedule.groups_start}-{rollout_schedule.groups_end}_"
            f"seed{config.seed}"
        )
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    trainer = ScheduledEMACountIPSTrainer(
        config,
        device=_resolve_device(args.device),
        exploration=exploration,
        rollout_schedule=rollout_schedule,
        history_ips=history_ips,
    )
    print(f"Run directory: {run_dir}")
    print(f"Device: {trainer.device}")
    print(f"DAG: {trainer.environment_summary()}")
    print(
        f"Scheduled EMA Count-IPS: mode={history_ips.propensity_mode}, "
        f"G={config.group_size}, "
        f"groups {rollout_schedule.groups_start}->{rollout_schedule.groups_end}, "
        f"alpha={history_ips.alpha:g} (window~{1.0 / history_ips.alpha:g} groups), "
        f"init={history_ips.initialization}, update={update_mode}, "
        f"ema_unit={history_ips.ema_update_unit}, "
        f"weight={history_ips.ips_weight_mode}"
    )

    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "train_config": asdict(config),
                "exploration_config": asdict(exploration),
                "rollout_schedule_config": asdict(rollout_schedule),
                "history_ips_config": asdict(history_ips),
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
    _plot_history_tracker(history, run_dir / "history_tracker.png")

    evaluation = trainer.evaluate(args.final_samples)
    sampling = _plot_final_counts(
        trainer,
        evaluation,
        samples=args.final_samples,
        output=run_dir / "sampling_counts.png",
        suptitle=(
            f"Scheduled {args.propensity_mode} history Count-IPS vs reward target"
        ),
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
        "history_ips": asdict(history_ips),
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
