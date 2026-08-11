"""Saturating global Count-IPS with rollout warm-up and annealed exploration.

This variant keeps a persistent pseudo-count for every terminal outcome that
has actually been observed. It does not enumerate outcomes or use coordinates
to guide exploration, so the mechanism applies to any DAG with hashable
terminal identifiers.

For a new terminal ``o``:

    c(o) = 1

For every later observation of that terminal:

    c(o) <- decay * c(o) + increment

Only sampled outcomes are updated. With the defaults
``decay=0.9947368421052631`` and ``increment=0.10526315789473684``,
the sequence is:

    1.0 -> 1.1 -> 1.1995 -> 1.2984 -> ... -> 20.0

The store is normalized across observed outcomes to form a relative
pseudo-propensity. For each independently normalized PPO group:

    p_sat(o) = c(o) / sum_observed c
    scaled_i = reward_i / p_sat(o_i) ** beta
    advantage_i = normalize_group(scaled_i)

The common normalization mass does not affect normalized advantages, but it
keeps the logged ``p_sat`` values interpretable as relative probabilities.

Before learning, a configurable number of rollout-only warm-up groups populate
the store. Training then uses the same PPO-consistent epsilon/temperature
mixture as ``epsilon_greedy_count_ips.py`` and anneals it over time.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from config import TrainConfig  # noqa: E402
from count_ips import Episode  # noqa: E402
from dag_env import State  # noqa: E402
from epsilon_greedy_count_ips import (  # noqa: E402
    EpsilonGreedyCountIPSTrainer,
    ExplorationConfig,
    _plot_exploration,
    _resolve_device,
)
from run_count_ips import (  # noqa: E402
    _plot_final_counts,
    _plot_training_curves,
    _plot_trajectory_diagnostics,
)


@dataclass(frozen=True)
class SaturatingGlobalCountConfig:
    """Persistent pseudo-count and pre-training warm-up parameters."""

    decay: float = 0.9947368421052631
    increment: float = 0.10526315789473684
    beta: float = 1.0
    initial_count: float = 1.0
    warmup_groups: int = 8

    @property
    def fixed_point(self) -> float:
        return self.increment / (1.0 - self.decay)

    def validate(self) -> None:
        if not 0.0 <= self.decay < 1.0:
            raise ValueError("decay must be in [0, 1)")
        if self.increment <= 0.0:
            raise ValueError("increment must be > 0")
        if self.beta <= 0.0:
            raise ValueError("beta must be > 0")
        if self.initial_count <= 0.0:
            raise ValueError("initial_count must be > 0")
        if self.warmup_groups < 0:
            raise ValueError("warmup_groups must be >= 0")
        if self.fixed_point <= self.initial_count:
            raise ValueError(
                "increment / (1 - decay) must exceed initial_count so repeats "
                "increase rather than decrease the pseudo-count"
            )


def update_saturating_counts(
    outcome_ids: Sequence[object],
    pseudo_counts: MutableMapping[object, float],
    *,
    decay: float,
    increment: float,
    initial_count: float = 1.0,
) -> dict[str, float]:
    """Update sampled outcomes in place, aggregated without sample-order effects.

    A first observation initializes an outcome to ``initial_count``. If a new
    outcome appears ``k`` times in the pool, the recurrence is applied ``k - 1``
    times. For an already tracked outcome it is applied ``k`` times.
    """
    if len(outcome_ids) == 0:
        raise ValueError("outcome_ids must be non-empty")
    if not 0.0 <= decay < 1.0:
        raise ValueError("decay must be in [0, 1)")
    if increment <= 0.0 or initial_count <= 0.0:
        raise ValueError("increment and initial_count must be > 0")

    fixed_point = increment / (1.0 - decay)
    if fixed_point <= initial_count:
        raise ValueError("the recurrence fixed point must exceed initial_count")

    batch_counts = Counter(outcome_ids)
    new_outcomes = 0
    repeat_observations = 0
    for outcome, observations in batch_counts.items():
        if outcome in pseudo_counts:
            previous = float(pseudo_counts[outcome])
            recurrence_steps = observations
            repeat_observations += observations
        else:
            previous = initial_count
            recurrence_steps = observations - 1
            new_outcomes += 1
            repeat_observations += observations - 1

        pseudo_counts[outcome] = float(
            fixed_point
            + (previous - fixed_point) * decay**recurrence_steps
        )

    return {
        "pool_new_outcomes": float(new_outcomes),
        "pool_repeat_observations": float(repeat_observations),
        "pool_unique_outcomes": float(len(batch_counts)),
        "pool_observations": float(len(outcome_ids)),
    }


def saturating_global_count_advantages(
    rewards: Sequence[float],
    outcome_ids: Sequence[object],
    pseudo_counts: Mapping[object, float],
    *,
    beta: float = 1.0,
    eps: float = 1e-8,
) -> tuple[np.ndarray, dict[str, float]]:
    """Compute normalized ``reward / relative pseudo-propensity**beta``."""
    if len(rewards) == 0 or len(rewards) != len(outcome_ids):
        raise ValueError("rewards and outcome_ids must have the same non-zero length")
    if beta <= 0.0:
        raise ValueError("beta must be > 0")
    if not pseudo_counts:
        raise ValueError("pseudo_counts must be non-empty")

    reward_array = np.asarray(rewards, dtype=np.float64)
    count_mass = float(sum(pseudo_counts.values()))
    if count_mass <= 0.0:
        raise ValueError("pseudo-count mass must be positive")

    count_values = np.asarray(
        [float(pseudo_counts[outcome]) for outcome in outcome_ids],
        dtype=np.float64,
    )
    if np.any(count_values <= 0.0):
        raise ValueError("every sampled outcome must have a positive pseudo-count")

    p_sat = count_values / count_mass
    inverse_weights = np.power(np.maximum(p_sat, eps), -beta)
    scaled = reward_array * inverse_weights
    centered = scaled - scaled.mean()
    std = float(scaled.std())
    advantages = centered if std < eps else centered / (std + eps)

    batch_counts = Counter(outcome_ids)
    inverse_square_sum = float(np.square(inverse_weights).sum())
    ess = float(
        inverse_weights.sum() ** 2 / max(inverse_square_sum, eps)
    )
    singleton_episodes = sum(
        count for count in batch_counts.values() if count == 1
    )
    return advantages, {
        "ips_prob_mean": float(p_sat.mean()),
        "ips_prob_min": float(p_sat.min()),
        "ips_prob_max": float(p_sat.max()),
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
        "batch_pseudo_count_mean": float(count_values.mean()),
        "batch_pseudo_count_min": float(count_values.min()),
        "batch_pseudo_count_max": float(count_values.max()),
        "batch_inverse_weight_mean": float(inverse_weights.mean()),
        "batch_inverse_weight_min": float(inverse_weights.min()),
        "batch_inverse_weight_max": float(inverse_weights.max()),
        "group_singleton_fraction": singleton_episodes / len(outcome_ids),
    }


class SaturatingGlobalCountIPSTrainer(EpsilonGreedyCountIPSTrainer):
    """Persistent saturating pseudo-counts plus annealed normal exploration."""

    probability_label = "p_sat"
    algorithm_name = "saturating_global_count_ips"

    def __init__(
        self,
        config: TrainConfig | None = None,
        *,
        device: str = "cpu",
        exploration: ExplorationConfig | None = None,
        saturating_count: SaturatingGlobalCountConfig | None = None,
    ) -> None:
        resolved_config = config or TrainConfig()
        if exploration is None:
            exploration = ExplorationConfig(
                epsilon_start=0.30,
                epsilon_end=0.10,
                temperature_start=2.0,
                temperature_end=1.25,
                anneal_updates=resolved_config.num_updates,
                schedule="cosine",
            )
        super().__init__(
            resolved_config, device=device, exploration=exploration
        )
        self.saturating_count = (
            saturating_count or SaturatingGlobalCountConfig()
        )
        self.saturating_count.validate()
        self._pseudo_counts: dict[State, float] = {}
        self._lifetime_terminal_counts: Counter[State] = Counter()
        self._warmup_complete = False
        self._current_update_step = 0
        self._store_history: list[dict[str, Any]] = []
        self._warmup_summary: dict[str, Any] = {
            "groups": 0,
            "rollouts": 0,
            "unique_outcomes": 0,
            "outcome_counts": {},
            "pseudo_counts_after_warmup": {},
        }

    @property
    def pseudo_counts(self) -> Mapping[State, float]:
        return self._pseudo_counts

    @property
    def lifetime_terminal_counts(self) -> Mapping[State, int]:
        return self._lifetime_terminal_counts

    @property
    def warmup_summary(self) -> Mapping[str, Any]:
        return self._warmup_summary

    @property
    def store_history(self) -> Sequence[Mapping[str, Any]]:
        return self._store_history

    def _on_update_start(self, update_step: int) -> None:
        super()._on_update_start(update_step)
        self._current_update_step = update_step

    def _store_metrics(self) -> dict[str, float]:
        if not self._pseudo_counts:
            return {
                "store_tracked_outcomes": 0.0,
                "store_coverage_fraction": 0.0,
                "store_pseudo_count_mass": 0.0,
                "store_pseudo_count_min": 0.0,
                "store_pseudo_count_mean": 0.0,
                "store_pseudo_count_median": 0.0,
                "store_pseudo_count_p90": 0.0,
                "store_pseudo_count_max": 0.0,
                "store_saturated_fraction": 0.0,
                "global_total_observations": 0.0,
            }

        values = np.asarray(list(self._pseudo_counts.values()), dtype=np.float64)
        fixed_point = self.saturating_count.fixed_point
        saturation_threshold = fixed_point - 0.01 * (
            fixed_point - self.saturating_count.initial_count
        )
        return {
            "store_tracked_outcomes": float(len(values)),
            "store_coverage_fraction": float(len(values) / len(self.terminals)),
            "store_pseudo_count_mass": float(values.sum()),
            "store_pseudo_count_min": float(values.min()),
            "store_pseudo_count_mean": float(values.mean()),
            "store_pseudo_count_median": float(np.median(values)),
            "store_pseudo_count_p90": float(np.quantile(values, 0.9)),
            "store_pseudo_count_max": float(values.max()),
            "store_saturated_fraction": float(
                np.mean(values >= saturation_threshold)
            ),
            "global_total_observations": float(
                sum(self._lifetime_terminal_counts.values())
            ),
        }

    def _update_store(self, outcomes: Sequence[State]) -> dict[str, float]:
        self._lifetime_terminal_counts.update(outcomes)
        update_metrics = update_saturating_counts(
            outcomes,
            self._pseudo_counts,
            decay=self.saturating_count.decay,
            increment=self.saturating_count.increment,
            initial_count=self.saturating_count.initial_count,
        )
        self._seen_terminals.update(outcomes)
        return {**update_metrics, **self._store_metrics()}

    def warmup_store(self) -> dict[str, Any]:
        """Populate the store with exploratory rollouts and no optimizer step."""
        if self._warmup_complete:
            return dict(self._warmup_summary)

        warmup_groups = self.saturating_count.warmup_groups
        if warmup_groups == 0:
            self._warmup_complete = True
            return dict(self._warmup_summary)

        self._on_update_start(1)
        warmup_rollouts = warmup_groups * self.config.group_size
        episodes = self.rollout_batch(warmup_rollouts, explore=True)
        outcomes = [episode.terminal for episode in episodes]
        update_metrics = self._update_store(outcomes)
        outcome_counts = Counter(outcomes)
        self._warmup_summary = {
            "groups": warmup_groups,
            "rollouts": warmup_rollouts,
            "unique_outcomes": len(outcome_counts),
            "mean_reward": float(np.mean([episode.reward for episode in episodes])),
            "mean_length": float(
                np.mean([len(episode.steps) for episode in episodes])
            ),
            "epsilon": self.current_epsilon,
            "temperature": self.current_temperature,
            "new_outcomes": int(update_metrics["pool_new_outcomes"]),
            "repeat_observations": int(
                update_metrics["pool_repeat_observations"]
            ),
            "outcome_counts": {
                state.signature: int(outcome_counts[state])
                for state in self.terminals
            },
            "pseudo_counts_after_warmup": {
                state.signature: float(value)
                for state, value in self._pseudo_counts.items()
            },
        }
        self._warmup_complete = True
        print(
            "warm-up complete  "
            f"groups={warmup_groups}  rollouts={warmup_rollouts}  "
            f"outcomes={len(outcome_counts)}  "
            f"repeats={int(update_metrics['pool_repeat_observations'])}  "
            f"eps={self.current_epsilon:.3f}  "
            f"temp={self.current_temperature:.3f}"
        )
        return dict(self._warmup_summary)

    def _assign_group_advantages(
        self,
        episodes: list[Episode],
        *,
        pool_metrics: Mapping[str, float],
    ) -> dict[str, float]:
        advantages, metrics = saturating_global_count_advantages(
            [episode.reward for episode in episodes],
            [episode.terminal for episode in episodes],
            self._pseudo_counts,
            beta=self.saturating_count.beta,
            eps=self.config.advantage_eps,
        )
        for episode, advantage in zip(episodes, advantages):
            for step in episode.steps:
                step.advantage = float(advantage)

        metrics.update(
            {
                **pool_metrics,
                "exploration_epsilon": self.current_epsilon,
                "exploration_temperature": self.current_temperature,
                "pseudo_count_decay": self.saturating_count.decay,
                "pseudo_count_increment": self.saturating_count.increment,
                "pseudo_count_beta": self.saturating_count.beta,
                "pseudo_count_fixed_point": self.saturating_count.fixed_point,
                "advantage_group_size": float(len(episodes)),
            }
        )
        return metrics

    def _collect_training_groups(
        self,
    ) -> tuple[list[list[Episode]], list[dict[str, float]]]:
        """Collect one frozen-policy pool and update the store once per pool."""
        group_size = self.config.group_size
        num_groups = self._groups_for_update()
        pool_size = group_size * num_groups
        pool = self.rollout_batch(pool_size, explore=True)
        groups = [
            pool[start : start + group_size]
            for start in range(0, pool_size, group_size)
        ]
        if not groups or any(len(group) != group_size for group in groups):
            raise RuntimeError("rollout pool must partition into full groups")

        outcomes = [episode.terminal for episode in pool]
        pool_metrics = self._update_store(outcomes)
        pool_metrics["propensity_pool_size"] = float(pool_size)
        count_mass = float(sum(self._pseudo_counts.values()))
        self._store_history.append(
            {
                "step": self._current_update_step,
                "pseudo_count_mass": count_mass,
                "tracked_outcomes": len(self._pseudo_counts),
                "new_outcomes": int(pool_metrics["pool_new_outcomes"]),
                "repeat_observations": int(
                    pool_metrics["pool_repeat_observations"]
                ),
                "pseudo_counts": {
                    state.signature: float(value)
                    for state, value in self._pseudo_counts.items()
                },
                "pseudo_probabilities": {
                    state.signature: float(value / count_mass)
                    for state, value in self._pseudo_counts.items()
                },
            }
        )

        group_metrics = [
            self._assign_group_advantages(group, pool_metrics=pool_metrics)
            for group in groups
        ]
        self._last_ips_metrics = dict(group_metrics[-1])
        return groups, group_metrics

    def _group_advantages(self, episodes: list[Episode]) -> float:
        """Single-group compatibility hook used by direct trainer callers."""
        pool_metrics = self._update_store(
            [episode.terminal for episode in episodes]
        )
        metrics = self._assign_group_advantages(
            episodes, pool_metrics=pool_metrics
        )
        self._last_ips_metrics = metrics
        return metrics["ips_ess"]

    def train(self, **kwargs) -> list[dict[str, Any]]:
        self.warmup_store()
        history = super().train(**kwargs)
        warmup_rollouts = int(self._warmup_summary["rollouts"])
        for row in history:
            row["warmup_groups"] = float(self._warmup_summary["groups"])
            row["warmup_rollouts"] = float(warmup_rollouts)
            row["total_rollouts_including_warmup"] = float(
                row["cumulative_rollouts"] + warmup_rollouts
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
                    "name": self.algorithm_name,
                    "exploration": asdict(self.exploration),
                    "saturating_count": asdict(self.saturating_count),
                    "current_epsilon": self.current_epsilon,
                    "current_temperature": self.current_temperature,
                    "warmup_complete": self._warmup_complete,
                    "warmup_summary": self._warmup_summary,
                    "pseudo_counts": {
                        state.signature: value
                        for state, value in self._pseudo_counts.items()
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
    ) -> "SaturatingGlobalCountIPSTrainer":
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
            saturating_count=SaturatingGlobalCountConfig(
                **algorithm["saturating_count"]
            ),
        )
        trainer.direction_policy.load_state_dict(payload["direction_policy"])
        trainer.step_policy.load_state_dict(payload["step_policy"])
        trainer.current_epsilon = float(algorithm["current_epsilon"])
        trainer.current_temperature = float(algorithm["current_temperature"])
        trainer._warmup_complete = bool(algorithm.get("warmup_complete", True))
        trainer._warmup_summary = dict(
            algorithm.get("warmup_summary", trainer._warmup_summary)
        )

        def state_from_signature(signature: str) -> State:
            x_text, y_text = signature.strip("()").split(",")
            return State(int(x_text), int(y_text))

        for signature, value in algorithm.get("pseudo_counts", {}).items():
            state = state_from_signature(signature)
            trainer._pseudo_counts[state] = float(value)
            trainer._seen_terminals.add(state)
        for signature, count in algorithm.get(
            "lifetime_terminal_counts", {}
        ).items():
            state = state_from_signature(signature)
            trainer._lifetime_terminal_counts[state] = int(count)
            trainer._seen_terminals.add(state)

        trainer.direction_policy.eval()
        trainer.step_policy.eval()
        return trainer


def _plot_saturating_store(
    history: list[dict[str, Any]],
    output: Path,
) -> None:
    steps = [row["step"] for row in history]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))

    ax = axes[0, 0]
    ax.plot(
        steps,
        [row["store_tracked_outcomes"] for row in history],
        color="#0984e3",
        label="tracked terminals",
    )
    coverage_axis = ax.twinx()
    coverage_axis.plot(
        steps,
        [row["store_coverage_fraction"] for row in history],
        color="#00b894",
        label="coverage",
    )
    coverage_axis.set_ylim(-0.02, 1.02)
    ax.set_ylabel("tracked terminals", color="#0984e3")
    coverage_axis.set_ylabel("coverage fraction", color="#00b894")
    ax.set_title("Persistent store coverage")

    ax = axes[0, 1]
    ax.plot(
        steps,
        [row["store_pseudo_count_min"] for row in history],
        label="min",
    )
    ax.plot(
        steps,
        [row["store_pseudo_count_median"] for row in history],
        label="median",
    )
    ax.plot(
        steps,
        [row["store_pseudo_count_p90"] for row in history],
        label="p90",
    )
    ax.plot(
        steps,
        [row["store_pseudo_count_max"] for row in history],
        label="max",
    )
    ax.axhline(
        history[0]["pseudo_count_fixed_point"],
        color="#2d3436",
        linestyle="--",
        label="fixed point",
    )
    ax.set_title("Saturating pseudo-count distribution")
    ax.legend()

    ax = axes[1, 0]
    ax.plot(
        steps,
        [row["ips_ess_fraction"] for row in history],
        color="#6c5ce7",
        label="inverse-weight ESS / G",
    )
    ax.plot(
        steps,
        [row["group_singleton_fraction"] for row in history],
        color="#fdcb6e",
        label="group singleton fraction",
    )
    ax.plot(
        steps,
        [row["store_saturated_fraction"] for row in history],
        color="#d63031",
        label="store saturated fraction",
    )
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Weight stability and saturation")
    ax.legend()

    ax = axes[1, 1]
    ax.plot(
        steps,
        [row["pool_new_outcomes"] for row in history],
        color="#00b894",
        label="new outcomes / update",
    )
    ax.plot(
        steps,
        [row["pool_repeat_observations"] for row in history],
        color="#e17055",
        label="repeat observations / update",
    )
    ax.set_title("Store updates from fresh rollouts")
    ax.legend()

    for ax in axes.flat:
        ax.set_xlabel("learning update")
        ax.grid(alpha=0.22)
    fig.suptitle("Saturating global Count-IPS diagnostics")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=128)
    parser.add_argument("--max-step", type=int, default=3)
    parser.add_argument("--num-updates", type=int, default=2_000)
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--num-groups", type=int, default=1)
    parser.add_argument("--warmup-groups", type=int, default=8)
    parser.add_argument(
        "--count-decay", type=float, default=0.9947368421052631
    )
    parser.add_argument(
        "--count-increment", type=float, default=0.10526315789473684
    )
    parser.add_argument("--count-beta", type=float, default=1.0)
    parser.add_argument("--initial-count", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--train-epochs", type=int, default=1)
    parser.add_argument("--epsilon-start", type=float, default=0.30)
    parser.add_argument("--epsilon-end", type=float, default=0.10)
    parser.add_argument("--temperature-start", type=float, default=2.0)
    parser.add_argument("--temperature-end", type=float, default=1.25)
    parser.add_argument(
        "--anneal-updates",
        type=int,
        default=2_000,
        help="cosine/linear exploration anneal horizon (full default run)",
    )
    parser.add_argument(
        "--schedule", choices=("linear", "cosine"), default="cosine"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--eval-every", type=int, default=200)
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
    args = parser.parse_args()

    config = TrainConfig(
        budget=args.budget,
        max_step=args.max_step,
        terminal_rewards=(
            tuple(args.terminal_rewards)
            if args.terminal_rewards is not None
            else None
        ),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        group_size=args.group_size,
        num_groups=args.num_groups,
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
        anneal_updates=args.anneal_updates,
        schedule=args.schedule,
    )
    saturating_count = SaturatingGlobalCountConfig(
        decay=args.count_decay,
        increment=args.count_increment,
        beta=args.count_beta,
        initial_count=args.initial_count,
        warmup_groups=args.warmup_groups,
    )
    run_dir = args.run_dir or (
        Path(__file__).resolve().parent
        / "data"
        / "saturating_global_count_ips_runs"
        / (
            f"{datetime.now():%Y%m%d_%H%M%S}_b{config.budget}_"
            f"g{config.group_size}_seed{config.seed}"
        )
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    trainer = SaturatingGlobalCountIPSTrainer(
        config,
        device=_resolve_device(args.device),
        exploration=exploration,
        saturating_count=saturating_count,
    )

    print(f"Run directory: {run_dir}")
    print(f"Device: {trainer.device}")
    print(f"DAG: {trainer.environment_summary()}")
    print(
        "Algorithm: reward / normalized saturating pseudo-count -> "
        "group-normalized PPO advantage"
    )
    print(
        "Pseudo-count: "
        f"initial={saturating_count.initial_count:g}, "
        f"decay={saturating_count.decay:g}, "
        f"increment={saturating_count.increment:g}, "
        f"fixed_point={saturating_count.fixed_point:g}, "
        f"beta={saturating_count.beta:g}"
    )
    print(
        "Exploration: "
        f"{saturating_count.warmup_groups} rollout-only warm-up groups; "
        f"epsilon {exploration.epsilon_start:g}->{exploration.epsilon_end:g}; "
        f"temperature "
        f"{exploration.temperature_start:g}->{exploration.temperature_end:g}; "
        f"{exploration.schedule} over {trainer.anneal_updates} updates"
    )

    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "train_config": asdict(config),
                "exploration_config": asdict(exploration),
                "saturating_count_config": asdict(saturating_count),
                "pseudo_count_fixed_point": saturating_count.fixed_point,
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
    (run_dir / "store_history.json").write_text(
        json.dumps(list(trainer.store_history), indent=2), encoding="utf-8"
    )
    trainer.save(run_dir / "checkpoint.pt", update_step=config.num_updates)

    _plot_training_curves(
        history,
        trainer,
        output=run_dir / "training_curves.png",
        propensity_title="Relative saturating pseudo-propensities",
        suptitle="Saturating global Count-IPS training diagnostics",
    )
    _plot_exploration(history, run_dir / "exploration_schedule.png")
    _plot_saturating_store(
        history, run_dir / "saturating_store_diagnostics.png"
    )

    evaluation = trainer.evaluate(args.final_samples)
    sampling = _plot_final_counts(
        trainer,
        evaluation,
        samples=args.final_samples,
        output=run_dir / "sampling_counts.png",
        suptitle="Saturating global Count-IPS vs ideal reward sampling",
    )
    trajectories = _plot_trajectory_diagnostics(
        history,
        trainer,
        evaluation,
        samples=args.final_samples,
        output=run_dir / "trajectory_sampling.png",
    )
    eval_rows = [row for row in history if "tv_reward_target" in row]
    store_snapshot = {
        state.signature: float(trainer.pseudo_counts[state])
        for state in trainer.pseudo_counts
    }
    summary = {
        "environment": trainer.environment_summary(),
        "exploration": asdict(exploration),
        "saturating_count": {
            **asdict(saturating_count),
            "fixed_point": saturating_count.fixed_point,
        },
        "warmup": dict(trainer.warmup_summary),
        "final_store_metrics": trainer._store_metrics(),
        "final_pseudo_counts": store_snapshot,
        "final_sampling": sampling,
        "trajectory_sampling": trajectories,
        "best_eval": (
            min(eval_rows, key=lambda row: row["tv_reward_target"])
            if eval_rows
            else None
        ),
        "plots": {
            "training_curves": "training_curves.png",
            "exploration_schedule": "exploration_schedule.png",
            "saturating_store": "saturating_store_diagnostics.png",
            "store_history": "store_history.json",
            "sampling_counts": "sampling_counts.png",
            "trajectory_sampling": "trajectory_sampling.png",
        },
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Final tracked outcomes: {len(trainer.pseudo_counts)}")
    print(f"Final ideal-line R^2: {sampling['r2_reward_target']:.4f}")
    print(f"Final TV distance: {sampling['tv_reward_target']:.4f}")
    print(f"Artifacts: {run_dir}")


if __name__ == "__main__":
    main()


'''
.venv/bin/python compound_action_rl/dag_toy_dataset/saturating_global_count_ips.py \
  --budget 128 \
  --max-step 3 \
  --group-size 16 \
  --num-groups 1 \
  --warmup-groups 8 \
  --num-updates 2000 \
  --count-decay 0.9947368421052631 \
  --count-increment 0.10526315789473684 \
  --count-beta 1.0 \
  --epsilon-start 0.30 \
  --epsilon-end 0.10 \
  --temperature-start 2.0 \
  --temperature-end 1.25 \
  --anneal-updates 2000 \
  --schedule cosine \
  --eval-every 200 \
  --eval-episodes 2000 \
  --final-samples 10000 \
  --checkpoint-every 500 \
  --device auto
'''
