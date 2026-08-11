"""Non-saturating global Count-IPS with warm-up and annealed exploration.

This is the non-saturating counterpart to ``saturating_global_count_ips.py``.
It stores a lazy pseudo-count for every observed terminal and updates only
outcomes present in the fresh rollout pool:

    first observation: c(o) = 1
    every repeat:       c(o) <- c(o) + step

With the default ``step=0.1``, an outcome's values are:

    1.0 -> 1.1 -> 1.2 -> 1.3 -> ...

Counts never converge to a shared fixed point, so terminals with very different
visit histories remain distinguishable. The relative pseudo-propensity and
group advantage are:

    p_linear(o) = c(o) / sum_observed c
    scaled_i = reward_i / p_linear(o_i) ** beta
    advantage_i = normalize_group(scaled_i)

The algorithm needs only hashable terminal identifiers. It does not enumerate
outcomes or use DAG coordinates to guide exploration.
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
    ExplorationConfig,
    _plot_exploration,
    _resolve_device,
)
from run_count_ips import (  # noqa: E402
    _plot_final_counts,
    _plot_training_curves,
    _plot_trajectory_diagnostics,
)
from saturating_global_count_ips import (  # noqa: E402
    SaturatingGlobalCountConfig,
    SaturatingGlobalCountIPSTrainer,
    saturating_global_count_advantages,
)


@dataclass(frozen=True)
class LinearGlobalCountConfig:
    """Non-saturating pseudo-count parameters."""

    count_step: float = 0.1
    beta: float = 1.0
    initial_count: float = 1.0
    warmup_groups: int = 8

    def validate(self) -> None:
        if self.count_step <= 0.0:
            raise ValueError("count_step must be > 0")
        if self.beta <= 0.0:
            raise ValueError("beta must be > 0")
        if self.initial_count <= 0.0:
            raise ValueError("initial_count must be > 0")
        if self.warmup_groups < 0:
            raise ValueError("warmup_groups must be >= 0")


def update_linear_counts(
    outcome_ids: Sequence[object],
    pseudo_counts: MutableMapping[object, float],
    *,
    count_step: float = 0.1,
    initial_count: float = 1.0,
) -> dict[str, float]:
    """Apply aggregated linear repeat increments without ordering effects."""
    if len(outcome_ids) == 0:
        raise ValueError("outcome_ids must be non-empty")
    if count_step <= 0.0 or initial_count <= 0.0:
        raise ValueError("count_step and initial_count must be > 0")

    batch_counts = Counter(outcome_ids)
    new_outcomes = 0
    repeat_observations = 0
    for outcome, observations in batch_counts.items():
        if outcome in pseudo_counts:
            pseudo_counts[outcome] = float(
                pseudo_counts[outcome] + count_step * observations
            )
            repeat_observations += observations
        else:
            pseudo_counts[outcome] = float(
                initial_count + count_step * (observations - 1)
            )
            new_outcomes += 1
            repeat_observations += observations - 1

    return {
        "pool_new_outcomes": float(new_outcomes),
        "pool_repeat_observations": float(repeat_observations),
        "pool_unique_outcomes": float(len(batch_counts)),
        "pool_observations": float(len(outcome_ids)),
    }


class LinearGlobalCountIPSTrainer(SaturatingGlobalCountIPSTrainer):
    """Persistent linear pseudo-counts plus normal annealed exploration."""

    probability_label = "p_linear"
    algorithm_name = "linear_global_count_ips"

    def __init__(
        self,
        config: TrainConfig | None = None,
        *,
        device: str = "cpu",
        exploration: ExplorationConfig | None = None,
        linear_count: LinearGlobalCountConfig | None = None,
    ) -> None:
        resolved_config = config or TrainConfig()
        self.linear_count = linear_count or LinearGlobalCountConfig()
        self.linear_count.validate()
        if exploration is None:
            exploration = ExplorationConfig(
                epsilon_start=0.30,
                epsilon_end=0.10,
                temperature_start=2.0,
                temperature_end=1.25,
                anneal_updates=resolved_config.num_updates,
                schedule="cosine",
            )

        # The parent owns the rollout, warm-up, PPO, store-history, and plotting
        # integration. Linear-specific store and advantage hooks below replace
        # every use of this compatibility configuration during learning.
        compatibility_config = SaturatingGlobalCountConfig(
            beta=self.linear_count.beta,
            initial_count=self.linear_count.initial_count,
            warmup_groups=self.linear_count.warmup_groups,
        )
        super().__init__(
            resolved_config,
            device=device,
            exploration=exploration,
            saturating_count=compatibility_config,
        )

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
        return {
            "store_tracked_outcomes": float(len(values)),
            "store_coverage_fraction": float(len(values) / len(self.terminals)),
            "store_pseudo_count_mass": float(values.sum()),
            "store_pseudo_count_min": float(values.min()),
            "store_pseudo_count_mean": float(values.mean()),
            "store_pseudo_count_median": float(np.median(values)),
            "store_pseudo_count_p90": float(np.quantile(values, 0.9)),
            "store_pseudo_count_max": float(values.max()),
            "store_saturated_fraction": 0.0,
            "global_total_observations": float(
                sum(self._lifetime_terminal_counts.values())
            ),
        }

    def _update_store(self, outcomes: Sequence[State]) -> dict[str, float]:
        self._lifetime_terminal_counts.update(outcomes)
        update_metrics = update_linear_counts(
            outcomes,
            self._pseudo_counts,
            count_step=self.linear_count.count_step,
            initial_count=self.linear_count.initial_count,
        )
        self._seen_terminals.update(outcomes)
        return {**update_metrics, **self._store_metrics()}

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
            beta=self.linear_count.beta,
            eps=self.config.advantage_eps,
        )
        for episode, advantage in zip(episodes, advantages):
            for record in episode.steps:
                record.advantage = float(advantage)

        metrics.update(
            {
                **pool_metrics,
                "exploration_epsilon": self.current_epsilon,
                "exploration_temperature": self.current_temperature,
                "pseudo_count_step": self.linear_count.count_step,
                "pseudo_count_beta": self.linear_count.beta,
                "pseudo_count_initial": self.linear_count.initial_count,
                "pseudo_count_is_linear": 1.0,
                "advantage_group_size": float(len(episodes)),
            }
        )
        return metrics

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
                    "linear_count": asdict(self.linear_count),
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
    ) -> "LinearGlobalCountIPSTrainer":
        payload = torch.load(Path(path), map_location=device, weights_only=False)
        algorithm = payload.get("algorithm", {})
        if algorithm.get("name") != cls.algorithm_name:
            raise ValueError(f"checkpoint is not a {cls.algorithm_name} run")
        trainer = cls(
            payload["config"],
            device=device,
            exploration=ExplorationConfig(**algorithm["exploration"]),
            linear_count=LinearGlobalCountConfig(**algorithm["linear_count"]),
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


def _plot_linear_store(
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
    )
    coverage_axis = ax.twinx()
    coverage_axis.plot(
        steps,
        [row["store_coverage_fraction"] for row in history],
        color="#00b894",
    )
    coverage_axis.set_ylim(-0.02, 1.02)
    ax.set_ylabel("tracked terminals", color="#0984e3")
    coverage_axis.set_ylabel("coverage fraction", color="#00b894")
    ax.set_title("Persistent store coverage")

    ax = axes[0, 1]
    for key, label in (
        ("store_pseudo_count_min", "min"),
        ("store_pseudo_count_median", "median"),
        ("store_pseudo_count_p90", "p90"),
        ("store_pseudo_count_max", "max"),
    ):
        ax.plot(steps, [row[key] for row in history], label=label)
    ax.set_title("Non-saturating pseudo-count distribution")
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
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Inverse-weight stability")
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
    fig.suptitle("Linear global Count-IPS diagnostics")
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
    parser.add_argument("--count-step", type=float, default=0.1)
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
    parser.add_argument("--anneal-updates", type=int, default=2_000)
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
    linear_count = LinearGlobalCountConfig(
        count_step=args.count_step,
        beta=args.count_beta,
        initial_count=args.initial_count,
        warmup_groups=args.warmup_groups,
    )
    run_dir = args.run_dir or (
        Path(__file__).resolve().parent
        / "data"
        / "linear_global_count_ips_runs"
        / (
            f"{datetime.now():%Y%m%d_%H%M%S}_b{config.budget}_"
            f"g{config.group_size}_seed{config.seed}"
        )
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    trainer = LinearGlobalCountIPSTrainer(
        config,
        device=_resolve_device(args.device),
        exploration=exploration,
        linear_count=linear_count,
    )

    print(f"Run directory: {run_dir}")
    print(f"Device: {trainer.device}")
    print(f"DAG: {trainer.environment_summary()}")
    print(
        "Algorithm: reward / normalized non-saturating pseudo-count -> "
        "group-normalized PPO advantage"
    )
    print(
        "Pseudo-count: "
        f"initial={linear_count.initial_count:g}, "
        f"repeat_step={linear_count.count_step:g}, "
        f"beta={linear_count.beta:g}"
    )
    print(
        "Exploration: "
        f"{linear_count.warmup_groups} rollout-only warm-up groups; "
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
                "linear_count_config": asdict(linear_count),
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
        propensity_title="Relative linear pseudo-propensities",
        suptitle="Linear global Count-IPS training diagnostics",
    )
    _plot_exploration(history, run_dir / "exploration_schedule.png")
    _plot_linear_store(history, run_dir / "linear_store_diagnostics.png")

    evaluation = trainer.evaluate(args.final_samples)
    sampling = _plot_final_counts(
        trainer,
        evaluation,
        samples=args.final_samples,
        output=run_dir / "sampling_counts.png",
        suptitle="Linear global Count-IPS vs ideal reward sampling",
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
        "exploration": asdict(exploration),
        "linear_count": asdict(linear_count),
        "warmup": dict(trainer.warmup_summary),
        "final_store_metrics": trainer._store_metrics(),
        "final_pseudo_counts": {
            state.signature: float(value)
            for state, value in trainer.pseudo_counts.items()
        },
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
            "linear_store": "linear_store_diagnostics.png",
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
