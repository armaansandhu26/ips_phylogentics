"""Count-IPS GRPO with a deduplicated terminal-outcome replay buffer.

The replay buffer contains exactly one entry for each terminal outcome observed
during training.  Its key is the terminal state, while its payload is the most
recent complete trajectory reaching that terminal (PPO still needs states,
actions, and behavior-policy log probabilities).

Each training group contains fresh on-policy trajectories and trajectories
sampled uniformly by terminal outcome from the buffer.  The two sources are
combined before count-IPS scaling and joint advantage normalization:

    p_hat(x_i) = count(x_i) / group_size
    scaled_i   = reward(x_i) / p_hat(x_i)
    advantage  = normalize(scaled)

For a frontier ``x + y == budget``, the buffer can contain at most
``budget + 1`` entries, independent of the number of paths to each outcome.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from config import TrainConfig
from count_ips import Episode
from dag_env import State
from deduplicated_buffer_ips import (
    BufferedTrajectory,
    DeduplicatedBufferIPSTrainer,
    _resolve_device,
)


@dataclass(frozen=True)
class OutcomeBufferAddStats:
    inserted_outcomes: int
    refreshed_outcomes: int


class DeduplicatedOutcomeBuffer:
    """Unbounded-in-time buffer with one replay trajectory per terminal outcome."""

    def __init__(self, *, seed: int = 0) -> None:
        self._entries: dict[State, BufferedTrajectory] = {}
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, terminal: State) -> bool:
        return terminal in self._entries

    @property
    def outcomes(self) -> tuple[State, ...]:
        return tuple(self._entries)

    def add(self, episodes: Iterable[Episode]) -> OutcomeBufferAddStats:
        """Insert new outcomes and refresh the representative for known outcomes."""
        inserted = 0
        refreshed = 0
        for episode in episodes:
            if not episode.trajectory:
                raise ValueError("cannot buffer an empty trajectory")
            if episode.terminal in self._entries:
                refreshed += 1
            else:
                inserted += 1
            self._entries[episode.terminal] = BufferedTrajectory.from_episode(
                episode
            )
        return OutcomeBufferAddStats(
            inserted_outcomes=inserted,
            refreshed_outcomes=refreshed,
        )

    def sample(self, size: int) -> tuple[list[BufferedTrajectory], bool]:
        """Sample outcomes uniformly, with replacement only when necessary."""
        if size < 0:
            raise ValueError("sample size must be non-negative")
        if size == 0:
            return [], False
        if not self._entries:
            raise RuntimeError("cannot sample from an empty outcome buffer")

        entries = list(self._entries.values())
        with_replacement = size > len(entries)
        if with_replacement:
            return self._rng.choices(entries, k=size), True
        return self._rng.sample(entries, k=size), False

    def state_dict(self) -> dict[str, Any]:
        return {
            "entries": [
                {
                    "trajectory": entry.trajectory,
                    "terminal": (entry.terminal.x, entry.terminal.y),
                    "reward": entry.reward,
                    "log_prob_directions": entry.log_prob_directions,
                    "log_prob_steps": entry.log_prob_steps,
                }
                for entry in self._entries.values()
            ],
            "rng_state": self._rng.getstate(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        entries = [
            BufferedTrajectory(
                trajectory=tuple(
                    (int(direction), int(length))
                    for direction, length in payload["trajectory"]
                ),
                terminal=State(*payload["terminal"]),
                reward=float(payload["reward"]),
                log_prob_directions=tuple(payload["log_prob_directions"]),
                log_prob_steps=tuple(payload["log_prob_steps"]),
            )
            for payload in state.get("entries", ())
        ]
        self._entries = {entry.terminal: entry for entry in entries}
        if "rng_state" in state:
            self._rng.setstate(state["rng_state"])


class DeduplicatedOutcomeBufferIPSTrainer(DeduplicatedBufferIPSTrainer):
    """Count-IPS trainer with uniformly sampled terminal-outcome replay."""

    def __init__(
        self,
        config: TrainConfig | None = None,
        *,
        device: str = "cpu",
        replay_fraction: float = 0.5,
    ) -> None:
        super().__init__(
            config,
            device=device,
            replay_fraction=replay_fraction,
        )
        # The parent supplies shared rollout/materialization machinery; this
        # variant deliberately has no trajectory-keyed replay buffer.
        del self.trajectory_buffer
        self.outcome_buffer = DeduplicatedOutcomeBuffer(seed=self.config.seed)

    def _build_mixed_group(self) -> tuple[list[Episode], dict[str, float]]:
        fresh = self.rollout_batch(self.on_policy_group_size)

        # Current outcomes are eligible immediately, which makes replay possible
        # during the first update even when the buffer starts empty.
        add_stats = self.outcome_buffer.add(fresh)
        replay_entries, sampled_with_replacement = self.outcome_buffer.sample(
            self.replay_group_size
        )
        replay = [self._materialize(entry) for entry in replay_entries]
        combined = fresh + replay

        # Estimate the outcome propensity and normalize advantages only after the
        # on-policy and outcome-buffer subgroups have been combined.
        self._group_advantages(combined)
        source_metrics = {
            "on_policy_count": float(len(fresh)),
            "replay_count": float(len(replay)),
            "replay_fraction": float(len(replay) / len(combined)),
            "outcome_buffer_size": float(len(self.outcome_buffer)),
            "buffer_inserted_outcomes": float(add_stats.inserted_outcomes),
            "buffer_refreshed_outcomes": float(add_stats.refreshed_outcomes),
            "replay_unique_outcomes": float(
                len({entry.terminal for entry in replay_entries})
            ),
            "replay_sampled_with_replacement": float(sampled_with_replacement),
        }
        return combined, source_metrics

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
            all_episodes: list[Episode] = []
            group_metrics: list[dict[str, float]] = []
            for _ in range(self.config.num_groups):
                group, source_metrics = self._build_mixed_group()
                group_metrics.append({**self._last_ips_metrics, **source_metrics})
                all_episodes.extend(group)

            stats = self.update(all_episodes)
            batch_counts = Counter(episode.terminal for episode in all_episodes)
            batch_size = len(all_episodes)
            self._seen_terminals.update(batch_counts)
            row: dict[str, Any] = {
                "step": update_step,
                "mean_reward": float(
                    np.mean([episode.reward for episode in all_episodes])
                ),
                "mean_length": float(
                    np.mean([len(episode.steps) for episode in all_episodes])
                ),
                "unique_terminals": len(
                    {episode.terminal for episode in all_episodes}
                ),
                "global_unique_outcomes": float(len(self._seen_terminals)),
                "batch_outcome_counts": {
                    state.signature: int(batch_counts[state])
                    for state in self.terminals
                },
                "batch_outcome_probs": {
                    state.signature: float(batch_counts[state] / batch_size)
                    for state in self.terminals
                },
                **{
                    key: float(np.mean([metrics[key] for metrics in group_metrics]))
                    for key in group_metrics[0]
                },
                "outcome_buffer_size": float(len(self.outcome_buffer)),
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
                    f"update {update_step:4d}  reward={row['mean_reward']:.3f}  "
                    f"fresh={self.on_policy_group_size}  "
                    f"replay={self.replay_group_size}  "
                    f"outcome_buffer={int(row['outcome_buffer_size'])}  "
                    f"outcomes={row['ips_unique_outcomes']:.1f}  "
                    f"p_hat={row['ips_prob_mean']:.3f}  "
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
                "optimizer": self.optimizer.state_dict(),
                "update_step": update_step,
                "algorithm": {
                    "name": "deduplicated_outcome_buffer_count_ips",
                    "replay_fraction": self.replay_fraction,
                    "on_policy_group_size": self.on_policy_group_size,
                    "replay_group_size": self.replay_group_size,
                },
                "outcome_buffer": self.outcome_buffer.state_dict(),
            },
            path,
        )
        return path

    @classmethod
    def load(
        cls, path: Path | str, *, device: str = "cpu"
    ) -> "DeduplicatedOutcomeBufferIPSTrainer":
        payload = torch.load(Path(path), map_location=device, weights_only=False)
        trainer = cls(
            payload["config"],
            device=device,
            replay_fraction=payload["algorithm"]["replay_fraction"],
        )
        trainer.direction_policy.load_state_dict(payload["direction_policy"])
        trainer.step_policy.load_state_dict(payload["step_policy"])
        if "optimizer" in payload:
            trainer.optimizer.load_state_dict(payload["optimizer"])
        trainer.outcome_buffer.load_state_dict(payload["outcome_buffer"])
        return trainer


def main() -> None:
    from run_count_ips import (
        _plot_final_counts,
        _plot_training_curves,
        _plot_trajectory_diagnostics,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=32)
    parser.add_argument("--max-step", type=int, default=3)
    parser.add_argument("--num-updates", type=int, default=500)
    parser.add_argument("--group-size", type=int, default=512)
    parser.add_argument("--num-groups", type=int, default=1)
    parser.add_argument(
        "--replay-fraction",
        type=float,
        default=0.5,
        help="fraction of each group sampled uniformly from known outcomes",
    )
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
        / "deduplicated_outcome_buffer_ips_runs"
        / (
            f"{datetime.now():%Y%m%d_%H%M%S}_b{config.budget}"
            f"_gs{config.group_size}_rf{args.replay_fraction:g}_seed{config.seed}"
        )
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    trainer = DeduplicatedOutcomeBufferIPSTrainer(
        config,
        device=_resolve_device(args.device),
        replay_fraction=args.replay_fraction,
    )
    print(f"Run directory: {run_dir}")
    print(f"Device: {trainer.device}")
    print(f"DAG: {trainer.environment_summary()}")
    print(
        "Algorithm: joint count-IPS over "
        f"fresh={trainer.on_policy_group_size} + "
        f"uniform-outcome replay={trainer.replay_group_size}"
    )

    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "train_config": asdict(config),
                "device": str(trainer.device),
                "algorithm": "deduplicated_outcome_buffer_count_ips",
                "buffer_key": "terminal_outcome",
                "buffer_payload": "latest_complete_trajectory",
                "replay_fraction": args.replay_fraction,
                "on_policy_group_size": trainer.on_policy_group_size,
                "replay_group_size": trainer.replay_group_size,
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
        suptitle="Outcome-buffer IPS sampling vs ideal reward sampling",
    )
    trajectory_plot = run_dir / "trajectory_sampling.png"
    trajectory_sampling = _plot_trajectory_diagnostics(
        history,
        trainer,
        final_evaluation,
        samples=args.final_samples,
        output=trajectory_plot,
        subtitle="Replay samples discovered terminal outcomes uniformly",
    )
    eval_rows = [row for row in history if "tv_reward_target" in row]
    summary = {
        "environment": trainer.environment_summary(),
        "outcome_buffer_size": len(trainer.outcome_buffer),
        "outcome_buffer_capacity": config.budget + 1,
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

    print(
        f"Final outcome buffer: {len(trainer.outcome_buffer)}/"
        f"{config.budget + 1} terminal outcomes"
    )
    print(f"Final counts: {sampling['actual_counts']}")
    print(f"Ideal counts: {sampling['ideal_counts']}")
    print(f"Final TV distance: {sampling['tv_reward_target']:.4f}")
    print(f"Training curves: {training_plot}")
    print(f"Sampling plots: {run_dir / 'sampling_counts.png'}")
    print(f"Trajectory plots: {trajectory_plot}")
    print(f"Artifacts: {run_dir}")


if __name__ == "__main__":
    main()
