"""Count-IPS GRPO with an all-history deduplicated trajectory buffer.

Each training group is assembled from two sources:

1. fresh trajectories sampled from the current policy;
2. trajectories sampled uniformly from a buffer containing one entry for every
   distinct action trajectory observed so far.

The two sources are concatenated *before* computing the usual count-IPS score,

    p_hat(x_i) = count(x_i) / group_size
    scaled_i   = reward(x_i) / p_hat(x_i)
    advantage  = normalize(scaled)

and the resulting advantages feed the same token-level clipped PPO loss as
``count_ips.py``.  Re-observing a trajectory refreshes its stored rollout data
without adding a duplicate.  The buffer is intentionally unbounded so that it
retains every unique trajectory encountered throughout training.
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
from count_ips import CountIPSTrainer, Episode, StepRecord
from dag_env import RIGHT, UP, State, Trajectory


@dataclass(frozen=True)
class BufferedTrajectory:
    """Compact replay representation for one unique complete trajectory."""

    trajectory: Trajectory
    terminal: State
    reward: float
    log_prob_directions: tuple[float, ...]
    log_prob_steps: tuple[float, ...]

    @classmethod
    def from_episode(cls, episode: Episode) -> "BufferedTrajectory":
        return cls(
            trajectory=episode.trajectory,
            terminal=episode.terminal,
            reward=float(episode.reward),
            log_prob_directions=tuple(
                float(step.log_prob_direction) for step in episode.steps
            ),
            log_prob_steps=tuple(float(step.log_prob_step) for step in episode.steps),
        )


@dataclass(frozen=True)
class BufferAddStats:
    inserted: int
    refreshed: int


class DeduplicatedTrajectoryBuffer:
    """Unbounded replay buffer with exactly one entry per action trajectory.

    Duplicate observations refresh the payload.  This keeps the most recent
    behavior-policy log probabilities while preserving deduplicated membership.
    """

    def __init__(self, *, seed: int = 0) -> None:
        self._entries: dict[Trajectory, BufferedTrajectory] = {}
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, trajectory: Trajectory) -> bool:
        return trajectory in self._entries

    @property
    def trajectories(self) -> tuple[Trajectory, ...]:
        return tuple(self._entries)

    def add(self, episodes: Iterable[Episode]) -> BufferAddStats:
        inserted = 0
        refreshed = 0
        for episode in episodes:
            if not episode.trajectory:
                raise ValueError("cannot buffer an empty trajectory")
            if episode.trajectory in self._entries:
                refreshed += 1
            else:
                inserted += 1
            self._entries[episode.trajectory] = BufferedTrajectory.from_episode(
                episode
            )
        return BufferAddStats(inserted=inserted, refreshed=refreshed)

    def sample(self, size: int) -> tuple[list[BufferedTrajectory], bool]:
        """Sample entries, using replacement only while the buffer is too small."""
        if size < 0:
            raise ValueError("sample size must be non-negative")
        if size == 0:
            return [], False
        if not self._entries:
            raise RuntimeError("cannot sample from an empty deduplicated buffer")

        entries = list(self._entries.values())
        with_replacement = size > len(entries)
        if with_replacement:
            return self._rng.choices(entries, k=size), True
        return self._rng.sample(entries, k=size), False

    def state_dict(self) -> dict[str, Any]:
        return {
            # Use only primitive containers so checkpoints written by the CLI do
            # not pickle this class under the non-importable ``__main__`` module.
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
        self._entries = {entry.trajectory: entry for entry in entries}
        if "rng_state" in state:
            self._rng.setstate(state["rng_state"])


def split_group_sizes(group_size: int, replay_fraction: float) -> tuple[int, int]:
    """Return fresh and replay counts while keeping both sources non-empty."""
    if group_size < 2:
        raise ValueError("group_size must be >= 2 for a fresh/replay split")
    if not 0.0 < replay_fraction < 1.0:
        raise ValueError("replay_fraction must be strictly between 0 and 1")
    replay_size = min(
        group_size - 1,
        max(1, int(round(group_size * replay_fraction))),
    )
    return group_size - replay_size, replay_size


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is not available")
    return requested


class DeduplicatedBufferIPSTrainer(CountIPSTrainer):
    """Count-IPS trainer using fresh and deduplicated-replay subgroups."""

    def __init__(
        self,
        config: TrainConfig | None = None,
        *,
        device: str = "cpu",
        replay_fraction: float = 0.5,
    ) -> None:
        super().__init__(config, device=device)
        self.on_policy_group_size, self.replay_group_size = split_group_sizes(
            self.config.group_size, replay_fraction
        )
        self.replay_fraction = float(replay_fraction)
        self.trajectory_buffer = DeduplicatedTrajectoryBuffer(seed=self.config.seed)

    def _materialize(self, entry: BufferedTrajectory) -> Episode:
        """Reconstruct deterministic observations/masks for a buffered path."""
        trajectory = entry.trajectory
        if not (
            len(trajectory)
            == len(entry.log_prob_directions)
            == len(entry.log_prob_steps)
        ):
            raise ValueError("buffered action and log-probability lengths differ")

        width = self.config.budget + 1
        x = 0
        y = 0
        steps: list[StepRecord] = []
        for index, (direction, physical_step) in enumerate(trajectory):
            remaining = self.config.budget - x - y
            if direction not in (RIGHT, UP):
                raise ValueError(f"invalid buffered direction: {direction}")
            if physical_step < 1 or physical_step > min(
                self.config.max_step, remaining
            ):
                raise ValueError(
                    f"invalid buffered step {physical_step} with remaining={remaining}"
                )

            obs = np.zeros(3 * width, dtype=np.float32)
            obs[x] = 1.0
            obs[width + y] = 1.0
            obs[2 * width + remaining] = 1.0
            step_mask = np.arange(self.config.max_step) < min(
                self.config.max_step, remaining
            )
            steps.append(
                StepRecord(
                    obs=obs,
                    direction_mask=np.ones(2, dtype=bool),
                    step_mask=step_mask,
                    direction=direction,
                    step_index=physical_step - 1,
                    log_prob_direction=entry.log_prob_directions[index],
                    log_prob_step=entry.log_prob_steps[index],
                )
            )
            if direction == RIGHT:
                x += physical_step
            else:
                y += physical_step

        terminal = State(x, y)
        if terminal != entry.terminal or terminal.depth != self.config.budget:
            raise ValueError(
                "buffered trajectory does not reconstruct its recorded terminal"
            )
        return Episode(
            steps=steps,
            terminal=terminal,
            signature=terminal.signature,
            trajectory=trajectory,
            reward=entry.reward,
        )

    def _build_mixed_group(self) -> tuple[list[Episode], dict[str, float]]:
        fresh = self.rollout_batch(self.on_policy_group_size)

        # Insert first so the replay half is available from the first update.
        add_stats = self.trajectory_buffer.add(fresh)
        replay_entries, sampled_with_replacement = self.trajectory_buffer.sample(
            self.replay_group_size
        )
        replay = [self._materialize(entry) for entry in replay_entries]
        combined = fresh + replay

        # IPS is intentionally computed over the joint fresh+replay group.
        self._group_advantages(combined)
        source_metrics = {
            "on_policy_count": float(len(fresh)),
            "replay_count": float(len(replay)),
            "replay_fraction": float(len(replay) / len(combined)),
            "deduplicated_buffer_size": float(len(self.trajectory_buffer)),
            "buffer_inserted": float(add_stats.inserted),
            "buffer_refreshed": float(add_stats.refreshed),
            "replay_unique_trajectories": float(
                len({entry.trajectory for entry in replay_entries})
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
                # Report the post-update buffer size exactly, rather than averaging
                # its size across groups within this optimizer step.
                "deduplicated_buffer_size": float(len(self.trajectory_buffer)),
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
                    f"buffer={int(row['deduplicated_buffer_size'])}  "
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
                    "name": "deduplicated_buffer_count_ips",
                    "replay_fraction": self.replay_fraction,
                    "on_policy_group_size": self.on_policy_group_size,
                    "replay_group_size": self.replay_group_size,
                },
                "trajectory_buffer": self.trajectory_buffer.state_dict(),
            },
            path,
        )
        return path

    @classmethod
    def load(
        cls, path: Path | str, *, device: str = "cpu"
    ) -> "DeduplicatedBufferIPSTrainer":
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
        trainer.trajectory_buffer.load_state_dict(payload["trajectory_buffer"])
        return trainer


def main() -> None:
    # Plotting is a CLI-only dependency; importing the trainer does not require it.
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
        help="fraction of each group sampled from the deduplicated buffer",
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
        / "deduplicated_buffer_ips_runs"
        / (
            f"{datetime.now():%Y%m%d_%H%M%S}_b{config.budget}"
            f"_gs{config.group_size}_rf{args.replay_fraction:g}_seed{config.seed}"
        )
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    trainer = DeduplicatedBufferIPSTrainer(
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
        f"deduplicated replay={trainer.replay_group_size}"
    )

    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "train_config": asdict(config),
                "device": str(trainer.device),
                "algorithm": "deduplicated_buffer_count_ips",
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
        suptitle="Deduplicated-buffer IPS sampling vs ideal reward sampling",
    )
    trajectory_plot = run_dir / "trajectory_sampling.png"
    trajectory_sampling = _plot_trajectory_diagnostics(
        history,
        trainer,
        final_evaluation,
        samples=args.final_samples,
        output=trajectory_plot,
        subtitle="Training mixes fresh policy rollouts with unique-trajectory replay",
    )
    eval_rows = [row for row in history if "tv_reward_target" in row]
    summary = {
        "environment": trainer.environment_summary(),
        "deduplicated_buffer_size": len(trainer.trajectory_buffer),
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

    print(f"Final buffer size: {len(trainer.trajectory_buffer)} unique trajectories")
    print(f"Final counts: {sampling['actual_counts']}")
    print(f"Ideal counts: {sampling['ideal_counts']}")
    print(f"Final TV distance: {sampling['tv_reward_target']:.4f}")
    print(f"Training curves: {training_plot}")
    print(f"Sampling plots: {run_dir / 'sampling_counts.png'}")
    print(f"Trajectory plots: {trajectory_plot}")
    print(f"Artifacts: {run_dir}")


if __name__ == "__main__":
    main()
