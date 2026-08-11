"""Scatter plot: unique trajectory sampling density vs reward (GRPO vs IPS-GRPO, 3x3)."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from grpo import agent_label_from_checkpoint, load_agent_from_checkpoint
from sampling_comparison import (
    UniqueTrajectoryStats,
    plot_unique_trajectory_scatter,
    sample_rewards_and_indices,
    write_unique_trajectory_summary,
)

from catalog import iter_trajectories
from grid_environment import GridEnv

DATA_DIR = Path(__file__).resolve().parent / "data"
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
DEFAULT_GRPO_CHECKPOINT = CHECKPOINT_DIR / "grpo.pt"
DEFAULT_IPS_CHECKPOINT = CHECKPOINT_DIR / "ips_grpo.pt"
DEFAULT_PLOT_PATH = DATA_DIR / "grpo_vs_ips_unique_trajectory_scatter.png"
DEFAULT_SUMMARY_PATH = DATA_DIR / "grpo_vs_ips_unique_trajectories.txt"


def trajectory_index_path(sample_path: Path) -> Path:
    return sample_path.with_name(sample_path.stem + "_trajectory_indices.npy")


def match_unique_trajectories(
    *,
    label: str,
    checkpoint: Path,
    env: GridEnv,
    grid_lookup: dict,
    reward_by_index: dict[int, float],
    episodes: int,
    greedy: bool,
    index_path: Path,
) -> UniqueTrajectoryStats:
    if index_path.exists():
        indices = np.load(index_path)
        hit_counts: Counter[int] = Counter()
        for idx in indices:
            if idx >= 0:
                hit_counts[int(idx)] += 1
        matched = sum(hit_counts.values())
        return UniqueTrajectoryStats(
            label=label,
            episodes_sampled=len(indices),
            enumerated_count=len(reward_by_index),
            matched_episodes=matched,
            unmatched_episodes=len(indices) - matched,
            unique_trajectories_hit=len(hit_counts),
            hit_counts=hit_counts,
        )

    agent = load_agent_from_checkpoint(checkpoint)
    _rewards, indices, unmatched = sample_rewards_and_indices(
        agent,
        env,
        num_episodes=episodes,
        grid_lookup=grid_lookup,
        greedy=greedy,
    )
    index_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(index_path, indices)
    hit_counts: Counter[int] = Counter(int(idx) for idx in indices if idx >= 0)
    return UniqueTrajectoryStats(
        label=label,
        episodes_sampled=episodes,
        enumerated_count=len(reward_by_index),
        matched_episodes=int(np.count_nonzero(indices >= 0)),
        unmatched_episodes=unmatched,
        unique_trajectories_hit=len(hit_counts),
        hit_counts=hit_counts,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scatter unique trajectory density vs reward for GRPO and IPS-GRPO."
    )
    parser.add_argument("--grpo-checkpoint", type=Path, default=DEFAULT_GRPO_CHECKPOINT)
    parser.add_argument("--ips-checkpoint", type=Path, default=DEFAULT_IPS_CHECKPOINT)
    parser.add_argument(
        "--grpo-indices",
        type=Path,
        default=DATA_DIR / "grpo_sample_1000_trajectory_indices.npy",
    )
    parser.add_argument(
        "--ips-indices",
        type=Path,
        default=DATA_DIR / "ips_grpo_sample_1000_trajectory_indices.npy",
    )
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--plot-path", type=Path, default=DEFAULT_PLOT_PATH)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--no-jitter", action="store_true")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    env = GridEnv()
    records = list(iter_trajectories(env))
    grid_lookup = {r.final_grid: r.index for r in records}
    reward_by_index = {r.index: r.reward for r in records}

    stats_list = []
    configs = (
        (agent_label_from_checkpoint(args.grpo_checkpoint), args.grpo_checkpoint, args.grpo_indices),
        (agent_label_from_checkpoint(args.ips_checkpoint), args.ips_checkpoint, args.ips_indices),
    )

    for label, checkpoint, index_path in configs:
        stats = match_unique_trajectories(
            label=label,
            checkpoint=checkpoint,
            env=env,
            grid_lookup=grid_lookup,
            reward_by_index=reward_by_index,
            episodes=args.episodes,
            greedy=args.greedy,
            index_path=index_path,
        )
        stats_list.append(stats)
        print(
            f"{label}: {stats.unique_trajectories_hit}/{stats.enumerated_count} unique trajectories "
            f"({stats.unmatched_episodes} unmatched of {stats.episodes_sampled})"
        )

    title = (
        f"Unique trajectory density vs reward "
        f"({env.grid_size}x{env.grid_size}, {args.episodes} episodes each)"
    )
    plot_path = plot_unique_trajectory_scatter(
        stats_list,
        reward_by_index,
        title=title,
        save_path=args.plot_path,
        colors=("#0984e3", "#e17055"),
        jitter=not args.no_jitter,
        show=args.show,
    )
    write_unique_trajectory_summary(args.summary_path, stats_list)
    print(f"Saved plot: {plot_path}")
    print(f"Saved summary: {args.summary_path}")


if __name__ == "__main__":
    main()
