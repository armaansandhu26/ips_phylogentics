"""Overlay IPS-GRPO sampling density: group_size 16 vs 64 (4x4)."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from grid_environment_2 import GridEnv
from grpo import load_agent_from_checkpoint
from sampling_comparison import (
    UniqueTrajectoryStats,
    format_unique_trajectory_annotation,
    plot_reward_density_overlay,
    sample_rewards,
    sample_rewards_and_indices,
    unique_trajectory_stats,
    write_unique_trajectory_summary,
)
from trajectory_catalog import iter_trajectories

DATA_DIR = Path(__file__).resolve().parent / "data"
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
DEFAULT_GS16_CHECKPOINT = CHECKPOINT_DIR / "ips_grpo.pt"
DEFAULT_GS64_CHECKPOINT = CHECKPOINT_DIR / "ips_grpo_gs64.pt"
DEFAULT_GS16_SAMPLE = DATA_DIR / "ips_grpo_gs16_sample_rewards_100k.npy"
DEFAULT_GS64_SAMPLE = DATA_DIR / "ips_grpo_gs64_sample_rewards_100k.npy"
DEFAULT_PLOT_PATH = DATA_DIR / "ips_grpo_gs16_vs_gs64_sampling_density.png"
DEFAULT_SUMMARY_PATH = DATA_DIR / "ips_grpo_gs16_vs_gs64_unique_trajectories.txt"


def trajectory_index_path(sample_path: Path) -> Path:
    return sample_path.with_name(sample_path.stem + "_trajectory_indices.npy")


def load_rewards(
    *,
    sample_path: Path | None,
    checkpoint: Path,
    env: GridEnv,
    episodes: int,
    greedy: bool,
) -> np.ndarray:
    if sample_path is not None and sample_path.exists():
        return np.load(sample_path).astype(np.float64)

    agent = load_agent_from_checkpoint(checkpoint)
    return sample_rewards(agent, env, num_episodes=episodes, greedy=greedy)


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
        description="Overlay IPS-GRPO group_size 16 vs 64 sampling density."
    )
    parser.add_argument("--gs16-label", type=str, default="IPS-GRPO (group_size=16)")
    parser.add_argument("--gs64-label", type=str, default="IPS-GRPO (group_size=64)")
    parser.add_argument("--gs16-checkpoint", type=Path, default=DEFAULT_GS16_CHECKPOINT)
    parser.add_argument("--gs64-checkpoint", type=Path, default=DEFAULT_GS64_CHECKPOINT)
    parser.add_argument("--gs16-sample", type=Path, default=DEFAULT_GS16_SAMPLE)
    parser.add_argument("--gs64-sample", type=Path, default=DEFAULT_GS64_SAMPLE)
    parser.add_argument("--episodes", type=int, default=100_000)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--plot-path", type=Path, default=DEFAULT_PLOT_PATH)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument(
        "--skip-matching",
        action="store_true",
        help="Skip trajectory matching unless cached indices exist.",
    )
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    env = GridEnv()
    records = list(iter_trajectories(env))
    enumerated_rewards = [r.reward for r in records]
    grid_lookup = {r.final_grid: r.index for r in records}
    reward_by_index = {r.index: r.reward for r in records}

    reward_sets: list[tuple[str, list[float]]] = []
    stats_list: list[UniqueTrajectoryStats] = []

    configs = (
        (args.gs16_label, args.gs16_checkpoint, args.gs16_sample),
        (args.gs64_label, args.gs64_checkpoint, args.gs64_sample),
    )

    for label, checkpoint, sample_path in configs:
        rewards = load_rewards(
            sample_path=sample_path,
            checkpoint=checkpoint,
            env=env,
            episodes=args.episodes,
            greedy=args.greedy,
        )
        reward_sets.append((label, rewards.tolist()))
        print(
            f"{label}: n={len(rewards)}  mean={rewards.mean():.6f}  "
            f"min={rewards.min():.6f}  max={rewards.max():.6f}"
        )

        index_path = trajectory_index_path(sample_path)
        if not args.skip_matching or index_path.exists():
            stats = match_unique_trajectories(
                label=label,
                checkpoint=checkpoint,
                env=env,
                grid_lookup=grid_lookup,
                reward_by_index=reward_by_index,
                episodes=len(rewards),
                greedy=args.greedy,
                index_path=index_path,
            )
            stats_list.append(stats)
            print(
                f"  unique trajectories: {stats.unique_trajectories_hit}/{stats.enumerated_count}  "
                f"matched={stats.matched_episodes}  unmatched={stats.unmatched_episodes}"
            )

    title = (
        f"IPS-GRPO group_size 16 vs 64 sampling density "
        f"({env.grid_size}x{env.grid_size}, {len(reward_sets[0][1])} episodes each)"
    )
    annotation = format_unique_trajectory_annotation(stats_list) if stats_list else None
    plot_path = plot_reward_density_overlay(
        reward_sets,
        title=title,
        save_path=args.plot_path,
        enumerated_rewards=enumerated_rewards,
        stats_annotation=annotation,
        show=args.show,
    )
    print(f"Saved plot: {plot_path}")

    if stats_list:
        write_unique_trajectory_summary(args.summary_path, stats_list)
        print(f"Saved summary: {args.summary_path}")


if __name__ == "__main__":
    main()
