"""Scatter plot: sampling density vs reward for each unique hit trajectory."""

from __future__ import annotations

import argparse
from pathlib import Path

from grid_environment_2 import GridEnv
from plot_ips_groupsize_overlay import (
    DEFAULT_GS16_SAMPLE,
    DEFAULT_GS64_SAMPLE,
    DEFAULT_GS16_CHECKPOINT,
    DEFAULT_GS64_CHECKPOINT,
    match_unique_trajectories,
    trajectory_index_path,
)
from sampling_comparison import plot_unique_trajectory_scatter
from trajectory_catalog import iter_trajectories

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_PLOT_PATH = DATA_DIR / "ips_grpo_gs16_vs_gs64_unique_trajectory_scatter.png"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scatter density vs reward for each unique sampled trajectory."
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
    parser.add_argument("--no-jitter", action="store_true")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    env = GridEnv()
    records = list(iter_trajectories(env))
    grid_lookup = {r.final_grid: r.index for r in records}
    reward_by_index = {r.index: r.reward for r in records}

    stats_list = []
    configs = (
        (args.gs16_label, args.gs16_checkpoint, args.gs16_sample),
        (args.gs64_label, args.gs64_checkpoint, args.gs64_sample),
    )

    for label, checkpoint, sample_path in configs:
        stats = match_unique_trajectories(
            label=label,
            checkpoint=checkpoint,
            env=env,
            grid_lookup=grid_lookup,
            reward_by_index=reward_by_index,
            episodes=args.episodes,
            greedy=args.greedy,
            index_path=trajectory_index_path(sample_path),
        )
        stats_list.append(stats)
        print(
            f"{label}: {stats.unique_trajectories_hit} unique trajectories plotted "
            f"(density = count / {stats.episodes_sampled})"
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
        jitter=not args.no_jitter,
        show=args.show,
    )
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
