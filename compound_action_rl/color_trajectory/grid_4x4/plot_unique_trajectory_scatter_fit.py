"""Scatter + linear fit of unique trajectory density vs reward (all group sizes)."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from grid_environment_2 import GridEnv
from sampling_comparison import (
    UniqueTrajectoryStats,
    plot_unique_trajectory_scatter_with_fit,
    write_linear_fit_summary,
)
from trajectory_catalog import iter_trajectories

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_PLOT_PATH = DATA_DIR / "ips_grpo_all_groupsizes_unique_trajectory_scatter_fit.png"
DEFAULT_SUMMARY_PATH = DATA_DIR / "ips_grpo_all_groupsizes_linear_fit.txt"

DEFAULT_CONFIGS = (
    ("IPS-GRPO (group_size=16)", "data/ips_grpo_gs16_sample_rewards_100k.npy"),
    ("IPS-GRPO (group_size=64)", "data/ips_grpo_gs64_sample_rewards_100k.npy"),
    ("IPS-GRPO (group_size=256)", "data/ips_grpo_gs256_sample_rewards_100k.npy"),
    ("IPS-GRPO (group_size=1024)", "data/ips_grpo_gs1024_sample_rewards_100k.npy"),
)


def load_stats(label: str, sample_path: Path, enumerated_count: int) -> UniqueTrajectoryStats:
    rewards = np.load(sample_path)
    indices = np.load(sample_path.with_name(sample_path.stem + "_trajectory_indices.npy"))
    hit_counts = Counter(int(i) for i in indices if i >= 0)
    return UniqueTrajectoryStats(
        label=label,
        episodes_sampled=len(rewards),
        enumerated_count=enumerated_count,
        matched_episodes=sum(hit_counts.values()),
        unmatched_episodes=int(np.count_nonzero(indices < 0)),
        unique_trajectories_hit=len(hit_counts),
        hit_counts=hit_counts,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scatter + linear fit for unique trajectory density vs reward."
    )
    parser.add_argument("--plot-path", type=Path, default=DEFAULT_PLOT_PATH)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--no-jitter", action="store_true")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    env = GridEnv()
    records = list(iter_trajectories(env))
    reward_by_index = {r.index: r.reward for r in records}

    stats_list = [
        load_stats(label, Path(sample), len(reward_by_index))
        for label, sample in DEFAULT_CONFIGS
    ]

    title = (
        f"Unique trajectory density vs reward with linear fit "
        f"({env.grid_size}x{env.grid_size}, 100k episodes each)"
    )
    plot_path, fit_results = plot_unique_trajectory_scatter_with_fit(
        stats_list,
        reward_by_index,
        title=title,
        save_path=args.plot_path,
        jitter=not args.no_jitter,
        show=args.show,
    )
    write_linear_fit_summary(args.summary_path, fit_results)

    print(f"Saved plot: {plot_path}")
    print(f"Saved summary: {args.summary_path}")
    for label, fit in fit_results:
        print(f"{label}: R²={fit.r2:.4f}  RMSE={fit.rmse:.6f}  slope={fit.slope:.6f}")


if __name__ == "__main__":
    main()
