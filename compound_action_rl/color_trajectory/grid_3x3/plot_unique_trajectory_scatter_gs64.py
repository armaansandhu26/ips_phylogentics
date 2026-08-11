"""Scatter plot: unique trajectory density vs reward (GRPO vs IPS-GRPO, group_size=64)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
GRID_4X4 = ROOT / "grid_4x4"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(GRID_4X4))
sys.path.insert(0, str(HERE))

from plot_unique_trajectory_scatter import (  # noqa: E402
    DATA_DIR,
    DEFAULT_GRPO_CHECKPOINT,
    DEFAULT_IPS_CHECKPOINT,
    match_unique_trajectories,
)
from catalog import iter_trajectories  # noqa: E402
from grpo import agent_label_from_checkpoint  # noqa: E402
from grid_environment import GridEnv  # noqa: E402
from sampling_comparison import (  # noqa: E402
    plot_unique_trajectory_scatter_with_fit,
    write_linear_fit_summary,
    write_unique_trajectory_summary,
)

DEFAULT_PLOT_PATH = DATA_DIR / "grpo_vs_ips_unique_trajectory_scatter_gs64.png"
DEFAULT_SUMMARY_PATH = DATA_DIR / "grpo_vs_ips_unique_trajectories_gs64.txt"
DEFAULT_FIT_SUMMARY_PATH = DATA_DIR / "grpo_vs_ips_unique_trajectories_gs64_linear_fit.txt"
DEFAULT_GRPO_INDICES = DATA_DIR / "grpo_sample_1000_gs64_trajectory_indices.npy"
DEFAULT_IPS_INDICES = DATA_DIR / "ips_grpo_sample_1000_gs64_trajectory_indices.npy"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scatter unique trajectory density vs reward (group_size=64)."
    )
    parser.add_argument("--grpo-checkpoint", type=Path, default=DEFAULT_GRPO_CHECKPOINT)
    parser.add_argument("--ips-checkpoint", type=Path, default=DEFAULT_IPS_CHECKPOINT)
    parser.add_argument("--grpo-indices", type=Path, default=DEFAULT_GRPO_INDICES)
    parser.add_argument("--ips-indices", type=Path, default=DEFAULT_IPS_INDICES)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--plot-path", type=Path, default=DEFAULT_PLOT_PATH)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--fit-summary-path", type=Path, default=DEFAULT_FIT_SUMMARY_PATH)
    parser.add_argument("--no-jitter", action="store_true")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    env = GridEnv()
    records = list(iter_trajectories(env))
    grid_lookup = {r.final_grid: r.index for r in records}
    reward_by_index = {r.index: r.reward for r in records}

    stats_list = []
    configs = (
        (f"{agent_label_from_checkpoint(args.grpo_checkpoint)} (group_size=64)", args.grpo_checkpoint, args.grpo_indices),
        (f"{agent_label_from_checkpoint(args.ips_checkpoint)} (group_size=64)", args.ips_checkpoint, args.ips_indices),
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
        f"({env.grid_size}x{env.grid_size}, group_size=64, {args.episodes} episodes each)"
    )
    plot_path, fit_results = plot_unique_trajectory_scatter_with_fit(
        stats_list,
        reward_by_index,
        title=title,
        save_path=args.plot_path,
        colors=("#0984e3", "#e17055"),
        jitter=not args.no_jitter,
        show=args.show,
        fit_indices=[1],
    )
    write_unique_trajectory_summary(args.summary_path, stats_list)
    write_linear_fit_summary(args.fit_summary_path, fit_results)
    print(f"Saved plot: {plot_path}")
    print(f"Saved summary: {args.summary_path}")
    print(f"Saved OLS fit summary: {args.fit_summary_path}")
    for label, fit in fit_results:
        print(f"{label}: OLS R²={fit.r2:.4f}  RMSE={fit.rmse:.6f}  slope={fit.slope:.6f}")


if __name__ == "__main__":
    main()
