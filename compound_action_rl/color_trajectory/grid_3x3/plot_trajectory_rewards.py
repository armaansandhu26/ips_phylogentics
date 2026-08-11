"""Plot reward for each enumerated 3x3 minimal trajectory."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from grid_environment import GridEnv
from catalog import DEFAULT_ENUM_PATH, iter_trajectories

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_PLOT_PATH = DATA_DIR / "trajectory_reward_distribution.png"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot 3x3 trajectory rewards.")
    parser.add_argument("--plot-path", type=Path, default=DEFAULT_PLOT_PATH)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    env = GridEnv()
    indices = []
    rewards = []
    for record in iter_trajectories(env):
        indices.append(record.index)
        rewards.append(record.reward)

    args.plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(indices, rewards, marker=".", linestyle="none", markersize=4, color="#0984e3")
    ax.set_xlabel("Trajectory number")
    ax.set_ylabel("Reward")
    ax.set_title(
        f"3x3 trajectory rewards (n={len(indices)}, max steps={env.max_episode_steps})"
    )
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.plot_path, dpi=160)
    if args.show:
        plt.show()
    else:
        plt.close(fig)

    print(f"Saved plot: {args.plot_path}")
    print(f"Enumerated trajectories: {DEFAULT_ENUM_PATH}")


if __name__ == "__main__":
    main()
