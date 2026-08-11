"""Plot reward for each enumerated minimal trajectory."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from grid_environment_2 import GridEnv
from trajectory_catalog import DEFAULT_ENUM_PATH, iter_trajectories

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_PLOT_PATH = DATA_DIR / "trajectory_reward_distribution.png"


def plot_trajectory_rewards(
    save_path: Path | str = DEFAULT_PLOT_PATH,
    *,
    show: bool = False,
) -> Path:
    env = GridEnv()
    indices: list[int] = []
    rewards: list[float] = []

    for record in iter_trajectories(env):
        indices.append(record.index)
        rewards.append(record.reward)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(indices, rewards, marker=".", linestyle="none", markersize=3, color="#0984e3")
    ax.set_xlabel("Trajectory number")
    ax.set_ylabel("Reward")
    ax.set_title(f"Trajectory rewards ({env.grid_size}x{env.grid_size}, n={len(indices)})")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return save_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot reward vs trajectory number for all minimal trajectories."
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=DEFAULT_PLOT_PATH,
        help="Where to save the plot.",
    )
    parser.add_argument("--show", action="store_true", help="Show the plot interactively.")
    args = parser.parse_args()

    plot_path = plot_trajectory_rewards(args.plot_path, show=args.show)
    print(f"Saved plot: {plot_path}")
    print(f"Enumerated trajectories listed in: {DEFAULT_ENUM_PATH}")


if __name__ == "__main__":
    main()
