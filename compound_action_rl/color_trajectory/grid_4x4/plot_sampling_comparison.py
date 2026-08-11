"""Overlay GRPO samples on the enumerated trajectory reward plot."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from grid_environment_2 import GridEnv
from grpo import GRPOAgent
from sampling_comparison import generate_comparison
from trajectory_catalog import iter_trajectories

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "checkpoints" / "grpo.pt"
DEFAULT_PLOT_PATH = DATA_DIR / "trajectory_sampling_comparison.png"
DEFAULT_SUMMARY_PATH = DATA_DIR / "trajectory_sampling_comparison.txt"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare GRPO samples against enumerated trajectories."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--greedy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--plot-path", type=Path, default=DEFAULT_PLOT_PATH)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    env = GridEnv()
    records = list(iter_trajectories(env))
    agent = GRPOAgent.from_checkpoint(args.checkpoint)
    matches, plot_path, summary_path, _density_path = generate_comparison(
        agent,
        env,
        records,
        num_episodes=args.episodes,
        plot_path=args.plot_path,
        summary_path=args.summary_path,
        greedy=args.greedy,
        show=args.show,
    )

    hit_counts = Counter(m.trajectory_index for m in matches)
    print(f"Saved comparison plot: {plot_path}")
    print(f"Saved summary: {summary_path}")
    print(
        f"Sampled {args.episodes} episodes, hit {len(hit_counts)} / {len(records)} "
        f"unique enumerated trajectories"
    )


if __name__ == "__main__":
    main()
