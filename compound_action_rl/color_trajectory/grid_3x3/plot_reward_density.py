"""Plot sampling density vs reward for a trained 3x3 policy checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from catalog import iter_trajectories
from grpo import agent_label_from_checkpoint, load_agent_from_checkpoint
from sampling_comparison import plot_reward_density, sample_rewards

from grid_environment import GridEnv

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_PLOT_PATH = DATA_DIR / "sampling_reward_density.png"
DEFAULT_SAMPLE_PATH = DATA_DIR / "grpo_sample_rewards.npy"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot policy sampling density over reward."
    )
    parser.add_argument("--checkpoint", type=str, default="checkpoints/ips_grpo.pt")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument(
        "--sample-path",
        type=Path,
        default=None,
        help="Use saved sample rewards instead of rolling out fresh episodes.",
    )
    parser.add_argument("--plot-path", type=Path, default=DEFAULT_PLOT_PATH)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    env = GridEnv()
    enumerated_rewards = [r.reward for r in iter_trajectories(env)]
    agent_label = agent_label_from_checkpoint(args.checkpoint)

    if args.sample_path is not None:
        sample_rewards_arr = np.load(args.sample_path).astype(np.float64)
    else:
        agent = load_agent_from_checkpoint(args.checkpoint)
        sample_rewards_arr = sample_rewards(
            agent,
            env,
            num_episodes=args.episodes,
            greedy=args.greedy,
        )

    title = (
        f"{agent_label} sampling density vs reward "
        f"({env.grid_size}x{env.grid_size}, {len(sample_rewards_arr)} episodes)"
    )
    plot_path = plot_reward_density(
        sample_rewards_arr,
        title=title,
        save_path=args.plot_path,
        enumerated_rewards=enumerated_rewards,
        show=args.show,
    )

    print(f"Saved plot: {plot_path}")
    print(
        f"Samples n={len(sample_rewards_arr)}  "
        f"mean={sample_rewards_arr.mean():.6f}  "
        f"min={sample_rewards_arr.min():.6f}  "
        f"max={sample_rewards_arr.max():.6f}"
    )


if __name__ == "__main__":
    main()
