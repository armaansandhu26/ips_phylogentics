"""Overlay GRPO vs IPS-GRPO sampling density over reward (3x3)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from grpo import agent_label_from_checkpoint, load_agent_from_checkpoint
from sampling_comparison import plot_reward_density_overlay, sample_rewards

from catalog import iter_trajectories
from grid_environment import GridEnv

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_GRPO_CHECKPOINT = Path(__file__).resolve().parent / "checkpoints" / "grpo.pt"
DEFAULT_IPS_CHECKPOINT = Path(__file__).resolve().parent / "checkpoints" / "ips_grpo.pt"
DEFAULT_PLOT_PATH = DATA_DIR / "grpo_vs_ips_sampling_density.png"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overlay GRPO and IPS-GRPO sampling density over reward."
    )
    parser.add_argument("--grpo-checkpoint", type=Path, default=DEFAULT_GRPO_CHECKPOINT)
    parser.add_argument("--ips-checkpoint", type=Path, default=DEFAULT_IPS_CHECKPOINT)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--plot-path", type=Path, default=DEFAULT_PLOT_PATH)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    env = GridEnv()
    enumerated_rewards = [r.reward for r in iter_trajectories(env)]
    reward_sets: list[tuple[str, list[float]]] = []

    for checkpoint in (args.grpo_checkpoint, args.ips_checkpoint):
        label = agent_label_from_checkpoint(checkpoint)
        agent = load_agent_from_checkpoint(checkpoint)
        rewards = sample_rewards(
            agent,
            env,
            num_episodes=args.episodes,
            greedy=args.greedy,
        )
        reward_sets.append((label, rewards.tolist()))
        print(
            f"{label}: n={len(rewards)}  mean={rewards.mean():.6f}  "
            f"min={rewards.min():.6f}  max={rewards.max():.6f}"
        )

    title = (
        f"GRPO vs IPS-GRPO sampling density "
        f"({env.grid_size}x{env.grid_size}, {args.episodes} episodes each)"
    )
    plot_path = plot_reward_density_overlay(
        reward_sets,
        title=title,
        save_path=args.plot_path,
        enumerated_rewards=enumerated_rewards,
        show=args.show,
    )
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
