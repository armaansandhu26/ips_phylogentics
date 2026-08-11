"""Enumerate all minimal 3x3 trajectories and their terminal rewards."""

from __future__ import annotations

import argparse
from collections import Counter

from grid_environment import GridEnv
from catalog import DEFAULT_ENUM_PATH, build_catalog, iter_trajectories, write_trajectory_list


def main() -> None:
    parser = argparse.ArgumentParser(description="Enumerate all 3x3 minimal trajectories.")
    parser.add_argument("--output", type=str, default=str(DEFAULT_ENUM_PATH))
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    env = GridEnv()
    catalog = build_catalog(env)
    reward_counts = Counter(round(r, 6) for r in catalog.rewards)

    print(
        f"{env.grid_size}x{env.grid_size}: "
        f"{catalog.num_paths} paths x 2^{catalog.steps_per_episode} = {catalog.num_trajectories}"
    )
    print(
        f"max_episode_steps={env.max_episode_steps}, "
        f"reward [{catalog.min_reward:.6f}, {catalog.max_reward:.6f}], "
        f"mean={catalog.mean_reward:.6f}"
    )
    print(f"distinct rewards: {len(reward_counts)}")
    for reward, count in sorted(reward_counts.items(), reverse=True):
        print(f"  {reward:.6f} x{count}")

    print(f"\nTop {args.top} trajectories:")
    for record in sorted(iter_trajectories(env), key=lambda t: (-t.reward, t.index))[: args.top]:
        print(record.format_line())

    output_path = write_trajectory_list(args.output, env=env)
    print(f"\nFull list written to {output_path}")


if __name__ == "__main__":
    main()
