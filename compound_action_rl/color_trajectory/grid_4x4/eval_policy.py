"""Sample rollouts from a trained GRPO checkpoint."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from grid_environment_2 import GREEN_COLOR, GridEnv, RED_COLOR, UNCOLORED
from grpo import GRPOAgent, load_agent_from_checkpoint
from sampling_comparison import generate_comparison, sample_rewards_and_indices
from trajectory_catalog import iter_trajectories

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_SAMPLE_PATH = DATA_DIR / "grpo_sample_rewards.npy"
DEFAULT_PLOT_PATH = DATA_DIR / "trajectory_sampling_comparison.png"
DEFAULT_SUMMARY_PATH = DATA_DIR / "trajectory_sampling_comparison.txt"

CELL_CHARS = {
    UNCOLORED: ".",
    RED_COLOR: "R",
    GREEN_COLOR: "G",
}


def format_grid(colors) -> str:
    lines = []
    for row in range(colors.shape[0]):
        lines.append(" ".join(CELL_CHARS[int(colors[row, col])] for col in range(colors.shape[1])))
    return "\n".join(lines)


def sample_episodes(
    agent: GRPOAgent,
    env: GridEnv,
    *,
    num_episodes: int,
    greedy: bool = False,
) -> np.ndarray:
    rewards = np.empty(num_episodes, dtype=np.float64)
    for episode_idx in range(num_episodes):
        obs, _, _ = env.reset()
        done = False
        while not done:
            if greedy:
                move_action, color_action, _info = agent.act_greedy(obs)
            else:
                move_action, color_action, _info = agent.act(obs)
            obs, reward, done, _ = env.step(move_action, color_action)
        rewards[episode_idx] = reward
    return rewards


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample rollouts from a trained GRPO checkpoint.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/grpo.pt")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument(
        "--greedy",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use greedy actions instead of sampling.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save sampled terminal rewards (.npy).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print summary statistics, not full rollouts.",
    )
    parser.add_argument(
        "--show-first",
        type=int,
        default=0,
        help="When --quiet is set, still print this many full rollouts.",
    )
    parser.add_argument(
        "--comparison",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate comparison plot against enumerated trajectories.",
    )
    parser.add_argument("--plot-path", type=Path, default=DEFAULT_PLOT_PATH)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--show", action="store_true", help="Show comparison plot interactively.")
    parser.add_argument(
        "--capture-trajectories",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When saving rewards, also match and save enumerated trajectory indices.",
    )
    args = parser.parse_args()

    agent = load_agent_from_checkpoint(args.checkpoint)
    env = GridEnv()

    if args.quiet or args.episodes > 20:
        mode = "greedy" if args.greedy else "sampled"

        if args.comparison:
            records = list(iter_trajectories(env))
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
            rewards = np.asarray([m.reward for m in matches], dtype=np.float64)
            print(f"Saved comparison plot: {plot_path}")
            print(f"Saved comparison summary: {summary_path}")
        else:
            if args.capture_trajectories:
                records = list(iter_trajectories(env))
                grid_lookup = {r.final_grid: r.index for r in records}
                rewards, indices, unmatched = sample_rewards_and_indices(
                    agent,
                    env,
                    num_episodes=args.episodes,
                    grid_lookup=grid_lookup,
                    greedy=args.greedy,
                )
                hit_counts = Counter(int(idx) for idx in indices if idx >= 0)
                print(
                    f"  unique trajectories: {len(hit_counts)}/{len(records)}  "
                    f"matched={int(np.count_nonzero(indices >= 0))}  unmatched={unmatched}"
                )
            else:
                rewards = sample_episodes(
                    agent, env, num_episodes=args.episodes, greedy=args.greedy
                )
                indices = None

        print(
            f"{args.episodes} {mode} episodes from {args.checkpoint}\n"
            f"  mean={rewards.mean():.6f}\n"
            f"  min={rewards.min():.6f}\n"
            f"  max={rewards.max():.6f}\n"
            f"  std={rewards.std():.6f}"
        )
        output_path = args.output
        if output_path is None and args.episodes >= 100:
            output_path = DEFAULT_SAMPLE_PATH
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_path, rewards)
            print(f"Saved rewards to {output_path}")
            if indices is not None:
                index_path = output_path.with_name(output_path.stem + "_trajectory_indices.npy")
                np.save(index_path, indices)
                print(f"Saved trajectory indices to {index_path}")

        for episode_idx in range(1, min(args.show_first, args.episodes) + 1):
            _print_episode(agent, env, episode_idx, greedy=args.greedy)
        return

    for episode_idx in range(1, args.episodes + 1):
        _print_episode(agent, env, episode_idx, greedy=args.greedy)


def _print_episode(agent: GRPOAgent, env: GridEnv, episode_idx: int, *, greedy: bool) -> None:
    obs, _, state = env.reset()
    step = 0
    print(f"\n=== Episode {episode_idx} ===")
    print(f"start {tuple(state)}")

    done = False
    while not done:
        step += 1
        if greedy:
            move_action, color_action, _info = agent.act_greedy(obs)
        else:
            move_action, color_action, _info = agent.act(obs)
        move_label, color_label = agent.action_labels(move_action, color_action)
        obs, reward, done, state = env.step(move_action, color_action)
        print(
            f"  step {step:2d}: move={move_label:5s} color={color_label:5s} "
            f"-> pos={tuple(state)}"
        )

    print(f"terminal reward: {reward:.3f}")
    print("final grid:")
    print(format_grid(env._colors))


if __name__ == "__main__":
    main()
