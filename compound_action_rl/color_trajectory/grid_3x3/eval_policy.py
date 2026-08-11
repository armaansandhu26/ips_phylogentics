"""Sample rollouts from a trained 3x3 GRPO checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from catalog import iter_trajectories
from grpo import agent_label_from_checkpoint, load_agent_from_checkpoint
from sampling_comparison import generate_comparison

from grid_environment import GREEN_COLOR, GridEnv, RED_COLOR, UNCOLORED

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_SAMPLE_PATH = DATA_DIR / "grpo_sample_rewards.npy"
DEFAULT_PLOT_PATH = DATA_DIR / "trajectory_sampling_comparison.png"
DEFAULT_DENSITY_PLOT_PATH = DATA_DIR / "sampling_reward_density.png"
DEFAULT_SUMMARY_PATH = DATA_DIR / "trajectory_sampling_comparison.txt"

CELL_CHARS = {UNCOLORED: ".", RED_COLOR: "R", GREEN_COLOR: "G"}


def format_grid(colors) -> str:
    return "\n".join(
        " ".join(CELL_CHARS[int(colors[row, col])] for col in range(colors.shape[1]))
        for row in range(colors.shape[0])
    )


def sample_episodes(agent, env: GridEnv, *, num_episodes: int, greedy: bool = False) -> np.ndarray:
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
    parser = argparse.ArgumentParser(description="Sample 3x3 GRPO rollouts.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/grpo.pt")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--greedy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--comparison",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate comparison plot against enumerated trajectories.",
    )
    parser.add_argument("--plot-path", type=Path, default=DEFAULT_PLOT_PATH)
    parser.add_argument("--density-plot-path", type=Path, default=DEFAULT_DENSITY_PLOT_PATH)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    agent = load_agent_from_checkpoint(args.checkpoint)
    env = GridEnv()
    agent_label = agent_label_from_checkpoint(args.checkpoint)

    if args.quiet or args.episodes > 20:
        mode = "greedy" if args.greedy else "sampled"

        if args.comparison:
            records = list(iter_trajectories(env))
            density_title = (
                f"{agent_label} sampling density vs reward "
                f"({env.grid_size}x{env.grid_size}, {args.episodes} episodes)"
            )
            matches, plot_path, summary_path, density_path = generate_comparison(
                agent,
                env,
                records,
                num_episodes=args.episodes,
                plot_path=args.plot_path,
                summary_path=args.summary_path,
                density_plot_path=args.density_plot_path,
                density_title=density_title,
                greedy=args.greedy,
                show=args.show,
            )
            rewards = np.asarray([m.reward for m in matches], dtype=np.float64)
            print(f"Saved comparison plot: {plot_path}")
            print(f"Saved comparison summary: {summary_path}")
            if density_path is not None:
                print(f"Saved density plot: {density_path}")
        else:
            rewards = sample_episodes(agent, env, num_episodes=args.episodes, greedy=args.greedy)

        print(
            f"{args.episodes} {mode} episodes (max {env.max_episode_steps} steps)\n"
            f"  mean={rewards.mean():.6f}  min={rewards.min():.6f}  "
            f"max={rewards.max():.6f}  std={rewards.std():.6f}"
        )
        output_path = args.output
        if output_path is None and args.episodes >= 100:
            output_path = DEFAULT_SAMPLE_PATH
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_path, rewards)
            print(f"Saved rewards to {output_path}")
        return

    for episode_idx in range(1, args.episodes + 1):
        obs, _, state = env.reset()
        step = 0
        print(f"\n=== Episode {episode_idx} ===")
        done = False
        while not done:
            step += 1
            if args.greedy:
                move_action, color_action, _info = agent.act_greedy(obs)
            else:
                move_action, color_action, _info = agent.act(obs)
            move_label, color_label = agent.action_labels(move_action, color_action)
            obs, reward, done, state = env.step(move_action, color_action)
            print(f"  step {step}: move={move_label} color={color_label} -> {tuple(state)}")
        print(f"terminal reward: {reward:.3f}")
        print(format_grid(env._colors))


if __name__ == "__main__":
    main()
