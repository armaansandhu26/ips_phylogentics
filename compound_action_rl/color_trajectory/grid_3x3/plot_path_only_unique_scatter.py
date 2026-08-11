"""Scatter plot: unique path sampling density vs reward (path model only, no color model).

On a 3x3 grid there are 6 minimal move sequences. With the color model added,
each path expands to 16 full trajectories (96 total).
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
GRID_4X4 = ROOT / "grid_4x4"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(GRID_4X4))
sys.path.insert(0, str(HERE))

from catalog import PathRecord, build_path_catalog, moves_to_path_index  # noqa: E402
from grpo import GRPOAgent, agent_label_from_checkpoint, load_agent_from_checkpoint  # noqa: E402
from grid_environment import GridEnv  # noqa: E402
from hierarchical import RandomModel2Policy  # noqa: E402
from sampling_comparison import (  # noqa: E402
    LinearFitStats,
    UniqueTrajectoryStats,
    _jitter_rewards,
    fit_linear,
    write_linear_fit_summary,
    write_unique_trajectory_summary,
)

DATA_DIR = HERE / "data"
CHECKPOINT_DIR = HERE / "checkpoints"
DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "ips_grpo.pt"
DEFAULT_PLOT_PATH = DATA_DIR / "path_only_unique_scatter.png"
DEFAULT_SUMMARY_PATH = DATA_DIR / "path_only_unique_scatter.txt"
DEFAULT_FIT_SUMMARY_PATH = DATA_DIR / "path_only_unique_scatter_linear_fit.txt"
DEFAULT_INDICES_PATH = DATA_DIR / "path_only_sample_1000_path_indices.npy"


@dataclass(frozen=True)
class PathSamplingStats:
    label: str
    episodes_sampled: int
    enumerated_paths: int
    matched_episodes: int
    unmatched_episodes: int
    unique_paths_hit: int
    hit_counts: Counter[int]


def path_reward_by_index(paths: tuple[PathRecord, ...], *, use: str) -> dict[int, float]:
    if use == "max":
        return {path.path_index: path.max_reward for path in paths}
    if use == "mean":
        return {path.path_index: path.mean_reward for path in paths}
    raise ValueError(f"Unknown reward mode: {use}")


def sample_path_indices(
    agent: GRPOAgent,
    env: GridEnv,
    *,
    num_episodes: int,
    moves_lookup: dict[tuple[int, ...], int],
    greedy: bool,
    random_color: bool,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Roll out episodes; return rewards, path indices (-1 if unmatched), unmatched count."""
    random_color_policy = RandomModel2Policy()
    rewards = np.empty(num_episodes, dtype=np.float64)
    indices = np.full(num_episodes, -1, dtype=np.int32)
    unmatched = 0

    for episode_idx in range(num_episodes):
        obs, _, _ = env.reset()
        done = False
        moves: list[int] = []
        while not done:
            if greedy:
                move_action, _log_p1, move_rep = agent.move_net._select_from_obs(obs, greedy=True)
            else:
                move_action, _log_p1, move_rep = agent.move_net.sample_with_rep(obs, agent.rng)
            if random_color:
                color_action, _log_p2 = random_color_policy.sample(obs, move_action, agent.rng)
            else:
                color_action, _log_p2 = agent.color_net.select_action(
                    move_rep, move_action, greedy=greedy
                )
            obs, reward, done, _ = env.step(move_action, color_action)
            moves.append(move_action)
        rewards[episode_idx] = reward
        move_key = tuple(moves)
        if move_key in moves_lookup:
            indices[episode_idx] = moves_lookup[move_key]
        else:
            unmatched += 1

    return rewards, indices, unmatched


def match_unique_paths(
    *,
    label: str,
    checkpoint: Path | None,
    env: GridEnv,
    moves_lookup: dict[tuple[int, ...], int],
    reward_by_index: dict[int, float],
    episodes: int,
    greedy: bool,
    random_color: bool,
    index_path: Path,
) -> PathSamplingStats:
    if index_path.exists():
        indices = np.load(index_path)
        hit_counts: Counter[int] = Counter()
        for idx in indices:
            if idx >= 0:
                hit_counts[int(idx)] += 1
        matched = sum(hit_counts.values())
        return PathSamplingStats(
            label=label,
            episodes_sampled=len(indices),
            enumerated_paths=len(reward_by_index),
            matched_episodes=matched,
            unmatched_episodes=len(indices) - matched,
            unique_paths_hit=len(hit_counts),
            hit_counts=hit_counts,
        )

    if checkpoint is None:
        raise ValueError("checkpoint is required when cached path indices are missing.")

    agent = load_agent_from_checkpoint(checkpoint)
    _rewards, indices, unmatched = sample_path_indices(
        agent,
        env,
        num_episodes=episodes,
        moves_lookup=moves_lookup,
        greedy=greedy,
        random_color=random_color,
    )
    index_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(index_path, indices)
    hit_counts = Counter(int(idx) for idx in indices if idx >= 0)
    return PathSamplingStats(
        label=label,
        episodes_sampled=episodes,
        enumerated_paths=len(reward_by_index),
        matched_episodes=int(np.count_nonzero(indices >= 0)),
        unmatched_episodes=unmatched,
        unique_paths_hit=len(hit_counts),
        hit_counts=hit_counts,
    )


def uniform_path_stats(
    label: str,
    paths: tuple[PathRecord, ...],
    *,
    reward_by_index: dict[int, float],
) -> PathSamplingStats:
    """Baseline: uniform sampling over enumerated minimal paths."""
    hit_counts = Counter({path.path_index: 1 for path in paths})
    return PathSamplingStats(
        label=label,
        episodes_sampled=len(paths),
        enumerated_paths=len(reward_by_index),
        matched_episodes=len(paths),
        unmatched_episodes=0,
        unique_paths_hit=len(paths),
        hit_counts=hit_counts,
    )


def path_density_points(
    stats: PathSamplingStats,
    reward_by_index: dict[int, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.fromiter(stats.hit_counts.keys(), dtype=np.int32)
    counts = np.fromiter(stats.hit_counts.values(), dtype=np.float64)
    rewards = np.asarray([reward_by_index[int(idx)] for idx in indices], dtype=np.float64)
    densities = counts / float(stats.episodes_sampled)
    return rewards, densities, indices


def path_stats_to_unique_stats(stats: PathSamplingStats) -> UniqueTrajectoryStats:
    return UniqueTrajectoryStats(
        label=stats.label,
        episodes_sampled=stats.episodes_sampled,
        enumerated_count=stats.enumerated_paths,
        matched_episodes=stats.matched_episodes,
        unmatched_episodes=stats.unmatched_episodes,
        unique_trajectories_hit=stats.unique_paths_hit,
        hit_counts=stats.hit_counts,
    )


def plot_path_unique_scatter(
    stats_list: list[PathSamplingStats],
    reward_by_index: dict[int, float],
    *,
    title: str,
    save_path: Path,
    colors: tuple[str, ...] = ("#636e72", "#0984e3", "#e17055"),
    jitter: bool = True,
    seed: int = 0,
    show: bool = False,
    fit_index: int | None = None,
) -> tuple[Path, list[tuple[str, LinearFitStats]]]:
    rng = np.random.default_rng(seed)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fit_results: list[tuple[str, LinearFitStats]] = []

    for idx, stats in enumerate(stats_list):
        rewards, densities, _indices = path_density_points(stats, reward_by_index)
        if jitter:
            rewards = _jitter_rewards(rewards, rng)
        color = colors[idx % len(colors)]
        ax.scatter(
            rewards,
            densities,
            s=80,
            alpha=0.75,
            color=color,
            edgecolors="white",
            linewidths=0.6,
            label=f"{stats.label} ({stats.unique_paths_hit}/{stats.enumerated_paths} paths)",
        )
        if fit_index is not None and idx == fit_index and rewards.size >= 2:
            fit = fit_linear(rewards, densities)
            fit_results.append((stats.label, fit))
            x_line = np.linspace(float(rewards.min()), float(rewards.max()), 100)
            y_line = fit.slope * x_line + fit.intercept
            ax.plot(
                x_line,
                y_line,
                color="#c0392b",
                linewidth=2.0,
                label=f"OLS R²={fit.r2:.3f}, slope={fit.slope:.4f}",
            )

    ax.set_xlabel("Reward (max achievable on path)")
    ax.set_ylabel("Sampling density")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return save_path, fit_results


def write_path_summary(path: Path, stats_list: list[PathSamplingStats], *, header: str) -> None:
    write_unique_trajectory_summary(
        path,
        [path_stats_to_unique_stats(stats) for stats in stats_list],
        header=header,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot unique minimal-path density vs reward (path model only)."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Move-policy checkpoint. Omit with --uniform-only to plot enumeration baseline only.",
    )
    parser.add_argument("--indices-path", type=Path, default=DEFAULT_INDICES_PATH)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument(
        "--random-color",
        action="store_true",
        default=True,
        help="Sample colors uniformly (ignore color model). Default: on.",
    )
    parser.add_argument(
        "--use-trained-color",
        action="store_true",
        help="Use the checkpoint color model instead of uniform random colors.",
    )
    parser.add_argument(
        "--reward-axis",
        choices=("max", "mean"),
        default="max",
        help="Reward on x-axis for each enumerated path.",
    )
    parser.add_argument(
        "--uniform-only",
        action="store_true",
        help="Plot only the uniform baseline over all 6 paths (no checkpoint sampling).",
    )
    parser.add_argument("--plot-path", type=Path, default=DEFAULT_PLOT_PATH)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--fit-summary-path", type=Path, default=DEFAULT_FIT_SUMMARY_PATH)
    parser.add_argument("--no-jitter", action="store_true")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    random_color = not args.use_trained_color
    env = GridEnv()
    path_catalog = build_path_catalog(env)
    paths = path_catalog.paths
    moves_lookup = moves_to_path_index(env)
    reward_by_index = path_reward_by_index(paths, use=args.reward_axis)

    print(
        f"{env.grid_size}x{env.grid_size}: {path_catalog.num_paths} minimal paths "
        f"x 2^{path_catalog.steps_per_episode} colorings "
        f"= {path_catalog.num_full_trajectories} full trajectories"
    )
    print("\nEnumerated minimal paths (path model only):")
    for path in paths:
        print(f"  {path.format_line()}")

    stats_list: list[PathSamplingStats] = []
    stats_list.append(
        uniform_path_stats(
            "Uniform over 6 paths",
            paths,
            reward_by_index=reward_by_index,
        )
    )

    fit_index: int | None = None
    if not args.uniform_only:
        if not args.checkpoint.exists():
            print(
                f"\nCheckpoint not found at {args.checkpoint}; plotting uniform baseline only."
            )
        else:
            label = agent_label_from_checkpoint(args.checkpoint)
            if random_color:
                label += " (move net + random color)"
            else:
                label += " (full hierarchical)"
            stats = match_unique_paths(
                label=label,
                checkpoint=args.checkpoint,
                env=env,
                moves_lookup=moves_lookup,
                reward_by_index=reward_by_index,
                episodes=args.episodes,
                greedy=args.greedy,
                random_color=random_color,
                index_path=args.indices_path,
            )
            stats_list.append(stats)
            fit_index = 1
            print(
                f"\n{label}: {stats.unique_paths_hit}/{stats.enumerated_paths} unique paths "
                f"({stats.unmatched_episodes} unmatched of {stats.episodes_sampled})"
            )

    title = (
        f"Path-only unique sampling density vs reward "
        f"({env.grid_size}x{env.grid_size}, {path_catalog.num_paths} paths)"
    )
    plot_path, fit_results = plot_path_unique_scatter(
        stats_list,
        reward_by_index,
        title=title,
        save_path=args.plot_path,
        jitter=not args.no_jitter,
        show=args.show,
        fit_index=fit_index,
    )
    write_path_summary(
        args.summary_path,
        stats_list,
        header="Unique minimal paths captured (path model only)",
    )
    if fit_results:
        write_linear_fit_summary(args.fit_summary_path, fit_results)

    print(f"\nSaved plot: {plot_path}")
    print(f"Saved summary: {args.summary_path}")
    if fit_results:
        print(f"Saved OLS fit summary: {args.fit_summary_path}")
        for fit_label, fit in fit_results:
            print(f"{fit_label}: OLS R²={fit.r2:.4f}  slope={fit.slope:.6f}")


if __name__ == "__main__":
    main()
