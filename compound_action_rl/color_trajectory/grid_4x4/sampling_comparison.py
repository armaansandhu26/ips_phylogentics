"""Map GRPO samples onto enumerated trajectories and plot comparisons."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import matplotlib.pyplot as plt
import numpy as np

from grpo import GRPOAgent


@dataclass(frozen=True)
class SampleMatch:
    episode: int
    reward: float
    trajectory_index: int
    enumerated_reward: float


class TrajectoryRecord(Protocol):
    index: int
    reward: float
    final_grid: tuple[tuple[int, ...], ...]


def grid_tuple(colors: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(cell) for cell in row) for row in colors)


def sample_rewards(
    agent: GRPOAgent,
    env,
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


def sample_and_match(
    agent: GRPOAgent,
    env,
    *,
    num_episodes: int,
    grid_lookup: dict[tuple[tuple[int, ...], ...], int],
    reward_by_index: dict[int, float],
    greedy: bool = False,
) -> list[SampleMatch]:
    matches: list[SampleMatch] = []

    for episode_idx in range(num_episodes):
        obs, _, _ = env.reset()
        done = False
        while not done:
            if greedy:
                move_action, color_action, _info = agent.act_greedy(obs)
            else:
                move_action, color_action, _info = agent.act(obs)
            obs, reward, done, _ = env.step(move_action, color_action)

        grid_key = grid_tuple(env._colors)
        if grid_key not in grid_lookup:
            raise RuntimeError(
                f"Sampled episode {episode_idx} final grid not found in enumeration."
            )
        traj_idx = grid_lookup[grid_key]
        matches.append(
            SampleMatch(
                episode=episode_idx,
                reward=float(reward),
                trajectory_index=traj_idx,
                enumerated_reward=reward_by_index[traj_idx],
            )
        )

    return matches


def sample_rewards_and_indices(
    agent: GRPOAgent,
    env,
    *,
    num_episodes: int,
    grid_lookup: dict[tuple[tuple[int, ...], ...], int],
    greedy: bool = False,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Sample rollouts; return rewards, trajectory indices (-1 if unmatched), unmatched count."""
    rewards = np.empty(num_episodes, dtype=np.float64)
    indices = np.full(num_episodes, -1, dtype=np.int32)
    unmatched = 0

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
        grid_key = grid_tuple(env._colors)
        if grid_key in grid_lookup:
            indices[episode_idx] = grid_lookup[grid_key]
        else:
            unmatched += 1

    return rewards, indices, unmatched


def sample_and_match_tolerant(
    agent: GRPOAgent,
    env,
    *,
    num_episodes: int,
    grid_lookup: dict[tuple[tuple[int, ...], ...], int],
    reward_by_index: dict[int, float],
    greedy: bool = False,
) -> tuple[list[SampleMatch], int]:
    """Match rollouts to enumerated trajectories; skip unmatched final grids."""
    matches: list[SampleMatch] = []
    unmatched = 0

    for episode_idx in range(num_episodes):
        obs, _, _ = env.reset()
        done = False
        while not done:
            if greedy:
                move_action, color_action, _info = agent.act_greedy(obs)
            else:
                move_action, color_action, _info = agent.act(obs)
            obs, reward, done, _ = env.step(move_action, color_action)

        grid_key = grid_tuple(env._colors)
        if grid_key not in grid_lookup:
            unmatched += 1
            continue
        traj_idx = grid_lookup[grid_key]
        matches.append(
            SampleMatch(
                episode=episode_idx,
                reward=float(reward),
                trajectory_index=traj_idx,
                enumerated_reward=reward_by_index[traj_idx],
            )
        )

    return matches, unmatched


@dataclass(frozen=True)
class UniqueTrajectoryStats:
    label: str
    episodes_sampled: int
    enumerated_count: int
    matched_episodes: int
    unmatched_episodes: int
    unique_trajectories_hit: int
    hit_counts: Counter[int]


def unique_trajectory_stats(
    label: str,
    matches: list[SampleMatch],
    *,
    episodes_sampled: int,
    enumerated_count: int,
    unmatched_episodes: int,
) -> UniqueTrajectoryStats:
    hit_counts = Counter(m.trajectory_index for m in matches)
    return UniqueTrajectoryStats(
        label=label,
        episodes_sampled=episodes_sampled,
        enumerated_count=enumerated_count,
        matched_episodes=len(matches),
        unmatched_episodes=unmatched_episodes,
        unique_trajectories_hit=len(hit_counts),
        hit_counts=hit_counts,
    )


def write_unique_trajectory_summary(
    path: Path,
    stats_list: Sequence[UniqueTrajectoryStats],
    *,
    header: str = "Unique enumerated trajectories captured",
) -> None:
    lines = [header, ""]
    for stats in stats_list:
        lines.extend(
            [
                stats.label,
                f"  episodes_sampled={stats.episodes_sampled}",
                f"  enumerated_trajectories={stats.enumerated_count}",
                f"  matched_episodes={stats.matched_episodes}",
                f"  unmatched_episodes={stats.unmatched_episodes}",
                f"  unique_enumerated_trajectories_hit={stats.unique_trajectories_hit}",
                "",
                "  Top sampled trajectory indices (index -> count):",
            ]
        )
        for index, count in stats.hit_counts.most_common(20):
            lines.append(f"    #{index:04d}  x{count}")
        lines.append("")

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def format_unique_trajectory_annotation(stats_list: Sequence[UniqueTrajectoryStats]) -> str:
    parts = []
    for stats in stats_list:
        parts.append(
            f"{stats.label}: {stats.unique_trajectories_hit}/{stats.enumerated_count} unique "
            f"({stats.unmatched_episodes} unmatched)"
        )
    return "\n".join(parts)


def unique_trajectory_density_points(
    stats: UniqueTrajectoryStats,
    reward_by_index: dict[int, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return rewards, sampling densities, and trajectory indices for hit trajectories."""
    indices = np.fromiter(stats.hit_counts.keys(), dtype=np.int32)
    counts = np.fromiter(stats.hit_counts.values(), dtype=np.float64)
    rewards = np.asarray([reward_by_index[int(idx)] for idx in indices], dtype=np.float64)
    densities = counts / float(stats.episodes_sampled)
    return rewards, densities, indices


def _jitter_rewards(
    rewards: np.ndarray,
    rng: np.random.Generator,
    *,
    span_fraction: float = 0.012,
) -> np.ndarray:
    if rewards.size == 0:
        return rewards
    span = max(float(rewards.max() - rewards.min()), 1e-6)
    jitter = rng.uniform(-1.0, 1.0, size=rewards.size)
    return rewards + jitter * span * span_fraction


def plot_unique_trajectory_scatter(
    stats_list: Sequence[UniqueTrajectoryStats],
    reward_by_index: dict[int, float],
    *,
    title: str,
    save_path: Path | str,
    colors: Sequence[str] | None = None,
    jitter: bool = True,
    seed: int = 0,
    show: bool = False,
) -> Path:
    """Scatter plot of sampling density vs reward for each uniquely hit trajectory."""
    if not stats_list:
        raise ValueError("stats_list must contain at least one UniqueTrajectoryStats entry.")

    default_colors = ("#0984e3", "#e17055", "#00b894", "#6c5ce7")
    palette = list(colors) if colors is not None else list(default_colors)
    rng = np.random.default_rng(seed)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for idx, stats in enumerate(stats_list):
        rewards, densities, _indices = unique_trajectory_density_points(stats, reward_by_index)
        if jitter:
            rewards = _jitter_rewards(rewards, rng)
        color = palette[idx % len(palette)]
        ax.scatter(
            rewards,
            densities,
            s=14,
            alpha=0.55,
            color=color,
            edgecolors="none",
            label=f"{stats.label} ({stats.unique_trajectories_hit} unique)",
        )

    ax.set_xlabel("Reward")
    ax.set_ylabel("Sampling density")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    if show:
        plt.show()
    else:
        plt.close(fig)

    return save_path


@dataclass(frozen=True)
class LinearFitStats:
    slope: float
    intercept: float
    r2: float
    rmse: float


def fit_linear(x: np.ndarray, y: np.ndarray) -> LinearFitStats:
    """Ordinary least-squares linear fit y = slope * x + intercept."""
    if x.size == 0:
        raise ValueError("Cannot fit linear model with zero points.")
    if x.size == 1:
        slope = 0.0
        intercept = float(y[0])
        return LinearFitStats(slope=slope, intercept=intercept, r2=1.0, rmse=0.0)

    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    residuals = y - predicted
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    rmse = float(np.sqrt(np.mean(residuals**2)))
    return LinearFitStats(
        slope=float(slope),
        intercept=float(intercept),
        r2=r2,
        rmse=rmse,
    )


def plot_unique_trajectory_scatter_with_fit(
    stats_list: Sequence[UniqueTrajectoryStats],
    reward_by_index: dict[int, float],
    *,
    title: str,
    save_path: Path | str,
    colors: Sequence[str] | None = None,
    jitter: bool = True,
    seed: int = 0,
    show: bool = False,
    fit_indices: Sequence[int] | None = None,
) -> tuple[Path, list[tuple[str, LinearFitStats]]]:
    """Scatter of density vs reward with optional linear fits; returns path and fit metrics."""
    if not stats_list:
        raise ValueError("stats_list must contain at least one UniqueTrajectoryStats entry.")

    default_colors = ("#0984e3", "#e17055", "#00b894", "#6c5ce7")
    palette = list(colors) if colors is not None else list(default_colors)
    rng = np.random.default_rng(seed)
    fit_set = set(range(len(stats_list))) if fit_indices is None else set(fit_indices)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fit_results: list[tuple[str, LinearFitStats]] = []

    for idx, stats in enumerate(stats_list):
        rewards, densities, _indices = unique_trajectory_density_points(stats, reward_by_index)
        color = palette[idx % len(palette)]
        plot_rewards = _jitter_rewards(rewards, rng) if jitter else rewards
        ax.scatter(
            plot_rewards,
            densities,
            s=14,
            alpha=0.45,
            color=color,
            edgecolors="none",
            zorder=2,
        )

        if idx not in fit_set:
            ax.plot(
                [],
                [],
                color=color,
                linewidth=0,
                label=f"{stats.label} ({stats.unique_trajectories_hit} unique)",
            )
            continue

        fit = fit_linear(rewards, densities)
        fit_results.append((stats.label, fit))

        x_line = np.linspace(float(rewards.min()), float(rewards.max()), 100)
        y_line = fit.slope * x_line + fit.intercept
        ax.plot(
            x_line,
            y_line,
            color=color,
            linewidth=2.0,
            label=(
                f"{stats.label}: OLS R²={fit.r2:.3f}, RMSE={fit.rmse:.4f}, "
                f"slope={fit.slope:.4f}"
            ),
            zorder=3,
        )

    ax.set_xlabel("Reward")
    ax.set_ylabel("Sampling density")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    if show:
        plt.show()
    else:
        plt.close(fig)

    return save_path, fit_results


def write_linear_fit_summary(
    path: Path,
    fit_results: Sequence[tuple[str, LinearFitStats]],
    *,
    header: str = "Linear fit: sampling density vs reward",
) -> None:
    lines = [header, ""]
    for label, fit in fit_results:
        lines.extend(
            [
                label,
                f"  slope={fit.slope:.6f}",
                f"  intercept={fit.intercept:.6f}",
                f"  R2={fit.r2:.6f}",
                f"  RMSE={fit.rmse:.6f}",
                "",
            ]
        )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_summary(
    path: Path,
    *,
    num_episodes: int,
    enumerated_count: int,
    matches: list[SampleMatch],
    hit_counts: Counter[int],
) -> None:
    rewards = np.asarray([m.reward for m in matches], dtype=np.float64)
    lines = [
        "GRPO sampling vs enumerated trajectories",
        f"episodes_sampled={num_episodes}",
        f"enumerated_trajectories={enumerated_count}",
        f"unique_enumerated_trajectories_hit={len(hit_counts)}",
        f"sample_mean={rewards.mean():.6f}",
        f"sample_min={rewards.min():.6f}",
        f"sample_max={rewards.max():.6f}",
        "",
        "Top sampled trajectory indices (index -> count):",
    ]
    for index, count in hit_counts.most_common(20):
        lines.append(f"  #{index:04d}  x{count}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _jitter_sample_coords(
    matches: list[SampleMatch],
    *,
    spread: float = 0.35,
) -> tuple[np.ndarray, np.ndarray]:
    """Spread duplicate trajectory hits horizontally so overlaps stay visible."""
    by_index: dict[int, list[int]] = {}
    for i, match in enumerate(matches):
        by_index.setdefault(match.trajectory_index, []).append(i)

    xs = np.empty(len(matches), dtype=np.float64)
    ys = np.empty(len(matches), dtype=np.float64)
    for index, order in by_index.items():
        count = len(order)
        if count == 1:
            offsets = [0.0]
        else:
            offsets = np.linspace(-spread, spread, count)
        for offset, match_idx in zip(offsets, order):
            xs[match_idx] = index + offset
            ys[match_idx] = matches[match_idx].reward
    return xs, ys


def plot_comparison(
    enumerated_indices: Sequence[int],
    enumerated_rewards: Sequence[float],
    matches: list[SampleMatch],
    *,
    title: str,
    save_path: Path | str,
    show: bool = False,
) -> Path:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    sample_x, sample_y = _jitter_sample_coords(matches)
    hit_counts = Counter(m.trajectory_index for m in matches)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]})

    ax = axes[0]
    ax.plot(
        enumerated_indices,
        enumerated_rewards,
        marker=".",
        linestyle="none",
        markersize=3,
        color="#b2bec3",
        alpha=0.85,
        label=f"All enumerated (n={len(enumerated_indices)})",
        zorder=1,
    )
    ax.scatter(
        sample_x,
        sample_y,
        s=16,
        color="#e17055",
        alpha=0.18,
        edgecolors="#c0392b",
        linewidths=0.25,
        label=f"GRPO samples (n={len(matches)}, jittered for overlap)",
        zorder=2,
    )

    ax.set_xlabel("Trajectory number")
    ax.set_ylabel("Reward")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)

    ax2 = axes[1]
    counts = np.zeros(len(enumerated_indices), dtype=np.int32)
    for index, count in hit_counts.items():
        counts[index] = count
    ax2.bar(enumerated_indices, counts, width=1.0, color="#e17055", alpha=0.75)
    ax2.set_xlabel("Trajectory number")
    ax2.set_ylabel("Sample count")
    ax2.set_title("How often each enumerated trajectory was sampled")
    ax2.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    if show:
        plt.show()
    else:
        plt.close(fig)

    return save_path


def _plot_enumerated_reward_bars(
    ax,
    enumerated_rewards: Sequence[float],
    bins: np.ndarray,
    *,
    label: str,
    color: str = "#636e72",
    zorder: int = 1,
) -> None:
    """Plot enumerated rewards with bar heights normalized so the tallest bar is 1."""
    enumerated = np.asarray(enumerated_rewards, dtype=np.float64)
    counts, _ = np.histogram(enumerated, bins=bins)
    max_count = float(counts.max())
    heights = counts / max_count if max_count > 0 else counts.astype(np.float64)
    bin_width = bins[1] - bins[0]
    ax.bar(
        bins[:-1] + bin_width / 2,
        heights,
        width=bin_width * 0.95,
        align="center",
        color=color,
        alpha=0.75,
        edgecolor="white",
        linewidth=0.4,
        label=label,
        zorder=zorder,
    )
    ax.set_ylim(0, 1.05)


def _reward_bins(
    *reward_groups: Sequence[float],
    num_bins: int = 40,
) -> np.ndarray:
    combined = np.concatenate([np.asarray(group, dtype=np.float64) for group in reward_groups])
    reward_min = float(combined.min())
    reward_max = float(combined.max())
    pad = max(0.01, 0.02 * (reward_max - reward_min)) if reward_max > reward_min else 0.01
    return np.linspace(reward_min - pad, reward_max + pad, num_bins + 1)


def _make_density_figure(
    enumerated_rewards: Sequence[float] | None,
    *,
    figsize: tuple[float, float] = (9, 5.5),
) -> tuple[plt.Figure, plt.Axes, plt.Axes | None]:
    if enumerated_rewards is not None:
        fig, (ax_enum, ax) = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=figsize,
            gridspec_kw={"height_ratios": [1, 3], "hspace": 0.08},
            layout="constrained",
        )
        ax_enum.set_ylabel("Count\n(norm.)")
        ax_enum.grid(axis="y", alpha=0.25)
        return fig, ax, ax_enum

    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax, None


def plot_reward_density(
    sample_rewards: Sequence[float],
    *,
    title: str,
    save_path: Path | str,
    enumerated_rewards: Sequence[float] | None = None,
    num_bins: int = 40,
    show: bool = False,
) -> Path:
    """Plot policy sampling density over reward."""
    samples = np.asarray(sample_rewards, dtype=np.float64)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    reward_groups: list[Sequence[float]] = [samples]
    if enumerated_rewards is not None:
        reward_groups.append(enumerated_rewards)
    bins = _reward_bins(*reward_groups, num_bins=num_bins)

    fig, ax, ax_enum = _make_density_figure(enumerated_rewards)
    if ax_enum is not None and enumerated_rewards is not None:
        _plot_enumerated_reward_bars(
            ax_enum,
            enumerated_rewards,
            bins,
            label=f"Enumerated trajectories (n={len(enumerated_rewards)})",
        )
        ax_enum.legend(loc="upper right", fontsize=9)
    ax.hist(
        samples,
        bins=bins,
        density=True,
        alpha=0.75,
        color="#e17055",
        edgecolor="white",
        linewidth=0.4,
        label=f"Policy samples (n={len(samples)})",
        zorder=2,
    )
    ax.set_xlabel("Reward")
    ax.set_ylabel("Density")
    fig.suptitle(title, y=1.02)
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    if ax_enum is None:
        fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    if show:
        plt.show()
    else:
        plt.close(fig)

    return save_path


def plot_reward_density_overlay(
    reward_sets: Sequence[tuple[str, Sequence[float]]],
    *,
    title: str,
    save_path: Path | str,
    enumerated_rewards: Sequence[float] | None = None,
    stats_annotation: str | None = None,
    num_bins: int = 40,
    colors: Sequence[str] | None = None,
    show: bool = False,
) -> Path:
    """Plot sampling density over reward for multiple policies on the same axes."""
    if not reward_sets:
        raise ValueError("reward_sets must contain at least one labeled sample set.")

    default_colors = ("#0984e3", "#e17055", "#00b894", "#6c5ce7", "#fdcb6e")
    palette = list(colors) if colors is not None else list(default_colors)

    reward_groups: list[Sequence[float]] = [rewards for _label, rewards in reward_sets]
    if enumerated_rewards is not None:
        reward_groups.append(enumerated_rewards)
    bins = _reward_bins(*reward_groups, num_bins=num_bins)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax, ax_enum = _make_density_figure(enumerated_rewards)
    if ax_enum is not None and enumerated_rewards is not None:
        _plot_enumerated_reward_bars(
            ax_enum,
            enumerated_rewards,
            bins,
            label=f"Enumerated trajectories (n={len(enumerated_rewards)})",
        )
        ax_enum.legend(loc="upper right", fontsize=9)
    for idx, (label, rewards) in enumerate(reward_sets):
        samples = np.asarray(rewards, dtype=np.float64)
        color = palette[idx % len(palette)]
        ax.hist(
            samples,
            bins=bins,
            density=True,
            alpha=0.55,
            color=color,
            edgecolor="white",
            linewidth=0.4,
            label=f"{label} (n={len(samples)})",
            zorder=2,
        )

    ax.set_xlabel("Reward")
    ax.set_ylabel("Density")
    fig.suptitle(title, y=1.02)
    if stats_annotation:
        fig.text(
            0.5,
            0.01,
            stats_annotation,
            ha="center",
            va="bottom",
            fontsize=9,
            transform=fig.transFigure,
        )
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    if ax_enum is None:
        fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    if show:
        plt.show()
    else:
        plt.close(fig)

    return save_path


def generate_comparison(
    agent: GRPOAgent,
    env,
    records: Sequence[TrajectoryRecord],
    *,
    num_episodes: int,
    plot_path: Path | str,
    summary_path: Path | str,
    density_plot_path: Path | str | None = None,
    density_title: str | None = None,
    greedy: bool = False,
    show: bool = False,
) -> tuple[list[SampleMatch], Path, Path, Path | None]:
    enumerated_indices = [r.index for r in records]
    enumerated_rewards = [r.reward for r in records]
    reward_by_index = {r.index: r.reward for r in records}
    grid_lookup = {r.final_grid: r.index for r in records}

    matches = sample_and_match(
        agent,
        env,
        num_episodes=num_episodes,
        grid_lookup=grid_lookup,
        reward_by_index=reward_by_index,
        greedy=greedy,
    )

    for match in matches:
        expected = reward_by_index[match.trajectory_index]
        if abs(match.reward - expected) > 1e-9:
            raise RuntimeError(
                f"Reward mismatch for trajectory #{match.trajectory_index}: "
                f"sample={match.reward}, enumerated={expected}"
            )

    hit_counts = Counter(m.trajectory_index for m in matches)
    title = f"Enumerated trajectories vs GRPO samples ({env.grid_size}x{env.grid_size})"
    saved_plot = plot_comparison(
        enumerated_indices,
        enumerated_rewards,
        matches,
        title=title,
        save_path=plot_path,
        show=show,
    )
    saved_summary = Path(summary_path)
    write_summary(
        saved_summary,
        num_episodes=num_episodes,
        enumerated_count=len(records),
        matches=matches,
        hit_counts=hit_counts,
    )

    saved_density: Path | None = None
    if density_plot_path is not None:
        sample_rewards_arr = sample_rewards(
            agent,
            env,
            num_episodes=num_episodes,
            greedy=greedy,
        )
        title = density_title or f"Sampling density vs reward ({env.grid_size}x{env.grid_size})"
        saved_density = plot_reward_density(
            sample_rewards_arr,
            title=title,
            save_path=density_plot_path,
            enumerated_rewards=enumerated_rewards,
            show=show,
        )

    return matches, saved_plot, saved_summary, saved_density
