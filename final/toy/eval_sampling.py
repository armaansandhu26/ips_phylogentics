"""Evaluate Hyper-Grid sampling against the target terminal distribution."""

from __future__ import annotations

from typing import Any

import numpy as np

from final.toy.hypergrid import count_modes
from final.toy.hypergrid_env import HyperGridDataset


def peak_mode_label_grid(rewards: np.ndarray, *, peak_reward: float | None = None) -> np.ndarray:
    """Label peak-reward connected components with mode ids 0..K-1; -1 elsewhere."""
    rewards = np.asarray(rewards)
    if peak_reward is None:
        peak_reward = float(rewards.max())
    mask = rewards >= peak_reward - 1e-8
    labels = np.full(rewards.shape, -1, dtype=np.int32)
    if rewards.ndim == 1:
        labels[mask] = np.arange(int(mask.sum()), dtype=np.int32)
        return labels

    visited = np.zeros_like(mask, dtype=bool)
    mode_id = 0
    height, width = mask.shape
    for i in range(height):
        for j in range(width):
            if not mask[i, j] or visited[i, j]:
                continue
            stack = [(i, j)]
            visited[i, j] = True
            while stack:
                x, y = stack.pop()
                labels[x, y] = mode_id
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if (
                        0 <= nx < height
                        and 0 <= ny < width
                        and mask[nx, ny]
                        and not visited[nx, ny]
                    ):
                        visited[nx, ny] = True
                        stack.append((nx, ny))
            mode_id += 1
    return labels


def modes_found_vs_samples(
    coords: np.ndarray,
    dataset: HyperGridDataset,
    *,
    sample_counts: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Cumulative peak modes discovered as more terminal samples are drawn."""
    spec = dataset.spec
    rewards = dataset.rewards
    peak_reward = float(rewards.max())
    expected_modes = count_modes(rewards, peak_reward=peak_reward)
    labels = peak_mode_label_grid(rewards, peak_reward=peak_reward)

    coords = np.asarray(coords, dtype=np.int64).reshape(-1, spec.D)
    n = len(coords)
    if n == 0:
        raise ValueError("coords must be non-empty")

    if sample_counts is None:
        sample_counts = np.unique(
            np.round(np.logspace(1, np.log10(n), num=40)).astype(np.int64)
        )
    sample_counts = np.asarray(sample_counts, dtype=np.int64)
    sample_counts = sample_counts[(sample_counts >= 1) & (sample_counts <= n)]

    modes_found: list[int] = []
    seen: set[int] = set()
    for k in sample_counts:
        for row in coords[: int(k)]:
            if spec.D == 1:
                mode = int(labels[int(row[0])])
            else:
                mode = int(labels[int(row[0]), int(row[1])])
            if mode >= 0:
                seen.add(mode)
        modes_found.append(len(seen))

    return {
        "sample_counts": sample_counts,
        "modes_found": np.asarray(modes_found, dtype=np.int64),
        "expected_modes": np.int64(expected_modes),
        "recovery_rate_pct": 100.0 * float(modes_found[-1]) / float(expected_modes),
    }


def empirical_terminal_grid(
    coords: np.ndarray,
    *,
    H: int,
    D: int = 2,
) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.int64)
    if D == 1:
        grid = np.zeros(H, dtype=np.float64)
        for x in coords.reshape(-1):
            grid[int(x)] += 1.0
        return grid / max(grid.sum(), 1.0)
    grid = np.zeros((H, H), dtype=np.float64)
    for row in coords:
        grid[int(row[0]), int(row[1])] += 1.0
    return grid / max(grid.sum(), 1.0)


def distribution_l1(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.abs(np.asarray(p) - np.asarray(q)).sum())


def distribution_tv(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * distribution_l1(p, q)


def evaluate_terminal_distribution(
    coords: np.ndarray,
    dataset: HyperGridDataset,
) -> dict[str, Any]:
    spec = dataset.spec
    target = dataset.load_target_probs()
    empirical = empirical_terminal_grid(coords, H=spec.H, D=spec.D)
    rewards = dataset.rewards
    sampled_rewards = dataset.reward_at(np.asarray(coords))

    peak_reward = float(rewards.max())
    peak_mask = rewards >= peak_reward - 1e-8
    peak_indices = np.argwhere(peak_mask)
    peak_mass = 0.0
    for idx in peak_indices:
        if spec.D == 1:
            peak_mass += empirical[int(idx[0])]
        else:
            peak_mass += empirical[int(idx[0]), int(idx[1])]

    labels = peak_mode_label_grid(rewards, peak_reward=peak_reward)
    modes_with_mass = 0
    seen_peak_modes: set[int] = set()
    for row in np.asarray(coords).reshape(-1, spec.D):
        if spec.D == 1:
            mode = int(labels[int(row[0])])
        else:
            mode = int(labels[int(row[0]), int(row[1])])
        if mode >= 0:
            seen_peak_modes.add(mode)
    modes_with_mass = len(seen_peak_modes)

    return {
        "l1_distance": distribution_l1(empirical, target),
        "total_variation": distribution_tv(empirical, target),
        "num_samples": int(len(coords)),
        "sampled_unique_terminals": int(len({tuple(row) for row in np.asarray(coords).reshape(-1, spec.D)})),
        "num_modes_with_mass": float(modes_with_mass),
        "expected_num_modes": float(count_modes(rewards, peak_reward=peak_reward)),
        "peak_mode_mass": float(peak_mass),
        "mean_reward": float(sampled_rewards.mean()),
        "sampled_coords": np.asarray(coords),
        "sampled_rewards": sampled_rewards,
    }
