"""Enumerate minimal-path trajectories and their terminal rewards."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from grid_environment_2 import GRID_SIZE, GREEN, GREEN_COLOR, RED, RED_COLOR, RIGHT, UNCOLORED, UP, GridEnv
from hierarchical import COLOR_NAMES, MOVE_NAMES

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_CATALOG_PATH = DATA_DIR / "trajectory_rewards.npz"
DEFAULT_PLOT_PATH = DATA_DIR / "trajectory_reward_distribution.png"


DEFAULT_ENUM_PATH = DATA_DIR / "all_trajectories.txt"


@dataclass(frozen=True)
class TrajectoryRecord:
    index: int
    path_index: int
    moves: tuple[int, ...]
    colors: tuple[int, ...]
    reward: float
    visited_cells: tuple[tuple[int, int], ...]
    painted_colors: tuple[int, ...]
    final_grid: tuple[tuple[int, ...], ...]

    @property
    def move_labels(self) -> str:
        return "".join(MOVE_NAMES[m][0].upper() for m in self.moves)

    @property
    def color_labels(self) -> str:
        return "".join(COLOR_NAMES[c][0].upper() for c in self.colors)

    def format_line(self) -> str:
        cells = ", ".join(
            f"({r},{c})={COLOR_NAMES[color][0].upper()}"
            for (r, c), color in zip(self.visited_cells, self.painted_colors)
        )
        return (
            f"#{self.index:04d}  path={self.path_index:02d}  "
            f"reward={self.reward:.6f}  moves={self.move_labels}  "
            f"colors={self.color_labels}  painted=[{cells}]"
        )


@dataclass(frozen=True)
class TrajectoryCatalog:
    """Rewards for every minimal-path trajectory and coloring."""

    rewards: np.ndarray
    max_normalized_reward: float
    grid_size: int = GRID_SIZE
    num_paths: int = 20
    steps_per_episode: int = 6

    @property
    def num_trajectories(self) -> int:
        return int(self.rewards.size)

    @property
    def min_reward(self) -> float:
        return float(self.rewards.min())

    @property
    def max_reward(self) -> float:
        return float(self.rewards.max())

    @property
    def mean_reward(self) -> float:
        return float(self.rewards.mean())

    def summary(self) -> dict[str, float | int]:
        return {
            "num_trajectories": self.num_trajectories,
            "min_reward": self.min_reward,
            "max_reward": self.max_reward,
            "mean_reward": self.mean_reward,
            "max_normalized_reward": self.max_normalized_reward,
        }


def _gen_minimal_move_sequences(
    grid_size: int = GRID_SIZE,
) -> Iterable[tuple[int, ...]]:
    target = grid_size - 1

    def walk(row: int, col: int, seq: list[int]) -> Iterable[tuple[int, ...]]:
        if row == target and col == target:
            yield tuple(seq)
            return
        if row < target:
            yield from walk(row + 1, col, seq + [UP])
        if col < target:
            yield from walk(row, col + 1, seq + [RIGHT])

    yield from walk(0, 0, [])


def _simulate_trajectory(
    env: GridEnv, moves: tuple[int, ...], colors: tuple[int, ...]
) -> tuple[float, tuple[tuple[int, int], ...], tuple[int, ...], tuple[tuple[int, ...], ...]]:
    env.reset()
    visited: list[tuple[int, int]] = []
    painted: list[int] = []
    reward = 0.0

    for move, color in zip(moves, colors):
        _, reward, _, state = env.step(move, color)
        visited.append((int(state[0]), int(state[1])))
        painted.append(color)

    final_grid = tuple(tuple(int(cell) for cell in row) for row in env._colors)
    return reward, tuple(visited), tuple(painted), final_grid


def iter_trajectories(env: GridEnv | None = None) -> Iterator[TrajectoryRecord]:
    """Yield every minimal-path trajectory with its terminal reward."""
    env = env or GridEnv()
    move_seqs = list(_gen_minimal_move_sequences(env.grid_size))
    steps = 2 * (env.grid_size - 1)
    index = 0

    for path_index, moves in enumerate(move_seqs):
        for colors in product([RED, GREEN], repeat=steps):
            reward, visited, painted, final_grid = _simulate_trajectory(env, moves, colors)
            yield TrajectoryRecord(
                index=index,
                path_index=path_index,
                moves=moves,
                colors=colors,
                reward=reward,
                visited_cells=visited,
                painted_colors=painted,
                final_grid=final_grid,
            )
            index += 1


def build_catalog(env: GridEnv | None = None) -> TrajectoryCatalog:
    """Simulate all minimal trajectories and collect terminal rewards."""
    env = env or GridEnv()
    move_seqs = list(_gen_minimal_move_sequences(env.grid_size))
    steps = 2 * (env.grid_size - 1)
    rewards = np.empty(len(move_seqs) * (2**steps), dtype=np.float64)

    for idx, record in enumerate(iter_trajectories(env)):
        rewards[idx] = record.reward

    return TrajectoryCatalog(
        rewards=rewards,
        max_normalized_reward=float(env._max_reward),
        grid_size=env.grid_size,
        num_paths=len(move_seqs),
        steps_per_episode=steps,
    )


def save_catalog(catalog: TrajectoryCatalog, path: Path | str = DEFAULT_CATALOG_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        rewards=catalog.rewards,
        max_normalized_reward=catalog.max_normalized_reward,
        grid_size=catalog.grid_size,
        num_paths=catalog.num_paths,
        steps_per_episode=catalog.steps_per_episode,
    )
    return path


def load_catalog(path: Path | str = DEFAULT_CATALOG_PATH) -> TrajectoryCatalog:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Trajectory catalog not found at {path}. "
            "Run `python plot_trajectory_rewards.py` to build it."
        )
    data = np.load(path)
    return TrajectoryCatalog(
        rewards=data["rewards"],
        max_normalized_reward=float(data["max_normalized_reward"]),
        grid_size=int(data["grid_size"]),
        num_paths=int(data["num_paths"]),
        steps_per_episode=int(data["steps_per_episode"]),
    )


def ensure_catalog(path: Path | str = DEFAULT_CATALOG_PATH) -> TrajectoryCatalog:
    path = Path(path)
    if path.exists():
        return load_catalog(path)
    catalog = build_catalog()
    save_catalog(catalog, path)
    return catalog


def format_grid(grid: tuple[tuple[int, ...], ...]) -> str:
    chars = {UNCOLORED: ".", RED_COLOR: "R", GREEN_COLOR: "G"}
    return "\n".join(" ".join(chars[cell] for cell in row) for row in grid)


def write_trajectory_list(
    path: Path | str = DEFAULT_ENUM_PATH,
    env: GridEnv | None = None,
) -> Path:
    """Write every minimal trajectory and reward to a text file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    env = env or GridEnv()
    catalog = build_catalog(env)

    with path.open("w", encoding="utf-8") as handle:
        handle.write(
            f"Color Trajectory — all minimal trajectories ({env.grid_size}x{env.grid_size})\n"
        )
        handle.write(
            f"paths={catalog.num_paths}, steps={catalog.steps_per_episode}, "
            f"total={catalog.num_trajectories}\n"
        )
        handle.write(
            f"reward range: [{catalog.min_reward:.6f}, {catalog.max_reward:.6f}], "
            f"mean={catalog.mean_reward:.6f}\n\n"
        )

        for record in iter_trajectories(env):
            handle.write(record.format_line() + "\n")
            handle.write(format_grid(record.final_grid) + "\n\n")

    return path
