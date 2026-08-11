"""Enumerate minimal-path trajectories and their terminal rewards (3x3)."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable, Iterator

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hierarchical import COLOR_NAMES, MOVE_NAMES

import numpy as np

from grid_environment import GRID_SIZE, GREEN, GREEN_COLOR, RED, RED_COLOR, RIGHT, UNCOLORED, UP, GridEnv

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_CATALOG_PATH = DATA_DIR / "trajectory_rewards.npz"
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
            f"#{self.index:04d}  path={self.index:02d}  "
            f"reward={self.reward:.6f}  moves={self.move_labels}  "
            f"colors={self.color_labels}  painted=[{cells}]"
        )


@dataclass(frozen=True)
class TrajectoryCatalog:
    rewards: np.ndarray
    max_normalized_reward: float
    grid_size: int = GRID_SIZE
    num_paths: int = 6
    steps_per_episode: int = 4

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


@dataclass(frozen=True)
class PathRecord:
    """One minimal move sequence (path-model outcome), ignoring color choices."""

    path_index: int
    moves: tuple[int, ...]
    min_reward: float
    max_reward: float
    mean_reward: float
    num_colorings: int

    @property
    def move_labels(self) -> str:
        return "".join(MOVE_NAMES[m][0].upper() for m in self.moves)

    def format_line(self) -> str:
        return (
            f"path={self.path_index:02d}  moves={self.move_labels}  "
            f"colorings={self.num_colorings}  "
            f"reward=[{self.min_reward:.6f}, {self.max_reward:.6f}]  "
            f"mean={self.mean_reward:.6f}"
        )


@dataclass(frozen=True)
class PathCatalog:
    paths: tuple[PathRecord, ...]
    grid_size: int = GRID_SIZE
    steps_per_episode: int = 4
    colorings_per_path: int = 16

    @property
    def num_paths(self) -> int:
        return len(self.paths)

    @property
    def num_full_trajectories(self) -> int:
        return self.num_paths * self.colorings_per_path


def _gen_minimal_move_sequences(grid_size: int = GRID_SIZE) -> Iterable[tuple[int, ...]]:
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


def _enumerate_raw_trajectories(env: GridEnv) -> list[TrajectoryRecord]:
    move_seqs = list(_gen_minimal_move_sequences(env.grid_size))
    steps = 2 * (env.grid_size - 1)
    records: list[TrajectoryRecord] = []

    for path_index, moves in enumerate(move_seqs):
        for colors in product([RED, GREEN], repeat=steps):
            reward, visited, painted, final_grid = _simulate_trajectory(env, moves, colors)
            records.append(
                TrajectoryRecord(
                    index=0,
                    path_index=path_index,
                    moves=moves,
                    colors=colors,
                    reward=reward,
                    visited_cells=visited,
                    painted_colors=painted,
                    final_grid=final_grid,
                )
            )

    return records


def build_path_catalog(env: GridEnv | None = None) -> PathCatalog:
    """Aggregate full trajectories by move sequence (path model only)."""
    env = env or GridEnv()
    move_seqs = list(_gen_minimal_move_sequences(env.grid_size))
    steps = 2 * (env.grid_size - 1)
    rewards_by_moves: dict[tuple[int, ...], list[float]] = {
        moves: [] for moves in move_seqs
    }

    for record in _enumerate_raw_trajectories(env):
        rewards_by_moves[record.moves].append(record.reward)

    paths: list[PathRecord] = []
    for path_index, moves in enumerate(move_seqs):
        rewards = rewards_by_moves[moves]
        paths.append(
            PathRecord(
                path_index=path_index,
                moves=moves,
                min_reward=min(rewards),
                max_reward=max(rewards),
                mean_reward=float(sum(rewards) / len(rewards)),
                num_colorings=len(rewards),
            )
        )

    return PathCatalog(
        paths=tuple(paths),
        grid_size=env.grid_size,
        steps_per_episode=steps,
        colorings_per_path=2**steps,
    )


def iter_paths(env: GridEnv | None = None) -> Iterator[PathRecord]:
    yield from build_path_catalog(env).paths


def moves_to_path_index(env: GridEnv | None = None) -> dict[tuple[int, ...], int]:
    env = env or GridEnv()
    return {
        moves: path_index
        for path_index, moves in enumerate(_gen_minimal_move_sequences(env.grid_size))
    }


def iter_trajectories(env: GridEnv | None = None) -> Iterator[TrajectoryRecord]:
    """Yield trajectories with index 0 = min reward, index n-1 = max reward."""
    env = env or GridEnv()
    records = _enumerate_raw_trajectories(env)
    records.sort(key=lambda record: (record.reward, record.path_index, record.colors))

    for index, record in enumerate(records):
        yield TrajectoryRecord(
            index=index,
            path_index=record.path_index,
            moves=record.moves,
            colors=record.colors,
            reward=record.reward,
            visited_cells=record.visited_cells,
            painted_colors=record.painted_colors,
            final_grid=record.final_grid,
        )


def build_catalog(env: GridEnv | None = None) -> TrajectoryCatalog:
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


def format_grid(grid: tuple[tuple[int, ...], ...]) -> str:
    chars = {UNCOLORED: ".", RED_COLOR: "R", GREEN_COLOR: "G"}
    return "\n".join(" ".join(chars[cell] for cell in row) for row in grid)


def write_trajectory_list(
    path: Path | str = DEFAULT_ENUM_PATH,
    env: GridEnv | None = None,
) -> Path:
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
