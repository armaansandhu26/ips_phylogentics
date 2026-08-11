"""Import 4×4 grid + catalog from color_trajectory."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

_CT = Path(__file__).resolve().parent.parent / "color_trajectory"
_G4 = _CT / "grid_4x4"

for _p in (_G4, _CT):
    p = str(_p)
    if p not in sys.path:
        sys.path.append(p)

from grid_environment_2 import GRID_SIZE, GridEnv  # noqa: E402
from trajectory_catalog import (  # noqa: E402
    TrajectoryRecord,
    build_catalog,
    iter_trajectories,
)

_PKG = Path(__file__).resolve().parent
if sys.path[0] != str(_PKG):
    sys.path.insert(0, str(_PKG))

NUM_MOVE_ACTIONS = 2
NUM_COLOR_ACTIONS = 2
NUM_TRAJECTORIES = 20 * (2 ** (2 * (GRID_SIZE - 1)))  # 1280 for 4×4

__all__ = [
    "GridEnv",
    "TrajectoryRecord",
    "build_catalog",
    "iter_trajectories",
    "make_env",
    "moves_to_path_index",
    "trajectory_lookup",
    "num_trajectories",
    "NUM_TRAJECTORIES",
]


def _gen_minimal_move_sequences(grid_size: int = GRID_SIZE) -> Iterable[tuple[int, ...]]:
    target = grid_size - 1

    def walk(row: int, col: int, seq: list[int]) -> Iterable[tuple[int, ...]]:
        if row == target and col == target:
            yield tuple(seq)
            return
        if row < target:
            yield from walk(row + 1, col, seq + [0])  # UP
        if col < target:
            yield from walk(row, col + 1, seq + [1])  # RIGHT

    yield from walk(0, 0, [])


def make_env(**kwargs) -> GridEnv:
    return GridEnv(**kwargs)


def num_trajectories(env: GridEnv | None = None) -> int:
    env = env or GridEnv()
    return len(list(_gen_minimal_move_sequences(env.grid_size))) * (2 ** (2 * (env.grid_size - 1)))


def moves_to_path_index(env: GridEnv | None = None) -> dict[tuple[int, ...], int]:
    env = env or GridEnv()
    return {
        moves: path_index
        for path_index, moves in enumerate(_gen_minimal_move_sequences(env.grid_size))
    }


def trajectory_lookup(env: GridEnv | None = None) -> dict[tuple[tuple[int, ...], tuple[int, ...]], int]:
    env = env or GridEnv()
    return {(r.moves, r.colors): r.index for r in iter_trajectories(env)}
