from typing import Optional, Tuple

import numpy as np

# network1 (move)
UP = 0
RIGHT = 1
NUM_MOVE_ACTIONS = 2

# network2 (color)
RED = 0
GREEN = 1
NUM_COLOR_ACTIONS = 2

# internal color grid values
UNCOLORED = 0
RED_COLOR = 1
GREEN_COLOR = 2
NUM_COLOR_CHANNELS = 3


def minimal_episode_steps(grid_size: int) -> int:
    return 2 * (grid_size - 1)


def grid_distribution(
    center: Tuple[int, int],
    grid_size: int,
    temperature: float = 2.0,
) -> np.ndarray:
    scores = np.zeros((grid_size, grid_size), dtype=np.float64)
    for row in range(grid_size):
        for col in range(grid_size):
            dist_sq = (row - center[0]) ** 2 + (col - center[1]) ** 2
            scores[row, col] = np.exp(-dist_sq / temperature)
    return scores / scores.sum()


def coloring_reward(
    colors: np.ndarray,
    red_dist: np.ndarray,
    green_dist: np.ndarray,
) -> float:
    red_mask = colors == RED_COLOR
    green_mask = colors == GREEN_COLOR
    return float(np.sum(red_dist * red_mask) + np.sum(green_dist * green_mask))


def max_coloring_reward(
    red_dist: np.ndarray,
    green_dist: np.ndarray,
    skip_cells: Tuple[Tuple[int, int], ...] = (),
) -> float:
    per_cell = np.maximum(red_dist, green_dist)
    for row, col in skip_cells:
        per_cell[row, col] = 0.0
    return float(per_cell.sum())


def compound_action_to_index(color_action: int, move_action: int) -> int:
    return move_action * NUM_COLOR_ACTIONS + color_action


def compound_action_from_index(flat_action: int) -> Tuple[int, int]:
    move_action = flat_action // NUM_COLOR_ACTIONS
    color_action = flat_action % NUM_COLOR_ACTIONS
    return color_action, move_action


class GridEnv:
    """
    Compound-action grid environment.

    Decision order (two networks):
    1. network1 chooses move (up or right)
    2. network2 chooses color (red or green) given that move

    Execution order (single env step):
    1. apply the move
    2. paint the new cell with the chosen color
    3. episode ends when the agent reaches the top-right corner
       or when max_episode_steps is reached
    """

    def __init__(
        self,
        *,
        grid_size: int,
        red_center: Tuple[int, int],
        green_center: Tuple[int, int],
        temperature: float = 2.0,
        red_dist: Optional[np.ndarray] = None,
        green_dist: Optional[np.ndarray] = None,
        max_episode_steps: Optional[int] = None,
    ) -> None:
        self.grid_size = grid_size
        self.temperature = temperature
        self.max_episode_steps = (
            minimal_episode_steps(grid_size)
            if max_episode_steps is None
            else max_episode_steps
        )

        if red_dist is None:
            red_dist = grid_distribution(red_center, grid_size, temperature)
        if green_dist is None:
            green_dist = grid_distribution(green_center, grid_size, temperature)

        self.red_dist = np.asarray(red_dist, dtype=np.float64)
        self.green_dist = np.asarray(green_dist, dtype=np.float64)
        if self.red_dist.shape != (grid_size, grid_size):
            raise ValueError(f"red_dist must have shape ({grid_size}, {grid_size})")
        if self.green_dist.shape != (grid_size, grid_size):
            raise ValueError(f"green_dist must have shape ({grid_size}, {grid_size})")

        self._unpaintable_cells = ((0, 0),)
        self._max_reward = max_coloring_reward(
            self.red_dist, self.green_dist, skip_cells=self._unpaintable_cells
        )

        self._row = 0
        self._col = 0
        self._colors = np.zeros((grid_size, grid_size), dtype=np.int32)
        self._step_count = 0
        self._done = False

    @property
    def obs_dim(self) -> int:
        return self.grid_size * 2 + self.grid_size * self.grid_size * NUM_COLOR_CHANNELS

    @property
    def move_action_dim(self) -> int:
        return NUM_MOVE_ACTIONS

    @property
    def color_action_dim(self) -> int:
        return NUM_COLOR_ACTIONS

    @property
    def goal_pos(self) -> Tuple[int, int]:
        return self.grid_size - 1, self.grid_size - 1

    def optimal_color_pattern(self) -> np.ndarray:
        pattern = np.empty((self.grid_size, self.grid_size), dtype=np.int32)
        choose_red = self.red_dist >= self.green_dist
        pattern[choose_red] = RED_COLOR
        pattern[~choose_red] = GREEN_COLOR
        for row, col in self._unpaintable_cells:
            pattern[row, col] = UNCOLORED
        return pattern

    def _at_goal(self) -> bool:
        goal_row, goal_col = self.goal_pos
        return self._row == goal_row and self._col == goal_col

    def _terminal_reward(self) -> float:
        raw = coloring_reward(self._colors, self.red_dist, self.green_dist)
        if self._max_reward <= 0.0:
            return 0.0
        return raw / self._max_reward

    def get_observation(self) -> np.ndarray:
        row_oh = np.zeros(self.grid_size, dtype=np.float32)
        col_oh = np.zeros(self.grid_size, dtype=np.float32)
        row_oh[self._row] = 1.0
        col_oh[self._col] = 1.0

        color_oh = np.zeros(
            (self.grid_size, self.grid_size, NUM_COLOR_CHANNELS), dtype=np.float32
        )
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                color_oh[row, col, self._colors[row, col]] = 1.0

        return np.concatenate([row_oh, col_oh, color_oh.ravel()])

    def reset(self):
        self._row = 0
        self._col = 0
        self._colors.fill(UNCOLORED)
        self._step_count = 0
        self._done = False
        state = np.array([self._row, self._col], dtype=np.int32)
        return self.get_observation(), 0.0, state

    def step(self, move_action: int, color_action: int):
        if self._done:
            raise RuntimeError("step() called after episode ended")

        if move_action == UP:
            self._row = min(self._row + 1, self.grid_size - 1)
        elif move_action == RIGHT:
            self._col = min(self._col + 1, self.grid_size - 1)
        else:
            raise ValueError(f"Invalid move action: {move_action}")

        self._colors[self._row, self._col] = RED_COLOR if color_action == RED else GREEN_COLOR
        self._step_count += 1

        if self._at_goal():
            self._done = True
            reward = self._terminal_reward()
        elif self._step_count >= self.max_episode_steps:
            self._done = True
            reward = 0.0
        else:
            reward = 0.0

        state = np.array([self._row, self._col], dtype=np.int32)
        return self.get_observation(), reward, self._done, state
