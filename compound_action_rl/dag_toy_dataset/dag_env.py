"""A small monotone DAG with a compound (direction, step count) action."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from config import default_terminal_rewards

RIGHT = 0
UP = 1
DIRECTION_NAMES = ("right", "up")
Trajectory = tuple[tuple[int, int], ...]


@dataclass(frozen=True, order=True)
class State:
    x: int
    y: int

    @property
    def depth(self) -> int:
        return self.x + self.y

    @property
    def signature(self) -> str:
        return f"({self.x},{self.y})"


def find_default_terminal_states(budget: int) -> tuple[State, ...]:
    """Return the complete monotone frontier ``x + y == budget``."""
    if budget < 1:
        raise ValueError("budget must be >= 1")
    return tuple(State(x, budget - x) for x in range(budget + 1))


def trajectory_signature(trajectory: Trajectory) -> str:
    """Compact action-sequence label, e.g. ``R1-U2``."""
    names = {RIGHT: "R", UP: "U"}
    return "-".join(f"{names[direction]}{length}" for direction, length in trajectory)


def uniform_backward_log_probability(
    trajectory: Trajectory,
    *,
    max_step: int,
) -> float:
    """Log probability under a uniform policy over valid parent edges.

    A child ``(x, y)`` has ``min(max_step, x)`` horizontal parents and
    ``min(max_step, y)`` vertical parents. Multiplying these locally normalized
    reverse probabilities defines a normalized distribution over all paths
    from any terminal back to the root.
    """
    if max_step < 1:
        raise ValueError("max_step must be >= 1")

    x = 0
    y = 0
    log_probability = 0.0
    for direction, length in trajectory:
        if direction not in (RIGHT, UP):
            raise ValueError(f"unknown direction {direction}")
        if length < 1 or length > max_step:
            raise ValueError(
                f"trajectory step length must be in [1, {max_step}], got {length}"
            )
        if direction == RIGHT:
            x += length
        else:
            y += length
        parent_count = min(max_step, x) + min(max_step, y)
        log_probability -= math.log(parent_count)
    return log_probability


@dataclass(frozen=True)
class RewardModel:
    budget: int = 32
    values: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.values is None:
            object.__setattr__(self, "values", default_terminal_rewards(self.budget))
        if self.values is None or len(self.values) != self.budget + 1:
            raise ValueError("reward values must contain budget + 1 entries")
        if any(not 0 < reward <= 1 for reward in self.values):
            raise ValueError("reward values must be in (0, 1]")

    def reward(self, state: State) -> float:
        if state.depth != self.budget:
            raise ValueError(f"reward is terminal-only, got nonterminal state {state}")
        assert self.values is not None
        return float(self.values[state.x])


def reward_per_terminal_state(
    budget: int,
    rewards: tuple[float, ...] | None = None,
) -> dict[State, float]:
    """Return a terminal-to-normal-reward mapping without path enumeration."""
    reward_model = RewardModel(budget=budget, values=rewards)
    return {
        state: reward_model.reward(state)
        for state in find_default_terminal_states(budget)
    }


class DAGEnv:
    """Monotone grid whose terminal states lie on ``x + y == budget``.

    An edge is selected hierarchically:

      model 1: direction in {RIGHT, UP}
      model 2: step count in {1, ..., max_step}

    Step counts are masked by the remaining budget. Coordinates only increase,
    hence the state graph is acyclic. Different direction/length compositions
    can merge into exactly the same state and receive exactly the same reward.
    """

    def __init__(
        self,
        *,
        budget: int = 3,
        max_step: int = 3,
        reward_model=None,
    ) -> None:
        if budget < 1:
            raise ValueError("budget must be >= 1")
        if max_step < 1:
            raise ValueError("max_step must be >= 1")
        self.budget = int(budget)
        self.max_step = int(max_step)
        self.reward_model = reward_model
        self.state = State(0, 0)
        self.done = False

    @property
    def obs_dim(self) -> int:
        # Exact, non-aliased encoding: one-hot x, y, and remaining distance.
        return 3 * (self.budget + 1)

    @property
    def remaining(self) -> int:
        return self.budget - self.state.depth

    def reset(self) -> np.ndarray:
        self.state = State(0, 0)
        self.done = False
        return self.observation()

    def observation(self) -> np.ndarray:
        width = self.budget + 1
        obs = np.zeros(3 * width, dtype=np.float32)
        obs[self.state.x] = 1.0
        obs[width + self.state.y] = 1.0
        obs[2 * width + self.remaining] = 1.0
        return obs

    def direction_mask(self) -> np.ndarray:
        valid = not self.done and self.remaining > 0
        return np.array([valid, valid], dtype=bool)

    def step_mask(self, direction: int) -> np.ndarray:
        if direction not in (RIGHT, UP):
            raise ValueError(f"unknown direction {direction}")
        mask = np.zeros(self.max_step, dtype=bool)
        if not self.done:
            mask[: min(self.max_step, self.remaining)] = True
        return mask

    def step(self, direction: int, step_index: int):
        """Apply an action; ``step_index=0`` means a physical step count of 1."""
        if self.done:
            raise RuntimeError("step() called after terminal state")
        mask = self.step_mask(direction)
        if step_index < 0 or step_index >= self.max_step or not mask[step_index]:
            raise ValueError(
                f"invalid step index {step_index}; remaining={self.remaining}, "
                f"valid physical lengths=1..{int(mask.sum())}"
            )
        length = step_index + 1
        if direction == RIGHT:
            self.state = State(self.state.x + length, self.state.y)
        elif direction == UP:
            self.state = State(self.state.x, self.state.y + length)
        else:
            raise ValueError(f"unknown direction {direction}")

        self.done = self.remaining == 0

        reward = 0.0
        info = {
            "state": self.state,
            "length": length,
        }
        if self.done:
            reward = (
                float(self.reward_model.reward(self.state))
                if self.reward_model is not None
                else 1.0
            )
            info.update(
                signature=self.state.signature,
                reward=reward,
            )
        return self.observation(), reward, self.done, info
