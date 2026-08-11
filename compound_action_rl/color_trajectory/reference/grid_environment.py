import itertools
from typing import Callable, Dict, Optional, Tuple

import numpy as np


REWARD_R0 = 0.0
REWARD_R1 = 0.5
REWARD_R2 = 2.0


def reward_corners(x: np.ndarray) -> float:
    ax = np.abs(x)
    return float(
        (ax > 0.5).prod(-1) * REWARD_R1
        + ((ax < 0.8) * (ax > 0.6)).prod(-1) * REWARD_R2
        + REWARD_R0
    )


def reward_cos_N(x: np.ndarray) -> float:
    # Matches the original "cos_N" style shape without scipy dependency.
    gaussian = np.exp(-0.5 * (x * 5.0) ** 2) / np.sqrt(2.0 * np.pi)
    return float(((np.cos(x * 50.0) + 1.0) * gaussian).prod(-1) + 0.01)


REWARD_FUNCS: Dict[str, Callable[[np.ndarray], float]] = {
    "corners": reward_corners,
    "cos_N": reward_cos_N,
}


class GridEnv:
    """
    Standalone N-dimensional discrete grid environment.

    State representation:
    - Internal state is an integer vector of length `num_dims`.
    - Each coordinate ranges from 0 to `horizon - 1`.
    - Observation is a concatenated one-hot encoding for each dimension
      with length `horizon * num_dims`.

    Actions:
    - 0 .. num_dims-1: increment the selected coordinate by 1
    - num_dims: stop action
    """

    def __init__(
        self,
        horizon: int,
        num_dims: int = 2,
        x_range: Tuple[float, float] = (-1.0, 1.0),
        reward_name: str = "cos_N",
        reward_func: Optional[Callable[[np.ndarray], float]] = None,
    ) -> None:
        self.horizon = horizon
        self.num_dims = num_dims
        self.x_range = x_range
        self.width = x_range[1] - x_range[0]

        if reward_func is not None:
            self.reward_func = reward_func
        else:
            self.reward_func = REWARD_FUNCS.get(reward_name, reward_cos_N)
        self.x_coords = np.linspace(*x_range, horizon, dtype=np.float32)

        self._state = np.zeros(self.num_dims, dtype=np.int32)
        self._step = 0
        self._true_density = None

    @staticmethod
    def _default_reward(x: np.ndarray) -> float:
        return reward_cos_N(x)

    def get_observation(self, state: Optional[np.ndarray] = None) -> np.ndarray:
        state = np.int32(self._state if state is None else state)
        obs = np.zeros((self.horizon * self.num_dims), dtype=np.float32)
        obs[np.arange(len(state)) * self.horizon + state] = 1.0
        return obs

    def state_to_x(self, state: np.ndarray) -> np.ndarray:
        obs = self.get_observation(state)
        return (obs.reshape((self.num_dims, self.horizon)) * self.x_coords[None, :]).sum(1)

    def reset(self):
        self._state = np.zeros(self.num_dims, dtype=np.int32)
        self._step = 0
        reward = float(self.reward_func(self.state_to_x(self._state)))
        return self.get_observation(), reward, self._state.copy()

    def get_parents(self, state: np.ndarray, used_stop_action: bool):
        if used_stop_action:
            return [self.get_observation(state)], [self.num_dims]

        parents = []
        actions = []
        for dim in range(self.num_dims):
            if state[dim] > 0:
                parent_state = state.copy()
                parent_state[dim] -= 1
                parents.append(self.get_observation(parent_state))
                actions.append(dim)
        return parents, actions

    def step(self, action: int, state: Optional[np.ndarray] = None):
        original_state = state
        state = (self._state if state is None else state).copy()

        if action < self.num_dims:
            state[action] = min(state[action] + 1, self.horizon - 1)

        at_terminal_corner = bool(np.all(state >= (self.horizon - 1)))
        done = bool(at_terminal_corner or action == self.num_dims)
        reward = 0.0 if not done else float(self.reward_func(self.state_to_x(state)))

        if original_state is None:
            self._state = state
            self._step += 1

        return self.get_observation(state), reward, done, state

    def true_density(self):
        if self._true_density is not None:
            return self._true_density

        all_states = np.int32(
            list(itertools.product(*[list(range(self.horizon))] * self.num_dims))
        )

        state_mask = np.array(
            [len(self.get_parents(s, False)[0]) > 0 or np.sum(s) == 0 for s in all_states]
        )

        all_x = (
            np.float32(all_states) / (self.horizon - 1) * (self.x_coords[-1] - self.x_coords[0])
            + self.x_coords[0]
        )
        rewards = np.asarray(self.reward_func(all_x))[state_mask]

        self._true_density = (
            rewards / rewards.sum(),
            list(map(tuple, all_states[state_mask])),
            rewards,
        )
        return self._true_density
