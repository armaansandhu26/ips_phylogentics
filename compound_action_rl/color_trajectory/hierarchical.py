"""Two-stage policy interface for the color-trajectory environment.

Model 1 chooses move in {up, right}.
Model 2 chooses color in {red, green} given that move.

Decision order each step:
    move ~ pi_1(· | obs)
    color ~ pi_2(· | rep, move)   where rep = encode(obs) from model 1

The environment then executes move and color together (move first, then paint).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from grid_environment_2 import (
    NUM_COLOR_ACTIONS,
    NUM_MOVE_ACTIONS,
    compound_action_to_index,
)


MOVE_NAMES = ("up", "right")
COLOR_NAMES = ("red", "green")


def model2_input_from_rep(move_rep: np.ndarray, move_action: int) -> np.ndarray:
    """Input for network 2: move-net hidden state + one-hot move from network 1."""
    one_hot = np.zeros(NUM_MOVE_ACTIONS, dtype=np.float32)
    one_hot[move_action] = 1.0
    return np.concatenate([move_rep.astype(np.float32), one_hot])


def model2_input_dim(move_hidden_size: int) -> int:
    return move_hidden_size + NUM_MOVE_ACTIONS


@dataclass(frozen=True)
class HierarchicalAction:
    move_action: int
    color_action: int

    @property
    def flat(self) -> int:
        return compound_action_to_index(self.color_action, self.move_action)


@dataclass(frozen=True)
class HierarchicalStepInfo:
    action: HierarchicalAction
    log_prob_model1: float
    log_prob_model2: float

    @property
    def log_prob_joint(self) -> float:
        return self.log_prob_model1 + self.log_prob_model2


class Model1Policy(Protocol):
    """Move policy: pi_1(move | obs)."""

    def action_probs(self, obs: np.ndarray) -> np.ndarray:
        """Return shape (NUM_MOVE_ACTIONS,) probabilities."""

    def sample(self, obs: np.ndarray, rng: np.random.Generator) -> tuple[int, float]:
        """Return (move_action, log_prob)."""


class Model2Policy(Protocol):
    """Color policy: pi_2(color | obs, move)."""

    def action_probs(self, obs: np.ndarray, move_action: int) -> np.ndarray:
        """Return shape (NUM_COLOR_ACTIONS,) probabilities."""

    def sample(
        self, obs: np.ndarray, move_action: int, rng: np.random.Generator
    ) -> tuple[int, float]:
        """Return (color_action, log_prob)."""


def _sample_from_probs(
    probs: np.ndarray, rng: np.random.Generator
) -> tuple[int, float]:
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()
    action = int(rng.choice(len(probs), p=probs))
    log_prob = float(np.log(probs[action] + 1e-12))
    return action, log_prob


class RandomModel1Policy:
    def action_probs(self, obs: np.ndarray) -> np.ndarray:
        del obs
        return np.full(NUM_MOVE_ACTIONS, 1.0 / NUM_MOVE_ACTIONS)

    def sample(self, obs: np.ndarray, rng: np.random.Generator) -> tuple[int, float]:
        return _sample_from_probs(self.action_probs(obs), rng)


class RandomModel2Policy:
    def action_probs(self, obs: np.ndarray, move_action: int) -> np.ndarray:
        del obs, move_action
        return np.full(NUM_COLOR_ACTIONS, 1.0 / NUM_COLOR_ACTIONS)

    def sample(
        self, obs: np.ndarray, move_action: int, rng: np.random.Generator
    ) -> tuple[int, float]:
        return _sample_from_probs(self.action_probs(obs, move_action), rng)


class HierarchicalAgent:
    """Network 1 picks move, network 2 picks color, then env executes both."""

    def __init__(
        self,
        model1: Model1Policy,
        model2: Model2Policy,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.model1 = model1
        self.model2 = model2
        self.rng = rng if rng is not None else np.random.default_rng()

    def act(self, obs: np.ndarray) -> tuple[int, int, HierarchicalStepInfo]:
        move_action, log_p1 = self.model1.sample(obs, self.rng)
        color_action, log_p2 = self.model2.sample(obs, move_action, self.rng)
        action = HierarchicalAction(move_action, color_action)
        info = HierarchicalStepInfo(action, log_p1, log_p2)
        return move_action, color_action, info

    @staticmethod
    def action_labels(move_action: int, color_action: int) -> tuple[str, str]:
        return MOVE_NAMES[move_action], COLOR_NAMES[color_action]


@dataclass
class Transition:
    obs: np.ndarray
    move_action: int
    color_action: int
    reward: float
    next_obs: np.ndarray
    done: bool
    log_prob_model1: float
    log_prob_model2: float


def rollout_episode(env, agent: HierarchicalAgent) -> list[Transition]:
    obs, _, _state = env.reset()
    transitions: list[Transition] = []
    done = False

    while not done:
        move_action, color_action, step_info = agent.act(obs)
        next_obs, reward, done, _state = env.step(move_action, color_action)
        transitions.append(
            Transition(
                obs=obs.copy(),
                move_action=move_action,
                color_action=color_action,
                reward=reward,
                next_obs=next_obs.copy(),
                done=done,
                log_prob_model1=step_info.log_prob_model1,
                log_prob_model2=step_info.log_prob_model2,
            )
        )
        obs = next_obs

    return transitions
