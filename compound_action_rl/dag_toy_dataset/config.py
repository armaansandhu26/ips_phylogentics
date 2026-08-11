"""Configuration for the direction/step DAG toy."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_TERMINAL_REWARDS = (1.0, 0.8, 0.2, 0.05)


def default_terminal_rewards(budget: int) -> tuple[float, ...]:
    """Scale the budget-3 reward profile over any terminal frontier size.

    Rewards are linearly interpolated against normalized x-coordinate anchors,
    so budget=3 remains exactly ``(1.0, 0.8, 0.2, 0.05)``.
    """
    if budget < 1:
        raise ValueError("budget must be >= 1")
    last_anchor = len(DEFAULT_TERMINAL_REWARDS) - 1
    rewards: list[float] = []
    for x in range(budget + 1):
        anchor_position = (x / budget) * last_anchor
        left = min(int(anchor_position), last_anchor - 1)
        fraction = anchor_position - left
        rewards.append(
            float(
                DEFAULT_TERMINAL_REWARDS[left] * (1.0 - fraction)
                + DEFAULT_TERMINAL_REWARDS[left + 1] * fraction
            )
        )
    return tuple(rewards)


@dataclass(frozen=True)
class TrainConfig:
    # The terminal frontier is x + y == budget. Each compound action advances
    # between 1 and max_step cells, so episode length is variable.
    budget: int = 32
    max_step: int = 3
    # Indexed by terminal x coordinate. None constructs a budget-scaled default.
    terminal_rewards: tuple[float, ...] | None = None

    hidden_size: int = 128
    num_layers: int = 2
    detach_step_rep: bool = True

    group_size: int = 128
    num_groups: int = 4
    num_updates: int = 500
    train_epochs: int = 1
    lr: float = 3e-4
    entropy_coef: float = 0.0
    clip_ratio: float = 0.2
    advantage_eps: float = 1e-8
    grad_clip_norm: float = 1.0
    seed: int = 0
    log_every: int = 25

    def __post_init__(self) -> None:
        if self.terminal_rewards is None:
            object.__setattr__(
                self, "terminal_rewards", default_terminal_rewards(self.budget)
            )

    def validate(self) -> None:
        if self.budget < 1:
            raise ValueError("budget must be >= 1")
        if self.max_step < 1:
            raise ValueError("max_step must be >= 1")
        if self.terminal_rewards is None or len(self.terminal_rewards) != self.budget + 1:
            raise ValueError("terminal_rewards must contain budget + 1 values")
        if any(not 0 < reward <= 1 for reward in self.terminal_rewards):
            raise ValueError("every terminal reward must be in (0, 1]")
        if self.hidden_size < 1 or self.num_layers < 1:
            raise ValueError("hidden_size and num_layers must be >= 1")
        if self.group_size < 1 or self.num_groups < 1:
            raise ValueError("group_size and num_groups must be >= 1")
        if self.num_updates < 1 or self.train_epochs < 1:
            raise ValueError("num_updates and train_epochs must be >= 1")
        if not 0 <= self.clip_ratio < 1:
            raise ValueError("clip_ratio must be in [0, 1)")
