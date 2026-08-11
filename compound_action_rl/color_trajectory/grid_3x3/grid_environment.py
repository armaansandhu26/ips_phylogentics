import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from grid_common import (
    GREEN,
    GREEN_COLOR,
    GridEnv as _GridEnv,
    NUM_COLOR_ACTIONS,
    NUM_COLOR_CHANNELS,
    NUM_MOVE_ACTIONS,
    RED,
    RED_COLOR,
    RIGHT,
    UNCOLORED,
    UP,
    compound_action_from_index,
    compound_action_to_index,
    minimal_episode_steps,
)

GRID_SIZE = 3


class GridEnv(_GridEnv):
    """3x3 compound-action grid environment for debugging."""

    def __init__(
        self,
        *,
        grid_size: int = GRID_SIZE,
        red_center=(1, 0),
        green_center=(2, 2),
        temperature: float = 2.0,
        red_dist=None,
        green_dist=None,
        max_episode_steps: int | None = None,
    ) -> None:
        super().__init__(
            grid_size=grid_size,
            red_center=red_center,
            green_center=green_center,
            temperature=temperature,
            red_dist=red_dist,
            green_dist=green_dist,
            max_episode_steps=max_episode_steps,
        )
