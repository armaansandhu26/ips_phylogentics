from __future__ import annotations

import numpy as np

from grid_paths import NUM_MOVE_ACTIONS


def model2_input_from_rep(state_rep: np.ndarray, move_action: int) -> np.ndarray:
    one_hot = np.zeros(NUM_MOVE_ACTIONS, dtype=np.float32)
    one_hot[move_action] = 1.0
    return np.concatenate([state_rep.astype(np.float32), one_hot])


def model2_input_dim(state_rep_dim: int) -> int:
    return state_rep_dim + NUM_MOVE_ACTIONS
