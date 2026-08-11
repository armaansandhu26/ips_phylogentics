"""4×4 grid with spatial red/green variation — IPS-GRPO v2 training config."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

CreditMode = Literal["joint", "reward_weighted"]
Trainable = Literal["both", "path_only", "color_only"]
PropensityMode = Literal["exact", "count"]
ColorCredit = Literal["trajectory", "counterfactual"]

# 4×4 analogues of the 3×3 profiles (corners scaled to 0..3).
COLOR_PROFILES: dict[str, dict[str, object]] = {
    "default": {
        "red_center": (2, 0),
        "green_center": (3, 3),
        "temperature": 2.0,
        "description": "Red upper-left bias, green lower-right (4×4 analogue of 3×3 default)",
    },
    "swapped": {
        "red_center": (3, 3),
        "green_center": (2, 0),
        "temperature": 2.0,
        "description": "Flip: green upper-left, red lower-right",
    },
    "split_corners": {
        "red_center": (0, 3),
        "green_center": (3, 0),
        "temperature": 2.0,
        "description": "Red top-right, green bottom-left — strong diagonal split",
    },
    "center_green": {
        "red_center": (0, 0),
        "green_center": (2, 2),
        "temperature": 1.5,
        "description": "Green peaks near center, red at start corner",
    },
}


@dataclass(frozen=True)
class TrainConfig:
    color_profile: str = "default"
    hidden_size: int = 128
    num_layers: int = 2
    grid_size: int = 4

    group_size: int = 256
    num_groups: int = 4
    num_updates: int = 500
    train_epochs: int = 1
    lr: float = 3e-4
    entropy_coef: float = 0.01
    clip_ratio: float = 0.4
    advantage_eps: float = 1e-8
    grad_clip_norm: float = 1.0
    seed: int = 0
    log_every: int = 25

    p_eps: float = 1e-8
    max_inverse_weight: float | None = 2560.0
    propensity_mode: PropensityMode = "exact"

    detach_color_rep: bool = True
    aux_pos_coef: float = 0.1
    color_credit: ColorCredit = "trajectory"

    credit_mode: CreditMode = "joint"
    trainable: Trainable = "both"

    def profile_kwargs(self) -> dict[str, object]:
        if self.color_profile not in COLOR_PROFILES:
            raise ValueError(
                f"Unknown color_profile={self.color_profile!r}. "
                f"Choose from {list(COLOR_PROFILES)}"
            )
        p = COLOR_PROFILES[self.color_profile]
        return {
            "red_center": p["red_center"],
            "green_center": p["green_center"],
            "temperature": p["temperature"],
        }
