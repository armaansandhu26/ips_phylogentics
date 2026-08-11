"""Training config + reward landscape presets for the merge (phylo) toy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RewardMode = Literal["linear", "exp", "log_score"]
PropensityMode = Literal["none", "exact", "marginal", "count"]

# Reward landscape presets. `score_seed` picks the (deterministic) per-topology
# quality landscape; `beta`/`mode` are defaults that can be overridden on the CLI.
REWARD_PROFILES: dict[str, dict[str, object]] = {
    "phylo_peaked": {
        "score_seed": 0,
        "beta": 57.0,
        "mode": "exp",
        "description": "R=exp(57*score): astronomically peaked posterior, mirrors phylo ~0..57 nat span",
    },
    "phylo_log_tilt": {
        "score_seed": 0,
        "beta": 57.0,
        "mode": "log_score",
        "description": "R=57*score: the near-flat 'uniform tilt' your current phylo config uses (reward = log_score)",
    },
    "gentle": {
        "score_seed": 0,
        "beta": 1.0,
        "mode": "linear",
        "description": "R=score in (0,1]: gentle target like the grid toy",
    },
    "mild_peaked": {
        "score_seed": 0,
        "beta": 8.0,
        "mode": "exp",
        "description": "R=exp(8*score): moderately peaked, easier to learn than phylo_peaked",
    },
}


@dataclass(frozen=True)
class TrainConfig:
    # ---- environment / reward ----
    n_leaves: int = 5
    reward_profile: str = "phylo_peaked"
    beta: float | None = None            # override profile beta
    reward_mode: RewardMode | None = None  # override profile mode

    # ---- policy net ----
    hidden_size: int = 128
    num_layers: int = 2

    # ---- optimisation ----
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

    # ---- IPS / credit ----
    # none     : plain GRPO (group-normalised reward)
    # exact    : trajectory IPS  w ∝ exp(-log P_F(tau))            -> biased to R*m
    # marginal : backward-corrected  w ∝ exp(log P_B(tau|x) - log P_F(tau)) -> R
    # count    : legacy within-group signature count weighting
    propensity_mode: PropensityMode = "exact"
    max_inverse_weight: float | None = None  # cap on 1/P_F(tau) (log-space)
    p_eps: float = 1e-8
    # If True, form advantages via naive exp-space reward*weight (overflows for
    # large beta) instead of the numerically-safe log-space SNIPS. Kept as a
    # switch to demonstrate the "broken log pi(tau) at scale" failure mode.
    naive_expspace: bool = False

    def resolved_beta(self) -> float:
        if self.beta is not None:
            return self.beta
        return float(REWARD_PROFILES[self.reward_profile]["beta"])  # type: ignore[index]

    def resolved_mode(self) -> RewardMode:
        if self.reward_mode is not None:
            return self.reward_mode
        return REWARD_PROFILES[self.reward_profile]["mode"]  # type: ignore[return-value]

    def score_seed(self) -> int:
        return int(REWARD_PROFILES[self.reward_profile]["score_seed"])  # type: ignore[index]

    def reward_model_kwargs(self) -> dict[str, object]:
        if self.reward_profile not in REWARD_PROFILES:
            raise ValueError(
                f"Unknown reward_profile={self.reward_profile!r}. "
                f"Choose from {list(REWARD_PROFILES)}"
            )
        return {
            "n_leaves": self.n_leaves,
            "score_seed": self.score_seed(),
            "beta": self.resolved_beta(),
            "mode": self.resolved_mode(),
        }
