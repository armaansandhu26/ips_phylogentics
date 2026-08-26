"""Hyper-Grid toy environment from the Trajectory Balance GFlowNet paper."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

# TB paper (Malkin et al., 2022), equivalent to user eq. (10) with signed coords
#   x_i = 2 * s_i / (H - 1) - 1
#   |x_i| > 0.5          <=> |s_i/(H-1) - 0.5| > 0.25
#   0.6 < |x_i| < 0.8    <=> |s_i/(H-1) - 0.5| in (0.3, 0.4)


@dataclass(frozen=True)
class HyperGridSpec:
    """Specification for a D-dimensional hyper-grid with side length H."""

    H: int = 4096
    D: int = 2
    R0: float = 0.1
    R1: float = 0.5
    R2: float = 2.0
    outer_lo: float = 0.25
    outer_hi: float = 0.5
    inner_lo: float = 0.3
    inner_hi: float = 0.4

    def validate(self) -> None:
        if self.H < 2:
            raise ValueError("H must be >= 2")
        if self.D < 1:
            raise ValueError("D must be >= 1")
        if not (0.0 < self.R0 < self.R1 < self.R2):
            raise ValueError("expected 0 < R0 < R1 < R2")
        if not (0.0 < self.outer_lo < self.outer_hi <= 0.5):
            raise ValueError("expected 0 < outer_lo < outer_hi <= 0.5")
        if not (self.outer_lo < self.inner_lo < self.inner_hi < self.outer_hi):
            raise ValueError("expected outer_lo < inner_lo < inner_hi < outer_hi")

    @property
    def num_terminals(self) -> int:
        return int(self.H**self.D)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["num_terminals"] = self.num_terminals
        payload["reward_formula"] = (
            "R(x) = R0 + R1 * prod_d I(|x_d/(H-1) - 0.5| in (outer_lo, outer_hi]) "
            "+ R2 * prod_d I(|x_d/(H-1) - 0.5| in (inner_lo, inner_hi))"
        )
        return payload


def signed_coords(coords: np.ndarray, H: int) -> np.ndarray:
    """Signed normalized terminal coords x_i = 2*s_i/(H-1) - 1 in [-1, 1]."""
    coords = np.asarray(coords, dtype=np.float64)
    return 2.0 * coords / float(H - 1) - 1.0


def normalized_offset(coords: np.ndarray, H: int) -> np.ndarray:
    """Return |coord / (H - 1) - 0.5| for integer terminal coordinates."""
    coords = np.asarray(coords, dtype=np.float64)
    return np.abs(coords / float(H - 1) - 0.5)


def hypergrid_reward(
    coords: np.ndarray,
    *,
    H: int,
    R0: float = 0.1,
    R1: float = 0.5,
    R2: float = 2.0,
    outer_lo: float = 0.25,
    outer_hi: float = 0.5,
    inner_lo: float = 0.3,
    inner_hi: float = 0.4,
) -> np.ndarray:
    """Evaluate terminal reward for coordinates with shape (..., D)."""
    offset = normalized_offset(coords, H)
    outer = (offset > outer_lo) & (offset <= outer_hi)
    inner = (offset > inner_lo) & (offset < inner_hi)
    outer_term = np.prod(outer, axis=-1)
    inner_term = np.prod(inner, axis=-1)
    return R0 + R1 * outer_term + R2 * inner_term


def build_reward_grid(spec: HyperGridSpec) -> np.ndarray:
    """Build a dense reward grid with shape (H,) * D."""
    spec.validate()
    if spec.D == 1:
        axis = np.arange(spec.H, dtype=np.int32)
        return hypergrid_reward(
            axis[:, None],
            H=spec.H,
            R0=spec.R0,
            R1=spec.R1,
            R2=spec.R2,
            outer_lo=spec.outer_lo,
            outer_hi=spec.outer_hi,
            inner_lo=spec.inner_lo,
            inner_hi=spec.inner_hi,
        ).astype(np.float32)
    if spec.D == 2:
        x = np.arange(spec.H, dtype=np.int32)
        y = np.arange(spec.H, dtype=np.int32)
        xx, yy = np.meshgrid(x, y, indexing="ij")
        coords = np.stack([xx, yy], axis=-1)
        return hypergrid_reward(
            coords,
            H=spec.H,
            R0=spec.R0,
            R1=spec.R1,
            R2=spec.R2,
            outer_lo=spec.outer_lo,
            outer_hi=spec.outer_hi,
            inner_lo=spec.inner_lo,
            inner_hi=spec.inner_hi,
        ).astype(np.float32)
    raise NotImplementedError(f"dense reward grids are only implemented for D <= 2, got D={spec.D}")


def target_distribution(rewards: np.ndarray) -> np.ndarray:
    """Return p(x) proportional to reward over a dense terminal grid."""
    rewards = np.asarray(rewards, dtype=np.float64)
    total = rewards.sum()
    if total <= 0:
        raise ValueError("reward grid must have positive mass")
    return (rewards / total).astype(np.float64)


def count_modes(rewards: np.ndarray, *, peak_reward: float | None = None) -> int:
    """Count connected components at the maximum reward level."""
    rewards = np.asarray(rewards)
    if peak_reward is None:
        peak_reward = float(rewards.max())
    mask = rewards >= peak_reward - 1e-8
    if rewards.ndim != 2:
        return int(mask.sum())
    visited = np.zeros_like(mask, dtype=bool)
    count = 0
    height, width = mask.shape
    for i in range(height):
        for j in range(width):
            if not mask[i, j] or visited[i, j]:
                continue
            count += 1
            stack = [(i, j)]
            visited[i, j] = True
            while stack:
                x, y = stack.pop()
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if (
                        0 <= nx < height
                        and 0 <= ny < width
                        and mask[nx, ny]
                        and not visited[nx, ny]
                    ):
                        visited[nx, ny] = True
                        stack.append((nx, ny))
    return count


def summarize_reward_grid(rewards: np.ndarray, spec: HyperGridSpec) -> dict[str, Any]:
    rewards = np.asarray(rewards)
    unique, counts = np.unique(rewards, return_counts=True)
    peak = float(rewards.max())
    return {
        "shape": list(rewards.shape),
        "reward_min": float(rewards.min()),
        "reward_max": peak,
        "reward_levels": [
            {"reward": float(value), "count": int(count)} for value, count in zip(unique, counts)
        ],
        "num_modes_at_peak": count_modes(rewards, peak_reward=peak),
        "expected_num_modes": 2**spec.D,
        "mass_by_level": {
            str(float(value)): float((rewards == value).sum() / rewards.size)
            for value in unique
        },
    }
