"""Hyper-Grid MDP for GRPO / IPS experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from final.toy.hypergrid import HyperGridSpec, build_reward_grid, hypergrid_reward


@dataclass
class HyperGridDataset:
    spec: HyperGridSpec
    rewards: np.ndarray
    root: Path

    @classmethod
    def load(cls, root: str | Path) -> HyperGridDataset:
        root = Path(root)
        meta_path = root / "meta.json"
        rewards_path = root / "rewards.npy"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            spec = HyperGridSpec(
                H=int(meta["H"]),
                D=int(meta["D"]),
                R0=float(meta["R0"]),
                R1=float(meta["R1"]),
                R2=float(meta["R2"]),
                outer_lo=float(meta.get("outer_lo", 0.25)),
                outer_hi=float(meta.get("outer_hi", 0.5)),
                inner_lo=float(meta.get("inner_lo", 0.3)),
                inner_hi=float(meta.get("inner_hi", 0.4)),
            )
        else:
            spec = HyperGridSpec()
        if rewards_path.exists():
            rewards = np.load(rewards_path)
        else:
            rewards = build_reward_grid(spec)
        if rewards.shape != (spec.H,) * spec.D:
            raise ValueError(f"reward shape {rewards.shape} != {(spec.H,) * spec.D}")
        return cls(spec=spec, rewards=rewards.astype(np.float32, copy=False), root=root)

    @property
    def terminate_action(self) -> int:
        return self.spec.D

    @property
    def num_actions(self) -> int:
        return self.spec.D + 1

    def reward_at(self, coords: np.ndarray) -> np.ndarray:
        if self.spec.D == 2:
            return self.rewards[coords[..., 0], coords[..., 1]]
        if self.spec.D == 1:
            return self.rewards[coords[..., 0]]
        return hypergrid_reward(
            coords,
            H=self.spec.H,
            R0=self.spec.R0,
            R1=self.spec.R1,
            R2=self.spec.R2,
            outer_lo=self.spec.outer_lo,
            outer_hi=self.spec.outer_hi,
            inner_lo=self.spec.inner_lo,
            inner_hi=self.spec.inner_hi,
        )

    def terminal_outcome_id(self, x: int, y: int) -> str:
        if self.spec.D == 1:
            return str(int(x))
        parts = [str(int(v)) for v in (x, y)[: self.spec.D]]
        return ",".join(parts)

    def coords_from_outcome_id(self, outcome_id: str) -> tuple[int, ...]:
        return tuple(int(part) for part in outcome_id.split(","))

    def load_target_probs(self) -> np.ndarray:
        target_path = self.root / "target_distribution.npz"
        if target_path.exists():
            payload = np.load(target_path)
            return np.asarray(payload["probs"], dtype=np.float64)
        rewards = self.rewards.astype(np.float64)
        return rewards / rewards.sum()
