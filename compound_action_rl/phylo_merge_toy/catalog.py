"""
Reward landscape + exact enumeration of the merge DAG.

Because the toy is small (n=5 -> 180 histories, 105 topologies) we can compute
*exact* ground truth:

  * multiplicity m(x)      = #histories (labeled orderings) that build topology x
  * marginal target        pi*(x)   proportional to R(x)          (what we WANT)
  * trajectory-IPS target  pi_ips(x) proportional to m(x) * R(x)  (the BIAS)

and we validate the uniform-backward identity  sum_{tau->x} P_B(tau|x) == 1.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from merge_env import (
    Subtree,
    canonical_leaf,
    canonical_merge,
    num_ordered_histories,
    num_rooted_topologies,
)

RewardMode = Literal["linear", "exp", "log_score"]

LOG_EPS = 1e-12


# ---------------------------------------------------------------------------
# Reward landscape
# ---------------------------------------------------------------------------
def _hash_unit(signature: str, seed: int) -> float:
    """Deterministic score in (0, 1) for a topology signature.

    Uses a stable hash so the landscape is reproducible across processes
    (Python's builtin hash is salted per-process).
    """
    key = f"{seed}:{signature}".encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()
    # take 13 hex digits -> 52 bits, map to (0, 1)
    val = int(digest[:13], 16) / float(16**13)
    return min(max(val, 1e-6), 1.0 - 1e-6)


@dataclass
class RewardModel:
    """Maps a topology signature -> (score, log_reward, reward).

    `score(x)` in (0,1) is the intrinsic "quality" of a topology (the toy
    analogue of a per-tree log-likelihood, normalised). `beta` is the
    dynamic-range / inverse-temperature knob and `mode` selects how score turns
    into reward:

      * linear    : R = score                     (gentle, ~[0,1] target)
      * exp       : R = exp(beta * score)         (astronomically peaked posterior;
                    beta=57 mirrors the phylo ~0..57 nat span)
      * log_score : R = beta * score              (the near-flat "uniform tilt"
                    your current phylo config uses by rewarding log_score itself)
    """

    n_leaves: int = 5
    score_seed: int = 0
    beta: float = 57.0
    mode: RewardMode = "exp"

    def score(self, signature: str) -> float:
        return _hash_unit(signature, self.score_seed)

    def log_reward(self, signature: str) -> float:
        s = self.score(signature)
        if self.mode == "linear":
            return float(np.log(max(s, LOG_EPS)))
        if self.mode == "exp":
            return float(self.beta * s)
        if self.mode == "log_score":
            return float(np.log(max(self.beta * s, LOG_EPS)))
        raise ValueError(f"unknown reward mode {self.mode!r}")

    def reward(self, signature: str) -> float:
        if self.mode == "linear":
            return float(self.score(signature))
        if self.mode == "exp":
            # kept finite: exp(beta*score) can overflow float64 for large beta,
            # so trainers should prefer log_reward. This is intentionally the
            # value that breaks naive exp-space weighting.
            return float(np.exp(self.beta * self.score(signature)))
        if self.mode == "log_score":
            return float(self.beta * self.score(signature))
        raise ValueError(f"unknown reward mode {self.mode!r}")


# ---------------------------------------------------------------------------
# Exact DAG enumeration
# ---------------------------------------------------------------------------
@dataclass
class Catalog:
    n_leaves: int
    signatures: list[str]
    multiplicity: dict[str, int]
    log_reward: dict[str, float]
    score: dict[str, float]
    pb_mass: dict[str, float]  # sum_{tau->x} P_B(tau|x), must be ~1
    num_histories: int
    reward_mode: str
    beta: float
    index: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.index:
            self.index = {sig: i for i, sig in enumerate(sorted(self.signatures))}

    # ---- ground-truth densities (in log space, then normalised) ----------
    def _normalised(self, log_unnorm: dict[str, float]) -> dict[str, float]:
        keys = list(log_unnorm)
        arr = np.array([log_unnorm[k] for k in keys], dtype=np.float64)
        arr = arr - arr.max()
        w = np.exp(arr)
        w = w / w.sum()
        return {k: float(v) for k, v in zip(keys, w)}

    def target_marginal(self) -> dict[str, float]:
        """pi*(x) proportional to R(x) — the unbiased posterior we want."""
        return self._normalised(dict(self.log_reward))

    def target_ips(self) -> dict[str, float]:
        """pi_ips(x) proportional to m(x) R(x) — the trajectory-IPS fixed point."""
        log_unnorm = {
            sig: self.log_reward[sig] + np.log(self.multiplicity[sig])
            for sig in self.signatures
        }
        return self._normalised(log_unnorm)

    def summary(self) -> dict[str, float | int]:
        mult = np.array(list(self.multiplicity.values()), dtype=np.float64)
        return {
            "n_leaves": self.n_leaves,
            "num_histories": self.num_histories,
            "num_topologies": len(self.signatures),
            "multiplicity_min": int(mult.min()),
            "multiplicity_max": int(mult.max()),
            "multiplicity_mean": float(mult.mean()),
            "reward_mode": self.reward_mode,
            "beta": self.beta,
        }


def _num_internal(forest: list[Subtree]) -> int:
    return sum(1 for s in forest if s.is_internal)


def build_catalog(reward_model: RewardModel) -> Catalog:
    """Enumerate every labeled history and collapse onto topologies."""
    n = reward_model.n_leaves
    multiplicity: dict[str, int] = {}
    pb_mass: dict[str, float] = {}
    num_histories = 0

    initial = [
        Subtree(canon=canonical_leaf(i), leaves=frozenset({i})) for i in range(n)
    ]

    def recurse(forest: list[Subtree], log_pb: float) -> None:
        nonlocal num_histories
        k = len(forest)
        if k == 1:
            sig = forest[0].canon
            multiplicity[sig] = multiplicity.get(sig, 0) + 1
            pb_mass[sig] = pb_mass.get(sig, 0.0) + float(np.exp(log_pb))
            num_histories += 1
            return
        for i in range(k):
            for j in range(i + 1, k):
                a, b = forest[i], forest[j]
                merged = Subtree(
                    canon=canonical_merge(a.canon, b.canon),
                    leaves=a.leaves | b.leaves,
                )
                new_forest = [forest[x] for x in range(k) if x not in (i, j)]
                new_forest.append(merged)
                step_log_pb = -np.log(_num_internal(new_forest))
                recurse(new_forest, log_pb + step_log_pb)

    recurse(initial, 0.0)

    signatures = sorted(multiplicity)
    log_reward = {sig: reward_model.log_reward(sig) for sig in signatures}
    score = {sig: reward_model.score(sig) for sig in signatures}

    return Catalog(
        n_leaves=n,
        signatures=signatures,
        multiplicity=multiplicity,
        log_reward=log_reward,
        score=score,
        pb_mass=pb_mass,
        num_histories=num_histories,
        reward_mode=reward_model.mode,
        beta=reward_model.beta,
    )


def validate_catalog(catalog: Catalog, *, tol: float = 1e-9) -> None:
    """Assert enumeration counts and the uniform-backward identity."""
    n = catalog.n_leaves
    assert catalog.num_histories == num_ordered_histories(n), (
        f"history count {catalog.num_histories} != {num_ordered_histories(n)}"
    )
    assert len(catalog.signatures) == num_rooted_topologies(n), (
        f"topology count {len(catalog.signatures)} != {num_rooted_topologies(n)}"
    )
    for sig, mass in catalog.pb_mass.items():
        assert abs(mass - 1.0) < tol, f"P_B mass for {sig} = {mass} != 1"
