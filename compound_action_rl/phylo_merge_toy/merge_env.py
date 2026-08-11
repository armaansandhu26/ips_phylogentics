"""
Coalescent / merge environment — a miniature of the phylogenetic tree env.

This toy reproduces the *structural* reason plain trajectory-IPS struggles on
phylo: the trajectory -> outcome map is a many-to-one DAG, not a bijection.

  * State      = a forest of subtrees. Start: `n` singleton leaves 0..n-1.
  * Action     = pick an unordered pair of current subtrees to merge into a new
                 internal node (exactly like the phylo `tree_pairs_dict`:
                 C(k, 2) choices when k subtrees remain).
  * Trajectory = the ordered sequence of merges (a "labeled history" /
                 ranked tree). Length n-1.
  * Outcome    = the resulting rooted binary topology ("signature"),
                 canonicalised so merge order does not matter.

For n=5: 180 ordered merge sequences collapse onto 105 rooted topologies with
*non-uniform* multiplicity m(x) = #histories -> x. This is the DAG
marginalisation problem GFlowNet's backward policy is built to solve.

The env is pure-python/numpy (no torch) so it can be enumerated exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Optional

import numpy as np


def canonical_leaf(leaf_id: int) -> str:
    return f"L{leaf_id}"


def canonical_merge(a: str, b: str) -> str:
    """Order-invariant canonical id for the tree formed by joining a and b."""
    lo, hi = sorted((a, b))
    return f"({lo},{hi})"


@dataclass(frozen=True)
class Subtree:
    """A rooted subtree in the current forest.

    `canon` is an order-invariant string id (two different build orders that
    yield the same topology share the same canon). `leaves` is the leaf set,
    used to build a permutation-aware observation. `is_internal` is True when
    the subtree is the result of at least one merge (has children).
    """

    canon: str
    leaves: frozenset[int]

    @property
    def is_internal(self) -> bool:
        return len(self.leaves) > 1


def pair_index_table(n: int) -> list[tuple[int, int]]:
    """Fixed enumeration of slot-index pairs (i<j). Action a -> PAIRS[a]."""
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


class MergeEnv:
    """Compact merge environment over `n_leaves` labeled leaves.

    The forest is kept as a compacted list: after merging slots i and j the two
    are removed and the merged subtree is appended, so active slots are always
    0..k-1. Valid actions at k subtrees are pairs (i, j) with j < k.
    """

    def __init__(self, *, n_leaves: int = 5, reward_model=None) -> None:
        if n_leaves < 2:
            raise ValueError("n_leaves must be >= 2")
        self.n_leaves = n_leaves
        self.reward_model = reward_model
        self.pairs = pair_index_table(n_leaves)
        self.num_actions = len(self.pairs)  # C(n, 2)

        self._forest: list[Subtree] = []
        self._step_count = 0
        self._done = False
        # per-step backward bookkeeping: num_parents of the resulting state
        self._num_parents_hist: list[int] = []

    # ---- static descriptors -------------------------------------------------
    @property
    def obs_dim(self) -> int:
        # per-slot leaf membership (n*n) + one-hot of current num_trees (n+1)
        return self.n_leaves * self.n_leaves + (self.n_leaves + 1)

    @property
    def num_trees(self) -> int:
        return len(self._forest)

    @property
    def horizon(self) -> int:
        return self.n_leaves - 1

    # ---- observation & masking ----------------------------------------------
    def action_mask(self) -> np.ndarray:
        """Boolean mask over the C(n,2) fixed pairs: valid iff both slots active."""
        k = self.num_trees
        mask = np.zeros(self.num_actions, dtype=bool)
        for a, (i, j) in enumerate(self.pairs):
            if j < k:
                mask[a] = True
        return mask

    def get_observation(self) -> np.ndarray:
        n = self.n_leaves
        membership = np.zeros((n, n), dtype=np.float32)
        for slot, sub in enumerate(self._forest):
            for leaf in sub.leaves:
                membership[slot, leaf] = 1.0
        num_trees_oh = np.zeros(n + 1, dtype=np.float32)
        num_trees_oh[self.num_trees] = 1.0
        return np.concatenate([membership.ravel(), num_trees_oh])

    # ---- dynamics ------------------------------------------------------------
    def reset(self) -> np.ndarray:
        self._forest = [
            Subtree(canon=canonical_leaf(i), leaves=frozenset({i}))
            for i in range(self.n_leaves)
        ]
        self._step_count = 0
        self._done = False
        self._num_parents_hist = []
        return self.get_observation()

    @staticmethod
    def _num_parents(forest: list[Subtree]) -> int:
        """#parents of a forest state = #internal top-level subtrees.

        A parent state is reached by un-merging (splitting) one internal
        top-level subtree into its two children; any internal top-level subtree
        is a valid last-merge. This is the toy analogue of the phylo rollout
        worker's `num_parents = has_children.sum(-1)`.
        """
        return sum(1 for s in forest if s.is_internal)

    def step(self, action: int):
        """Merge the pair indexed by `action`. Returns (obs, reward, done, info)."""
        if self._done:
            raise RuntimeError("step() called after episode ended")
        i, j = self.pairs[action]
        k = self.num_trees
        if not (i < j < k):
            raise ValueError(
                f"invalid action {action} -> pair ({i},{j}) with num_trees={k}"
            )

        a = self._forest[i]
        b = self._forest[j]
        merged = Subtree(
            canon=canonical_merge(a.canon, b.canon),
            leaves=a.leaves | b.leaves,
        )
        # remove j then i (j > i) and append merged -> compacted forest
        del self._forest[j]
        del self._forest[i]
        self._forest.append(merged)

        self._step_count += 1
        self._num_parents_hist.append(self._num_parents(self._forest))

        reward = 0.0
        log_reward = 0.0
        info: dict = {}
        if self.num_trees == 1:
            self._done = True
            signature = self._forest[0].canon
            info["signature"] = signature
            info["log_pb"] = self.log_pb()
            if self.reward_model is not None:
                reward = float(self.reward_model.reward(signature))
                log_reward = float(self.reward_model.log_reward(signature))
            info["reward"] = reward
            info["log_reward"] = log_reward

        return self.get_observation(), reward, self._done, info

    # ---- backward policy (uniform over parents) ------------------------------
    def log_pb(self) -> float:
        """log P_B(tau | x) for the completed trajectory under uniform backward.

        = sum_t -log(num_parents(state_t)); the terminal state contributes
        -log(1) = 0. Property (validated in catalog.py): for any signature x,
        sum over histories tau->x of P_B(tau|x) == 1.
        """
        total = 0.0
        for np_count in self._num_parents_hist:
            total += -np.log(np_count)
        return float(total)

    @property
    def signature(self) -> Optional[str]:
        if self.num_trees == 1:
            return self._forest[0].canon
        return None


def num_ordered_histories(n: int) -> int:
    """#labeled histories (ranked trees): prod_{k=2..n} C(k,2) = n!(n-1)!/2^{n-1}."""
    total = 1
    for k in range(2, n + 1):
        total *= comb(k, 2)
    return total


def num_rooted_topologies(n: int) -> int:
    """#rooted binary topologies on n labeled leaves = (2n-3)!! for n>=2."""
    if n <= 1:
        return 1
    total = 1
    for k in range(3, 2 * n - 2 + 1, 2):  # 3,5,...,(2n-3)
        total *= k
    return total
