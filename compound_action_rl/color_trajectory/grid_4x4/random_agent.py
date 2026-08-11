"""Random hierarchical agent: network 1 -> move, network 2 -> color."""

from __future__ import annotations

import numpy as np

from hierarchical import (
    HierarchicalAgent,
    RandomModel1Policy,
    RandomModel2Policy,
)


def make_random_agent(seed: int | None = None) -> HierarchicalAgent:
    rng = np.random.default_rng(seed)
    return HierarchicalAgent(RandomModel1Policy(), RandomModel2Policy(), rng=rng)
