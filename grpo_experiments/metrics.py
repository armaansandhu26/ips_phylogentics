"""Outcome diversity metrics for phylogenetic tree sampling experiments."""

from __future__ import annotations

from collections import Counter
from typing import Literal, Sequence

OutcomeLevel = Literal["signature", "topology"]


def extract_outcome_ids(trees, level: OutcomeLevel = "topology") -> tuple[list[str], list[str]]:
    """Return (outcome_ids, topology_ids) for a batch of PhylogeneticTree objects."""
    signatures = [t.signature for t in trees]
    topology_ids = [t.tree_topology_id for t in trees]
    if level == "topology":
        return topology_ids, topology_ids
    return signatures, topology_ids


def batch_diversity_stats(outcome_ids: Sequence[str], topology_ids: Sequence[str]) -> dict:
    """Per-step diversity statistics for the current batch."""
    total = float(len(outcome_ids))
    unique_outcomes = float(len(set(outcome_ids)))
    unique_topologies = float(len(set(topology_ids)))
    return {
        "batch_size": total,
        "batch_unique_outcomes": unique_outcomes,
        "batch_duplicate_fraction": (total - unique_outcomes) / total if total > 0 else 0.0,
        "batch_unique_topologies": unique_topologies,
        "batch_duplicate_topology_fraction": (total - unique_topologies) / total if total > 0 else 0.0,
    }


class OutcomeTracker:
    """Cumulative outcome statistics across the full run (monitoring only)."""

    def __init__(self) -> None:
        self.outcome_counts: Counter = Counter()
        self.topology_counts: Counter = Counter()
        self.total = 0

    def update(self, outcome_ids: Sequence[str], topology_ids: Sequence[str]) -> None:
        for oid, tid in zip(outcome_ids, topology_ids):
            self.outcome_counts[oid] += 1
            self.topology_counts[tid] += 1
            self.total += 1

    def stats(self) -> dict:
        unique = len(self.outcome_counts)
        t = float(self.total)
        topo_unique = len(self.topology_counts)
        return {
            "global_total_samples": t,
            "global_unique_outcomes": float(unique),
            "global_duplicate_fraction": float((self.total - unique) / t) if t > 0 else 0.0,
            "global_unique_topologies": float(topo_unique),
            "global_duplicate_topology_fraction": float((self.total - topo_unique) / t) if t > 0 else 0.0,
        }

    def to_dict(self) -> dict:
        return {
            "outcome_counts": dict(self.outcome_counts),
            "topology_counts": dict(self.topology_counts),
            "total": self.total,
        }

    @classmethod
    def from_dict(cls, data: dict) -> OutcomeTracker:
        tracker = cls()
        tracker.outcome_counts = Counter(data.get("outcome_counts", {}))
        tracker.topology_counts = Counter(data.get("topology_counts", {}))
        tracker.total = int(data.get("total", sum(tracker.outcome_counts.values())))
        return tracker
