"""IPS-GRPO v2 experiment package for split tree/edge policies."""

from grpo_experiments.tree_edge_ips_v2.config import TrainConfig
from grpo_experiments.tree_edge_ips_v2.ips_grpo import (
    IPSMetrics,
    compute_group_advantages,
    compute_ips_weights,
    snips_exact_weights,
    solve_temperature_for_ess,
)

__all__ = [
    "IPSMetrics",
    "TrainConfig",
    "compute_group_advantages",
    "compute_ips_weights",
    "snips_exact_weights",
    "solve_temperature_for_ess",
]
