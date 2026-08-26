"""Backward-compatible re-exports for the MLP reverse policy.

Prefer ``learned_reverse_ips.mlp_policy`` for new code.
"""

from learned_reverse_ips.mlp_policy import (
    PhyloLearnedReverseConfig,
    PhyloLearnedReversePolicy,
    PhyloReverseBatch,
    build_reverse_batch,
    max_merge_actions,
    num_merge_actions,
    path_log_probabilities,
    path_log_probabilities_tensor,
    reverse_action_mask,
    reverse_context,
    update_mlp_reverse_policy,
)

__all__ = [
    "PhyloLearnedReverseConfig",
    "PhyloLearnedReversePolicy",
    "PhyloReverseBatch",
    "build_reverse_batch",
    "max_merge_actions",
    "num_merge_actions",
    "path_log_probabilities",
    "path_log_probabilities_tensor",
    "reverse_action_mask",
    "reverse_context",
    "update_mlp_reverse_policy",
]
