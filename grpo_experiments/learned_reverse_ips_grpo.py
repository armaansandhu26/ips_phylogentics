"""Backward-compatible re-exports for learned-reverse IPS-GRPO.

Prefer the clean package at repo-root ``learned_reverse_ips/``:

    python -m learned_reverse_ips.train
"""

from learned_reverse_ips.advantages import (
    RunningLogWeightNormalizer,
    learned_reverse_advantages,
    terminal_log_rewards_from_scores,
)
from learned_reverse_ips.checkpoint import METHOD
from learned_reverse_ips.config import LearnedReverseExperimentConfig, parse_config
from learned_reverse_ips.mlp_policy import (
    PhyloLearnedReverseConfig,
    PhyloLearnedReversePolicy,
    build_reverse_batch,
    path_log_probabilities,
    update_mlp_reverse_policy,
)
from learned_reverse_ips.reverse_policy import (
    TabularTerminalReversePolicy,
    enumerate_tree_action_catalog,
    rollout_tree_action_paths,
    trajectory_indices_from_paths,
    update_reverse_policy,
)
from learned_reverse_ips.runner import run_experiment, validate_config

__all__ = [
    "METHOD",
    "LearnedReverseExperimentConfig",
    "PhyloLearnedReverseConfig",
    "PhyloLearnedReversePolicy",
    "RunningLogWeightNormalizer",
    "TabularTerminalReversePolicy",
    "build_reverse_batch",
    "enumerate_tree_action_catalog",
    "learned_reverse_advantages",
    "parse_config",
    "path_log_probabilities",
    "rollout_tree_action_paths",
    "run_experiment",
    "terminal_log_rewards_from_scores",
    "trajectory_indices_from_paths",
    "update_mlp_reverse_policy",
    "update_reverse_policy",
    "validate_config",
]


def main() -> None:
    from learned_reverse_ips.train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
