from __future__ import annotations

from grpo_experiments.tree_edge_ips_v2.config import TrainConfig
from grpo_experiments.tree_edge_ips_v2.ips_grpo import compute_group_advantages
from grpo_experiments.tree_edge_ips_v2.losses import split_tree_edge_ppo_loss
import torch


class TreeEdgeIPSGRPOTrainer:
    """Small v2 trainer core for exact/SNIPS advantages and split PPO updates.

    The phylo-specific rollout/replay code should feed this trainer flattened
    per-step log-probs plus per-episode old joint log-probs.
    """

    def __init__(self, params, config: TrainConfig):
        self.config = config
        self.optimizer = torch.optim.Adam(params, lr=config.lr)

    def compute_group_advantages(
        self,
        returns: torch.Tensor,
        old_log_prob_joint: torch.Tensor,
        *,
        outcome_ids=None,
    ) -> tuple[torch.Tensor, dict]:
        advantages, metrics = compute_group_advantages(
            returns,
            old_log_prob_joint,
            outcome_ids=outcome_ids,
            propensity_mode=self.config.propensity_mode,
            max_inverse_weight=self.config.max_inverse_weight,
            count_eps=self.config.count_propensity_eps,
            advantage_eps=self.config.advantage_eps,
            weight_temperature=self.config.ips_weight_temperature,
            snips_truncate_ratio=self.config.snips_truncate_ratio,
            target_ess_fraction=self.config.ips_target_ess_fraction,
        )
        return advantages, metrics.as_flat_dict()

    def update_from_log_probs(
        self,
        *,
        log_prob_tree: torch.Tensor,
        log_prob_edge: torch.Tensor,
        old_log_prob_tree: torch.Tensor,
        old_log_prob_edge: torch.Tensor,
        advantage_tree: torch.Tensor,
        advantage_edge: torch.Tensor | None = None,
        entropy_tree: torch.Tensor | None = None,
        entropy_edge: torch.Tensor | None = None,
        aux_loss: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> dict:
        self.optimizer.zero_grad(set_to_none=True)
        loss, metrics = split_tree_edge_ppo_loss(
            log_prob_tree,
            log_prob_edge,
            old_log_prob_tree,
            old_log_prob_edge,
            advantage_tree,
            advantage_edge,
            clip_eps=self.config.clip_eps,
            entropy_tree=entropy_tree,
            entropy_edge=entropy_edge,
            entropy_coef=self.config.entropy_coef,
            aux_loss=aux_loss,
            mask=mask,
        )
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for group in self.optimizer.param_groups for p in group["params"]],
            self.config.max_grad_norm,
        )
        self.optimizer.step()
        metrics["grad_norm"] = float(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm)
        return metrics

    def state_dict(self) -> dict:
        return {"optimizer": self.optimizer.state_dict(), "config": self.config.to_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.optimizer.load_state_dict(state["optimizer"])
