"""GRPO-family objectives adapted to RGFN's variable-length trajectories.

The policy loss and SNIPS implementation mirror the phylogenetics experiments;
they live in a dependency-light module so installing RGFN does not also require
the phylogenetics-only ETE/fvcore stack.
"""

from __future__ import annotations

from typing import Iterator

import gin
import torch
from torch import Tensor
from torch.nn import Parameter
from torch.nn.utils.rnn import pad_sequence

from molecule_synthesis.objective_math import compute_grpo_policy_loss
from rgfn.api.objective_base import ObjectiveBase, ObjectiveOutput
from rgfn.api.policy_base import PolicyBase
from rgfn.api.trajectories import Trajectories


def _pack_actions(
    flat: Tensor, trajectory_index: Tensor, n_trajectories: int
) -> tuple[Tensor, Tensor]:
    """Convert RGFN's action-flat tensor into a padded (batch, steps) tensor."""
    trajectory_index = trajectory_index.to(flat.device)
    rows = [flat[trajectory_index == i] for i in range(n_trajectories)]
    if not rows or any(row.numel() == 0 for row in rows):
        raise ValueError("Every sampled trajectory must contain at least one action")
    packed = pad_sequence(rows, batch_first=True, padding_value=0.0)
    mask = pad_sequence(
        [torch.ones_like(row, dtype=packed.dtype) for row in rows],
        batch_first=True,
        padding_value=0.0,
    )
    return packed, mask


def _stable_linear_rewards(log_reward: Tensor) -> Tensor:
    """Return rewards proportional to exp(log_reward) without overflow."""
    detached = log_reward.detach()
    return torch.exp(detached - detached.max())


def _standardize(values: Tensor, eps: float) -> Tensor:
    if values.numel() < 2:
        return torch.zeros_like(values)
    return (values - values.mean()) / values.std(unbiased=False).clamp_min(eps)


class _PolicyGradientObjective(ObjectiveBase):
    def __init__(
        self,
        forward_policy: PolicyBase,
        backward_policy: PolicyBase,
        clip_eps: float = 0.2,
        advantage_eps: float = 1e-8,
        reward_mode: str = "linear",
    ):
        super().__init__(forward_policy=forward_policy, backward_policy=backward_policy)
        if reward_mode not in {"linear", "log"}:
            raise ValueError("reward_mode must be 'linear' or 'log'")
        self.clip_eps = float(clip_eps)
        self.advantage_eps = float(advantage_eps)
        self.reward_mode = reward_mode

    def _forward_log_probs(self, trajectories: Trajectories) -> tuple[Tensor, Tensor, Tensor]:
        actions = trajectories.get_actions_flat()
        flat = self.forward_policy.compute_action_log_probs(
            states=trajectories.get_non_last_states_flat(),
            action_spaces=trajectories.get_forward_action_spaces_flat(),
            actions=actions,
        )
        packed, mask = _pack_actions(flat, trajectories.get_index_flat(), len(trajectories))
        return flat, packed, mask

    def _policy_loss(self, packed: Tensor, mask: Tensor, advantages: Tensor) -> tuple[Tensor, dict]:
        return compute_grpo_policy_loss(
            packed,
            advantages.detach(),
            log_paths_pf_old=packed.detach(),
            clip_eps=self.clip_eps,
            mask=mask,
        )

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        # GRPO-family objectives update only the forward generative policy. The
        # backward policy is a fixed proposal used by MIPS correction.
        if isinstance(self.forward_policy, torch.nn.Module):
            yield from self.forward_policy.parameters(recurse)


@gin.configurable()
class GRPOObjective(_PolicyGradientObjective):
    """Plain on-policy GRPO over one batch of reaction trajectories."""

    def compute_objective_output(self, trajectories: Trajectories) -> ObjectiveOutput:
        _, packed, mask = self._forward_log_probs(trajectories)
        log_reward = trajectories.get_reward_outputs().log_reward.to(packed.device)
        score = _stable_linear_rewards(log_reward) if self.reward_mode == "linear" else log_reward
        advantages = _standardize(score, self.advantage_eps)
        loss, metrics = self._policy_loss(packed, mask, advantages)
        metrics.update(
            mean_advantage=float(advantages.mean().item()),
            std_advantage=float(advantages.std(unbiased=False).item()),
            mean_log_reward=float(log_reward.mean().item()),
        )
        return ObjectiveOutput(loss=loss, metrics=metrics)


@gin.configurable()
class CountIPSGRPOObjective(_PolicyGradientObjective):
    """Original IPS-GRPO using empirical terminal-outcome frequencies.

    ``count_ips_grpo`` is the local disambiguating name for IPS-GRPO from
    Sinha et al. (2026): p_hat(o) = count(o) / group_size and the reward used
    by GRPO is R(o) / max(p_hat(o), epsilon). It has no auxiliary model.
    """

    def __init__(
        self,
        forward_policy: PolicyBase,
        backward_policy: PolicyBase,
        clip_eps: float = 0.2,
        advantage_eps: float = 1e-8,
        reward_mode: str = "linear",
        probability_floor: float = 1e-6,
    ):
        super().__init__(
            forward_policy=forward_policy,
            backward_policy=backward_policy,
            clip_eps=clip_eps,
            advantage_eps=advantage_eps,
            reward_mode=reward_mode,
        )
        self.probability_floor = float(probability_floor)

    @staticmethod
    def _outcome_id(state) -> str:
        molecule = getattr(state, "molecule", None)
        smiles = getattr(molecule, "smiles", None)
        return str(smiles) if smiles is not None else type(state).__name__

    def compute_objective_output(self, trajectories: Trajectories) -> ObjectiveOutput:
        _, packed, mask = self._forward_log_probs(trajectories)
        log_reward = trajectories.get_reward_outputs().log_reward.to(packed.device)
        base_reward = (
            _stable_linear_rewards(log_reward) if self.reward_mode == "linear" else log_reward
        )
        outcome_ids = [self._outcome_id(state) for state in trajectories.get_last_states_flat()]
        counts: dict[str, int] = {}
        for outcome_id in outcome_ids:
            counts[outcome_id] = counts.get(outcome_id, 0) + 1
        probabilities = torch.tensor(
            [counts[outcome_id] / len(outcome_ids) for outcome_id in outcome_ids],
            dtype=base_reward.dtype,
            device=base_reward.device,
        ).clamp_min(self.probability_floor)
        scaled_reward = base_reward / probabilities
        advantages = _standardize(scaled_reward, self.advantage_eps)
        loss, metrics = self._policy_loss(packed, mask, advantages)
        metrics.update(
            ips_unique_outcomes=float(len(counts)),
            ips_duplicate_fraction=float(1.0 - len(counts) / len(outcome_ids)),
            ips_probability_mean=float(probabilities.mean().item()),
            ips_probability_min=float(probabilities.min().item()),
            ips_clipped_fraction=float(
                (probabilities <= self.probability_floor).to(torch.float32).mean().item()
            ),
            mean_advantage=float(advantages.mean().item()),
            std_advantage=float(advantages.std(unbiased=False).item()),
            mean_log_reward=float(log_reward.mean().item()),
        )
        return ObjectiveOutput(loss=loss, metrics=metrics)


@gin.configurable()
class MIPSGRPOObjective(_PolicyGradientObjective):
    """MIPS-GRPO using a learned reverse policy q_phi(trajectory|molecule)."""

    def __init__(
        self,
        forward_policy: PolicyBase,
        backward_policy: PolicyBase,
        clip_eps: float = 0.2,
        advantage_eps: float = 1e-8,
        reward_mode: str = "linear",
        reverse_loss_weight: float = 1.0,
    ):
        super().__init__(
            forward_policy=forward_policy,
            backward_policy=backward_policy,
            clip_eps=clip_eps,
            advantage_eps=advantage_eps,
            reward_mode=reward_mode,
        )
        if reward_mode != "linear":
            raise ValueError("MIPS-GRPO requires a positive linear reward target")
        self.reverse_loss_weight = float(reverse_loss_weight)
        self.train_backward_policy = True

    def compute_objective_output(self, trajectories: Trajectories) -> ObjectiveOutput:
        flat_pf, packed, mask = self._forward_log_probs(trajectories)
        index = trajectories.get_index_flat().to(flat_pf.device)
        log_pf = torch.zeros(len(trajectories), dtype=flat_pf.dtype, device=flat_pf.device)
        log_pf.scatter_add_(0, index, flat_pf)

        flat_pb = self.backward_policy.compute_action_log_probs(
            states=trajectories.get_non_source_states_flat(),
            action_spaces=trajectories.get_backward_action_spaces_flat(),
            actions=trajectories.get_actions_flat(),
        ).to(flat_pf.device)
        log_pb = torch.zeros_like(log_pf)
        log_pb.scatter_add_(0, index, flat_pb)
        log_reward = trajectories.get_reward_outputs().log_reward.to(flat_pf.device)
        log_weight = log_reward.detach() + log_pb.detach() - log_pf.detach()
        scaled_reward = torch.exp(log_weight - log_weight.max())
        advantages = _standardize(scaled_reward, self.advantage_eps)
        forward_loss, metrics = self._policy_loss(packed, mask, advantages)
        reverse_loss = -log_pb.mean()
        loss = forward_loss + self.reverse_loss_weight * reverse_loss
        normalized_weight = scaled_reward / scaled_reward.sum().clamp_min(1e-8)
        ess = scaled_reward.sum().square() / scaled_reward.square().sum().clamp_min(1e-8)
        metrics.update(
            forward_loss=float(forward_loss.item()),
            reverse_loss=float(reverse_loss.item()),
            ips_ess=float(ess.item()),
            ips_ess_fraction=float((ess / len(trajectories)).item()),
            max_normalized_weight=float(normalized_weight.max().item()),
            mean_advantage=float(advantages.mean().item()),
            std_advantage=float(advantages.std(unbiased=False).item()),
            mean_log_reward=float(log_reward.mean().item()),
            forward_log_probability_mean=float(log_pf.mean().item()),
            reverse_log_probability_mean=float(log_pb.mean().item()),
        )
        return ObjectiveOutput(loss=loss, metrics=metrics)

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        if isinstance(self.forward_policy, torch.nn.Module):
            yield from self.forward_policy.parameters(recurse)
        if isinstance(self.backward_policy, torch.nn.Module):
            yield from self.backward_policy.parameters(recurse)
