"""GRPO-family objectives adapted to RGFN's variable-length trajectories.

The policy loss and SNIPS implementation mirror the phylogenetics experiments;
they live in a dependency-light module so installing RGFN does not also require
the phylogenetics-only ETE/fvcore stack.
"""

from __future__ import annotations

import math
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
        advantage_normalization: str = "running",
        running_scale_decay: float = 0.99,
        advantage_clip: float = 10.0,
        log_ratio_clip: float = 20.0,
        exploration_rate: float = 0.0,
        separate_reverse_updates: bool = True,
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
        if advantage_normalization not in {"batch", "running"}:
            raise ValueError("advantage_normalization must be 'batch' or 'running'")
        if not 0.0 <= running_scale_decay < 1.0:
            raise ValueError("running_scale_decay must be in [0, 1)")
        if advantage_clip <= 0.0 or log_ratio_clip <= 0.0:
            raise ValueError("advantage_clip and log_ratio_clip must be positive")
        if not 0.0 <= exploration_rate < 1.0:
            raise ValueError("exploration_rate must be in [0, 1)")
        self.reverse_loss_weight = float(reverse_loss_weight)
        self.advantage_normalization = advantage_normalization
        self.running_scale_decay = float(running_scale_decay)
        self.advantage_clip = float(advantage_clip)
        self.log_ratio_clip = float(log_ratio_clip)
        self.exploration_rate = float(exploration_rate)
        self.separate_reverse_updates = bool(separate_reverse_updates)
        self.train_backward_policy = True
        # Buffers make the running importance-weight scale checkpoint-safe.
        self.register_buffer("running_log_first_moment", torch.tensor(float("nan")))
        self.register_buffer("running_log_second_moment", torch.tensor(float("nan")))
        self.register_buffer("running_scale_updates", torch.tensor(0, dtype=torch.long))
        self._latest_trajectories: Trajectories | None = None

    def _running_advantages(
        self,
        log_weight: Tensor,
        target_to_behavior_ratio: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, float]]:
        values = log_weight.detach().to(torch.float64)
        log_n = math.log(max(values.numel(), 1))
        batch_log_first = torch.logsumexp(values, dim=0) - log_n
        batch_log_second = torch.logsumexp(2.0 * values, dim=0) - log_n
        bootstrap = int(self.running_scale_updates.item()) == 0
        scale_log_first = (
            batch_log_first
            if bootstrap
            else self.running_log_first_moment.to(values.dtype)
        )
        scale_log_second = (
            batch_log_second
            if bootstrap
            else self.running_log_second_moment.to(values.dtype)
        )
        log_rms = 0.5 * scale_log_second
        stabilized = torch.exp(
            torch.clamp(values - log_rms, -self.log_ratio_clip, self.log_ratio_clip)
        )
        baseline = torch.exp(
            torch.clamp(
                scale_log_first - log_rms,
                -self.log_ratio_clip,
                self.log_ratio_clip,
            )
        )
        if target_to_behavior_ratio is None:
            target_to_behavior_ratio = torch.ones_like(stabilized)
        else:
            target_to_behavior_ratio = target_to_behavior_ratio.detach().to(torch.float64)
        # For exploratory (off-policy) samples, a plain constant baseline is
        # biased because E_mu[grad log P_F] != 0.  Multiplying the baseline by
        # P_F/mu restores the zero-expectation control variate exactly.
        centered = stabilized - baseline * target_to_behavior_ratio
        advantages = torch.clamp(centered, -self.advantage_clip, self.advantage_clip)

        # RGFN evaluates the objective inside ``torch.no_grad()``. Validation
        # must read, but never mutate, the on-policy training normalizer.
        if torch.is_grad_enabled():
            decay = self.running_scale_decay
            if bootstrap:
                updated_first = batch_log_first
                updated_second = batch_log_second
            elif decay == 0.0:
                updated_first = batch_log_first
                updated_second = batch_log_second
            else:
                log_decay = math.log(decay)
                log_new = math.log1p(-decay)
                updated_first = torch.logaddexp(
                    self.running_log_first_moment.to(values.dtype) + log_decay,
                    batch_log_first + log_new,
                )
                updated_second = torch.logaddexp(
                    self.running_log_second_moment.to(values.dtype) + log_decay,
                    batch_log_second + log_new,
                )
            self.running_log_first_moment.copy_(
                updated_first.to(self.running_log_first_moment)
            )
            self.running_log_second_moment.copy_(
                updated_second.to(self.running_log_second_moment)
            )
            self.running_scale_updates.add_(1)

        metrics = {
            "running_scale_bootstrap": float(bootstrap),
            "running_scale_updates": float(self.running_scale_updates.item()),
            "running_log_weight_rms": float(log_rms.item()),
            "running_scaled_weight_baseline": float(baseline.item()),
            "running_scaled_weight_mean": float(stabilized.mean().item()),
            "running_scaled_weight_std": float(stabilized.std(unbiased=False).item()),
            "target_to_behavior_ratio_mean": float(target_to_behavior_ratio.mean().item()),
            "target_to_behavior_ratio_min": float(target_to_behavior_ratio.min().item()),
            "target_to_behavior_ratio_max": float(target_to_behavior_ratio.max().item()),
            "running_preclip_advantage_min": float(centered.min().item()),
            "running_preclip_advantage_max": float(centered.max().item()),
            "running_advantage_clip_fraction": float(
                (advantages != centered).to(torch.float64).mean().item()
            ),
        }
        return advantages.to(dtype=log_weight.dtype, device=log_weight.device), metrics

    def compute_reverse_loss(self, trajectories: Trajectories | None = None) -> Tensor:
        trajectories = trajectories or self._latest_trajectories
        if trajectories is None:
            raise RuntimeError("No MIPS trajectories are available for a reverse update")
        flat_pb = self.backward_policy.compute_action_log_probs(
            states=trajectories.get_non_source_states_flat(),
            action_spaces=trajectories.get_backward_action_spaces_flat(),
            actions=trajectories.get_actions_flat(),
        )
        # q_phi(tau | x) is a *trajectory* probability.  Match the
        # phylogenetics implementation by summing edge log-probabilities per
        # route and then averaging routes.  Averaging the flat edges instead
        # would unintentionally give longer synthesis routes more MLE weight.
        index = trajectories.get_index_flat().to(flat_pb.device)
        log_pb = torch.zeros(
            len(trajectories), dtype=flat_pb.dtype, device=flat_pb.device
        )
        log_pb.scatter_add_(0, index, flat_pb)
        return -log_pb.mean()

    def compute_objective_output(self, trajectories: Trajectories) -> ObjectiveOutput:
        flat_pf, packed, mask = self._forward_log_probs(trajectories)
        index = trajectories.get_index_flat().to(flat_pf.device)
        log_pf = torch.zeros(len(trajectories), dtype=flat_pf.dtype, device=flat_pf.device)
        log_pf.scatter_add_(0, index, flat_pf)

        # The small-batch molecule setup optionally samples each action from
        # (1-epsilon) P_F + epsilon Uniform.  Correcting by that exact behavior
        # probability preserves support without changing the reward-
        # proportional terminal target.  At epsilon=0 this is ordinary
        # on-policy MIPS.
        if self.exploration_rate > 0.0:
            action_spaces = trajectories.get_forward_action_spaces_flat()
            uniform_log_prob = -torch.log(
                torch.tensor(
                    [max(len(action_space), 1) for action_space in action_spaces],
                    dtype=flat_pf.dtype,
                    device=flat_pf.device,
                )
            )
            flat_behavior = torch.logaddexp(
                flat_pf.detach() + math.log1p(-self.exploration_rate),
                uniform_log_prob + math.log(self.exploration_rate),
            )
        else:
            flat_behavior = flat_pf.detach()
        log_behavior = torch.zeros_like(log_pf)
        log_behavior.scatter_add_(0, index, flat_behavior)
        target_to_behavior_ratio = torch.exp(
            torch.clamp(log_pf.detach() - log_behavior, -self.log_ratio_clip, self.log_ratio_clip)
        )

        flat_pb = self.backward_policy.compute_action_log_probs(
            states=trajectories.get_non_source_states_flat(),
            action_spaces=trajectories.get_backward_action_spaces_flat(),
            actions=trajectories.get_actions_flat(),
        ).to(flat_pf.device)
        log_pb = torch.zeros_like(log_pf)
        log_pb.scatter_add_(0, index, flat_pb)
        log_reward = trajectories.get_reward_outputs().log_reward.to(flat_pf.device)
        log_weight = log_reward.detach() + log_pb.detach() - log_behavior
        scaled_reward = torch.exp(log_weight - log_weight.max())
        normalization_metrics: dict[str, float] = {}
        if self.advantage_normalization == "running":
            advantages, normalization_metrics = self._running_advantages(
                log_weight, target_to_behavior_ratio
            )
        else:
            scale = scaled_reward.std(unbiased=False).clamp_min(self.advantage_eps)
            advantages = (
                scaled_reward - scaled_reward.mean() * target_to_behavior_ratio
            ) / scale
        forward_loss, metrics = self._policy_loss(packed, mask, advantages)
        reverse_loss = -log_pb.mean()
        loss = (
            forward_loss
            if self.separate_reverse_updates
            else forward_loss + self.reverse_loss_weight * reverse_loss
        )
        if torch.is_grad_enabled():
            self._latest_trajectories = trajectories
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
            log_importance_weight_mean=float(log_weight.mean().item()),
            log_importance_weight_std=float(log_weight.std(unbiased=False).item()),
            log_importance_weight_min=float(log_weight.min().item()),
            log_importance_weight_max=float(log_weight.max().item()),
            forward_log_probability_mean=float(log_pf.mean().item()),
            behavior_log_probability_mean=float(log_behavior.mean().item()),
            exploration_rate=self.exploration_rate,
            reverse_log_probability_mean=float(log_pb.mean().item()),
            separate_reverse_updates=float(self.separate_reverse_updates),
        )
        metrics.update(normalization_metrics)
        return ObjectiveOutput(loss=loss, metrics=metrics)

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        if isinstance(self.forward_policy, torch.nn.Module):
            yield from self.forward_policy.parameters(recurse)
        if isinstance(self.backward_policy, torch.nn.Module):
            yield from self.backward_policy.parameters(recurse)
