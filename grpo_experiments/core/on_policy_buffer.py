"""On-policy rollout reuse for TRL num_iterations (mu). Hybrid/policy-IS use update_cycles instead."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from grpo_experiments.core.advantage_logging import log_advantage_group
from grpo_experiments.core.policy_replay import ReplayBuffer, reevaluate_log_paths_pf, trajectory_actions
from src.gfn.action_tensors import TensorActionBatch


@dataclass
class OnPolicyBuffer:
    actions_set: list[list[dict]]
    log_paths_pf_old: torch.Tensor
    log_rewards: torch.Tensor
    log_scores: torch.Tensor
    advantages: torch.Tensor
    random_spec: dict | None
    advantage_metrics: dict
    outcome_ids: list[str] | None = None
    trajectories: list | None = None
    action_tensors: TensorActionBatch | None = None
    log_paths_pf_tree_old: torch.Tensor | None = None
    log_paths_pf_edge_old: torch.Tensor | None = None

    @property
    def size(self) -> int:
        if self.action_tensors is not None:
            return self.action_tensors.size
        return len(self.actions_set)


def capture_on_policy_buffer(
    batch: dict,
    trajectories,
    *,
    advantages: torch.Tensor,
    log_paths_pf_old: torch.Tensor,
    advantage_metrics: dict | None = None,
    outcome_ids: Sequence[str] | None = None,
    random_spec: dict | None = None,
    log_paths_pf_tree_old: torch.Tensor | None = None,
    log_paths_pf_edge_old: torch.Tensor | None = None,
) -> OnPolicyBuffer:
    action_tensors = batch.get("action_tensors")
    if action_tensors is not None:
        actions_set: list[list[dict]] = []
    else:
        actions_set = [trajectory_actions(traj) for traj in trajectories]
    return OnPolicyBuffer(
        actions_set=actions_set,
        log_paths_pf_old=log_paths_pf_old,
        log_rewards=batch["log_rewards"],
        log_scores=batch["log_scores"],
        advantages=advantages.detach(),
        random_spec=random_spec,
        advantage_metrics=dict(advantage_metrics or {}),
        outcome_ids=list(outcome_ids) if outcome_ids is not None else None,
        trajectories=trajectories,
        action_tensors=action_tensors,
        log_paths_pf_tree_old=log_paths_pf_tree_old,
        log_paths_pf_edge_old=log_paths_pf_edge_old,
    )


def reevaluate_on_policy_log_paths_pf(
    rollout_worker,
    generator,
    buffer: OnPolicyBuffer,
    *,
    chunk_size: int,
    device: str,
    return_split: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    replay = ReplayBuffer(
        actions_set=buffer.actions_set,
        trajectories=buffer.trajectories or [],
        log_paths_pf_old=buffer.log_paths_pf_old,
        log_rewards=buffer.log_rewards,
        log_scores=buffer.log_scores,
        random_spec=buffer.random_spec,
        action_tensors=buffer.action_tensors,
        log_paths_pf_tree_old=buffer.log_paths_pf_tree_old,
        log_paths_pf_edge_old=buffer.log_paths_pf_edge_old,
    )
    return reevaluate_log_paths_pf(
        rollout_worker,
        generator,
        replay,
        chunk_size=chunk_size,
        device=device,
        return_split=return_split,
    )


def prepare_on_policy_buffer(
    trainer,
    rollout_worker,
    generator,
    batch: dict,
    trajectories,
    *,
    random_spec: dict | None,
    chunk_size: int,
    device: str,
    outcome_ids: Sequence[str] | None = None,
    fixed_advantages: torch.Tensor | None = None,
    fixed_advantage_metrics: dict | None = None,
    ips_log_paths_pf: torch.Tensor | None = None,
) -> OnPolicyBuffer:
    if fixed_advantages is not None:
        advantages = fixed_advantages
        advantage_metrics = dict(fixed_advantage_metrics or {})
    else:
        advantages, advantage_metrics = trainer.precompute_advantages(
            batch["log_scores"],
            outcome_ids=list(outcome_ids) if outcome_ids is not None else None,
            log_paths_pf=ips_log_paths_pf,
        )
    scratch = capture_on_policy_buffer(
        batch,
        trajectories,
        advantages=advantages,
        log_paths_pf_old=batch["log_paths_pf"].detach(),
        advantage_metrics=advantage_metrics,
        outcome_ids=outcome_ids,
        random_spec=random_spec,
    )
    if trainer.num_iterations > 1:
        with torch.no_grad():
            if getattr(trainer, "uses_split_log_probs", False):
                old_pf, _, old_tree, old_edge = reevaluate_on_policy_log_paths_pf(
                    rollout_worker,
                    generator,
                    scratch,
                    chunk_size=chunk_size,
                    device=device,
                    return_split=True,
                )
                return capture_on_policy_buffer(
                    batch,
                    trajectories,
                    advantages=advantages,
                    log_paths_pf_old=old_pf.detach(),
                    log_paths_pf_tree_old=old_tree.detach(),
                    log_paths_pf_edge_old=old_edge.detach(),
                    advantage_metrics=advantage_metrics,
                    outcome_ids=outcome_ids,
                    random_spec=random_spec,
                )
            old_pf, _ = reevaluate_on_policy_log_paths_pf(
                rollout_worker,
                generator,
                scratch,
                chunk_size=chunk_size,
                device=device,
            )
        return capture_on_policy_buffer(
            batch,
            trajectories,
            advantages=advantages,
            log_paths_pf_old=old_pf.detach(),
            advantage_metrics=advantage_metrics,
            outcome_ids=outcome_ids,
            random_spec=random_spec,
        )
    if getattr(trainer, "uses_split_log_probs", False):
        with torch.no_grad():
            _, _, old_tree, old_edge = reevaluate_on_policy_log_paths_pf(
                rollout_worker,
                generator,
                scratch,
                chunk_size=chunk_size,
                device=device,
                return_split=True,
            )
        return capture_on_policy_buffer(
            batch,
            trajectories,
            advantages=advantages,
            log_paths_pf_old=batch["log_paths_pf"].detach(),
            log_paths_pf_tree_old=old_tree.detach(),
            log_paths_pf_edge_old=old_edge.detach(),
            advantage_metrics=advantage_metrics,
            outcome_ids=outcome_ids,
            random_spec=random_spec,
        )
    return scratch


def _trainer_update_kwargs(gen_buffer: OnPolicyBuffer, common: dict[str, Any]) -> dict[str, Any]:
    kwargs = {
        "log_paths_pf_old": common.get("log_paths_pf_old"),
        "fixed_advantages": common.get("fixed_advantages"),
        "log_paths_pf_old_for_metrics": common.get("log_paths_pf_old_for_metrics"),
        "extra_metrics": gen_buffer.advantage_metrics or None,
    }
    if gen_buffer.outcome_ids is not None:
        kwargs["outcome_ids"] = gen_buffer.outcome_ids
    for key in ("paths_entropy", "mask", "log_pf_old"):
        if key in common:
            kwargs[key] = common[key]
    for key in (
        "log_paths_pf_tree",
        "log_paths_pf_edge",
        "log_paths_pf_tree_old",
        "log_paths_pf_edge_old",
    ):
        if key in common:
            kwargs[key] = common[key]
    return kwargs


def run_on_policy_grpo_step(
    trainer,
    rollout_worker,
    generator,
    batch: dict,
    trajectories,
    *,
    random_spec: dict | None,
    generation_state: dict | None,
    chunk_size: int,
    device: str,
    extra_update_kwargs: dict | None = None,
    advantage_dump_dir: str | None = None,
    group_meta: dict | None = None,
) -> tuple[dict, dict]:
    """One on-policy GRPO step with optional mu-iteration buffering (trainer.num_iterations)."""
    extra_update_kwargs = extra_update_kwargs or {}
    iteration = 0 if generation_state is None else int(generation_state.get("iteration", 0))
    gen_buffer = None if generation_state is None else generation_state.get("buffer")
    outcome_ids = extra_update_kwargs.get("outcome_ids")

    if gen_buffer is None or iteration >= trainer.num_iterations:
        gen_buffer = prepare_on_policy_buffer(
            trainer,
            rollout_worker,
            generator,
            batch,
            trajectories,
            random_spec=random_spec,
            chunk_size=chunk_size,
            device=device,
            outcome_ids=outcome_ids,
            fixed_advantages=extra_update_kwargs.get("fixed_advantages"),
            fixed_advantage_metrics=extra_update_kwargs.get("fixed_advantage_metrics"),
            ips_log_paths_pf=extra_update_kwargs.get("ips_log_paths_pf"),
        )
        iteration = 0

    if getattr(trainer, "uses_split_log_probs", False):
        log_paths_pf, paths_entropy, log_paths_pf_tree, log_paths_pf_edge = reevaluate_on_policy_log_paths_pf(
            rollout_worker,
            generator,
            gen_buffer,
            chunk_size=chunk_size,
            device=device,
            return_split=True,
        )
    else:
        log_paths_pf, paths_entropy = reevaluate_on_policy_log_paths_pf(
            rollout_worker,
            generator,
            gen_buffer,
            chunk_size=chunk_size,
            device=device,
        )
        log_paths_pf_tree = None
        log_paths_pf_edge = None

    use_frozen_old = trainer.num_iterations > 1
    common = {
        "log_paths_pf_old": gen_buffer.log_paths_pf_old if use_frozen_old else None,
        "fixed_advantages": gen_buffer.advantages,
        "log_paths_pf_old_for_metrics": gen_buffer.log_paths_pf_old,
        "paths_entropy": paths_entropy if trainer.entropy_coef > 0 else None,
        "log_paths_pf_tree": log_paths_pf_tree,
        "log_paths_pf_edge": log_paths_pf_edge,
        "log_paths_pf_tree_old": gen_buffer.log_paths_pf_tree_old if use_frozen_old else None,
        "log_paths_pf_edge_old": gen_buffer.log_paths_pf_edge_old if use_frozen_old else None,
    }
    common.update({k: v for k, v in extra_update_kwargs.items() if k not in {
        "outcome_ids",
        "fixed_advantages",
        "fixed_advantage_metrics",
        "fixed_ips_metrics",
    }})
    if "fixed_ips_metrics" in extra_update_kwargs and extra_update_kwargs["fixed_ips_metrics"]:
        merged = dict(gen_buffer.advantage_metrics)
        merged.update(extra_update_kwargs["fixed_ips_metrics"])
        gen_buffer.advantage_metrics = merged

    train_info = trainer.update(
        log_paths_pf,
        gen_buffer.log_rewards,
        log_scores=gen_buffer.log_scores,
        **_trainer_update_kwargs(gen_buffer, common),
    )

    if group_meta is not None and iteration == 0:
        adv_stats = log_advantage_group(
            trainer,
            gen_buffer,
            dump_dir=advantage_dump_dir,
            epoch=int(group_meta["epoch"]),
            step=int(group_meta["step"]),
            global_step=int(group_meta["global_step"]),
            method=str(group_meta["method"]),
        )
        train_info.update(adv_stats)

    if trainer.num_iterations == 1:
        gen_buffer.log_paths_pf_old = log_paths_pf.detach()
        if log_paths_pf_tree is not None and log_paths_pf_edge is not None:
            gen_buffer.log_paths_pf_tree_old = log_paths_pf_tree.detach()
            gen_buffer.log_paths_pf_edge_old = log_paths_pf_edge.detach()

    return train_info, {"buffer": gen_buffer, "iteration": iteration + 1}


def run_policy_is_grpo_cycles(
    trainer,
    rollout_worker,
    generator,
    buffer: ReplayBuffer,
    *,
    advantages: torch.Tensor,
    advantage_metrics: dict,
    outcome_ids: Sequence[str] | None,
    update_cycles: int,
    chunk_size: int,
    device: str,
    reevaluate_fn=reevaluate_log_paths_pf,
    extra_update_kwargs: dict | None = None,
) -> list[dict]:
    """Policy-IS inner loop: frozen buffer, reevaluate pi_new each cycle, shared with GRPO core."""
    extra_update_kwargs = extra_update_kwargs or {}
    train_infos: list[dict] = []
    for _cycle in range(update_cycles):
        reeval_out = reevaluate_fn(
            rollout_worker,
            generator,
            buffer,
            chunk_size=chunk_size,
            device=device,
            return_split=getattr(trainer, "uses_split_log_probs", False),
        )
        paths_entropy = None
        log_paths_pf_tree = None
        log_paths_pf_edge = None
        if isinstance(reeval_out, tuple):
            if len(reeval_out) == 4:
                log_paths_pf, paths_entropy, log_paths_pf_tree, log_paths_pf_edge = reeval_out
            else:
                log_paths_pf, paths_entropy = reeval_out
        else:
            log_paths_pf = reeval_out

        update_kwargs = {
            "log_paths_pf_old": buffer.log_paths_pf_old,
            "fixed_advantages": advantages,
            "log_paths_pf_old_for_metrics": buffer.log_paths_pf_old,
            "extra_metrics": advantage_metrics or None,
            "paths_entropy": paths_entropy,
            "log_paths_pf_tree": log_paths_pf_tree,
            "log_paths_pf_edge": log_paths_pf_edge,
            "log_paths_pf_tree_old": getattr(buffer, "log_paths_pf_tree_old", None),
            "log_paths_pf_edge_old": getattr(buffer, "log_paths_pf_edge_old", None),
        }
        update_kwargs.update(extra_update_kwargs)
        if outcome_ids is not None:
            update_kwargs["outcome_ids"] = list(outcome_ids)

        train_info = trainer.update(
            log_paths_pf,
            buffer.log_rewards,
            log_scores=buffer.log_scores,
            **update_kwargs,
        )
        train_infos.append(train_info)
    return train_infos
