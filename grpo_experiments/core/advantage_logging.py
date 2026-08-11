"""Per-group advantage dumps and stats for distribution analysis."""

from __future__ import annotations

import json
import os
from typing import Any

import torch

from grpo_experiments.core.advantages import advantage_distribution_stats


def merge_advantage_stats(advantages: torch.Tensor) -> dict[str, float]:
    return advantage_distribution_stats(advantages)


def save_advantage_group_dump(
    dump_dir: str,
    *,
    epoch: int,
    step: int,
    global_step: int,
    method: str,
    advantages: torch.Tensor,
    rewards: torch.Tensor,
    log_scores: torch.Tensor,
    scaled_rewards: torch.Tensor | None = None,
    reward_mode: str = "exp_linear",
    extra: dict[str, Any] | None = None,
) -> str:
    os.makedirs(dump_dir, exist_ok=True)
    adv_cpu = advantages.detach().float().cpu().tolist()
    rew_cpu = rewards.detach().float().cpu().tolist()
    log_scores_cpu = log_scores.detach().float().cpu().tolist()
    payload: dict[str, Any] = {
        "epoch": epoch,
        "step": step,
        "global_step": global_step,
        "method": method,
        "group_size": len(adv_cpu),
        "reward_mode": reward_mode,
        "advantages": adv_cpu,
        "rewards": rew_cpu,
        "log_scores": log_scores_cpu,
        "stats": merge_advantage_stats(advantages),
        "reward_stats": advantage_distribution_stats(rewards, prefix="reward"),
    }
    if scaled_rewards is not None:
        payload["scaled_rewards"] = scaled_rewards.detach().float().cpu().tolist()
        payload["scaled_reward_stats"] = advantage_distribution_stats(
            scaled_rewards, prefix="scaled_reward"
        )
    if extra:
        payload.update(extra)

    path = os.path.join(dump_dir, f"group_{global_step:04d}_e{epoch}_s{step}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def log_advantage_group(
    trainer,
    buffer,
    *,
    dump_dir: str | None,
    epoch: int,
    step: int,
    global_step: int,
    method: str,
) -> dict[str, float]:
    """Return stats dict; optionally write full per-group vectors to dump_dir."""
    stats = merge_advantage_stats(buffer.advantages)
    if dump_dir is None:
        return stats

    rewards = trainer.batch_rewards(buffer.log_scores)
    scaled_rewards = None
    extra: dict[str, Any] = {
        "advantage_metrics": buffer.advantage_metrics,
        "reward_mode": trainer.advantage_reward_mode,
    }
    if buffer.outcome_ids is not None:
        extra["outcome_ids"] = buffer.outcome_ids
        from grpo_experiments.ips_grpo.trainer import scale_rewards_ips

        if hasattr(trainer, "ips_prob_floor"):
            scaled_rewards, _, _ = scale_rewards_ips(
                buffer.log_scores,
                buffer.outcome_ids,
                trainer.ips_prob_floor,
                reward_c=trainer.reward_c,
                reward_scale=trainer.reward_scale,
                mode=trainer.advantage_reward_mode,
            )

    save_advantage_group_dump(
        dump_dir,
        epoch=epoch,
        step=step,
        global_step=global_step,
        method=method,
        advantages=buffer.advantages,
        rewards=rewards,
        log_scores=buffer.log_scores,
        scaled_rewards=scaled_rewards,
        reward_mode=trainer.advantage_reward_mode,
        extra=extra,
    )
    return stats
