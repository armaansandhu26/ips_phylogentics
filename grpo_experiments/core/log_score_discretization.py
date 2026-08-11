"""Consistent log-score discretization for training and signature outcomes."""

from __future__ import annotations

from typing import Any

import torch

from src.env.binary_tree_env_one_step_likelihood import shaped_log_reward_from_log_score


def discretize_log_score(value: float, decimals: int | None) -> float:
    if decimals is None:
        return float(value)
    return round(float(value), int(decimals))


def discretize_log_scores(tensor: torch.Tensor, decimals: int | None) -> torch.Tensor:
    if decimals is None:
        return tensor
    factor = 10 ** int(decimals)
    return (tensor * factor).round() / factor


def reward_shaping_from_cfg(cfg) -> tuple[float, float]:
    reward_cfg = cfg.ENV.REWARD
    return float(reward_cfg.C), float(reward_cfg.SCALE)


def apply_log_score_discretization_to_tensors(
    log_scores: torch.Tensor,
    log_rewards: torch.Tensor | None,
    *,
    decimals: int | None,
    reward_c: float | None = None,
    reward_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if decimals is None:
        return log_scores, log_rewards
    log_scores = discretize_log_scores(log_scores, decimals)
    if log_rewards is not None and reward_c is not None and reward_scale is not None:
        # Keep TB log(log_reward) ablation consistent with PhyloTreeReward.
        log_rewards = shaped_log_reward_from_log_score(
            log_scores,
            reward_c=reward_c,
            reward_scale=reward_scale,
        )
    return log_scores, log_rewards


def sync_tree_log_scores(trees: list[Any], log_scores: torch.Tensor) -> None:
    for tree, score in zip(trees, log_scores):
        value = float(score.item())
        if hasattr(tree, "update_log_score"):
            tree.update_log_score(value)
        else:
            tree.log_score = value


def apply_experiment_log_score_discretization(obj: Any, exp_cfg: Any, cfg: Any) -> None:
    """Round batch/buffer log_scores (and log_rewards) when log_score_decimals is set."""
    decimals = getattr(exp_cfg, "log_score_decimals", None)
    if decimals is None:
        return

    reward_c, reward_scale = reward_shaping_from_cfg(cfg)

    if isinstance(obj, dict):
        scores, rewards = apply_log_score_discretization_to_tensors(
            obj["log_scores"],
            obj.get("log_rewards"),
            decimals=decimals,
            reward_c=reward_c,
            reward_scale=reward_scale,
        )
        obj["log_scores"] = scores
        if rewards is not None:
            obj["log_rewards"] = rewards
        return

    if not hasattr(obj, "log_scores"):
        return

    scores, rewards = apply_log_score_discretization_to_tensors(
        obj.log_scores,
        getattr(obj, "log_rewards", None),
        decimals=decimals,
        reward_c=reward_c,
        reward_scale=reward_scale,
    )
    obj.log_scores = scores
    if rewards is not None and hasattr(obj, "log_rewards"):
        obj.log_rewards = rewards
    if hasattr(obj, "trees"):
        sync_tree_log_scores(obj.trees, scores)
