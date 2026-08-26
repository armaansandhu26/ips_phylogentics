"""Batch-level statistics for enriched step logging."""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any, Sequence

import numpy as np
import torch

from final.logging.schema import LOG_W_BIN_EDGES


def signature_hash(signature: str) -> str:
    return hashlib.sha256(signature.encode("utf-8")).hexdigest()[:16]


def tensor_stats(x: torch.Tensor) -> dict[str, float]:
    x = x.detach().float().reshape(-1)
    if x.numel() == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
    }


def traj_lengths(log_paths_pf: torch.Tensor) -> dict[str, float]:
    lengths = log_paths_pf.detach().float().ne(0).sum(dim=-1).float()
    return {
        "traj_len_mean": float(lengths.mean().item()),
        "traj_len_std": float(lengths.std(unbiased=False).item()),
    }


def pf_entropy_from_paths(log_paths_pf: torch.Tensor) -> float:
    """Mean per-step entropy proxy: -mean(log p_f per token)."""
    values = -log_paths_pf.detach().float()
    return float(values.mean().item())


def log_w_histogram(log_w: torch.Tensor | np.ndarray) -> list[int]:
    arr = np.asarray(
        log_w.detach().cpu().numpy() if torch.is_tensor(log_w) else log_w,
        dtype=np.float64,
    ).reshape(-1)
    counts, _ = np.histogram(arr, bins=LOG_W_BIN_EDGES)
    return counts.astype(int).tolist()


def psis_khat(log_w: torch.Tensor | np.ndarray) -> float:
    """Pareto-smoothed importance sampling k-hat diagnostic (Vehtari et al.)."""
    arr = np.asarray(
        log_w.detach().cpu().numpy() if torch.is_tensor(log_w) else log_w,
        dtype=np.float64,
    ).reshape(-1)
    if arr.size < 2:
        return float("nan")
    lw = arr - arr.max()
    w = np.exp(lw)
    w /= w.sum()
    sorted_w = np.sort(w)[::-1]
    cum = np.cumsum(sorted_w)
    tail = 1.0 - cum
    ratios = tail / sorted_w
    finite = ratios[np.isfinite(ratios)]
    if finite.size == 0:
        return float("nan")
    return float(min(0.5, np.max(finite)))


def grpo_group_stats(
    outcome_ids: Sequence[str],
    rewards: torch.Tensor,
    *,
    group_size: int,
) -> dict[str, float]:
    ids = list(outcome_ids)
    rewards_list = rewards.detach().cpu().tolist()
    n = len(ids)
    if n == 0 or group_size <= 0:
        return {
            "group_reward_std": float("nan"),
            "distinct_terminals_per_group": float("nan"),
        }
    group_stds: list[float] = []
    distinct_counts: list[int] = []
    for start in range(0, n, group_size):
        chunk_ids = ids[start : start + group_size]
        chunk_rewards = rewards_list[start : start + group_size]
        if not chunk_ids:
            continue
        distinct_counts.append(len(set(chunk_ids)))
        if len(chunk_rewards) > 1:
            group_stds.append(float(np.std(chunk_rewards)))
    return {
        "group_reward_std": float(np.mean(group_stds)) if group_stds else 0.0,
        "distinct_terminals_per_group": float(np.mean(distinct_counts)),
    }


def count_ips_terminal_stats(outcome_ids: Sequence[str], counts: Counter[str]) -> dict[str, float]:
    batch_counts = [counts[oid] for oid in outcome_ids]
    p_hats = [1.0 / max(c, 1) for c in batch_counts]
    return {
        "n_terminals_with_count_ge_2": float(sum(1 for c in batch_counts if c >= 2)),
        "p_hat_mean": float(np.mean(p_hats)) if p_hats else float("nan"),
        "p_hat_min": float(np.min(p_hats)) if p_hats else float("nan"),
        "p_hat_max": float(np.max(p_hats)) if p_hats else float("nan"),
    }


def enrich_common_record(
    record: dict[str, Any],
    *,
    log_rewards: torch.Tensor,
    log_paths_pf: torch.Tensor,
    outcome_ids: Sequence[str],
    lr: float,
    wall_clock_s: float,
    gpu_seconds: float,
) -> dict[str, Any]:
    stats = tensor_stats(log_rewards)
    record.update(
        {
            "wall_clock_s": wall_clock_s,
            "gpu_seconds": gpu_seconds,
            "batch_size": int(log_rewards.numel()),
            "log_R_mean": stats["mean"],
            "log_R_std": stats["std"],
            "log_R_min": stats["min"],
            "log_R_max": stats["max"],
            "unique_terminals_in_batch": len(set(outcome_ids)),
            "pf_entropy": pf_entropy_from_paths(log_paths_pf),
            "lr": lr,
            **traj_lengths(log_paths_pf),
        }
    )
    if "mean_log_reward" in record and "log_R_mean" not in record:
        record["log_R_mean"] = record["mean_log_reward"]
    return record
