"""Dual linear / log-scale metrics and step prints for GRPO experiments."""

from __future__ import annotations

import math
from typing import Any

# exp underflows to 0 below this (float64).
_LOG_EXP_MIN = -745.0


def safe_exp(log_value: float) -> float:
    if not math.isfinite(log_value) or log_value <= _LOG_EXP_MIN:
        return 0.0
    return math.exp(log_value)


def safe_log(linear_value: float, *, floor: float = 1e-300) -> float:
    if not math.isfinite(linear_value) or linear_value <= 0.0:
        return float("-inf")
    return math.log(max(linear_value, floor))


def enrich_dual_scale_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Add linear partners for log-scale fields and log partners for positive linear fields."""
    out = dict(metrics)

    log_to_linear = {
        "mean_log_pf": "mean_pf_traj",
        "mean_log_pf_old": "mean_pf_traj_old",
        "mean_step_logprob": "mean_pf_step",
        "mean_step_logprob_old": "mean_pf_step_old",
        "mean_log_reward": "mean_reward",
        "mean_log_reward_behavior": "mean_reward_behavior",
    }
    for log_key, lin_key in log_to_linear.items():
        if log_key not in out:
            continue
        v = out[log_key]
        if isinstance(v, (int, float)) and math.isfinite(v):
            out[lin_key] = safe_exp(float(v))

    # Prefer direct linear IS stats when present; add log partners.
    for lin_key, log_key in (
        ("mean_importance_ratio", "mean_log_importance_ratio"),
        ("min_importance_ratio", "log_min_importance_ratio"),
        ("max_importance_ratio", "log_max_importance_ratio"),
    ):
        if lin_key not in out:
            continue
        v = float(out[lin_key])
        if math.isfinite(v) and v > 0.0 and log_key not in out:
            out[log_key] = safe_log(v)

    if "ips_scaled_reward_mean" in out:
        r = float(out["ips_scaled_reward_mean"])
        if math.isfinite(r) and r > 0.0:
            out["log_ips_scaled_reward_mean"] = safe_log(r)
    if "ips_prob_mean" in out:
        p = float(out["ips_prob_mean"])
        if math.isfinite(p) and p > 0.0:
            out["log_ips_prob_mean"] = safe_log(p)

    return out


def _pair(log_key: str, lin_key: str, metrics: dict[str, Any], label: str) -> str | None:
    if log_key not in metrics and lin_key not in metrics:
        return None
    log_v = metrics.get(log_key)
    if log_v is None and lin_key in metrics:
        lin_v = float(metrics[lin_key])
        log_v = safe_log(lin_v) if lin_v > 0 else float("-inf")
    else:
        log_v = float(log_v) if log_v is not None else float("nan")
    lin_v = metrics.get(lin_key)
    if lin_v is None:
        lin_v = safe_exp(log_v)
    else:
        lin_v = float(lin_v)
    if not math.isfinite(log_v):
        return f"{label}_log=nan {label}_lin={lin_v:.3e}"
    return f"{label}_log={log_v:.2f} {label}_lin={lin_v:.3e}"


def _scalar(name: str, value: float, fmt: str = ".4f") -> str:
    if not math.isfinite(value):
        return f"{name}=nan"
    return f"{name}={value:{fmt}}"


def format_hybrid_step_line(
    *,
    global_step: int,
    resample_round: int,
    cycle: int,
    train_info: dict[str, Any],
    mean_log_reward: float,
    replay_fraction: float,
    added: int,
    found_in_replay_buffer: int,
    replay_replaced: int,
    div: dict[str, Any],
    entropy_coef: float,
    ips: bool = False,
) -> str:
    """One training step line with linear and log counterparts where defined."""
    m = enrich_dual_scale_metrics({**train_info, "mean_log_reward": mean_log_reward})

    head = (
        f"step={global_step:04d} round={resample_round} cycle={cycle} "
        f"{_scalar('loss', float(train_info.get('loss', float('nan'))))} "
        f"{_scalar('pg_loss', float(train_info.get('pg_loss', float('nan'))))} "
        f"{_scalar('ent_loss', float(train_info.get('entropy_loss', float('nan'))))}"
    )

    policy_parts = [
        _pair("mean_log_pf", "mean_pf_traj", m, "pi_new_traj"),
        _pair("mean_log_pf_old", "mean_pf_traj_old", m, "pi_old_traj"),
        _pair("mean_step_logprob", "mean_pf_step", m, "pi_new_step"),
        _pair("mean_step_logprob_old", "mean_pf_step_old", m, "pi_old_step"),
    ]
    policy = " ".join(p for p in policy_parts if p)

    reward = _pair("mean_log_reward", "mean_reward", m, "R")

    is_parts = []
    if "mean_importance_ratio" in m or "mean_log_importance_ratio" in m:
        w_lin = float(m.get("mean_importance_ratio", safe_exp(float(m.get("mean_log_importance_ratio", 0)))))
        w_log = float(m.get("mean_log_importance_ratio", safe_log(w_lin) if w_lin > 0 else float("-inf")))
        w_min = float(m.get("min_importance_ratio", w_lin))
        w_max = float(m.get("max_importance_ratio", w_lin))
        is_parts.append(
            f"w_log={w_log:.3f} w_lin={w_lin:.3f} "
            f"w_min_lin={w_min:.3f} w_max_lin={w_max:.3f} "
            f"logw_min={safe_log(w_min) if w_min > 0 else float('-inf'):.3f} "
            f"logw_max={safe_log(w_max) if w_max > 0 else float('-inf'):.3f}"
        )
    is_str = " ".join(is_parts)

    ent = (
        f"H={float(train_info.get('mean_policy_entropy', 0)):.3f} "
        f"ent_coef={entropy_coef:g}"
    )

    adv = (
        f"adv_lin={float(train_info.get('mean_advantage', 0)):.3f} "
        f"adv_std={float(train_info.get('std_advantage', 0)):.3f}"
    )

    ips_str = ""
    if ips:
        p_hat = float(train_info.get("ips_prob_mean", 0))
        r_tilde = float(train_info.get("ips_scaled_reward_mean", 0))
        ips_str = (
            f"p_hat_lin={p_hat:.3f} "
            f"log_p_hat={safe_log(p_hat) if p_hat > 0 else float('-inf'):.3f} "
            f"r_tilde_lin={r_tilde:.3f} "
            f"log_r_tilde={safe_log(r_tilde) if r_tilde > 0 else float('-inf'):.3f}"
        )

    meta = (
        f"replay_frac={replay_fraction:.2f} added={added} found={found_in_replay_buffer} "
        f"replaced={replay_replaced} dup_batch={div.get('batch_duplicate_fraction', 0):.3f} "
        f"dup_global={div.get('global_duplicate_fraction', 0):.3f}"
    )

    return " | ".join(
        x for x in (head, policy, reward, is_str, ent, adv, ips_str, meta) if x
    )
