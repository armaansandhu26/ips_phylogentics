"""Shared reward–probability plot reference for cross-method comparison."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REFERENCE_RUN_DIR = (
    REPO_ROOT
    / "grpo_experiments/learned_reverse_runs"
    / "20260730_160341_learned_reverse_5taxa_mlp_shifted_linear_b4096_learned_reverse_ips_grpo"
)


def resolve_reference_run_dir(path: Path | None) -> Path:
    candidate = DEFAULT_REFERENCE_RUN_DIR if path is None else path
    if not candidate.is_dir():
        raise FileNotFoundError(f"missing reference run directory: {candidate}")
    return candidate


def load_reference_spec(
    reference_run_dir: Path | None = None,
    *,
    samples_name: str = "sampled_full_diagnostics_1000000.npz",
    metrics_subdir: str = "plots/mlp_shifted_linear_reference_1000k",
) -> dict[str, float]:
    """Load a shared log partition and axis ranges from the reference run."""
    run_dir = resolve_reference_run_dir(reference_run_dir)
    metrics_path = run_dir / metrics_subdir / "comparison_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"missing reference metrics: {metrics_path}")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    samples_path = run_dir / samples_name
    if not samples_path.exists():
        raise FileNotFoundError(f"missing reference samples: {samples_path}")

    with np.load(samples_path) as payload:
        log_pf = payload["log_pf"].astype(np.float64)
        log_q_reverse = payload["log_q_reverse"].astype(np.float64)
        metadata_path = samples_path.with_suffix(".json")
        metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.exists()
            else {}
        )
        shift = float(metadata.get("log_score_shift", 3600.0))
        if "raw_log_likelihood" in payload:
            log_reward = payload["raw_log_likelihood"].astype(np.float64)
        else:
            log_score = payload["log_score"].astype(np.float64)
            log_reward = log_score - shift

    log_probability = log_pf - log_q_reverse
    reward = np.exp(log_reward)
    probability = np.exp(log_probability)

    reward_min = float(reward.min())
    reward_max = float(reward.max())
    log_reward_min = float(log_reward.min())
    log_reward_max = float(log_reward.max())
    probability_min = float(probability.min())
    probability_max = float(probability.max())
    log_probability_min = float(log_probability.min())
    log_probability_max = float(log_probability.max())

    reward_padding = 0.02 * (reward_max - reward_min)
    log_reward_padding = 0.02 * (log_reward_max - log_reward_min)
    probability_padding = max(probability_max * 0.05, probability_min * 0.5, 1e-18)
    log_probability_padding = 0.02 * (log_probability_max - log_probability_min)

    return {
        "reference_run_dir": str(run_dir),
        "reference_log_partition": float(metrics["estimated_log_partition"]),
        "reward_min": reward_min - reward_padding,
        "reward_max": reward_max + reward_padding,
        "log_reward_min": log_reward_min - log_reward_padding,
        "log_reward_max": log_reward_max + log_reward_padding,
        "probability_min": max(probability_min - probability_padding, 1e-20),
        "probability_max": probability_max + probability_padding,
        "log_probability_min": log_probability_min - log_probability_padding,
        "log_probability_max": log_probability_max + log_probability_padding,
    }


def ideal_line_label(line_log_partition: float, *, show_log_z: bool = False) -> str:
    if show_log_z:
        return (
            r"Ideal: $P(x)\propto R(x)$"
            f" ($\\log Z={line_log_partition:.3f}$)"
        )
    return r"Ideal: $P(x)\propto R(x)$"


def pearson_vs_ideal_sampling(
    log_model_probability: np.ndarray,
    log_target_reward: np.ndarray,
    line_log_partition: float,
) -> float:
    """Pearson r between model-implied log P(x) and ideal log P*(x)=log R(x)-log Z."""
    log_ideal = log_target_reward - line_log_partition
    if log_model_probability.size < 2:
        return float("nan")
    if np.std(log_model_probability) == 0.0 or np.std(log_ideal) == 0.0:
        return float("nan")
    return float(np.corrcoef(log_model_probability, log_ideal)[0, 1])


def pearson_vs_ideal_sampling_linear(
    log_model_probability: np.ndarray,
    log_target_reward: np.ndarray,
    line_log_partition: float,
) -> float:
    """Pearson r between model P(x) and ideal P*(x)=R(x)/Z."""
    model_prob = np.exp(log_model_probability.astype(np.float64))
    ideal_prob = np.exp((log_target_reward - line_log_partition).astype(np.float64))
    if model_prob.size < 2:
        return float("nan")
    if np.std(model_prob) == 0.0 or np.std(ideal_prob) == 0.0:
        return float("nan")
    return float(np.corrcoef(model_prob, ideal_prob)[0, 1])


def shared_reward_axis_bounds(
    reference_run_dir: Path | None = None,
    *,
    samples_name: str = "sampled_full_diagnostics_1000000.npz",
) -> dict[str, float]:
    """Reward-axis bounds for cross-method comparison plots."""
    spec = load_reference_spec(
        reference_run_dir,
        samples_name=samples_name,
    )
    return {
        "reference_run_dir": spec["reference_run_dir"],
        "reward_min": spec["reward_min"],
        "reward_max": spec["reward_max"],
        "log_reward_min": spec["log_reward_min"],
        "log_reward_max": spec["log_reward_max"],
    }


def merge_reward_axis_bounds(
    axis_spec: dict[str, float],
    *,
    reward: np.ndarray,
    log_reward: np.ndarray,
) -> dict[str, float]:
    """Expand shared reward bounds so the current sample is fully visible."""
    reward_min = min(axis_spec["reward_min"], float(reward.min()))
    reward_max = max(axis_spec["reward_max"], float(reward.max()))
    log_reward_min = min(axis_spec["log_reward_min"], float(log_reward.min()))
    log_reward_max = max(axis_spec["log_reward_max"], float(log_reward.max()))
    reward_padding = 0.02 * (reward_max - reward_min)
    log_reward_padding = 0.02 * (log_reward_max - log_reward_min)
    return {
        **axis_spec,
        "reward_min": reward_min - reward_padding,
        "reward_max": reward_max + reward_padding,
        "log_reward_min": log_reward_min - log_reward_padding,
        "log_reward_max": log_reward_max + log_reward_padding,
    }


def linear_probability_limits(
    model_probability: np.ndarray,
    line_x: np.ndarray,
    partition: float,
) -> tuple[float, float]:
    ideal_y = line_x / partition
    y_min = min(float(model_probability.min()), float(ideal_y.min()))
    y_max = max(float(model_probability.max()), float(ideal_y.max()))
    padding = 0.03 * (y_max - y_min if y_max > y_min else max(y_max, 1e-20))
    return y_min - padding, y_max + padding


def log_probability_limits(
    log_model_probability: np.ndarray,
    line_x: np.ndarray,
    line_log_partition: float,
) -> tuple[float, float]:
    ideal_y = line_x - line_log_partition
    y_min = min(float(log_model_probability.min()), float(ideal_y.min()))
    y_max = max(float(log_model_probability.max()), float(ideal_y.max()))
    padding = 0.03 * (y_max - y_min if y_max > y_min else 1.0)
    return y_min - padding, y_max + padding
