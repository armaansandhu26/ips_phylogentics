#!/usr/bin/env python3
"""Paper-style outcome probability versus reward for GRPO / count IPS samples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
import plot_5taxa_sampling_comparison_grid as grid  # noqa: E402

BLUE = "#1976d2"
GRAY = "#555555"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Count IPS-GRPO")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scatter-points", type=int, default=0)
    return parser.parse_args()


def load_empirical_signature_frequency(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    with np.load(path) as payload:
        reward = payload["log_score"].astype(np.float64)
        topology_index = payload["topology_index"].astype(np.int16)

    rounded_reward = np.rint(reward * 1000.0).astype(np.int64)
    signatures = np.empty(
        len(reward),
        dtype=[("topology", np.int16), ("reward_milli", np.int64)],
    )
    signatures["topology"] = topology_index
    signatures["reward_milli"] = rounded_reward
    _, first_indices, counts = np.unique(
        signatures,
        return_index=True,
        return_counts=True,
    )
    frequency = counts.astype(np.float64) / len(reward)
    return reward[first_indices], frequency, len(counts), len(reward)


def linear_fit(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    centered_x = x - x_mean
    centered_y = y - y_mean
    slope = float(np.dot(centered_x, centered_y) / np.dot(centered_x, centered_x))
    intercept = y_mean - slope * x_mean
    fitted = slope * x + intercept
    residual_sum_squares = float(np.sum(np.square(y - fitted)))
    total_sum_squares = float(np.sum(np.square(centered_y)))
    r_squared = (
        1.0 - residual_sum_squares / total_sum_squares
        if total_sum_squares > 0.0
        else float("nan")
    )
    pearson = (
        float(np.corrcoef(x, y)[0, 1])
        if np.std(x) > 0.0 and np.std(y) > 0.0
        else float("nan")
    )
    return {
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_squared,
        "pearson_r": pearson,
    }


def main() -> None:
    args = parse_args()
    reward, probability, unique_signatures, sample_count = (
        load_empirical_signature_frequency(args.samples)
    )
    fit = linear_fit(reward, probability) if unique_signatures >= 2 else None

    reward_min = float(reward.min())
    reward_max = float(reward.max())
    reward_padding = 0.02 * (reward_max - reward_min)
    reward_limits = (reward_min - reward_padding, reward_max + reward_padding)

    selected = grid.select_points(len(reward), args.scatter_points, args.seed)
    marker_size, alpha = grid.scatter_style(len(selected))

    fig, ax = plt.subplots(figsize=(10, 10), dpi=220, constrained_layout=True)
    ax.scatter(
        reward[selected],
        probability[selected],
        s=marker_size,
        alpha=alpha,
        color=BLUE,
        edgecolors="none",
        rasterized=True,
        label=(
            r"Empirical $\hat q(x)=\mathrm{count}(x)/N$"
            f"\n{sample_count:,} sampled trajectories"
            f"\n{unique_signatures:,} unique signatures"
        ),
    )

    line_y = np.array([], dtype=np.float64)
    if fit is not None:
        line_x = np.linspace(reward_min, reward_max, 200)
        line_y = fit["slope"] * line_x + fit["intercept"]
        ax.plot(
            line_x,
            line_y,
            color=GRAY,
            linestyle="--",
            linewidth=1.5,
            label=(
                r"OLS best fit: $\hat P=aR+b$"
                f"\n$a={fit['slope']:.3e}$, $b={fit['intercept']:.3e}$"
                f"\n$R^2={fit['r_squared']:.4f}$"
            ),
        )

    ax.set_xlim(*reward_limits)
    y_min = float(probability.min())
    y_max = float(probability.max())
    if line_y.size:
        y_min = min(y_min, float(line_y.min()))
        y_max = max(y_max, float(line_y.max()))
    padding = 0.03 * (y_max - y_min if y_max > y_min else max(abs(y_max), 1e-20))
    ax.set_ylim(0.0, y_max + padding)
    ax.set_xlabel(r"Terminal reward: $R(x)=3600+\log L(x)$")
    ax.set_ylabel(r"Outcome probability: $\hat q(x)=\mathrm{count}(x)/N$")
    ax.set_title(args.title)
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False, loc="best", fontsize=9)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)

    metrics = {
        "samples": sample_count,
        "unique_signatures": unique_signatures,
        "quantity": "empirical_signature_frequency",
        "reward_min": reward_min,
        "reward_max": reward_max,
        "fit": fit,
        "samples_file": str(args.samples),
        "plot": str(args.output),
    }
    metrics_path = args.output.with_suffix(".json")
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    comparison_metrics = {
        "samples": sample_count,
        "unique_observed_signatures": unique_signatures,
        "empirical_fit_r_squared": None if fit is None else fit["r_squared"],
        "empirical_fit_pearson_r": None if fit is None else fit["pearson_r"],
        "reward_min": reward_min,
        "reward_max": reward_max,
        "plot_type": "empirical_signature_frequency",
    }
    comparison_path = args.output.parent / "comparison_metrics.json"
    comparison_path.write_text(
        json.dumps(comparison_metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output.resolve()}")
    print(f"wrote {metrics_path.resolve()}")
    print(f"wrote {comparison_path.resolve()}")


if __name__ == "__main__":
    main()
