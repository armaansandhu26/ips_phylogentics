#!/usr/bin/env python3
"""Side-by-side 100k early comparison for 10-taxa learned-reverse vs GFlowNet."""

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
from reward_probability_plot_reference import (  # noqa: E402
    merge_reward_axis_bounds,
    shared_reward_axis_bounds,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
GRAY = "#777777"
BLUE = "#1976d2"
LOG_SCORE_SHIFT = 5000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        default=REPO_ROOT
        / "grpo_experiments/comparisons/10taxa/early_epoch1000_100k",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Default: <comparison-dir>/sampling_comparison_grid.png",
    )
    parser.add_argument("--checkpoint-epoch", type=int, default=1000)
    return parser.parse_args()


def scatter_style(displayed_points: int) -> tuple[float, float]:
    if displayed_points >= 500_000:
        return 8.0, 0.10
    if displayed_points >= 100_000:
        return 12.0, 0.15
    return 20.0, 0.30


def panel_log_partition(metrics: dict) -> float:
    return float(
        metrics.get(
            "importance_estimated_log_partition",
            metrics.get(
                "estimated_log_partition",
                metrics.get("checkpoint_log_partition", 0.0),
            ),
        )
    )


def calibrated_axis_limits(
    reward: np.ndarray,
    calibrated_probability: np.ndarray,
) -> tuple[float, float]:
    """Match learned_reverse/partition_calibrated_model_probability_vs_reward."""
    lower = min(float(reward.min()), float(calibrated_probability.min()))
    upper = max(float(reward.max()), float(calibrated_probability.max()))
    padding = 0.03 * (upper - lower)
    return lower - padding, upper + padding


def pearson_calibrated_vs_reward(
    calibrated_probability: np.ndarray,
    reward: np.ndarray,
) -> float:
    if calibrated_probability.size < 2:
        return float("nan")
    if np.std(calibrated_probability) == 0.0 or np.std(reward) == 0.0:
        return float("nan")
    return float(np.corrcoef(calibrated_probability, reward)[0, 1])


def load_panel(comparison_dir: Path, name: str) -> dict:
    plot_dir = comparison_dir / name
    metrics_path = plot_dir / "comparison_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"missing metrics: {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if name == "learned_reverse":
        samples_path = comparison_dir / "learned_reverse_samples_100k.npz"
        with np.load(samples_path) as payload:
            log_score = payload["log_score"].astype(np.float64)
            log_pf = payload["log_pf"].astype(np.float64)
            log_q_reverse = payload["log_q_reverse"].astype(np.float64)
        log_reward = np.log(np.clip(log_score, 1e-8, None))
        log_probability = log_pf - log_q_reverse
        estimator = r"$\log P_F(\tau)-\log q_\phi(\tau|x)$"
    else:
        samples_path = plot_dir / "og_gflownet_reward_probability_samples.npz"
        with np.load(samples_path) as payload:
            log_reward = payload["log_reward"].astype(np.float64)
            log_probability = payload["log_probability"].astype(np.float64)
        estimator = r"$\log P_F(\tau)-\log P_B(\tau)$"
    return {
        "title": "Learned reverse IPS" if name == "learned_reverse" else "GFlowNet",
        "log_reward": log_reward,
        "log_probability": log_probability,
        "metrics": metrics,
        "estimator_label": estimator,
    }


def learned_reverse_axis_spec(log_reward: np.ndarray, reward: np.ndarray) -> dict[str, float]:
    axis_spec = shared_reward_axis_bounds()
    return merge_reward_axis_bounds(
        axis_spec,
        reward=reward,
        log_reward=log_reward,
    )


def main() -> None:
    args = parse_args()
    comparison_dir = args.comparison_dir.resolve()
    output_path = args.output or (comparison_dir / "sampling_comparison_grid.png")

    panels = [
        load_panel(comparison_dir, "learned_reverse"),
        load_panel(comparison_dir, "gflownet"),
    ]
    lr_panel = panels[0]
    lr_reward = np.exp(lr_panel["log_reward"])
    axis_spec = learned_reverse_axis_spec(lr_panel["log_reward"], lr_reward)

    lr_log_partition = panel_log_partition(lr_panel["metrics"])
    lr_calibrated = np.exp(lr_panel["log_probability"] + lr_log_partition)
    shared_lower, shared_upper = calibrated_axis_limits(lr_reward, lr_calibrated)
    ideal_line = np.linspace(shared_lower, shared_upper, 200)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 7),
        dpi=220,
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    marker_size, alpha = scatter_style(len(lr_panel["log_probability"]))

    for ax, panel in zip(axes, panels):
        log_reward = panel["log_reward"]
        log_probability = panel["log_probability"]
        reward = np.exp(log_reward)
        line_log_partition = panel_log_partition(panel["metrics"])
        calibrated_probability = np.exp(log_probability + line_log_partition)
        pearson = pearson_calibrated_vs_reward(calibrated_probability, reward)
        unique_signatures = int(panel["metrics"]["unique_observed_signatures"])
        observed_topologies = int(panel["metrics"]["observed_topologies"])
        ax.scatter(
            reward,
            calibrated_probability,
            color=BLUE,
            s=marker_size,
            alpha=alpha,
            edgecolors="none",
            rasterized=True,
            label=(
                f"{panel['title']}: {panel['estimator_label']}\n"
                f"{len(log_reward):,} trajectories\n"
                f"{unique_signatures:,} unique signatures; "
                f"{observed_topologies:,} unique topologies\n"
                f"Pearson r vs ideal={pearson:.4f}"
            ),
        )
        ax.plot(
            ideal_line,
            ideal_line,
            color=GRAY,
            linestyle="--",
            linewidth=1.3,
            label="ideal: reward = calibrated model probability",
        )
        ax.set_xlim(shared_lower, shared_upper)
        ax.set_title(panel["title"])
        ax.set_xlabel(rf"Terminal reward: $R(x)={LOG_SCORE_SHIFT:g}+\log L(x)$")
        ax.grid(True, alpha=0.2)
        ax.legend(frameon=False, loc="upper left", fontsize=9)

    axes[0].set_ylim(shared_lower, shared_upper)
    axes[0].set_ylabel(r"Partition-calibrated terminal probability: $\hat ZP(x)$")
    fig.suptitle(
        f"10-taxa early comparison (epoch {args.checkpoint_epoch} checkpoint, "
        "100k samples)",
        fontsize=14,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "comparison_dir": str(comparison_dir),
        "output": str(output_path),
        "plot_kind": "partition_calibrated_probability_vs_reward",
        "shared_axis": {
            "source": "learned_reverse/partition_calibrated_model_probability_vs_reward",
            "min": shared_lower,
            "max": shared_upper,
            "log_partition": lr_log_partition,
        },
        "panels": {
            panel["title"]: panel["metrics"] for panel in panels
        },
    }
    summary_path = comparison_dir / "sampling_comparison_grid.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
