#!/usr/bin/env python3
"""Learned-reverse IPS-GRPO training curves from metrics.jsonl."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from taxa_comparison_utils import load_metrics_rows  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]

BLUE = "#1976d2"
TEAL = "#00897b"
GRAY = "#555555"
GRID = "#e0e0e0"
RAW_ALPHA = 0.28
MARKER_EPOCH = 10_000

DEFAULT_RUNS = {
    "5 taxa": REPO_ROOT
    / "grpo_experiments/learned_reverse_runs/20260730_160341_learned_reverse_5taxa_mlp_shifted_linear_b4096_learned_reverse_ips_grpo",
    "10 taxa": REPO_ROOT
    / "grpo_experiments/learned_reverse_runs/20260803_124837_learned_reverse_10taxa_mlp_shifted_linear_b4096_learned_reverse_ips_grpo",
    "27 taxa": REPO_ROOT
    / "grpo_experiments/learned_reverse_runs/20260806_144004_learned_reverse_27taxa_mlp_shifted_linear_b1024_rlr1e-3_rev8x_learned_reverse_ips_grpo",
}


def aggregate_by_epoch(rows: list[dict]) -> dict[str, np.ndarray]:
    buckets: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        epoch = int(row["epoch"])
        for key, value in row.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                buckets[epoch][key].append(float(value))

    epochs = np.array(sorted(buckets), dtype=np.int64)
    out: dict[str, np.ndarray] = {"epoch": epochs}
    metric_keys = {
        key
        for epoch_rows in buckets.values()
        for key in epoch_rows
    }
    for key in sorted(metric_keys):
        out[key] = np.array(
            [float(np.mean(buckets[e][key])) for e in epochs],
            dtype=np.float64,
        )
    return out


def smooth_series(y: np.ndarray, *, window: int) -> np.ndarray:
    if y.size <= 2 or window <= 1:
        return y
    window = min(window, max(3, y.size // 20))
    if window <= 1:
        return y
    kernel = np.ones(window, dtype=np.float64) / window
    half = window // 2
    pad_right = window - 1 - half
    padded = np.pad(y, (half, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _plot_line(
    ax,
    x,
    y,
    *,
    color,
    label: str | None,
    logy: bool = False,
    alpha: float = 1.0,
    smooth_window: int = 0,
    show_raw: bool = True,
) -> None:
    if y.size == 0:
        return

    smooth_kwargs = {"color": color, "linewidth": 1.8, "alpha": alpha, "zorder": 2}
    if label is not None:
        smooth_kwargs["label"] = label

    if logy:
        y_raw = np.clip(y, 1e-12, None)
        plot_fn = ax.semilogy
    else:
        y_raw = y
        plot_fn = ax.plot

    if smooth_window > 0:
        if show_raw:
            plot_fn(
                x,
                y_raw,
                color=color,
                linewidth=0.7,
                alpha=RAW_ALPHA,
                zorder=1,
            )
        y_smooth = smooth_series(y, window=smooth_window)
        if logy:
            y_smooth = np.clip(y_smooth, 1e-12, None)
        plot_fn(x, y_smooth, **smooth_kwargs)
    else:
        plot_fn(x, y_raw, **smooth_kwargs)


def _add_smoothing_legend(fig) -> None:
    handles = [
        Line2D([0], [0], color=GRAY, linewidth=0.7, alpha=RAW_ALPHA, label="per-epoch"),
        Line2D([0], [0], color=GRAY, linewidth=1.8, label="smoothed"),
    ]
    fig.legend(
        handles=handles,
        loc="upper right",
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.99, 1.0),
    )


def _style_axis(ax, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.7)
    ax.tick_params(labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _maybe_mark_epoch(ax, epoch_max: int, *, mark_epoch: int | None) -> None:
    if mark_epoch is None or mark_epoch <= 0 or mark_epoch >= epoch_max:
        return
    ax.axvline(
        mark_epoch,
        color=GRAY,
        linestyle="--",
        linewidth=1.0,
        alpha=0.7,
        zorder=0,
    )


def _plot_core_panels(
    axes,
    epoch: np.ndarray,
    data: dict[str, np.ndarray],
    *,
    smooth: int,
    mark_epoch: int | None = None,
    panel_titles: tuple[str, str, str] | None = None,
) -> None:
    specs = [
        ("reverse_loss", "reverse NLL", True, TEAL),
        ("ips_ess_fraction", "IPS ESS fraction", False, BLUE),
        ("mean_log_score", "mean log score", False, BLUE),
    ]
    for col, (key, ylabel, logy, color) in enumerate(specs):
        ax = axes[col]
        if panel_titles is not None:
            ax.set_title(panel_titles[col], fontsize=10)
        _plot_line(
            ax,
            epoch,
            data[key],
            color=color,
            label=None,
            logy=logy,
            smooth_window=smooth,
        )
        if key == "ips_ess_fraction":
            ax.set_ylim(0.0, 1.02)
        _maybe_mark_epoch(ax, int(epoch[-1]), mark_epoch=mark_epoch)
        _style_axis(ax, "epoch", ylabel)


def plot_training_panel(
    run_dir: Path,
    *,
    title: str,
    output: Path,
    mark_epoch: int | None = None,
) -> dict[str, float | int | str]:
    rows = load_metrics_rows(run_dir / "metrics.jsonl")
    if not rows:
        raise FileNotFoundError(f"no metrics rows in {run_dir / 'metrics.jsonl'}")

    data = aggregate_by_epoch(rows)
    epoch = data["epoch"]
    smooth = max(25, int(epoch.size // 200))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), dpi=180, constrained_layout=True)
    fig.suptitle(title, fontsize=13, y=1.04)
    _add_smoothing_legend(fig)
    _plot_core_panels(axes, epoch, data, smooth=smooth, mark_epoch=mark_epoch)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)

    return {
        "run_dir": str(run_dir),
        "epochs": int(epoch[-1]) + 1,
        "final_mean_log_score": float(data["mean_log_score"][-1]),
        "final_reverse_loss": float(data["reverse_loss"][-1]),
        "final_ips_ess_fraction": float(data["ips_ess_fraction"][-1]),
        "output": str(output),
    }


def plot_combined_grid(
    specs: list[tuple[str, Path, int | None]],
    *,
    output: Path,
) -> None:
    n = len(specs)
    fig, axes = plt.subplots(n, 3, figsize=(13.5, 3.2 * n), dpi=180, constrained_layout=True)
    if n == 1:
        axes = np.array([axes])
    _add_smoothing_legend(fig)

    for row_idx, (title, run_dir, mark_epoch) in enumerate(specs):
        rows = load_metrics_rows(run_dir / "metrics.jsonl")
        data = aggregate_by_epoch(rows)
        epoch = data["epoch"]
        smooth = max(25, int(epoch.size // 200))
        row_axes = axes[row_idx]
        panel_titles = (
            f"{title} — reverse NLL",
            f"{title} — IPS ESS",
            f"{title} — mean log score",
        )
        _plot_core_panels(
            row_axes,
            epoch,
            data,
            smooth=smooth,
            mark_epoch=mark_epoch,
            panel_titles=panel_titles,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--combined-output",
        type=Path,
        default=REPO_ROOT
        / "grpo_experiments/comparisons/learned_reverse_training_curves_5_10_27.png",
    )
    parser.add_argument("--run-5", type=Path, default=DEFAULT_RUNS["5 taxa"])
    parser.add_argument("--run-10", type=Path, default=DEFAULT_RUNS["10 taxa"])
    parser.add_argument("--run-27", type=Path, default=DEFAULT_RUNS["27 taxa"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_map = {
        "5 taxa": (args.run_5, None),
        "10 taxa": (args.run_10, None),
        "27 taxa": (args.run_27, MARKER_EPOCH),
    }
    outputs = {
        "5 taxa": REPO_ROOT / "grpo_experiments/comparisons/5taxa/learned_reverse_training_curves.png",
        "10 taxa": REPO_ROOT
        / "grpo_experiments/comparisons/10taxa/learned_reverse_training_curves.png",
        "27 taxa": REPO_ROOT
        / "grpo_experiments/comparisons/27taxa/learned_reverse_noreplay_training_curves.png",
    }
    titles = {
        "5 taxa": "Learned-reverse IPS-GRPO — 5 taxa (B=4096, 10k epochs)",
        "10 taxa": "Learned-reverse IPS-GRPO — 10 taxa (B=4096, 10k epochs)",
        "27 taxa": "Learned-reverse IPS-GRPO — 27 taxa, no replay (B=1024, 32k epochs)",
    }

    summaries = []
    combined_specs: list[tuple[str, Path, int | None]] = []
    for title, (run_dir, mark_epoch) in run_map.items():
        out = outputs[title]
        summary = plot_training_panel(
            run_dir,
            title=titles[title],
            output=out,
            mark_epoch=mark_epoch,
        )
        summaries.append(summary)
        combined_specs.append((title, run_dir, mark_epoch))
        print(f"wrote {out}")

    plot_combined_grid(combined_specs, output=args.combined_output)
    print(f"wrote {args.combined_output}")

    meta_path = args.combined_output.with_suffix(".json")
    meta_path.write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {meta_path}")


if __name__ == "__main__":
    main()
