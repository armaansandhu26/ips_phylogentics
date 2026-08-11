#!/usr/bin/env python3
"""GFlowNet training curves from TensorBoard logs (og_code train.py)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

REPO_ROOT = Path(__file__).resolve().parents[2]

BLUE = "#1976d2"
TEAL = "#00897b"
GRAY = "#555555"
GRID = "#e0e0e0"
RAW_ALPHA = 0.28

DEFAULT_RUN = (
    REPO_ROOT
    / "og_code/experiments/full_model/20260806_150835_phylgfn_logreward_27taxa_g1024_noreplay_shift12000"
)


def load_tb_series(run_dir: Path, tag: str) -> tuple[np.ndarray, np.ndarray]:
    tb_dir = run_dir / "tb_log"
    if not tb_dir.is_dir():
        raise FileNotFoundError(f"tensorboard log dir not found: {tb_dir}")
    ea = EventAccumulator(str(tb_dir))
    ea.Reload()
    scalars = ea.Tags().get("scalars", [])
    if tag not in scalars:
        raise KeyError(f"tag {tag!r} not in tensorboard scalars: {scalars}")
    events = ea.Scalars(tag)
    steps = np.array([event.step for event in events], dtype=np.int64)
    values = np.array([event.value for event in events], dtype=np.float64)
    return steps, values


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


def plot_training_panel(
    run_dir: Path,
    *,
    title: str,
    output: Path,
) -> dict[str, float | int | str]:
    epoch, loss = load_tb_series(run_dir, "/loss")
    _, log_z = load_tb_series(run_dir, "/log_partition")

    smooth = max(25, int(epoch.size // 200))

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), dpi=180, constrained_layout=True)
    fig.suptitle(title, fontsize=13, y=1.04)
    _add_smoothing_legend(fig)

    ax = axes[0]
    _plot_line(ax, epoch, loss, color=BLUE, label=None, logy=True, smooth_window=smooth)
    _style_axis(ax, "epoch", "TB loss")

    ax = axes[1]
    _plot_line(ax, epoch, log_z, color=TEAL, label=None, smooth_window=smooth)
    _style_axis(ax, "epoch", "log partition (log Z)")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)

    return {
        "run_dir": str(run_dir),
        "epochs": int(epoch[-1]),
        "final_loss": float(loss[-1]),
        "final_log_partition": float(log_z[-1]),
        "output": str(output),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT
        / "grpo_experiments/comparisons/27taxa/gflownet_noreplay_training_curves.png",
    )
    parser.add_argument(
        "--title",
        default="PhyloGFN training — 27 taxa, no replay (B=1024, 32k epochs)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = plot_training_panel(
        args.run_dir.resolve(),
        title=args.title,
        output=args.output,
    )
    meta_path = args.output.with_suffix(".json")
    meta_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    print(f"wrote {meta_path}")


if __name__ == "__main__":
    main()
