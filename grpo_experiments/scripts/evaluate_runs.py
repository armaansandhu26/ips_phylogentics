#!/usr/bin/env python3
"""Plot training metrics and diversity curves for one or more experiment runs."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from grpo_experiments.eval_utils import (
    auto_smoothing_window,
    load_manifest,
    load_metrics,
    manifest_run_entries,
    moving_average,
    percentile_ylim,
    raw_series,
    resolve_run_artifacts,
    rolling_finite_mean,
    rolling_nonfinite_fraction,
    rolling_quantiles,
    sample_series,
    save_json,
    subsample_xy,
)


@dataclass(frozen=True)
class PanelSpec:
    key: str
    title: str
    kind: str = "line"
    y_clip: float | None = None


# Curated panels: avoid redundant dup-topology plots; emphasize interpretable summaries.
PANEL_SPECS = [
    PanelSpec("loss", "Loss (rolling median & IQR)", kind="loss_band"),
    PanelSpec("loss", "|Loss| (rolling mean)", kind="loss_abs"),
    PanelSpec("mean_log_reward", "Mean log reward", kind="reward"),
    PanelSpec("batch_unique_topologies", "Batch unique topologies", kind="batch_unique"),
    PanelSpec("global_duplicate_fraction", "Global duplicate fraction", kind="dup_fraction"),
    PanelSpec("cumulative_unique_outcomes", "Cumulative unique outcomes", kind="unique_count"),
    PanelSpec(
        "mean_importance_ratio",
        "IS weight mean (finite, capped)",
        kind="importance_finite",
        y_clip=10.0,
    ),
    PanelSpec(
        "mean_importance_ratio",
        "IS non-finite fraction",
        kind="importance_bad_frac",
    ),
]

OPTIONAL_PANELS = [
    PanelSpec("grad_norm", "Grad norm", kind="log_line"),
    PanelSpec("param_norm", "Param norm", kind="log_line"),
    PanelSpec("ips_prob_mean", "IPS mean p_hat", kind="line"),
    PanelSpec("ips_scaled_reward_mean", "IPS scaled reward mean", kind="line"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate training runs from metrics.jsonl and plot diversity curves.",
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--manifest",
        type=Path,
        help="Matrix manifest JSON written by run_sanity_matrix.sh",
    )
    src.add_argument(
        "--run-dirs",
        nargs="+",
        type=Path,
        help="One or more run directories containing metrics.jsonl",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Optional labels aligned with --run-dirs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for plots and evaluation_summary.json",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=0,
        help="Rolling window (0 = auto from run length).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Manual subsample stride. Default: auto from --max-points.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=2000,
        help="Target max points per series on the plot.",
    )
    parser.add_argument(
        "--importance-ratio-cap",
        type=float,
        default=10.0,
        help="Cap for finite IS-weight display.",
    )
    parser.add_argument(
        "--include-stability",
        action="store_true",
        help="Add grad_norm and param_norm panels when present in metrics.",
    )
    return parser.parse_args()


def collect_runs(args: argparse.Namespace) -> list[tuple[str, Path, list[dict]]]:
    runs: list[tuple[str, Path, list[dict]]] = []
    if args.manifest is not None:
        manifest = load_manifest(args.manifest)
        for row in manifest_run_entries(manifest):
            label = row.get("label") or row.get("id") or Path(row["run_dir"]).name
            run_dir = Path(row["run_dir"])
            runs.append((label, run_dir, load_metrics(run_dir)))
        return runs

    labels = args.labels or [path.name for path in args.run_dirs]
    if len(labels) != len(args.run_dirs):
        raise ValueError("--labels count must match --run-dirs count")
    for label, run_dir in zip(labels, args.run_dirs):
        runs.append((label, run_dir, load_metrics(run_dir)))
    return runs


def available_panels(
    rows_list: list[list[dict]],
    *,
    include_stability: bool,
) -> list[PanelSpec]:
    keys_present = {key for rows in rows_list for row in rows for key in row}
    panels: list[PanelSpec] = []
    seen: set[tuple[str, str]] = set()
    for spec in PANEL_SPECS + (OPTIONAL_PANELS if include_stability else []):
        pair = (spec.key, spec.kind)
        if spec.key in keys_present and pair not in seen:
            panels.append(spec)
            seen.add(pair)
    return panels


def _downsample(
    steps: list[int],
    *series: list[float],
    max_points: int,
) -> tuple[list[int], list[list[float]]]:
    stride = max(1, (len(steps) + max_points - 1) // max_points)
    idxs = list(range(0, len(steps), stride))
    if idxs[-1] != len(steps) - 1:
        idxs.append(len(steps) - 1)
    out_steps = [steps[i] for i in idxs]
    out_series = [[s[i] for i in idxs] for s in series]
    return out_steps, out_series


def _plot_ribbon(
    ax: plt.Axes,
    steps: list[int],
    low: list[float],
    mid: list[float],
    high: list[float],
    *,
    label: str,
    color,
) -> None:
    ax.fill_between(steps, low, high, color=color, alpha=0.22, linewidth=0)
    ax.plot(steps, mid, label=label, color=color, linewidth=1.6)


def _annotate_loss_stats(ax: plt.Axes, values: list[float]) -> None:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return
    pos_frac = sum(v > 0 for v in finite) / len(finite)
    max_abs = max(abs(v) for v in finite)
    ax.text(
        0.02,
        0.97,
        f"positive loss: {100 * pos_frac:.0f}% of steps\nmax |loss|: {max_abs:.3e}",
        transform=ax.transAxes,
        fontsize=7,
        va="top",
        color="0.35",
    )


def _plot_panel(
    ax: plt.Axes,
    rows: list[dict],
    spec: PanelSpec,
    *,
    label: str,
    color,
    smooth_window: int,
    max_points: int,
    importance_cap: float,
) -> None:
    steps, values = raw_series(rows, spec.key)

    if spec.kind == "loss_band":
        bands = rolling_quantiles(values, smooth_window, (0.25, 0.5, 0.75))
        ds, series = _downsample(
            steps,
            bands[0.25],
            bands[0.5],
            bands[0.75],
            max_points=max_points,
        )
        _plot_ribbon(ax, ds, series[0], series[1], series[2], label=label, color=color)
        ax.axhline(0.0, color="0.5", linewidth=0.8, linestyle="--", zorder=0)
        band_vals = [v for s in series for v in s if math.isfinite(v)]
        ylim = percentile_ylim(band_vals, lo=1, hi=99, include_zero=True)
        if ylim:
            ax.set_ylim(*ylim)
        _annotate_loss_stats(ax, values)

    elif spec.kind == "loss_abs":
        abs_vals = [abs(v) for v in values if math.isfinite(v)]
        abs_full = [abs(v) if math.isfinite(v) else float("nan") for v in values]
        smooth = moving_average(abs_full, smooth_window)
        ds, sv = _downsample(steps, smooth, max_points=max_points)
        ax.plot(ds, sv[0], label=label, color=color, linewidth=1.6)
        ylim = percentile_ylim(sv[0], lo=5, hi=95)
        if ylim:
            ax.set_ylim(0.0, ylim[1])

    elif spec.kind == "reward":
        smooth = moving_average(values, smooth_window)
        ds, sv = _downsample(steps, smooth, max_points=max_points)
        ax.plot(ds, sv[0], label=label, color=color, linewidth=1.6)
        ylim = percentile_ylim(sv[0], lo=2, hi=98)
        if ylim:
            ax.set_ylim(*ylim)

    elif spec.kind == "batch_unique":
        smooth = moving_average(values, smooth_window)
        ds, sv = _downsample(steps, smooth, max_points=max_points)
        ax.plot(ds, sv[0], label=label, color=color, linewidth=1.6)
        ymax = max(sv[0]) if sv[0] else 1.0
        ax.set_ylim(0, ymax * 1.08 + 1)

    elif spec.kind == "dup_fraction":
        smooth = moving_average(values, smooth_window)
        ds, sv = _downsample(steps, smooth, max_points=max_points)
        ax.plot(ds, sv[0], label=label, color=color, linewidth=1.6)
        finite = [v for v in sv[0] if math.isfinite(v)]
        if finite:
            lo = max(0.0, min(finite))
            hi = min(1.0, max(finite))
            span = hi - lo
            if span < 1e-6:
                center = 0.5 * (lo + hi)
                ax.set_ylim(max(0.0, center - 0.02), min(1.0, center + 0.02))
            else:
                pad = max(0.005, 0.08 * span)
                ax.set_ylim(max(0.0, lo - pad), min(1.0, hi + pad))

    elif spec.kind == "unique_count":
        smooth = moving_average(values, max(1, smooth_window // 4))
        ds, sv = _downsample(steps, smooth, max_points=max_points)
        ax.plot(ds, sv[0], label=label, color=color, linewidth=1.6)
        ymin = min(sv[0]) if sv[0] else 0
        ymax = max(sv[0]) if sv[0] else 1
        pad = max(1.0, 0.02 * (ymax - ymin + 1))
        ax.set_ylim(ymin - pad, ymax + pad)

    elif spec.kind == "importance_finite":
        cap = spec.y_clip if spec.y_clip is not None else importance_cap
        smooth = rolling_finite_mean(values, smooth_window, cap=cap)
        ds, sv = _downsample(steps, smooth, max_points=max_points)
        ax.plot(ds, sv[0], label=label, color=color, linewidth=1.6)
        ax.axhline(1.0, color="0.5", linewidth=0.8, linestyle=":", zorder=0)
        finite = [v for v in sv[0] if math.isfinite(v)]
        if finite:
            lo = min(finite)
            hi = max(finite)
            span = hi - lo
            if span < 1e-4:
                ax.set_ylim(max(0.0, lo - 0.05), min(cap + 0.25, hi + 0.05))
            else:
                pad = max(0.02, 0.12 * span)
                ax.set_ylim(max(0.0, lo - pad), min(cap + 0.25, hi + pad))
        else:
            ax.set_ylim(0.0, cap + 0.25)

    elif spec.kind == "importance_bad_frac":
        frac = rolling_nonfinite_fraction(values, smooth_window)
        ds, sv = _downsample(steps, frac, max_points=max_points)
        ax.plot(ds, sv[0], label=label, color=color, linewidth=1.6)
        ax.set_ylim(0.0, 1.02)
        n_bad = sum(1 for v in values if not math.isfinite(v))
        ax.text(
            0.02,
            0.97,
            f"total non-finite: {n_bad}/{len(values)}",
            transform=ax.transAxes,
            fontsize=7,
            va="top",
            color="0.35",
        )

    elif spec.kind == "log_line":
        pos = [max(v, 1e-12) for v in values]
        smooth = moving_average(pos, smooth_window)
        ds, sv = _downsample(steps, smooth, max_points=max_points)
        ax.plot(ds, sv[0], label=label, color=color, linewidth=1.6)
        if all(v > 0 for v in sv[0]):
            ax.set_yscale("log")

    else:
        plot_steps, plot_values = sample_series(
            rows,
            spec.key,
            smooth_window,
            None,
            max_points=max_points,
        )
        ax.plot(plot_steps, plot_values, label=label, color=color, linewidth=1.6)


def plot_training_curves(
    runs: list[tuple[str, Path, list[dict]]],
    output_path: Path,
    panels: list[PanelSpec],
    *,
    smoothing_window: int,
    max_points: int,
    importance_ratio_cap: float,
) -> None:
    n_cols = 2
    n_rows = (len(panels) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(13, 3.2 * n_rows),
        dpi=200,
        constrained_layout=True,
    )
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
    cmap = plt.get_cmap("tab10")

    for ax_idx, spec in enumerate(panels):
        ax = axes_flat[ax_idx]
        smooth_window = auto_smoothing_window(
            len(runs[0][2]) if runs else 0,
            smoothing_window,
        )

        for run_idx, (label, _, rows) in enumerate(runs):
            if spec.key not in rows[0]:
                continue
            _plot_panel(
                ax,
                rows,
                spec,
                label=label,
                color=cmap(run_idx % 10),
                smooth_window=smooth_window,
                max_points=max_points,
                importance_cap=importance_ratio_cap,
            )

        ax.set_title(spec.title, fontsize=10)
        ax.set_xlabel("Global step")
        ax.grid(True, alpha=0.2)
        if ax_idx == 0:
            ax.legend(frameon=False, fontsize=7, loc="best", ncol=1)

    for ax in axes_flat[len(panels):]:
        ax.axis("off")

    fig.suptitle(
        "Training & diversity (rolling summaries; IS capped for display)",
        fontsize=12,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def build_summary(runs: list[tuple[str, Path, list[dict]]]) -> dict:
    summary_runs = []
    for label, run_dir, rows in runs:
        last = rows[-1]
        first = rows[0]
        try:
            artifacts = resolve_run_artifacts(str(run_dir), label=label)
            method = artifacts.method
            checkpoint = str(artifacts.checkpoint_path)
        except Exception:
            method = last.get("method", "unknown")
            checkpoint = str(run_dir / "final_checkpoint.pt")

        summary_runs.append({
            "label": label,
            "run_dir": str(run_dir),
            "method": method,
            "checkpoint": checkpoint,
            "global_steps": int(last.get("global_step", len(rows) - 1)),
            "final": {
                "loss": last.get("loss"),
                "mean_log_reward": last.get("mean_log_reward"),
                "global_duplicate_fraction": last.get("global_duplicate_fraction"),
                "global_duplicate_topology_fraction": last.get(
                    "global_duplicate_topology_fraction"
                ),
                "cumulative_unique_outcomes": last.get("cumulative_unique_outcomes"),
                "global_unique_outcomes": last.get("global_unique_outcomes"),
                "global_unique_topologies": last.get("global_unique_topologies"),
            },
            "initial": {
                "global_duplicate_fraction": first.get("global_duplicate_fraction"),
                "cumulative_unique_outcomes": first.get("cumulative_unique_outcomes"),
            },
        })
    return {"runs": summary_runs}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runs = collect_runs(args)
    panels = available_panels(
        [rows for _, _, rows in runs],
        include_stability=args.include_stability,
    )

    plot_path = args.output_dir / "training_curves.png"
    plot_training_curves(
        runs,
        plot_path,
        panels,
        smoothing_window=args.smoothing_window,
        max_points=args.max_points,
        importance_ratio_cap=args.importance_ratio_cap,
    )

    summary = build_summary(runs)
    summary_path = args.output_dir / "evaluation_summary.json"
    save_json(summary_path, summary)

    print(f"saved training plot to: {plot_path}")
    print(f"saved summary to: {summary_path}")
    for row in summary["runs"]:
        final = row["final"]
        print(
            f"  {row['label']}: log_R={final.get('mean_log_reward')} "
            f"global_dup={final.get('global_duplicate_fraction')} "
            f"unique_outcomes={final.get('cumulative_unique_outcomes')}"
        )


if __name__ == "__main__":
    main()
