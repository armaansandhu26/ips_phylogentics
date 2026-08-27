"""Build the compact paper figure and table for one sEH checkpoint.

The input is a frozen JSON snapshot rather than live run directories.  This
keeps a checkpoint comparison reproducible after ``samples/summary.json`` is
overwritten by a later evaluation of the same run.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


METHOD_LABELS = {
    "grpo": "GRPO",
    "count_ips_grpo": "Count IPS-GRPO",
    "mips_grpo": "MIPS-GRPO",
    "rgfn": "RGFN",
}
METHOD_COLORS = {
    "grpo": "#EF5350",
    "count_ips_grpo": "#4C78A8",
    "mips_grpo": "#2CA02C",
    "rgfn": "#FF7F0E",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dpi", type=int, default=240)
    return parser


def _format_percent(value: float) -> str:
    percent = 100.0 * value
    if percent < 0.1:
        return f"{percent:.3f}%"
    if percent < 10:
        return f"{percent:.2f}%"
    if percent >= 99.9:
        return f"{percent:.2f}%"
    return f"{percent:.1f}%"


def _load_snapshot(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        snapshot = json.load(handle)
    rows = snapshot.get("methods", [])
    if not rows:
        raise ValueError("Checkpoint snapshot contains no method rows")
    checkpoint = int(snapshot["checkpoint_updates"])
    n_sampled = int(snapshot["evaluation_samples"])
    for row in rows:
        if int(row["checkpoint_updates"]) != checkpoint:
            raise ValueError("All methods must use the same checkpoint")
        if int(row["n_sampled"]) != n_sampled:
            raise ValueError("All methods must use the same evaluation budget")
    return snapshot


def _write_tables(snapshot: dict, output_dir: Path) -> tuple[Path, Path]:
    fields = (
        "method",
        "checkpoint_updates",
        "training_oracle_calls",
        "n_sampled",
        "valid_fraction",
        "n_unique",
        "unique_fraction",
        "mean_proxy",
        "n_mode_candidates",
        "n_modes",
        "top1_count",
        "top1_share",
        "status",
    )
    csv_path = output_dir / "table1_seed0_500_updates.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in snapshot["methods"]:
            writer.writerow({field: row[field] for field in fields})

    markdown_path = output_dir / "table1_seed0_500_updates.md"
    with markdown_path.open("w", encoding="utf-8") as handle:
        handle.write("# Table 1. Full-space sEH at 500 updates (seed 0)\n\n")
        handle.write(
            "All methods use 50,000 training oracle calls and 5,000 "
            "independent final-policy samples. This is an interim, single-seed result.\n\n"
        )
        handle.write(
            "| Method | Valid | Unique | Unique fraction | Mean proxy | "
            "Candidates >7 | Leader modes >7 | Top-1 mass | Status |\n"
        )
        handle.write(
            "|---|---:|---:|---:|---:|---:|---:|---:|---|\n"
        )
        for row in snapshot["methods"]:
            handle.write(
                f"| {METHOD_LABELS[row['method']]} "
                f"| {_format_percent(float(row['valid_fraction']))} "
                f"| {int(row['n_unique']):,} "
                f"| {_format_percent(float(row['unique_fraction']))} "
                f"| {float(row['mean_proxy']):.3f} "
                f"| {int(row['n_mode_candidates'])} "
                f"| {int(row['n_modes'])} "
                f"| {_format_percent(float(row['top1_share']))} "
                f"| {row['status']} |\n"
            )
        handle.write(
            "\nLeader modes use proxy > 7 and greedy Morgan-fingerprint leader "
            "clustering with maximum Tanimoto similarity ≤ 0.5.\n"
        )
        handle.write(
            "GRPO and Count IPS-GRPO checkpoint values come from the frozen "
            "500-update results record because their live summaries were later "
            "replaced by 2,000-update evaluations.\n"
        )
    return csv_path, markdown_path


def _bar_labels(axis, bars, labels: list[str], *, padding: int = 3) -> None:
    axis.bar_label(bars, labels=labels, padding=padding, fontsize=7)


def _write_figure(snapshot: dict, output_dir: Path, dpi: int) -> tuple[Path, Path]:
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "font.size": 8,
            "figure.dpi": dpi,
        }
    )

    rows = snapshot["methods"]
    labels = [METHOD_LABELS[row["method"]] for row in rows]
    colors = [METHOD_COLORS[row["method"]] for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.6))

    unique_percent = [100.0 * float(row["unique_fraction"]) for row in rows]
    bars = axes[0, 0].bar(x, unique_percent, color=colors, alpha=0.9)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylim(0.02, 160)
    axes[0, 0].set_ylabel("unique final samples (%)")
    axes[0, 0].set_title("(a) sampled non-repetition")
    _bar_labels(axes[0, 0], bars, [_format_percent(v / 100) for v in unique_percent])

    top1_percent = [100.0 * float(row["top1_share"]) for row in rows]
    bars = axes[0, 1].bar(x, top1_percent, color=colors, alpha=0.9)
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_ylim(0.04, 160)
    axes[0, 1].set_ylabel("probability mass on top molecule (%)")
    axes[0, 1].set_title("(b) collapse diagnostic")
    _bar_labels(axes[0, 1], bars, [_format_percent(v / 100) for v in top1_percent])

    mean_proxy = [float(row["mean_proxy"]) for row in rows]
    bars = axes[1, 0].bar(x, mean_proxy, color=colors, alpha=0.9)
    threshold = float(snapshot["mode_proxy_threshold"])
    axes[1, 0].axhline(threshold, color="#333333", linestyle="--", linewidth=0.9)
    axes[1, 0].text(
        len(rows) - 0.55,
        threshold + 0.08,
        "mode threshold",
        ha="right",
        va="bottom",
        fontsize=6.5,
    )
    axes[1, 0].set_ylim(0, threshold + 0.8)
    axes[1, 0].set_ylabel("mean sampled sEH proxy")
    axes[1, 0].set_title("(c) final-policy quality")
    _bar_labels(axes[1, 0], bars, [f"{value:.2f}" for value in mean_proxy])

    modes = [int(row["n_modes"]) for row in rows]
    candidates = [int(row["n_mode_candidates"]) for row in rows]
    bars = axes[1, 1].bar(x, modes, color=colors, alpha=0.9)
    axes[1, 1].set_ylim(0, max(4.6, max(modes) + 1.4))
    axes[1, 1].set_ylabel("leader modes with proxy > 7")
    axes[1, 1].set_title("(d) high-quality diverse discoveries")
    _bar_labels(
        axes[1, 1],
        bars,
        [f"{mode} ({candidate} hits)" for mode, candidate in zip(modes, candidates)],
    )

    for axis in axes.flat:
        axis.set_xticks(x, labels, rotation=18, ha="right")
        axis.grid(axis="y", color="#B0B0B0", linewidth=0.45, alpha=0.35)

    fig.suptitle(
        "Full-space sEH, seed 0 — 500 updates / 50k training oracle calls\n"
        "5,000 independent final-policy samples per method",
        fontsize=10,
    )
    fig.text(
        0.5,
        0.005,
        "Single-seed interim result; bars summarize samples and do not represent run-to-run uncertainty.",
        ha="center",
        fontsize=7,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.94), pad=1.0)

    png_path = output_dir / "figure1_seed0_500_updates.png"
    pdf_path = output_dir / "figure1_seed0_500_updates.pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def run(args: argparse.Namespace) -> list[Path]:
    metrics_path = Path(args.metrics_json).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = _load_snapshot(metrics_path)
    written = [*_write_tables(snapshot, output_dir), *_write_figure(snapshot, output_dir, args.dpi)]
    for path in written:
        print(f"WROTE={path}")
    return written


def main(argv: list[str] | None = None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
