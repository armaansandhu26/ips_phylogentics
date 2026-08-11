"""Evaluate observed trajectory coverage from a saved DAG policy checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Iterable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch

from count_ips import CountIPSTrainer
from run_count_ips import _plot_final_counts
from trajectory_ips import terminal_multiplicities


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is not available")
    return requested


def _fits_float(value: int) -> bool:
    try:
        float(value)
    except OverflowError:
        return False
    return True


def _safe_log10(count: int) -> float:
    if count <= 0:
        return float("-inf")
    try:
        return math.log10(float(count))
    except OverflowError:
        text = str(count)
        return len(text) - 1 + math.log10(int(text[0]))


def _coverage_percent(observed: int, possible: int) -> float:
    if possible <= 0:
        return 0.0
    return 100.0 * (observed / possible)


def _format_large_int(value: int) -> str:
    if value < 10**12:
        return f"{value:,}"
    text = str(value)
    exponent = len(text) - 1
    mantissa = int(text[0])
    if len(text) > 1:
        mantissa = float(f"{text[0]}.{text[1]}")
    return f"{mantissa}e{exponent}"


def _plot_trajectory_coverage(
    report: dict[str, object], *, output: Path
) -> None:
    possible_by_terminal = report["possible_by_terminal"]
    coverage_by_terminal = report["coverage_by_terminal"]
    assert isinstance(possible_by_terminal, dict)
    assert isinstance(coverage_by_terminal, dict)

    labels = list(possible_by_terminal)
    possible_counts = [int(possible_by_terminal[label]) for label in labels]
    observed_counts = [int(coverage_by_terminal[label]) for label in labels]
    missing_counts = [
        possible_count - observed_count
        for possible_count, observed_count in zip(possible_counts, observed_counts)
    ]
    coverage_percent = [
        _coverage_percent(observed, possible)
        for observed, possible in zip(observed_counts, possible_counts)
    ]
    use_log_scale = any(not _fits_float(count) for count in possible_counts + missing_counts)
    if use_log_scale:
        possible = np.asarray([_safe_log10(count) for count in possible_counts])
        observed = np.asarray([_safe_log10(count) for count in observed_counts])
        missing = np.asarray([_safe_log10(count) for count in missing_counts])
        counts_ylabel = "log10(unique trajectories)"
    else:
        possible = np.asarray(possible_counts, dtype=np.float64)
        observed = np.asarray(observed_counts, dtype=np.float64)
        missing = np.asarray(missing_counts, dtype=np.float64)
        counts_ylabel = "Number of unique trajectories"
    positions = np.arange(len(labels))
    width = 0.38

    fig, (ax_counts, ax_missing) = plt.subplots(2, 1, figsize=(13, 9))
    ax_counts.bar(
        positions - width / 2,
        possible,
        width,
        color="#0984e3",
        label="all possible trajectories",
    )
    ax_counts.bar(
        positions + width / 2,
        observed,
        width,
        color="#00b894",
        label="observed unique trajectories",
    )
    annotate_bars = len(labels) <= 32
    if annotate_bars:
        for position, count, plotted_count, percent in zip(
            positions, observed_counts, observed, coverage_percent
        ):
            ax_counts.annotate(
                f"{count:,}\n{percent:.1f}%",
                (position + width / 2, plotted_count),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax_counts.set_ylabel(counts_ylabel)
    ax_counts.set_title("Possible versus observed trajectory support")
    ax_counts.legend()
    ax_counts.grid(axis="y", alpha=0.22)
    if use_log_scale:
        ax_counts.set_yscale("linear")
    else:
        positive_observed = observed[observed > 0]
        if positive_observed.size and possible.max() / positive_observed.min() > 1_000:
            ax_counts.set_yscale("log")

    ax_missing.bar(positions, missing, color="#d63031")
    if annotate_bars:
        for position, count, plotted_count in zip(
            positions, missing_counts, missing
        ):
            ax_missing.annotate(
                f"{count:,}",
                (position, plotted_count),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    tick_step = max(1, int(np.ceil(len(labels) / 16)))
    tick_indices = list(range(0, len(labels), tick_step))
    if tick_indices[-1] != len(labels) - 1:
        tick_indices.append(len(labels) - 1)
    ax_missing.set_xticks(
        positions[tick_indices],
        [labels[index] for index in tick_indices],
        rotation=45,
        ha="right",
    )
    ax_missing.set_xlabel("Terminal outcome")
    ax_missing.set_ylabel("Unobserved trajectories")
    ax_missing.set_title("Remaining unseen trajectories by terminal")
    ax_missing.grid(axis="y", alpha=0.22)

    total_possible = int(report["total_possible_trajectories"])
    total_label = _format_large_int(total_possible)
    coverage_percent_value = float(report["coverage_percent"])
    coverage_label = (
        f"{coverage_percent_value:.2f}%"
        if coverage_percent_value >= 0.01
        else f"{coverage_percent_value:.3e}%"
    )
    scale_note = " (log10 scale)" if use_log_scale else ""
    fig.suptitle(
        f"Trajectory coverage from {int(report['samples']):,} policy samples\n"
        f"overall: {int(report['unique_trajectories_hit']):,} / "
        f"{total_label} ({coverage_label}){scale_note}"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _format_duration(seconds: float) -> str:
    if seconds < 0 or not float("inf") > seconds:
        return "?"
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def _evaluate_with_progress(
    trainer: CountIPSTrainer,
    samples: int,
    *,
    batch_size: int,
    print_every: int,
) -> dict[str, object]:
    """Sample rollouts in batches and log throughput / ETA."""
    if print_every < 1:
        raise ValueError("print_every must be >= 1")

    total_batches = (samples + batch_size - 1) // batch_size
    print(
        f"Sampling {samples:,} episodes in batches of {batch_size:,} "
        f"({total_batches} batches)",
        flush=True,
    )
    started = time.perf_counter()
    last_log = started
    completed = 0
    batch_index = 0

    def sample_batches() -> Iterable[object]:
        nonlocal completed, batch_index, last_log
        remaining = samples
        while remaining:
            current_batch_size = min(batch_size, remaining)
            batch_index += 1
            yield from trainer.rollout_batch(current_batch_size)
            completed += current_batch_size
            remaining -= current_batch_size

            should_log = (
                completed >= samples
                or completed % print_every < current_batch_size
                or batch_index == 1
            )
            now = time.perf_counter()
            if should_log or now - last_log >= 30.0:
                elapsed = now - started
                rate = completed / max(elapsed, 1e-9)
                remaining_samples = samples - completed
                eta = remaining_samples / rate if rate > 0 else float("inf")
                print(
                    f"[eval] batch {batch_index}/{total_batches} "
                    f"samples={completed:,}/{samples:,} "
                    f"({100.0 * completed / samples:.1f}%) "
                    f"rate={rate:,.1f}/s "
                    f"elapsed={_format_duration(elapsed)} "
                    f"eta={_format_duration(eta)}",
                    flush=True,
                )
                last_log = now

    evaluation = trainer.summarize_rollouts(sample_batches())
    elapsed = time.perf_counter() - started
    rate = samples / max(elapsed, 1e-9)
    print(
        f"[eval] finished {samples:,} samples in {_format_duration(elapsed)} "
        f"({rate:,.1f}/s); unique trajectories="
        f"{int(evaluation['eval_unique_trajectories']):,}",
        flush=True,
    )
    return evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=4_096)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="plot an existing output report without sampling again",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="default: evaluation_<samples>_coverage.json beside the checkpoint",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=10_000,
        help="Log sampling progress every this many completed samples (default: 10000).",
    )
    args = parser.parse_args()

    if args.samples < 1:
        parser.error("--samples must be >= 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if args.print_every < 1:
        parser.error("--print-every must be >= 1")

    output = args.output or (
        args.checkpoint.parent / f"evaluation_{args.samples}_coverage.json"
    )
    print(f"Loading checkpoint: {args.checkpoint}", flush=True)
    device = _resolve_device(args.device)
    trainer = CountIPSTrainer.load(args.checkpoint, device=device)
    print(f"Checkpoint: {args.checkpoint}", flush=True)
    print(
        f"Device: {device}  budget={trainer.config.budget}  max_step={trainer.config.max_step}",
        flush=True,
    )

    if args.plot_only:
        if not output.is_file():
            partial = output.with_suffix(".partial.json")
            if partial.is_file():
                output = partial
            else:
                parser.error(f"coverage report does not exist: {output}")
        report = json.loads(output.read_text(encoding="utf-8"))
        evaluation = {
            "eval_outcome_counts": report["sample_count_by_terminal"]
        }
    else:
        evaluation = _evaluate_with_progress(
            trainer,
            args.samples,
            batch_size=args.batch_size,
            print_every=args.print_every,
        )
        possible = terminal_multiplicities(
            trainer.config.budget, trainer.config.max_step
        )
        total_possible = sum(possible.values())
        unique_hit = int(evaluation["eval_unique_trajectories"])
        coverage_fraction = unique_hit / total_possible
        report = {
            "checkpoint": str(args.checkpoint),
            "device": device,
            "budget": trainer.config.budget,
            "max_step": trainer.config.max_step,
            "samples": args.samples,
            "batch_size": args.batch_size,
            "unique_trajectories_hit": unique_hit,
            "total_possible_trajectories": total_possible,
            "coverage_fraction": coverage_fraction,
            "coverage_percent": 100.0 * coverage_fraction,
            "coverage_by_terminal": evaluation["eval_trajectory_coverage"],
            "possible_by_terminal": {
                state.signature: count for state, count in possible.items()
            },
            "sample_count_by_terminal": evaluation["eval_outcome_counts"],
        }
        partial_output = output.with_suffix(".partial.json")
        partial_output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved sampling report: {partial_output}", flush=True)

    report_samples = int(report["samples"])
    plot_stem = output.stem.removesuffix("_coverage")
    terminal_plot = output.with_name(f"{plot_stem}_terminal_sampling.png")
    trajectory_plot = output.with_name(f"{plot_stem}_trajectory_coverage.png")

    report["terminal_sampling"] = _plot_final_counts(
        trainer,
        evaluation,
        samples=report_samples,
        output=terminal_plot,
    )
    _plot_trajectory_coverage(report, output=trajectory_plot)
    report["plots"] = {
        "terminal_sampling": terminal_plot.name,
        "trajectory_coverage": trajectory_plot.name,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"Saved to: {output}")
    print(f"Terminal sampling plot: {terminal_plot}")
    print(f"Trajectory coverage plot: {trajectory_plot}")


if __name__ == "__main__":
    main()
