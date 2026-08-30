"""Collect the final three-seed sEH results and build a paper-ready figure.

The figure reports arithmetic means and sample standard deviations across
three seeds. All seed values remain visible so the aggregate and its
run-to-run variability can be assessed together.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from pathlib import Path


METHOD_ORDER = ("grpo", "count_ips_grpo", "mips_grpo", "rgfn")
METHOD_LABELS = {
    "grpo": "GRPO",
    "count_ips_grpo": "Count IPS-GRPO",
    "mips_grpo": "MIPS-GRPO",
    "rgfn": "RGFN",
}
METHOD_COLORS = {
    "grpo": "#E45756",
    "count_ips_grpo": "#4C78A8",
    "mips_grpo": "#2CA02C",
    "rgfn": "#F58518",
}
METRICS = (
    "valid_fraction",
    "n_unique",
    "unique_fraction",
    "mean_proxy",
    "n_mode_candidates",
    "n_modes",
    "top_modes_mean_proxy",
    "n_scaffolds_proxy_gt_7",
    "n_scaffolds_proxy_gt_8",
    "importance_ess_fraction",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite-dir",
        default="molecule_synthesis/runs/seh_paper_medium",
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_run_dir(suite_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    repository_root = suite_dir.parents[2]
    candidate = repository_root / path
    return candidate if candidate.is_dir() else path.resolve()


def load_rows(suite_dir: Path) -> list[dict]:
    suite = _load_json(suite_dir / "suite.json")
    rows: list[dict] = []
    errors: list[str] = []
    for method in METHOD_ORDER:
        method_runs = suite.get("runs", {}).get(method, {})
        if not isinstance(method_runs, dict):
            errors.append(f"{method}: suite manifest does not contain per-seed runs")
            continue
        for seed in (0, 1, 2):
            raw_path = method_runs.get(str(seed))
            if raw_path is None:
                errors.append(f"{method} seed {seed}: absent from suite manifest")
                continue
            run_dir = _resolve_run_dir(suite_dir, raw_path)
            summary_path = run_dir / "samples" / "summary.json"
            epoch_path = run_dir / "train" / "checkpoints" / "last_epoch.txt"
            if not summary_path.is_file() or not epoch_path.is_file():
                errors.append(f"{method} seed {seed}: missing summary or checkpoint")
                continue
            summary = _load_json(summary_path)
            checkpoint_epoch = int(epoch_path.read_text(encoding="utf-8").strip())
            summary_epoch = int(summary.get("training_checkpoint_metrics", {}).get("epoch", -1))
            if checkpoint_epoch != 2499 or summary_epoch != 2499:
                errors.append(
                    f"{method} seed {seed}: expected epoch 2499, got "
                    f"checkpoint={checkpoint_epoch}, summary={summary_epoch}"
                )
            if int(summary.get("n_sampled", -1)) != 50_000:
                errors.append(
                    f"{method} seed {seed}: expected 50,000 samples, "
                    f"got {summary.get('n_sampled')}"
                )
            row = {
                "method": method,
                "method_label": METHOD_LABELS[method],
                "seed": seed,
                "training_updates": checkpoint_epoch + 1,
                "training_oracle_calls": (checkpoint_epoch + 1) * 100,
                "n_sampled": int(summary["n_sampled"]),
                "run_dir": str(run_dir),
            }
            for metric in METRICS:
                row[metric] = summary.get(metric)
            rows.append(row)
    if errors:
        raise ValueError("Final-result validation failed:\n- " + "\n- ".join(errors))
    if len(rows) != 12:
        raise ValueError(f"Expected 12 final rows, found {len(rows)}")
    return rows


def summarize_rows(rows: list[dict]) -> list[dict]:
    summaries: list[dict] = []
    for method in METHOD_ORDER:
        method_rows = [row for row in rows if row["method"] == method]
        aggregate: dict[str, str | int | float | None] = {
            "method": method,
            "method_label": METHOD_LABELS[method],
            "n_seeds": len(method_rows),
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in method_rows if row[metric] is not None]
            aggregate[f"{metric}_mean"] = statistics.fmean(values) if values else None
            aggregate[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else None
            aggregate[f"{metric}_n"] = len(values)
        summaries.append(aggregate)
    return summaries


def choose_best_rows(rows: list[dict]) -> list[dict]:
    """Choose by mode count, breaking ties by final-policy mean proxy."""
    selected: list[dict] = []
    for method in METHOD_ORDER:
        method_rows = [row for row in rows if row["method"] == method]
        selected.append(
            max(
                method_rows,
                key=lambda row: (int(row["n_modes"]), float(row["mean_proxy"])),
            )
        )
    return selected


def _write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _mean_sd(row: dict, metric: str, digits: int = 2, scale: float = 1.0) -> str:
    mean = row[f"{metric}_mean"]
    std = row[f"{metric}_std"]
    if mean is None:
        return "—"
    if std is None:
        return f"{scale * float(mean):.{digits}f}"
    return f"{scale * float(mean):.{digits}f} ± {scale * float(std):.{digits}f}"


def _write_markdown(path: Path, summaries: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Final sEH comparison (2,500 updates, 50,000 samples)\n\n")
        handle.write(
            "Values are mean ± sample standard deviation over three independent "
            "seeds. Modes follow the RGFN protocol: sEH proxy > 7 and maximum "
            "pairwise Tanimoto similarity ≤ 0.5 under greedy leader selection.\n\n"
        )
        handle.write(
            "| Method | Valid (%) | Unique (%) | Unique molecules | Mean proxy | "
            "Modes ↑ | Scaffolds >7 ↑ | Top-mode proxy ↑ |\n"
        )
        handle.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in summaries:
            handle.write(
                f"| {row['method_label']} "
                f"| {_mean_sd(row, 'valid_fraction', 2, 100)} "
                f"| {_mean_sd(row, 'unique_fraction', 3, 100)} "
                f"| {_mean_sd(row, 'n_unique', 1)} "
                f"| {_mean_sd(row, 'mean_proxy', 3)} "
                f"| {_mean_sd(row, 'n_modes', 1)} "
                f"| {_mean_sd(row, 'n_scaffolds_proxy_gt_7', 1)} "
                f"| {_mean_sd(row, 'top_modes_mean_proxy', 3)} |\n"
            )
        handle.write(
            "\nTop-mode proxy is undefined for GRPO and Count IPS-GRPO because "
            "none of their final samples met the mode threshold in any seed.\n"
        )


def _plot(rows: list[dict], summaries: list[dict], output_dir: Path, dpi: int) -> list[Path]:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/molecule_synthesis_matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "font.size": 9,
            "figure.dpi": dpi,
        }
    )
    x = np.arange(len(METHOD_ORDER))
    labels = [METHOD_LABELS[method] for method in METHOD_ORDER]
    colors = [METHOD_COLORS[method] for method in METHOD_ORDER]
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.25))
    panel_specs = (
        ("mean_proxy", "mean sampled sEH proxy", "(a) final-policy quality", False),
        ("n_modes", "modes (proxy > 7)", "(b) diverse high-reward modes", False),
        ("unique_fraction", "unique final samples (%)", "(c) non-collapse", True),
    )
    for axis, (metric, ylabel, title, log_scale) in zip(axes, panel_specs):
        scale = 100.0 if metric == "unique_fraction" else 1.0
        mean_values = [scale * float(row[f"{metric}_mean"]) for row in summaries]
        std_values = [scale * float(row[f"{metric}_std"]) for row in summaries]
        bars = axis.bar(x, mean_values, color=colors, alpha=0.78, width=0.65)
        if log_scale:
            display_floor = 0.001
            lower_errors = [min(std, max(mean - display_floor, 0.0)) for mean, std in zip(mean_values, std_values)]
        else:
            lower_errors = std_values
        axis.errorbar(
            x,
            mean_values,
            yerr=np.array([lower_errors, std_values]),
            fmt="none",
            ecolor="#222222",
            elinewidth=1.0,
            capsize=3,
            capthick=1.0,
            zorder=4,
        )
        seed_values_by_method: list[list[float]] = []
        for index, method in enumerate(METHOD_ORDER):
            seed_rows = [row for row in rows if row["method"] == method]
            seed_values = [scale * float(row[metric]) for row in seed_rows]
            seed_values_by_method.append(seed_values)
            offsets = (-0.14, 0.0, 0.14)
            for offset, value in zip(offsets, seed_values):
                axis.scatter(
                    index + offset,
                    value,
                    s=25,
                    marker="o",
                    facecolor=colors[index],
                    edgecolor="white",
                    linewidth=0.8,
                    zorder=5,
                )
        if metric == "mean_proxy":
            axis.axhline(7.0, color="#333333", linestyle="--", linewidth=0.9)
            axis.text(3.38, 7.05, "mode threshold", ha="right", va="bottom", fontsize=7)
            axis.set_ylim(0, 7.8)
        elif metric == "n_modes":
            upper = max(
                max(mean + std for mean, std in zip(mean_values, std_values)),
                max(max(values) for values in seed_values_by_method),
            )
            axis.set_ylim(0, max(1.0, upper * 1.22))
        else:
            axis.set_yscale("log")
            axis.set_ylim(0.001, 180)
        for bar, mean, std, seed_values in zip(bars, mean_values, std_values, seed_values_by_method):
            if metric == "unique_fraction":
                label = f"{mean:.3g} ± {std:.3g}%"
            elif metric == "mean_proxy":
                label = f"{mean:.2f} ± {std:.2f}"
            else:
                label = f"{mean:.1f} ± {std:.1f}"
            label_y = max(mean + std, max(seed_values), 0.001)
            axis.annotate(
                label,
                (bar.get_x() + bar.get_width() / 2, label_y),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
            )
        axis.set_xticks(x, labels, rotation=18, ha="right")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(axis="y", color="#B0B0B0", linewidth=0.45, alpha=0.35)
    fig.suptitle(
        "Full-space sEH: three-seed final-policy summary",
        fontsize=11,
    )
    fig.text(
        0.5,
        0.005,
        "Bars: arithmetic mean. Whiskers: ±1 sample SD. Circles: individual seeds. "
        "Lower uniqueness whiskers are clipped to the log-axis floor.",
        ha="center",
        fontsize=7.5,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.93), pad=1.0)
    paths = [output_dir / "figure_best_runs.png", output_dir / "figure_best_runs.pdf"]
    fig.savefig(paths[0], dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(paths[1], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return paths


def run(args: argparse.Namespace) -> list[Path]:
    suite_dir = Path(args.suite_dir).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else suite_dir / "results" / "final_2500"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(suite_dir)
    summaries = summarize_rows(rows)
    best_rows = choose_best_rows(rows)
    outputs = [
        output_dir / "per_seed_results.csv",
        output_dir / "table1_all_seeds.csv",
        output_dir / "table1_all_seeds.md",
        output_dir / "best_runs.csv",
    ]
    _write_csv(outputs[0], rows)
    _write_csv(outputs[1], summaries)
    _write_markdown(outputs[2], summaries)
    _write_csv(outputs[3], best_rows)
    outputs.extend(_plot(rows, summaries, output_dir, args.dpi))
    for path in outputs:
        print(f"WROTE={path}")
    return outputs


def main(argv: list[str] | None = None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
