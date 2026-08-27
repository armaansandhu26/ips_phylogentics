"""Create paper-style plots for a non-enumerable biological-oracle suite."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path


METHOD_ORDER = ("grpo", "count_ips_grpo", "mips_grpo", "rgfn")
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
METHOD_MARKERS = {
    "grpo": "o",
    "count_ips_grpo": "s",
    "mips_grpo": "^",
    "rgfn": "D",
}

SUITE_LABELS = {
    "seh_reduced_a100": "reduced-space sEH",
    "seh_paper_main": "paper-scale sEH",
}


def _load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _normalized_runs(suite: dict) -> dict[str, dict[str, Path]]:
    result: dict[str, dict[str, Path]] = {}
    for method, raw in suite["runs"].items():
        values = {"legacy": raw} if isinstance(raw, str) else raw
        result[method] = {str(seed): Path(path) for seed, path in values.items()}
    return result


def _suite_label(suite: dict) -> str:
    suite_id = str(suite.get("suite", "")).strip()
    if suite_id in SUITE_LABELS:
        return SUITE_LABELS[suite_id]
    return suite_id.replace("_", " ") or "sEH"


def _forward_trajectories(run_dir: Path, default: int = 100) -> int:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        return default
    bindings = _load_json(manifest_path).get("bindings", [])
    prefix = "Trainer.train_forward_n_trajectories="
    for binding in bindings:
        if str(binding).startswith(prefix):
            return int(str(binding)[len(prefix) :])
    return default


def _result_caption(summaries: dict[str, list[dict]]) -> str:
    seeds = sorted(
        {
            int(row["seed"])
            for method_summaries in summaries.values()
            for row in method_summaries
            if "seed" in row
        }
    )
    if len(seeds) == 1:
        return f"seed {seeds[0]}"
    if len(seeds) > 1:
        return f"mean ± SD over {len(seeds)} seeds"
    return "completed runs"


def _save(fig, output_dir: Path, stem: str, dpi: int) -> list[Path]:
    paths = [output_dir / f"{stem}.png", output_dir / f"{stem}.pdf"]
    fig.savefig(paths[0], dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(paths[1], bbox_inches="tight", facecolor="white")
    return paths


def run(args: argparse.Namespace) -> list[Path]:
    suite_dir = Path(args.suite_dir).expanduser().resolve()
    suite = _load_json(suite_dir / "suite.json")
    runs = _normalized_runs(suite)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else suite_dir / "results" / "plots"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "font.size": 8,
        }
    )
    methods = [method for method in METHOD_ORDER if method in runs]
    summaries: dict[str, list[dict]] = defaultdict(list)
    proxy_samples: dict[str, list[float]] = defaultdict(list)
    for method in methods:
        for run_dir in runs[method].values():
            summary_path = run_dir / "samples" / "summary.json"
            sample_path = run_dir / "samples" / "samples.jsonl"
            if not summary_path.is_file() or not sample_path.is_file():
                continue
            summaries[method].append(_load_json(summary_path))
            values = []
            with sample_path.open(encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    values.append(float(row["proxy"]))
            if len(values) > args.max_points_per_run:
                indices = np.linspace(0, len(values) - 1, args.max_points_per_run, dtype=int)
                values = [values[index] for index in indices]
            proxy_samples[method].extend(values)

    if not any(summaries.values()):
        raise RuntimeError(f"No completed sampled runs found under {suite_dir}")

    written: list[Path] = []

    # Mirrors the paper's per-task reward distribution figure.
    available = [method for method in methods if proxy_samples[method]]
    fig, axis = plt.subplots(figsize=(5.4, 2.8))
    boxes = axis.boxplot(
        [proxy_samples[method] for method in available],
        tick_labels=[METHOD_LABELS[method] for method in available],
        showfliers=False,
        patch_artist=True,
        widths=0.62,
        medianprops={"color": "black", "linewidth": 1.0},
        whiskerprops={"linewidth": 0.8},
        capprops={"linewidth": 0.8},
    )
    for patch, method in zip(boxes["boxes"], available):
        patch.set_facecolor(METHOD_COLORS[method])
        patch.set_alpha(0.8)
    axis.set_ylabel("sEH proxy score")
    axis.set_title("(a) final-policy reward distributions")
    axis.grid(axis="y", color="#B0B0B0", linewidth=0.45, alpha=0.4)
    fig.tight_layout(pad=0.7)
    written.extend(_save(fig, output_dir, "seh_reward_distributions", args.dpi))
    plt.close(fig)

    # The upstream metric writes leader modes every 500 iterations. Plot these
    # against oracle calls, exactly as the RGFN paper normalizes its x-axis.
    fig, axis = plt.subplots(figsize=(5.4, 2.8))
    any_mode_curve = False
    for method in methods:
        seed_curves: list[tuple[list[int], list[int]]] = []
        for run_dir in runs[method].values():
            points = []
            for path in sorted((run_dir / "modes").glob("modes_*.xlsx")):
                match = re.search(r"modes_(\d+)\.xlsx$", path.name)
                if match is None:
                    continue
                iteration = int(match.group(1))
                n_modes = len(pd.read_excel(path))
                points.append((iteration * _forward_trajectories(run_dir), n_modes))
            if points:
                points.sort()
                seed_curves.append(([point[0] for point in points], [point[1] for point in points]))
        for x_values, y_values in seed_curves:
            axis.plot(x_values, y_values, color=METHOD_COLORS[method], alpha=0.2, linewidth=0.7)
        if seed_curves:
            any_mode_curve = True
            common_x = sorted(set.intersection(*(set(curve[0]) for curve in seed_curves)))
            means = [
                np.mean([curve[1][curve[0].index(x_value)] for curve in seed_curves])
                for x_value in common_x
            ]
            axis.plot(
                common_x,
                means,
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                markersize=3,
                linewidth=1.2,
                label=METHOD_LABELS[method],
            )
    if any_mode_curve:
        axis.set_yscale("log")
        axis.set_xlabel("training oracle calls")
        mode_threshold = next(
            (
                float(row["mode_proxy_threshold"])
                for method in methods
                for row in summaries[method]
                if "mode_proxy_threshold" in row
            ),
            8.0,
        )
        axis.set_ylabel(f"discovered modes (proxy > {mode_threshold:g})")
        axis.set_title("(b) mode discovery during training")
        axis.grid(color="#B0B0B0", linewidth=0.45, alpha=0.4, which="both")
        axis.legend(frameon=False)
        fig.tight_layout(pad=0.7)
        written.extend(_save(fig, output_dir, "seh_training_modes", args.dpi))
    plt.close(fig)

    # Three outcomes needed for our claim: diverse high-score modes, scaffold
    # diversity, and reward-proportional importance-weight stability.
    metric_specs = (
        ("n_modes", "(a) final modes, proxy > 7", None),
        ("n_scaffolds_proxy_gt_7", "(b) scaffolds, proxy > 7", None),
        ("importance_ess_fraction", "(c) importance ESS fraction", (0, 1.05)),
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35))
    for axis, (key, title, ylim) in zip(axes, metric_specs):
        plot_methods = [
            method for method in methods if any(key in row for row in summaries[method])
        ]
        means = [
            np.mean([float(row[key]) for row in summaries[method] if key in row])
            for method in plot_methods
        ]
        errors = [
            np.std([float(row[key]) for row in summaries[method] if key in row], ddof=1)
            if sum(key in row for row in summaries[method]) > 1
            else 0.0
            for method in plot_methods
        ]
        x = np.arange(len(plot_methods))
        axis.bar(
            x,
            means,
            yerr=errors,
            capsize=2,
            color=[METHOD_COLORS[method] for method in plot_methods],
            alpha=0.85,
        )
        axis.set_xticks(
            x,
            [METHOD_LABELS[method] for method in plot_methods],
            rotation=24,
            ha="right",
        )
        axis.set_title(title)
        if ylim is not None:
            axis.set_ylim(*ylim)
        axis.grid(axis="y", color="#B0B0B0", linewidth=0.45, alpha=0.4)
    fig.suptitle(
        f"{_suite_label(suite)} final-policy evaluation ({_result_caption(summaries)})",
        fontsize=9,
    )
    fig.tight_layout(pad=0.7)
    written.extend(_save(fig, output_dir, "seh_final_summary", args.dpi))
    plt.close(fig)

    for path in written:
        print(f"PLOT={path}")
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--max-points-per-run", type=int, default=20000)
    return parser


def main(argv: list[str] | None = None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
