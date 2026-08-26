"""Plot final-checkpoint reward-proportional sampling diagnostics."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path


# Match the method order and color semantics used throughout draft_1.pdf.
METHOD_ORDER = ("grpo", "count_ips_grpo", "mips_grpo", "rgfn")
METHOD_LABELS = {
    "rgfn": "RGFN",
    "grpo": "GRPO",
    "count_ips_grpo": "Count IPS-GRPO",
    "mips_grpo": "MIPS-GRPO",
}
METHOD_COLORS = {
    "rgfn": "#FF7F0E",
    "grpo": "#EF5350",
    "count_ips_grpo": "#4C78A8",
    "mips_grpo": "#2CA02C",
}
METHOD_MARKERS = {
    "rgfn": "D",
    "grpo": "o",
    "count_ips_grpo": "s",
    "mips_grpo": "^",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dpi", type=int, default=180)
    return parser


def _load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _save(fig, output_dir: Path, stem: str, dpi: int) -> list[Path]:
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    return [png, pdf]


def run(args: argparse.Namespace) -> list[Path]:
    suite_dir = Path(args.suite_dir).expanduser().resolve()
    suite = _load_json(suite_dir / "suite.json")
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

    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "normal",
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "lines.linewidth": 1.25,
            "figure.dpi": args.dpi,
            "font.size": 8,
        }
    )

    target_path = Path(suite["target_distribution"])
    target = _load_json(target_path)
    target_rows = target["outcomes"]
    smiles = [row["smiles"] for row in target_rows]
    qed = np.asarray([float(row["qed"]) for row in target_rows])
    target_probability = np.asarray(
        [float(row["target_probability"]) for row in target_rows]
    )
    selected_runs: dict[str, Path] = {}
    for method, raw_runs in suite["runs"].items():
        if isinstance(raw_runs, str):
            selected_runs[method] = Path(raw_runs)
            continue
        requested_seed = str(args.seed) if args.seed is not None else sorted(
            raw_runs, key=int
        )[0]
        if requested_seed not in raw_runs:
            raise ValueError(f"Seed {requested_seed} is unavailable for {method}")
        selected_runs[method] = Path(raw_runs[requested_seed])
    methods = [method for method in METHOD_ORDER if method in selected_runs]
    comparison = {
        method: _load_json(selected_runs[method] / "samples" / "summary.json")
        for method in methods
    }

    empirical: dict[str, dict[str, float]] = {}
    sampled_qed: dict[str, list[float]] = {}
    sampled_smiles: dict[str, list[str]] = {}
    for method in methods:
        sample_path = selected_runs[method] / "samples" / "samples.jsonl"
        counts: Counter[str] = Counter()
        method_qed: list[float] = []
        method_smiles: list[str] = []
        with sample_path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                molecule = row.get("smiles")
                if molecule is None:
                    continue
                counts[molecule] += 1
                method_qed.append(float(row["proxy"]))
                method_smiles.append(str(molecule))
        n = sum(counts.values())
        empirical[method] = {key: value / n for key, value in counts.items()}
        sampled_qed[method] = method_qed
        sampled_smiles[method] = method_smiles

    written: list[Path] = []

    # High-level comparison of the three most interpretable pilot metrics.
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35))
    labels = [METHOD_LABELS[method] for method in methods]
    colors = [METHOD_COLORS[method] for method in methods]
    x = np.arange(len(methods))
    metric_specs = (
        ("tv_to_reward_target", "(a) TV distance ↓", (0, 1.0)),
        ("sample_support_coverage", "(b) terminal support coverage ↑", (0, 1.05)),
        ("mean_proxy", "(c) mean sampled QED", (0.65, 0.91)),
    )
    for axis, (key, title, ylim) in zip(axes, metric_specs):
        values = [float(comparison[method][key]) for method in methods]
        bars = axis.bar(x, values, color=colors, edgecolor="white", linewidth=0.8)
        axis.set_title(title)
        axis.set_ylim(*ylim)
        axis.set_xticks(x, labels, rotation=24, ha="right")
        axis.grid(axis="y", color="#B0B0B0", linewidth=0.45, alpha=0.45)
        axis.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=2, fontsize=6.5)
        if key == "mean_proxy":
            target_mean = float(np.dot(qed, target_probability))
            axis.axhline(target_mean, color="black", linestyle="--", linewidth=1.3)
            axis.text(
                0.02,
                target_mean + 0.004,
                f"target = {target_mean:.3f}",
                transform=axis.get_yaxis_transform(),
                fontsize=6.5,
            )
    fig.suptitle("MiniChem CPU pilot: 30 updates, 5,000 final-policy samples", fontsize=9)
    fig.tight_layout(pad=0.7)
    written.extend(_save(fig, output_dir, "pilot_summary", args.dpi))
    plt.close(fig)

    # Calibration in molecule probability space. Ideal sampling lies on y=x.
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.1), sharex=True, sharey=True)
    lower = min(float(target_probability.min()), 0.5 / 5000)
    upper = max(float(target_probability.max()), 0.2)
    for panel_index, (axis, method) in enumerate(zip(axes.flat, methods)):
        observed = np.asarray([empirical[method].get(key, 0.0) for key in smiles])
        observed_for_log = np.maximum(observed, 0.5 / 5000)
        axis.scatter(
            target_probability,
            observed_for_log,
            s=12,
            alpha=0.55,
            color=METHOD_COLORS[method],
            edgecolors="none",
        )
        axis.plot([lower, upper], [lower, upper], color="black", linestyle="--", linewidth=1)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(lower, upper)
        axis.set_ylim(lower, upper)
        axis.set_title(
            f"({chr(97 + panel_index)}) {METHOD_LABELS[method]}  "
            f"TV={float(comparison[method]['tv_to_reward_target']):.3f}"
        )
        axis.grid(color="#B0B0B0", linewidth=0.4, alpha=0.35, which="both")
    fig.supxlabel("Exact reward-proportional probability", y=0.047)
    fig.supylabel("Empirical final-checkpoint probability")
    fig.suptitle("Reward-proportional calibration by terminal molecule", fontsize=9)
    fig.text(
        0.5,
        0.012,
        "Unseen outcomes are displayed at the 0.5 / N pseudocount level.",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0.02, 0.075, 1, 0.96))
    written.extend(_save(fig, output_dir, "probability_calibration", args.dpi))
    plt.close(fig)

    # Compare the target and empirical QED marginals.
    fig, axis = plt.subplots(figsize=(7.2, 3.1))
    bins = np.linspace(min(0.25, float(qed.min())), max(0.95, float(qed.max())), 15)
    target_hist, edges = np.histogram(qed, bins=bins, weights=target_probability)
    centers = (edges[:-1] + edges[1:]) / 2
    axis.step(centers, target_hist, where="mid", color="black", linewidth=2.5, label="Exact target")
    for method in methods:
        hist, _ = np.histogram(sampled_qed[method], bins=bins)
        hist = hist / hist.sum()
        axis.step(
            centers,
            hist,
            where="mid",
            linewidth=1.8,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
    axis.set_xlabel("QED")
    axis.set_ylabel("Probability mass")
    axis.set_title("Final sampled QED distribution against the exact target")
    axis.grid(color="#B0B0B0", linewidth=0.45, alpha=0.4)
    axis.legend(frameon=False, ncol=2)
    fig.tight_layout()
    written.extend(_save(fig, output_dir, "qed_distribution", args.dpi))
    plt.close(fig)

    # Paper-style terminal distribution panels. Molecules are ordered once by
    # QED so every panel has a shared, interpretable horizontal coordinate.
    outcome_order = np.argsort(qed)
    ordered_target = target_probability[outcome_order]
    fig, axes = plt.subplots(1, 5, figsize=(7.2, 1.85), sharex=True, sharey=True)
    profile_floor = 0.5 / 5000
    axes[0].plot(
        np.arange(len(smiles)),
        np.maximum(ordered_target, profile_floor),
        color="#333333",
        linewidth=1.0,
    )
    axes[0].set_title("(a) exact target")
    axes[0].set_ylabel("probability")
    for panel_index, (axis, method) in enumerate(zip(axes[1:], methods), start=1):
        observed = np.asarray([empirical[method].get(key, 0.0) for key in smiles])
        axis.plot(
            np.arange(len(smiles)),
            np.maximum(observed[outcome_order], profile_floor),
            color=METHOD_COLORS[method],
            linewidth=0.85,
        )
        axis.plot(
            np.arange(len(smiles)),
            ordered_target,
            color="black",
            linestyle="--",
            linewidth=0.65,
            alpha=0.8,
        )
        axis.set_title(
            f"({chr(97 + panel_index)}) {METHOD_LABELS[method]}\n"
            f"TV={float(comparison[method]['tv_to_reward_target']):.3f}"
        )
    for axis in axes:
        axis.set_xlim(0, len(smiles) - 1)
        axis.set_yscale("log")
        axis.set_ylim(profile_floor, 1.0)
        axis.set_xlabel("terminal rank")
        axis.grid(axis="y", color="#B0B0B0", linewidth=0.35, alpha=0.3, which="both")
    fig.suptitle("Terminal distributions ordered by increasing QED", fontsize=9)
    fig.tight_layout(pad=0.6)
    written.extend(_save(fig, output_dir, "terminal_distributions", args.dpi))
    plt.close(fig)

    # Match the paper's two-view evaluation figure: diversity and reward as
    # progressively more samples are drawn from each trained policy.
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    for method in methods:
        seen: set[str] = set()
        unique_curve: list[int] = []
        cumulative_qed: list[float] = []
        running_total = 0.0
        for index, (molecule, value) in enumerate(
            zip(sampled_smiles[method], sampled_qed[method]), start=1
        ):
            seen.add(molecule)
            running_total += value
            unique_curve.append(len(seen))
            cumulative_qed.append(running_total / index)
        draw_index = np.arange(1, len(unique_curve) + 1)
        plot_kwargs = {
            "color": METHOD_COLORS[method],
            "label": METHOD_LABELS[method],
            "marker": METHOD_MARKERS[method],
            "markevery": max(1, len(draw_index) // 10),
            "markersize": 2.4,
            "linewidth": 1.05,
        }
        axes[0].plot(draw_index, unique_curve, **plot_kwargs)
        axes[1].plot(draw_index, cumulative_qed, **plot_kwargs)
    axes[0].axhline(len(smiles), color="black", linestyle="--", linewidth=0.7)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("evaluation samples drawn (log scale)")
    axes[0].set_ylabel("distinct terminal molecules")
    axes[0].set_title("(a) support discovery against samples drawn")
    target_mean = float(np.dot(qed, target_probability))
    axes[1].axhline(
        target_mean,
        color="black",
        linestyle="--",
        linewidth=0.8,
        label="exact target mean",
    )
    axes[1].set_xlabel("evaluation samples drawn")
    axes[1].set_ylabel("cumulative mean QED")
    axes[1].set_title("(b) mean terminal QED during evaluation")
    for axis in axes:
        axis.grid(color="#B0B0B0", linewidth=0.45, alpha=0.4)
        axis.legend(frameon=False, fontsize=6.5)
    fig.tight_layout(pad=0.7)
    written.extend(_save(fig, output_dir, "sample_discovery", args.dpi))
    plt.close(fig)

    for path in written:
        print(f"PLOT={path}")
    return written


def main(argv: list[str] | None = None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
