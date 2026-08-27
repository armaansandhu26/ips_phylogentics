"""Plot training and sampling diagnostics for completed seh_paper_medium seed-0 runs."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

METHOD_ORDER = ("grpo", "count_ips_grpo")
METHOD_LABELS = {
    "grpo": "GRPO",
    "count_ips_grpo": "Count IPS-GRPO",
}
METHOD_COLORS = {
    "grpo": "#EF5350",
    "count_ips_grpo": "#4C78A8",
}
METHOD_MARKERS = {
    "grpo": "o",
    "count_ips_grpo": "s",
}

METRIC_PREFIX = b"train/"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite-dir",
        default="molecule_synthesis/runs/seh_paper_medium",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dpi", type=int, default=180)
    return parser


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _save(fig, output_dir: Path, stem: str, dpi: int) -> list[Path]:
    paths = [output_dir / f"{stem}.png", output_dir / f"{stem}.pdf"]
    fig.savefig(paths[0], dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(paths[1], bbox_inches="tight", facecolor="white")
    return paths


def _parse_metric_value(part: bytes) -> float | None:
    idx = part.find(b"\x82\x01")
    if idx == -1:
        return None
    chunk = part[idx + 3 : idx + 40]
    numeric = "".join(chr(byte) for byte in chunk if (48 <= byte <= 57) or byte in b".+-eE")
    if not numeric:
        ascii_chunk = "".join(chr(byte) for byte in chunk if 32 <= byte < 127)
        numeric = ascii_chunk.split("\\")[0].strip(": ")
    try:
        return float(numeric)
    except ValueError:
        return None


def _parse_metrics_blob(blob: bytes) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for part in blob.split(METRIC_PREFIX)[1:]:
        name_bytes = part.split(b"\x82", 1)[0]
        name = name_bytes.decode("ascii", errors="ignore").strip("\x00\n\r ")
        value = _parse_metric_value(part)
        if name and value is not None:
            metrics[f"train/{name}"] = value
    return metrics


def _load_wandb_history(wandb_path: Path) -> list[dict[str, float]]:
    from wandb.sdk.internal import datastore

    rows: list[dict[str, float]] = []
    store = datastore.DataStore()
    store.open_for_scan(str(wandb_path))
    while True:
        try:
            record = store.scan_record()
        except AssertionError:
            break
        if record is None:
            break
        record_type, payload = record
        if record_type == 1 and b"train/proxy_mean" in payload:
            metrics = _parse_metrics_blob(payload)
            if metrics:
                rows.append(metrics)
    return rows


def _load_training_history(run_dir: Path) -> list[dict[str, float]]:
    wandb_root = run_dir / "logs" / "wandb"
    history: list[dict[str, float]] = []
    offset = 0
    for run_path in sorted(wandb_root.glob("offline-run-*")):
        wandb_files = list(run_path.glob("run-*.wandb"))
        if not wandb_files:
            continue
        segment = _load_wandb_history(wandb_files[0])
        if not segment:
            continue
        for index, row in enumerate(segment):
            point = dict(row)
            point["_iter"] = offset + index
            history.append(point)
        offset += len(segment)
    return history


def _load_sample_rows(run_dir: Path) -> list[dict]:
    sample_path = run_dir / "samples" / "samples.jsonl"
    if not sample_path.is_file():
        return []
    rows: list[dict] = []
    with sample_path.open(encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def _discovery_curves(rows: list[dict]) -> tuple[list[int], list[int], list[float]]:
    seen: set[str] = set()
    unique_curve: list[int] = []
    mean_proxy_curve: list[float] = []
    running_total = 0.0
    for index, row in enumerate(rows, start=1):
        smiles = row.get("smiles")
        if smiles is not None:
            seen.add(smiles)
        running_total += float(row["proxy"])
        unique_curve.append(len(seen))
        mean_proxy_curve.append(running_total / index)
    return list(range(1, len(rows) + 1)), unique_curve, mean_proxy_curve


def run(args: argparse.Namespace) -> list[Path]:
    suite_dir = Path(args.suite_dir).expanduser().resolve()
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
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "font.size": 8,
            "figure.dpi": args.dpi,
        }
    )

    run_dirs = {
        "grpo": suite_dir / "grpo" / "seed_0" / "batch",
        "count_ips_grpo": suite_dir / "count_ips_grpo" / "seed_0" / "batch",
    }
    training = {method: _load_training_history(path) for method, path in run_dirs.items()}
    samples = {method: _load_sample_rows(path) for method, path in run_dirs.items()}
    summaries = {
        method: _load_json(path / "samples" / "summary.json")
        for method, path in run_dirs.items()
        if (path / "samples" / "summary.json").is_file()
    }

    if not any(training.values()) and not any(samples.values()):
        raise RuntimeError("No training or sampling data found for seed 0")

    written: list[Path] = []

    # Training curves
    training_specs = (
        ("train/proxy_mean", "Train proxy mean", "sEH proxy"),
        ("train/num_unique_molecules", "Train unique molecules (batch)", "unique molecules"),
        ("train/loss", "Train loss", "loss"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.8))
    for axis, (metric_key, title, ylabel) in zip(axes, training_specs):
        for method in METHOD_ORDER:
            history = training.get(method, [])
            if not history:
                continue
            x_values = [int(row["_iter"]) for row in history if metric_key in row]
            y_values = [float(row[metric_key]) for row in history if metric_key in row]
            if not x_values:
                continue
            axis.plot(
                x_values,
                y_values,
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
                linewidth=1.1,
                alpha=0.9,
            )
        axis.set_title(title)
        axis.set_xlabel("iteration")
        axis.set_ylabel(ylabel)
        axis.grid(color="#B0B0B0", linewidth=0.4, alpha=0.35)
        if metric_key == "train/num_unique_molecules":
            axis.set_yscale("log")
    axes[0].legend(frameon=False, loc="upper right")
    fig.suptitle("Training diagnostics — seh_paper_medium seed 0", fontsize=10)
    fig.tight_layout(pad=0.8)
    written.extend(_save(fig, output_dir, "training_curves", args.dpi))
    plt.close(fig)

    # Count IPS duplicate fraction
    fig, axis = plt.subplots(figsize=(5.2, 2.8))
    count_history = training.get("count_ips_grpo", [])
    if count_history:
        x_values = [
            int(row["_iter"])
            for row in count_history
            if "train/ips_duplicate_fraction" in row
        ]
        y_values = [
            float(row["train/ips_duplicate_fraction"])
            for row in count_history
            if "train/ips_duplicate_fraction" in row
        ]
        axis.plot(
            x_values,
            y_values,
            color=METHOD_COLORS["count_ips_grpo"],
            linewidth=1.2,
        )
        axis.set_ylim(0, 1.02)
        axis.set_xlabel("iteration")
        axis.set_ylabel("ips_duplicate_fraction")
        axis.set_title("Count IPS-GRPO — duplicate outcome fraction during training")
        axis.grid(color="#B0B0B0", linewidth=0.4, alpha=0.35)
        fig.tight_layout(pad=0.8)
        written.extend(_save(fig, output_dir, "training_ips_duplicate_fraction", args.dpi))
    plt.close(fig)

    # Sampling proxy distributions (50k)
    available = [method for method in METHOD_ORDER if samples[method]]
    if available:
        fig, axis = plt.subplots(figsize=(5.4, 2.8))
        proxy_values = [[float(row["proxy"]) for row in samples[method]] for method in available]
        boxes = axis.boxplot(
            proxy_values,
            tick_labels=[METHOD_LABELS[method] for method in available],
            showfliers=False,
            patch_artist=True,
            widths=0.62,
            medianprops={"color": "black", "linewidth": 1.0},
        )
        for patch, method in zip(boxes["boxes"], available):
            patch.set_facecolor(METHOD_COLORS[method])
            patch.set_alpha(0.85)
        axis.axhline(7.0, color="black", linestyle="--", linewidth=0.9, label="mode threshold = 7.0")
        axis.set_ylabel("sampled sEH proxy")
        axis.set_title("Final-policy sampling @ 2,000 iterations (50,000 samples)")
        axis.grid(axis="y", color="#B0B0B0", linewidth=0.45, alpha=0.4)
        axis.legend(frameon=False, fontsize=6.5)
        fig.tight_layout(pad=0.7)
        written.extend(_save(fig, output_dir, "sampling_proxy_distribution", args.dpi))
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(8.8, 2.8))
        for method in available:
            draw_index, unique_curve, mean_proxy_curve = _discovery_curves(samples[method])
            plot_kwargs = {
                "color": METHOD_COLORS[method],
                "label": METHOD_LABELS[method],
                "marker": METHOD_MARKERS[method],
                "markevery": max(1, len(draw_index) // 12),
                "markersize": 2.6,
                "linewidth": 1.05,
            }
            axes[0].plot(draw_index, unique_curve, **plot_kwargs)
            axes[1].plot(draw_index, mean_proxy_curve, **plot_kwargs)
        axes[0].set_xscale("log")
        axes[0].set_xlabel("samples drawn (log scale)")
        axes[0].set_ylabel("distinct SMILES")
        axes[0].set_title("(a) support discovery")
        axes[1].set_xlabel("samples drawn")
        axes[1].set_ylabel("cumulative mean proxy")
        axes[1].set_title("(b) running mean proxy")
        for axis in axes:
            axis.grid(color="#B0B0B0", linewidth=0.45, alpha=0.35)
            axis.legend(frameon=False, fontsize=6.5)
        fig.suptitle("Sampling trajectories — 50k final-checkpoint draws", fontsize=10)
        fig.tight_layout(pad=0.8)
        written.extend(_save(fig, output_dir, "sampling_discovery_curves", args.dpi))
        plt.close(fig)

        fig, axis = plt.subplots(figsize=(5.8, 2.8))
        for method in available:
            counts = Counter(row["smiles"] for row in samples[method] if row.get("smiles"))
            if not counts:
                continue
            top_counts = counts.most_common(10)
            y = np.arange(len(top_counts))
            axis.barh(
                y + (0.18 if method == "grpo" else -0.18),
                [count for _, count in top_counts],
                height=0.35,
                color=METHOD_COLORS[method],
                alpha=0.85,
                label=METHOD_LABELS[method],
            )
        axis.set_xscale("log")
        axis.set_xlabel("sample count (log scale)")
        axis.set_ylabel("rank (top 10 SMILES)")
        axis.set_title("Top sampled molecules at 2,000 iterations")
        axis.grid(axis="x", color="#B0B0B0", linewidth=0.4, alpha=0.35)
        axis.legend(frameon=False)
        fig.tight_layout(pad=0.8)
        written.extend(_save(fig, output_dir, "sampling_top_molecules", args.dpi))
        plt.close(fig)

    # Summary bars from final 50k sampling
    if summaries:
        fig, axes = plt.subplots(1, 4, figsize=(9.6, 2.6))
        metric_specs = (
            ("n_unique", "unique SMILES", None),
            ("mean_proxy", "mean proxy", None),
            ("importance_ess_fraction", "ESS fraction", (0, 1.05)),
            ("unique_fraction", "unique fraction", (0, max(0.001, max(float(summaries[m]["unique_fraction"]) for m in summaries) * 1.4))),
        )
        methods = [method for method in METHOD_ORDER if method in summaries]
        x = np.arange(len(methods))
        for axis, (key, ylabel, ylim) in zip(axes, metric_specs):
            values = [float(summaries[method][key]) for method in methods]
            axis.bar(
                x,
                values,
                color=[METHOD_COLORS[method] for method in methods],
                alpha=0.9,
                edgecolor="white",
            )
            axis.set_xticks(x, [METHOD_LABELS[method] for method in methods], rotation=20, ha="right")
            axis.set_ylabel(ylabel)
            if ylim is not None:
                axis.set_ylim(*ylim)
            axis.grid(axis="y", color="#B0B0B0", linewidth=0.4, alpha=0.35)
            axis.bar_label(axis.containers[0], labels=[f"{value:.4g}" for value in values], padding=2, fontsize=6)
        fig.suptitle("Final sampling summary @ 2,000 iterations (50k)", fontsize=10)
        fig.tight_layout(pad=0.8)
        written.extend(_save(fig, output_dir, "sampling_summary_bars", args.dpi))
        plt.close(fig)

    # Train vs sample proxy at documented checkpoints
    checkpoint_rows = [
        ("grpo", 500, 8.05, 4.15),
        ("grpo", 2000, 3.43, 3.73),
        ("count_ips_grpo", 500, 7.77, 6.04),
        ("count_ips_grpo", 2000, 5.43, 4.80),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8), sharex=True)
    for method in METHOD_ORDER:
        method_rows = [row for row in checkpoint_rows if row[0] == method]
        iters = [row[1] for row in method_rows]
        train_vals = [row[2] for row in method_rows]
        sample_vals = [row[3] for row in method_rows]
        axes[0].plot(
            iters,
            train_vals,
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            label=METHOD_LABELS[method],
            linewidth=1.2,
        )
        axes[1].plot(
            iters,
            sample_vals,
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            label=METHOD_LABELS[method],
            linewidth=1.2,
        )
    for axis, title in zip(axes, ("Training batch proxy", "Sampled proxy (5k @500, 50k @2000)")):
        axis.set_title(title)
        axis.set_xlabel("iteration")
        axis.set_ylabel("proxy")
        axis.grid(color="#B0B0B0", linewidth=0.4, alpha=0.35)
        axis.legend(frameon=False)
    fig.suptitle("Train vs sampling proxy — collapse with more training", fontsize=10)
    fig.tight_layout(pad=0.8)
    written.extend(_save(fig, output_dir, "train_vs_sample_proxy", args.dpi))
    plt.close(fig)

    for path in written:
        print(f"PLOT={path}")
    return written


def main(argv: list[str] | None = None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
