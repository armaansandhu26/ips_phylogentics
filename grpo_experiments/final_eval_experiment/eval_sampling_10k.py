#!/usr/bin/env python3
"""Checkpoint sampling eval for final follow-up runs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from grpo_experiments.eval_utils import choose_device, load_json, save_json
from grpo_experiments.scripts.compare_sampling import (
    compute_bin_edges,
    compute_bin_frequencies,
    json_ready_summary,
    load_all_raw_summaries,
    plot_sampling_comparison,
    plot_sampling_distributions,
    plot_sampling_overlay,
    plot_score_density,
    save_raw_sample_bundle,
    save_scores_cache,
    sample_run,
)


DEFAULT_MANIFEST = ROOT / "manifest.json"
DEFAULT_OUTPUT_DIR = ROOT / "eval" / "topo" / "checkpoint_sampling_10k_40bins"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--n-bins", type=int, default=40)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--run-index",
        type=int,
        default=None,
        help="Sample only the manifest run at this index (for parallel GPU jobs).",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--sample-only",
        action="store_true",
        help="Sample trees, save raw bundles, and skip plot generation.",
    )
    mode.add_argument(
        "--plot-only",
        action="store_true",
        help="Load saved raw bundles and generate plots/summary outputs.",
    )
    return parser.parse_args()


def label_for_row(row: dict[str, Any]) -> str:
    method = row.get("method")
    run_id = str(row.get("id", ""))
    replay = row.get("replay_batch", "r?")
    if method == "phylgfn":
        return f"phylgfn_r{replay}"
    if method == "hybrid_ips_grpo":
        pfloor = row.get("ips_prob_floor")
        if pfloor is None:
            return f"hyb_ips_r{replay}"
        if abs(float(pfloor) - 0.002) < 1e-12:
            ptag = "p002"
        elif abs(float(pfloor) - 1e-6) < 1e-18:
            ptag = "p1em6"
        else:
            ptag = f"p{pfloor:g}".replace(".", "p").replace("-", "m")
        return f"hyb_ips_{ptag}_r{replay}"
    return run_id or Path(str(row["run_dir"])).name


def load_runs(manifest_path: Path) -> list[tuple[str, Path, dict[str, Any]]]:
    manifest = load_json(manifest_path)
    rows = manifest.get("runs", [])
    if not rows:
        raise SystemExit(f"No runs found in {manifest_path}")
    return [(label_for_row(row), Path(row["run_dir"]), row) for row in rows]


def order_summaries(
    summaries: list[dict[str, Any]],
    specs: list[tuple[str, Path, dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    if not specs:
        return summaries
    order = [label for label, _, _ in specs]
    by_label = {row["label"]: row for row in summaries}
    missing = [label for label in order if label not in by_label]
    if missing:
        raise SystemExit(f"missing raw bundles for labels: {missing}")
    return [by_label[label] for label in order]


def plot_logscore_logcount_points(
    summaries: list[dict[str, Any]],
    output_path: Path,
    *,
    n_bins: int,
    samples: int,
    with_fit: bool = False,
) -> list[dict[str, Any]]:
    cmap = plt.get_cmap("tab10")
    combined = np.concatenate([np.asarray(row["log_scores"], dtype=np.float64) for row in summaries])
    bin_edges = np.linspace(float(combined.min()), float(combined.max()), n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    fig, ax = plt.subplots(figsize=(11, 6), dpi=220, constrained_layout=True)
    fit_rows: list[dict[str, Any]] = []

    for idx, row in enumerate(summaries):
        scores = np.asarray(row["log_scores"], dtype=np.float64)
        counts, _ = np.histogram(scores, bins=bin_edges)
        mask = counts > 0
        xs = bin_centers[mask]
        ys = np.log(counts[mask].astype(np.float64))
        color = cmap(idx % 10)

        ax.scatter(
            xs,
            ys,
            s=28,
            alpha=0.9,
            color=color,
            label=row["label"],
        )

        fit_payload: dict[str, Any] = {
            "label": row["label"],
            "nonzero_bins": int(mask.sum()),
        }
        if with_fit and mask.sum() >= 2:
            slope, intercept = np.polyfit(xs, ys, deg=1)
            xfit = np.linspace(float(xs.min()), float(xs.max()), 200)
            yfit = slope * xfit + intercept
            ax.plot(xfit, yfit, color=color, linewidth=1.6, alpha=0.9)
            fit_payload["fit"] = {
                "slope": float(slope),
                "intercept": float(intercept),
            }
        fit_rows.append(fit_payload)

    ax.set_title(f"Checkpoint sampling: ln(score) vs ln(count) ({samples} samples/run, {n_bins} bins)")
    ax.set_xlabel("ln(score) bin center")
    ax.set_ylabel("ln(sample count in bin)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=9, loc="best")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return fit_rows


def sample_specs(
    specs: list[tuple[str, Path, dict[str, Any]]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    device = choose_device(args.device)
    selected = specs
    if args.run_index is not None:
        if args.run_index < 0 or args.run_index >= len(specs):
            raise SystemExit(f"--run-index {args.run_index} out of range for {len(specs)} runs")
        selected = [specs[args.run_index]]

    summaries = []
    for idx, (label, run_dir, _row) in enumerate(selected):
        seed = args.seed + (args.run_index if args.run_index is not None else idx)
        print(f"sampling {args.samples} trees from {label} ({run_dir.name}) on {device}")
        summary = sample_run(
            run_dir,
            label,
            device=device,
            samples=args.samples,
            batch_size=args.batch_size,
            seed=seed,
            checkpoint_name="final_checkpoint.pt",
            estimate_mll=False,
        )
        raw_path = save_raw_sample_bundle(summary, args.output_dir)
        print(f"saved raw samples -> {raw_path}")
        summaries.append(summary)
    return summaries


def write_outputs(
    summaries: list[dict[str, Any]],
    args: argparse.Namespace,
    *,
    device: str | None = None,
) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bin_edges = compute_bin_edges(summaries, args.n_bins)
    frequencies = compute_bin_frequencies(summaries, bin_edges)

    payload = {
        "metadata": {
            "experiment": "final_eval_followup",
            "samples_per_run": args.samples,
            "batch_size": args.batch_size,
            "seed_base": args.seed,
            "n_bins": args.n_bins,
            "device": device,
            "reward_bin_edges": [float(x) for x in bin_edges],
            "raw_samples_dir": str(args.output_dir / "raw_samples"),
        },
        "runs": [
            json_ready_summary(
                row,
                bin_edges=bin_edges,
                bin_frequencies=frequencies[row["label"]],
            )
            for row in summaries
        ],
    }
    save_json(args.output_dir / "sampling_summary.json", payload)
    save_scores_cache(summaries, args.output_dir / "sampling_scores.npz")

    title = "Final eval follow-up checkpoint sampling"
    plot_sampling_comparison(
        summaries,
        bin_edges,
        frequencies,
        args.output_dir / "sampling_comparison.png",
        args.top_k,
        samples=args.samples,
        n_bins=args.n_bins,
        title_context=title,
    )
    plot_sampling_distributions(
        summaries,
        args.output_dir / "sampling_distributions.png",
        samples=args.samples,
        title_context=title,
    )
    plot_sampling_overlay(
        summaries,
        args.output_dir / "sampling_distributions_overlay.png",
        title_context=title,
    )
    plot_score_density(
        summaries,
        args.output_dir / "sampling_score_density.png",
        title_context=title,
    )

    fit_rows = plot_logscore_logcount_points(
        summaries,
        args.output_dir / f"sampling_logscore_logcount_points_{args.samples//1000}k_{args.n_bins}bins.png",
        n_bins=args.n_bins,
        samples=args.samples,
        with_fit=False,
    )
    fit_rows = plot_logscore_logcount_points(
        summaries,
        args.output_dir / f"sampling_logscore_logcount_points_fit_{args.samples//1000}k_{args.n_bins}bins.png",
        n_bins=args.n_bins,
        samples=args.samples,
        with_fit=True,
    )
    save_json(args.output_dir / "sampling_logscore_logcount_fits.json", {"runs": fit_rows})

    baseline = summaries[0]["log_scores"]
    baseline_mean = float(baseline.mean())
    lines = []
    for row in summaries:
        scores = row["log_scores"]
        lines.append(
            f"  {row['label']}: mean={scores.mean():.2f} ({scores.mean() - baseline_mean:+.2f}) "
            f"topo={row['unique_topologies']} dup={row['topology_duplicate_fraction']:.3f}"
        )
    (args.output_dir / "sampling_report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nsaved outputs under: {args.output_dir}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        specs = load_runs(args.manifest) if args.manifest.exists() else None
        summaries = order_summaries(load_all_raw_summaries(args.output_dir), specs)
        write_outputs(summaries, args)
        return

    specs = load_runs(args.manifest)
    summaries = sample_specs(specs, args)
    if args.sample_only:
        print(f"\nsaved raw sample bundles under: {args.output_dir / 'raw_samples'}")
        return

    write_outputs(summaries, args, device=choose_device(args.device))


if __name__ == "__main__":
    main()
