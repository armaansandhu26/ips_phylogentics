#!/usr/bin/env python3
"""Plot empirical log-mass against log likelihood at signature or topology level."""

from __future__ import annotations

import argparse
from collections import defaultdict
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
from grpo_experiments.scripts.compare_sampling import load_all_raw_summaries, sample_run


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
    parser.add_argument(
        "--checkpoint",
        default="final_checkpoint.pt",
        help="Checkpoint filename inside each run dir.",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=1,
        help="Only plot signatures observed at least this many times.",
    )
    parser.add_argument(
        "--group-by",
        choices=("signature", "topology"),
        default="signature",
        help="Aggregate counts at the signature or topology level.",
    )
    parser.add_argument(
        "--from-raw",
        action="store_true",
        help="Load per-tree raw bundles from <output-dir>/raw_samples instead of resampling.",
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


def grouped_points(
    summary: dict[str, Any],
    *,
    group_by: str,
    min_count: int,
) -> dict[str, np.ndarray]:
    counts = defaultdict(int)
    signature_totals = defaultdict(float)
    signature_counts = defaultdict(int)
    topology_signatures = defaultdict(set)

    signatures = summary["signatures"]
    topology_ids = summary["topology_ids"]
    log_scores = summary["log_scores"]

    for signature, topology_id, log_score in zip(
        signatures,
        topology_ids,
        log_scores,
        strict=True,
    ):
        signature = str(signature)
        topology_id = str(topology_id)
        log_score = float(log_score)

        signature_totals[signature] += log_score
        signature_counts[signature] += 1
        topology_signatures[topology_id].add(signature)

        if group_by == "signature":
            group_id = signature
        else:
            group_id = topology_id
        counts[group_id] += 1

    xs = []
    ys = []
    plotted_counts = []
    n_samples = int(summary["samples"])
    log_total = float(np.log(n_samples))

    for group_id, count in counts.items():
        if count < min_count:
            continue

        if group_by == "signature":
            x_value = signature_totals[group_id] / signature_counts[group_id]
        else:
            observed_signature_log_scores = np.asarray(
                [
                    signature_totals[signature] / signature_counts[signature]
                    for signature in topology_signatures[group_id]
                ],
                dtype=np.float64,
            )
            x_value = float(np.logaddexp.reduce(observed_signature_log_scores))

        xs.append(x_value)
        ys.append(float(np.log(count) - log_total))
        plotted_counts.append(count)

    x_arr = np.asarray(xs, dtype=np.float64)
    y_arr = np.asarray(ys, dtype=np.float64)
    c_arr = np.asarray(plotted_counts, dtype=np.int64)

    if x_arr.size == 0:
        return {
            "log_scores": x_arr,
            "log_qhat": y_arr,
            "counts": c_arr,
        }

    order = np.argsort(x_arr)
    return {
        "log_scores": x_arr[order],
        "log_qhat": y_arr[order],
        "counts": c_arr[order],
    }


def plot_signature_mass_scatter(
    summaries: list[dict[str, Any]],
    output_path: Path,
    *,
    group_by: str,
    samples: int,
    min_count: int,
    with_fit: bool,
) -> list[dict[str, Any]]:
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(11, 6), dpi=220, constrained_layout=True)
    fit_rows: list[dict[str, Any]] = []
    annotation_lines: list[str] = []

    for idx, row in enumerate(summaries):
        points = grouped_points(row, group_by=group_by, min_count=min_count)
        xs = points["log_scores"]
        ys = points["log_qhat"]
        counts = points["counts"]
        color = cmap(idx % 10)
        if group_by == "signature":
            unique_items = int(row["unique_signatures"])
            item_label = "signatures"
        else:
            unique_items = int(row["unique_topologies"])
            item_label = "topologies"
        annotation_lines.append(f"{row['label']}: {unique_items} unique {item_label}")

        ax.scatter(
            xs,
            ys,
            s=10,
            alpha=0.45,
            color=color,
            label=f"{row['label']} ({unique_items} {item_label})",
        )

        fit_payload: dict[str, Any] = {
            "label": row["label"],
            "group_by": group_by,
            "samples": int(samples),
            "unique_items_plotted": int(xs.size),
            "min_count": int(min_count),
            "count_min": int(counts.min()) if counts.size else None,
            "count_max": int(counts.max()) if counts.size else None,
        }
        if with_fit and xs.size >= 2:
            slope, intercept = np.polyfit(xs, ys, deg=1)
            corr = float(np.corrcoef(xs, ys)[0, 1]) if xs.size >= 2 else None
            xfit = np.linspace(float(xs.min()), float(xs.max()), 200)
            yfit = slope * xfit + intercept
            ax.plot(xfit, yfit, color=color, linewidth=1.6, alpha=0.9)
            fit_payload["fit"] = {
                "slope": float(slope),
                "intercept": float(intercept),
                "pearson_r": corr,
            }
        fit_rows.append(fit_payload)

    title_target = "Per-signature" if group_by == "signature" else "Per-topology"
    xlabel = (
        "Terminal-tree log likelihood"
        if group_by == "signature"
        else "Observed-support topology log-mass proxy"
    )
    ax.set_title(f"{title_target} empirical ln-mass vs log likelihood ({samples} samples/run)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("ln q_hat(x)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=9, loc="best")
    ax.text(
        0.015,
        0.985,
        "\n".join(annotation_lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return fit_rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.from_raw:
        summaries = load_all_raw_summaries(args.output_dir)
        if args.manifest.exists():
            order = [label_for_row(row) for row in load_json(args.manifest).get("runs", [])]
            by_label = {row["label"]: row for row in summaries}
            summaries = [by_label[label] for label in order]
        print(f"loaded {len(summaries)} raw sample bundles from {args.output_dir / 'raw_samples'}")
    else:
        specs = load_runs(args.manifest)
        device = choose_device(args.device)
        summaries = []
        for idx, (label, run_dir, _row) in enumerate(specs):
            print(f"sampling {args.samples} trees from {label} ({run_dir.name}) on {device}")
            summaries.append(
                sample_run(
                    run_dir,
                    label,
                    device=device,
                    samples=args.samples,
                    batch_size=args.batch_size,
                    seed=args.seed + idx,
                    checkpoint_name=args.checkpoint,
                    estimate_mll=False,
                )
            )

    base_name = f"{args.group_by}_logqhat_vs_loglikelihood_{args.samples//1000}k"
    if args.min_count > 1:
        base_name += f"_mincount{args.min_count}"

    fit_rows = plot_signature_mass_scatter(
        summaries,
        args.output_dir / f"{base_name}.png",
        group_by=args.group_by,
        samples=args.samples,
        min_count=args.min_count,
        with_fit=False,
    )
    fit_rows = plot_signature_mass_scatter(
        summaries,
        args.output_dir / f"{base_name}_fit.png",
        group_by=args.group_by,
        samples=args.samples,
        min_count=args.min_count,
        with_fit=True,
    )
    save_json(args.output_dir / f"{base_name}_fits.json", {"runs": fit_rows})
    print(f"\nsaved outputs under: {args.output_dir}")


if __name__ == "__main__":
    main()
