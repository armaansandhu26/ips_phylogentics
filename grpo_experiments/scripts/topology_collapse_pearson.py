#!/usr/bin/env python3
"""Collapse signature-level Pearson outputs to topology-level diagnostics.

This script reads the `pairs.json` files produced by `eval_tree_logq_pearson.py`,
groups rows by `topology_id`, collapses each topology to a single point, and
computes topology-level Pearson correlation.

It is intended as a diagnostic sanity check on small datasets where
signature-level evaluation may over-emphasize branch-length calibration.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from grpo_experiments.eval_utils import load_json, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collapse signature-level Pearson outputs to topology-level diagnostics.",
    )
    parser.add_argument(
        "--pearson-eval-dir",
        type=Path,
        required=True,
        help="Directory produced by eval_tree_logq_pearson.py containing per-run subdirectories.",
    )
    parser.add_argument(
        "--run-label",
        action="append",
        default=[],
        help="Optional specific run label(s) to process. Defaults to all runs in summary.json.",
    )
    parser.add_argument(
        "--collapse-logq",
        choices=["mean", "logsumexp"],
        default="mean",
        help="How to collapse signature-level estimated log q(tree) within a topology.",
    )
    parser.add_argument(
        "--collapse-score",
        choices=["mean", "best"],
        default="mean",
        help="How to collapse true log scores within a topology.",
    )
    return parser.parse_args()


def pearson_or_none(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    corr = np.corrcoef(np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64))
    return float(corr[0, 1])


def collapse_logq(values: list[float], mode: str) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if mode == "mean":
        return float(arr.mean())
    if mode == "logsumexp":
        vmax = float(arr.max())
        return float(vmax + math.log(np.exp(arr - vmax).sum()))
    raise ValueError(f"Unknown collapse-logq mode: {mode}")


def collapse_score(values: list[float], mode: str) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if mode == "mean":
        return float(arr.mean())
    if mode == "best":
        return float(arr.max())
    raise ValueError(f"Unknown collapse-score mode: {mode}")


def choose_band(rows: list[dict[str, Any]]) -> str:
    counts = Counter(row["score_band"] for row in rows)
    return counts.most_common(1)[0][0]


def topology_colors(bands: list[str]) -> list[str]:
    color_map = {"low": "#d55e00", "medium": "#0072b2", "high": "#009e73"}
    return [color_map.get(band, "#444444") for band in bands]


def plot_collapsed(rows: list[dict[str, Any]], output_path: Path, *, title: str) -> None:
    x = np.asarray([row["collapsed_log_q"] for row in rows], dtype=np.float64)
    y = np.asarray([row["collapsed_true_log_score"] for row in rows], dtype=np.float64)
    bands = [row["collapsed_band"] for row in rows]

    fig, ax = plt.subplots(figsize=(7, 5), dpi=180, constrained_layout=True)
    ax.scatter(x, y, s=40, c=topology_colors(bands), alpha=0.8, edgecolors="none")
    ax.set_xlabel("Collapsed estimated log q(topology)")
    ax.set_ylabel("Collapsed true log score")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)

    for band, color in [("low", "#d55e00"), ("medium", "#0072b2"), ("high", "#009e73")]:
        ax.scatter([], [], s=36, c=color, label=band)
    ax.legend(frameon=False, title="Majority band")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def collapse_pairs(
    pairs: list[dict[str, Any]],
    *,
    collapse_logq_mode: str,
    collapse_score_mode: str,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pairs:
        grouped[row["topology_id"]].append(row)

    collapsed_rows: list[dict[str, Any]] = []
    for topology_id, rows in grouped.items():
        collapsed_rows.append(
            {
                "topology_id": topology_id,
                "collapsed_log_q": collapse_logq(
                    [row["estimated_log_q"] for row in rows],
                    collapse_logq_mode,
                ),
                "collapsed_true_log_score": collapse_score(
                    [row["true_log_score"] for row in rows],
                    collapse_score_mode,
                ),
                "collapsed_band": choose_band(rows),
                "signature_count": len(rows),
                "source_label_counts": dict(Counter(row["source_label"] for row in rows)),
            }
        )
    collapsed_rows.sort(key=lambda row: row["collapsed_true_log_score"])
    return collapsed_rows


def main() -> None:
    args = parse_args()
    summary_path = args.pearson_eval_dir / "summary.json"
    summary = load_json(summary_path)

    labels = args.run_label or [row["label"] for row in summary["runs"]]
    out_dir = args.pearson_eval_dir / "topology_collapsed"
    out_dir.mkdir(parents=True, exist_ok=True)

    combined = {
        "pearson_eval_dir": str(args.pearson_eval_dir),
        "collapse_logq": args.collapse_logq,
        "collapse_score": args.collapse_score,
        "runs": [],
    }

    for label in labels:
        run_dir = args.pearson_eval_dir / label
        pairs_payload = load_json(run_dir / "pairs.json")
        pairs = pairs_payload["pairs"]
        collapsed_rows = collapse_pairs(
            pairs,
            collapse_logq_mode=args.collapse_logq,
            collapse_score_mode=args.collapse_score,
        )

        overall = pearson_or_none(
            [row["collapsed_log_q"] for row in collapsed_rows],
            [row["collapsed_true_log_score"] for row in collapsed_rows],
        )
        by_band = {}
        for band in ("low", "medium", "high"):
            band_rows = [row for row in collapsed_rows if row["collapsed_band"] == band]
            by_band[band] = pearson_or_none(
                [row["collapsed_log_q"] for row in band_rows],
                [row["collapsed_true_log_score"] for row in band_rows],
            )

        payload = {
            "metadata": {
                "label": label,
                "input_pairs_json": str(run_dir / "pairs.json"),
                "collapse_logq": args.collapse_logq,
                "collapse_score": args.collapse_score,
            },
            "collapsed_pairs": collapsed_rows,
            "summary": {
                "label": label,
                "topology_count": len(collapsed_rows),
                "pearson_overall": overall,
                "pearson_by_band": by_band,
            },
        }

        run_out_dir = out_dir / label
        run_out_dir.mkdir(parents=True, exist_ok=True)
        save_json(run_out_dir / "collapsed_pairs.json", payload)
        plot_collapsed(
            collapsed_rows,
            run_out_dir / "collapsed_scatter.png",
            title=(
                f"{label} — topology-collapsed Pearson={overall:.4f}"
                if overall is not None
                else f"{label} — topology-collapsed Pearson=n/a"
            ),
        )
        combined["runs"].append(payload["summary"])

        overall_str = f"{overall:.4f}" if overall is not None else "n/a"
        print(f"{label}: topology-collapsed overall pearson={overall_str} (n={len(collapsed_rows)})")

    save_json(out_dir / "summary.json", combined)
    print(f"\nsaved outputs under: {out_dir}")


if __name__ == "__main__":
    main()
