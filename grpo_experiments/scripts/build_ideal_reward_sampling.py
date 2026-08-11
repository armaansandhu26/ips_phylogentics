#!/usr/bin/env python3
"""Build and plot ideal reward-proportional signature sampling for a fixed catalog."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grpo_experiments.eval_utils import load_json
from grpo_experiments.ideal_sampling import (
    build_signature_reward_catalog_from_trees,
    compute_ideal_signature_sampling_table,
    empirical_qhat_by_signature,
    plot_ideal_qhat_vs_log_score,
    plot_ideal_qhat_vs_reward,
    plot_qhat_vs_reward_comparison,
    save_ideal_signature_sampling_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute ideal reward-proportional counts for each signature in a "
            "sampled_trees JSON catalog and write table + plots."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="sampled_trees JSON containing one row per signature (log_score, log_reward).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for ideal_sampling_100k.json and PNG plots.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100_000,
        help="Total samples to allocate proportionally to reward.",
    )
    parser.add_argument(
        "--compare-input",
        nargs="+",
        type=Path,
        default=None,
        help="Optional sampled_trees JSON files to overlay against the ideal reward axis.",
    )
    parser.add_argument(
        "--compare-labels",
        nargs="+",
        default=None,
        help="Labels for --compare-input files (must match count).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = load_json(args.input)
    trees = payload.get("trees")
    if not trees:
        raise ValueError(f"no trees found in {args.input}")

    catalog = build_signature_reward_catalog_from_trees(trees)
    rows = compute_ideal_signature_sampling_table(catalog, n_samples=args.n_samples)
    output_dir = args.output_dir
    table_path = output_dir / "ideal_sampling_100k.json"
    save_ideal_signature_sampling_table(
        table_path,
        rows,
        metadata={
            "source_json": str(args.input),
            "n_samples": args.n_samples,
            "reward_definition": "R(x) = exp(log_reward)",
            "sampling_target": "q*(x) proportional to R(x)",
        },
    )
    plot_ideal_qhat_vs_log_score(
        rows,
        output_dir / "ideal_qhat_vs_loglikelihood_100k.png",
        n_samples=args.n_samples,
    )
    plot_ideal_qhat_vs_reward(
        rows,
        output_dir / "ideal_qhat_vs_reward_100k.png",
        n_samples=args.n_samples,
    )
    print(f"wrote {table_path}")
    print(f"wrote {output_dir / 'ideal_qhat_vs_loglikelihood_100k.png'}")
    print(f"wrote {output_dir / 'ideal_qhat_vs_reward_100k.png'}")

    if args.compare_input:
        compare_labels = args.compare_labels
        if compare_labels is not None and len(compare_labels) != len(args.compare_input):
            raise ValueError("--compare-labels count must match --compare-input count")
        empirical_runs: list[tuple[str, dict[str, float]]] = []
        for idx, compare_path in enumerate(args.compare_input):
            compare_payload = load_json(compare_path)
            compare_trees = compare_payload.get("trees")
            if not compare_trees:
                raise ValueError(f"no trees found in {compare_path}")
            label = (
                compare_labels[idx]
                if compare_labels is not None
                else compare_path.stem
            )
            empirical_runs.append(
                (
                    label,
                    empirical_qhat_by_signature(
                        compare_trees,
                        n_samples=args.n_samples,
                    ),
                )
            )
        comparison_path = output_dir / "signature_qhat_vs_reward_100k.png"
        comparison_fit_path = output_dir / "signature_qhat_vs_reward_100k_fit.png"
        plot_qhat_vs_reward_comparison(
            rows,
            empirical_runs,
            comparison_path,
            n_samples=args.n_samples,
            with_fit=False,
        )
        plot_qhat_vs_reward_comparison(
            rows,
            empirical_runs,
            comparison_fit_path,
            n_samples=args.n_samples,
            with_fit=True,
        )
        print(f"wrote {comparison_path}")
        print(f"wrote {comparison_fit_path}")


if __name__ == "__main__":
    main()
