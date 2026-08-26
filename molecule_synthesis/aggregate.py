"""Aggregate per-method sample summaries into a comparison table."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def aggregate_suite(suite_dir: Path) -> tuple[Path, Path]:
    manifest_path = suite_dir / "suite.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Suite manifest not found: {manifest_path}")
    with manifest_path.open(encoding="utf-8") as handle:
        suite = json.load(handle)

    rows = []
    optional_metrics = (
        "tv_to_reward_target",
        "js_to_reward_target",
        "l1_to_reward_target",
        "target_mass_covered",
        "sample_support_coverage",
        "out_of_support_fraction",
        "log_probability_correlation",
        "log_probability_calibration_slope",
        "median_proxy",
        "top_10_unique_mean_reward",
        "top_100_unique_mean_reward",
        "high_reward_outcomes_found",
        "importance_ess",
        "importance_ess_fraction",
        "log_importance_weight_mean",
        "log_importance_weight_std",
        "n_mode_candidates",
        "n_modes",
        "top_modes_mean_proxy",
        "top_modes_mean_reward",
        "top_modes_mean_qed",
        "top_modes_mean_molecular_weight",
        "top_modes_mean_sa_score",
        "top_modes_unique_scaffolds",
        "n_scaffolds_proxy_gt_7",
        "n_scaffolds_proxy_gt_8",
        "train_final_ips_duplicate_fraction",
        "train_final_ips_unique_outcomes",
        "train_final_ips_clipped_fraction",
        "train_final_ips_ess_fraction",
        "train_final_reverse_loss",
    )
    for method, method_runs in suite["runs"].items():
        normalized_runs = (
            {"legacy": method_runs} if isinstance(method_runs, str) else method_runs
        )
        for seed_raw, run_dir_raw in normalized_runs.items():
            summary_path = Path(run_dir_raw) / "samples" / "summary.json"
            if not summary_path.is_file():
                continue
            with summary_path.open(encoding="utf-8") as handle:
                summary = json.load(handle)
            seed = summary.get("seed")
            if seed is None and seed_raw != "legacy":
                seed = int(seed_raw)
            row = {
                "method": method,
                "seed": seed,
                "n_sampled": summary["n_sampled"],
                "valid_fraction": summary["valid_fraction"],
                "n_unique": summary["n_unique"],
                "unique_fraction": summary["unique_fraction"],
                "mean_log_reward": summary["mean_log_reward"],
                "mean_proxy": summary["mean_proxy"],
                "run_dir": run_dir_raw,
            }
            for key in optional_metrics:
                if key in summary:
                    row[key] = summary[key]
            rows.append(row)
    if not rows:
        raise RuntimeError(f"No sample summaries found for {manifest_path}")

    results_dir = suite_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / "comparison.json"
    csv_path = results_dir / "comparison.csv"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)
    summary_rows = []
    for method, method_rows in grouped.items():
        aggregate_row: dict[str, float | int | str] = {
            "method": method,
            "n_seeds": len(method_rows),
        }
        numeric_keys = sorted(
            {
                key
                for row in method_rows
                for key, value in row.items()
                if key not in {"seed", "n_sampled"}
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
            }
        )
        for key in numeric_keys:
            values = [float(row[key]) for row in method_rows if key in row]
            aggregate_row[f"{key}_mean"] = statistics.fmean(values)
            aggregate_row[f"{key}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summary_rows.append(aggregate_row)
    summary_json_path = results_dir / "comparison_summary.json"
    summary_csv_path = results_dir / "comparison_summary.csv"
    with summary_json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_rows, handle, indent=2, sort_keys=True)
        handle.write("\n")
    summary_fieldnames = list(
        dict.fromkeys(key for row in summary_rows for key in row)
    )
    with summary_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    return json_path, csv_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", required=True)
    args = parser.parse_args(argv)
    json_path, csv_path = aggregate_suite(Path(args.suite_dir).expanduser().resolve())
    print(f"COMPARISON_JSON={json_path}")
    print(f"COMPARISON_CSV={csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
