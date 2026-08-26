"""Aggregate comparison metrics across methods in a suite."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from final.configs import load_suite
from final.methods import get_runner
from final.methods.base import write_json
from final.paths import METHODS, RESULTS_DIR, RUNS_DIR


METRIC_KEYS = [
    ("probability_vs_reward_pearson_vs_ideal", "pearson_linear"),
    ("log_probability_vs_log_reward_pearson_vs_ideal", "pearson_loglog"),
    ("importance_ess_fraction", "ess_fraction"),
    ("unique_observed_signatures", "unique_signatures"),
    ("observed_topologies", "observed_topologies"),
    ("checkpoint_log_partition_error", "logZ_error"),
]


def _load_metrics(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate_suite(suite_id: str) -> dict:
    suite = load_suite(suite_id)
    num_trees = suite.sampling.num_trees
    rows: list[dict] = []
    merged: dict[str, dict] = {}

    for method in METHODS:
        runner = get_runner(method)
        suite_manifest_path = RUNS_DIR / suite.id / "suite.json"
        run_dir: Path | None = None
        if suite_manifest_path.exists():
            suite_manifest = json.loads(suite_manifest_path.read_text(encoding="utf-8"))
            method_info = suite_manifest.get("methods", {}).get(method, {})
            if method_info.get("run_dir"):
                run_dir = Path(method_info["run_dir"])

        metrics_path = None
        if run_dir is not None:
            metrics_path = runner.comparison_metrics_path(run_dir, num_trees)
        if metrics_path is None or not metrics_path.exists():
            method_root = RUNS_DIR / suite.id / method
            candidates = sorted(method_root.glob("**/comparison_metrics.json"))
            metrics_path = candidates[-1] if candidates else None

        metrics = _load_metrics(metrics_path) if metrics_path else {}
        row = {
            "suite_id": suite.id,
            "method": method,
            "taxa": suite.taxa,
            "log_score_shift": suite.log_score_shift,
            "run_dir": metrics.get("run_dir") or (str(run_dir) if run_dir else ""),
            "metrics_path": str(metrics_path) if metrics_path else "",
        }
        for src_key, dst_key in METRIC_KEYS:
            value = metrics.get(src_key)
            row[dst_key] = value
        rows.append(row)
        merged[method] = row

    results_dir = RESULTS_DIR / suite.id
    results_dir.mkdir(parents=True, exist_ok=True)

    json_path = results_dir / "comparison_table.json"
    write_json(json_path, {"suite_id": suite.id, "rows": rows})

    csv_path = results_dir / "comparison_table.csv"
    fieldnames = list(rows[0].keys()) if rows else ["suite_id", "method"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_path = results_dir / "comparison_summary.json"
    write_json(summary_path, merged)
    return {"json": str(json_path), "csv": str(csv_path), "rows": rows}


def parse_args(argv: list[str] | None = None):
    import argparse

    parser = argparse.ArgumentParser(description="Aggregate suite comparison metrics into tables.")
    parser.add_argument("--suite", required=True, help="Suite id (e.g. 27taxa_noreplay).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = aggregate_suite(args.suite)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
