"""Helpers for multi-taxa comparison runs under grpo_experiments/comparisons/."""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any


def num_labeled_rooted_tree_topologies(n_taxa: int) -> int:
    if n_taxa < 2:
        return 0
    result = 1
    for value in range(2 * n_taxa - 3, 0, -2):
        result *= value
    return result


def load_num_taxa(dataset_path: Path) -> int:
    with dataset_path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"expected species dict in {dataset_path}")
    return len(payload)


def load_metrics_rows(metrics_path: Path) -> list[dict[str, Any]]:
    if not metrics_path.exists():
        return []
    rows = []
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def summarize_learned_reverse_health(run_dir: Path) -> dict[str, Any]:
    rows = load_metrics_rows(run_dir / "metrics.jsonl")
    summary: dict[str, Any] = {"run_dir": str(run_dir), "metrics_rows": len(rows)}
    if rows:
        first = rows[0]
        last = rows[-1]
        summary["epoch0"] = {
            "batch_unique_outcomes": first.get("batch_unique_outcomes"),
            "batch_unique_topologies": first.get("batch_unique_topologies"),
            "ips_ess_fraction": first.get("ips_ess_fraction"),
            "loss": first.get("loss"),
            "reverse_loss": first.get("reverse_loss"),
        }
        summary["final"] = {
            "batch_unique_outcomes": last.get("batch_unique_outcomes"),
            "batch_unique_topologies": last.get("batch_unique_topologies"),
            "ips_ess_fraction": last.get("ips_ess_fraction"),
            "loss": last.get("loss"),
            "reverse_loss": last.get("reverse_loss"),
            "cumulative_unique_outcomes": last.get("cumulative_unique_outcomes"),
            "global_duplicate_fraction": last.get("global_duplicate_fraction"),
        }
    epoch_summaries_path = run_dir / "epoch_summaries.json"
    if epoch_summaries_path.exists():
        epoch_summaries = json.loads(epoch_summaries_path.read_text(encoding="utf-8"))
        if epoch_summaries:
            summary["last_epoch_summary"] = epoch_summaries[-1]
    return summary


def summarize_gflownet_health_from_log(log_path: Path) -> dict[str, Any]:
    if not log_path.exists():
        return {"log_path": str(log_path), "parsed_epochs": 0}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    epoch_re = re.compile(
        r"Epoch\s+(\d+)/(\d+)\s+\|\s+loss:\s+([-\d.eE+]+)\s+\|\s+log_Z:\s+([-\d.eE+]+)"
    )
    matches = epoch_re.findall(text)
    summary: dict[str, Any] = {
        "log_path": str(log_path),
        "parsed_epochs": len(matches),
    }
    if matches:
        epoch, total, loss, log_z = matches[-1]
        summary["last_progress"] = {
            "epoch": int(epoch),
            "total_epochs": int(total),
            "loss": float(loss),
            "log_Z": float(log_z),
        }
    eval_re = re.compile(r"Epoch\s+(\d+),\s+MLL\s+([-\d.eE+]+),\s+PEARSONR\s+([-\d.eE+]+)")
    eval_matches = eval_re.findall(text)
    if eval_matches:
        epoch, mll, pearson = eval_matches[-1]
        summary["last_eval"] = {
            "epoch": int(epoch),
            "mll": float(mll),
            "log_pearsonr": float(pearson),
        }
    return summary


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
