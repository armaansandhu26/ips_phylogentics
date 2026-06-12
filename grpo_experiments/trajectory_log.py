"""Log every training-sample signature/topology/score for later diversity plots."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from grpo_experiments.utils import append_jsonl


class TrajectoryLogger:
    """Append compact per-tree records and per-step diversity summaries."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        enabled: bool = True,
        flush_every: int = 1,
    ) -> None:
        self.enabled = enabled
        self.flush_every = max(1, int(flush_every))
        root = Path(output_dir)
        self.samples_path = root / "trajectory_samples.jsonl"
        self.summary_path = root / "trajectory_step_summary.jsonl"
        self._pending_samples: list[dict[str, Any]] = []
        self._pending_summaries: list[dict[str, Any]] = []
        self._rounds_since_flush = 0
        self.total_trees_logged = 0

    def log_batch(
        self,
        *,
        global_step: int,
        resample_round: int,
        trees: Sequence[Any],
        source_tags: Sequence[str] | None = None,
        update_cycle: int = 0,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled or not trees:
            return

        signatures: list[str] = []
        topologies: list[str] = []
        log_scores: list[float] = []
        sample_rows: list[dict[str, Any]] = []

        for idx, tree in enumerate(trees):
            sig = str(getattr(tree, "signature", "") or "")
            topo = str(getattr(tree, "tree_topology_id", "") or "")
            score = float(getattr(tree, "log_score", float("nan")))
            signatures.append(sig)
            topologies.append(topo)
            log_scores.append(score)
            row: dict[str, Any] = {
                "gs": int(global_step),
                "rr": int(resample_round),
                "uc": int(update_cycle),
                "sig": sig,
                "topo": topo,
                "ls": score,
            }
            if source_tags is not None and idx < len(source_tags):
                row["src"] = source_tags[idx]
            sample_rows.append(row)

        sig_counts = Counter(signatures)
        topo_counts = Counter(topologies)
        summary: dict[str, Any] = {
            "global_step": int(global_step),
            "resample_round": int(resample_round),
            "update_cycle": int(update_cycle),
            "batch_size": len(trees),
            "unique_signatures": len(sig_counts),
            "unique_topologies": len(topo_counts),
            "signature_duplicate_fraction": float(
                1.0 - len(sig_counts) / len(signatures)
            ),
            "topology_duplicate_fraction": float(
                1.0 - len(topo_counts) / len(topologies)
            ),
            "mean_log_score": float(sum(log_scores) / len(log_scores)),
            "min_log_score": float(min(log_scores)),
            "max_log_score": float(max(log_scores)),
            "top_signature_share": float(max(sig_counts.values()) / len(signatures)),
            "top_topology_share": float(max(topo_counts.values()) / len(topologies)),
        }
        if extra:
            summary.update(extra)

        self._pending_samples.extend(sample_rows)
        self._pending_summaries.append(summary)
        self.total_trees_logged += len(trees)
        self._rounds_since_flush += 1
        if self._rounds_since_flush >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self.enabled:
            return
        for row in self._pending_samples:
            append_jsonl(str(self.samples_path), row)
        for row in self._pending_summaries:
            append_jsonl(str(self.summary_path), row)
        self._pending_samples.clear()
        self._pending_summaries.clear()
        self._rounds_since_flush = 0

    def close(self) -> None:
        self.flush()
        if self.enabled:
            meta_path = self.samples_path.parent / "trajectory_log_meta.json"
            meta_path.write_text(
                json.dumps(
                    {
                        "total_trees_logged": self.total_trees_logged,
                        "samples_path": str(self.samples_path),
                        "summary_path": str(self.summary_path),
                    },
                    indent=2,
                )
            )
