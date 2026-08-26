"""Run-level logger: enriched step metrics, eval dumps, checkpoints, meta.json."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from final.logging.batch_stats import (
    enrich_common_record,
    log_w_histogram,
    psis_khat,
    signature_hash,
)
from final.logging.schema import LOG_W_BIN_EDGES, should_eval_dump
from grpo_experiments.utils import append_jsonl


class FinalRunLogger:
    """Activated for runs under final/runs/. Writes metrics_detailed.jsonl + eval dumps."""

    def __init__(
        self,
        output_dir: Path,
        *,
        method: str,
        seed: int = 0,
        enabled: bool | None = None,
    ) -> None:
        self.output_dir = Path(output_dir).resolve()
        self.method = method
        self.seed = seed
        self.enabled = (
            enabled
            if enabled is not None
            else "final/runs" in str(self.output_dir).replace("\\", "/")
        )
        self._t0 = time.perf_counter()
        self._gpu_t0 = time.perf_counter()
        self.metrics_path = self.output_dir / "metrics_detailed.jsonl"
        self.eval_root = self.output_dir / "eval_dumps"
        self.checkpoint_root = self.output_dir / "checkpoints_eval"
        self._log_w_bins = LOG_W_BIN_EDGES
        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.eval_root.mkdir(parents=True, exist_ok=True)
            self.checkpoint_root.mkdir(parents=True, exist_ok=True)
            self._write_meta_stub()

    @classmethod
    def maybe_create(cls, output_dir: Path, *, method: str, seed: int = 0) -> FinalRunLogger | None:
        logger = cls(output_dir, method=method, seed=seed)
        return logger if logger.enabled else None

    def _write_meta_stub(self) -> None:
        from final.logging.meta import write_run_meta

        write_run_meta(
            self.output_dir,
            method=self.method,
            seed=self.seed,
            log_w_bin_edges=self._log_w_bins,
            partial=True,
        )

    def wall_clock_s(self) -> float:
        return time.perf_counter() - self._t0

    def on_step(
        self,
        record: dict[str, Any],
        *,
        log_rewards: torch.Tensor,
        log_paths_pf: torch.Tensor,
        outcome_ids: list[str],
        topology_ids: list[str] | None = None,
        lr: float,
        extra: dict[str, Any] | None = None,
        final_step: int | None = None,
    ) -> None:
        if not self.enabled:
            return
        enrich_common_record(
            record,
            log_rewards=log_rewards,
            log_paths_pf=log_paths_pf,
            outcome_ids=outcome_ids,
            lr=lr,
            wall_clock_s=self.wall_clock_s(),
            gpu_seconds=self.wall_clock_s(),  # placeholder until CUDA events wired
        )
        if extra:
            record.update(extra)
        append_jsonl(str(self.metrics_path), record)

        try:
            from final.logging.wandb_logger import FinalWandbLogger

            wandb_logger = FinalWandbLogger.maybe_get()
            if wandb_logger is not None:
                gs = int(record.get("global_step", record.get("step", -1)))
                if gs >= 0:
                    wandb_logger.log_metrics(gs, record)
        except (ImportError, Exception):
            pass

        gs = int(record.get("global_step", -1))
        if should_eval_dump(gs, final_step=final_step):
            self._schedule_eval_note(gs)

    def _schedule_eval_note(self, global_step: int) -> None:
        marker = self.eval_root / f"eval_step_{global_step:06d}.pending"
        marker.write_text(json.dumps({"global_step": global_step}) + "\n")

    def finalize(self) -> None:
        if not self.enabled:
            return
        from final.logging.meta import write_run_meta

        write_run_meta(
            self.output_dir,
            method=self.method,
            seed=self.seed,
            log_w_bin_edges=self._log_w_bins,
            partial=False,
            total_wall_clock_s=self.wall_clock_s(),
        )


def ppo_extra_metrics(
    record: dict[str, Any],
    *,
    outcome_ids: list[str],
    rewards: torch.Tensor,
    group_size: int,
) -> dict[str, Any]:
    from final.logging.batch_stats import grpo_group_stats

    extra = {
        "policy_loss": record.get("pg_loss", record.get("loss")),
        "clip_frac": record.get("clip_ratio/region_mean"),
        "ratio_mean": record.get("mean_importance_ratio"),
        "ratio_std": record.get("std_log_importance_ratio"),
        "advantage_mean": record.get("mean_advantage"),
        "advantage_std": record.get("std_advantage"),
        "ema_baseline": record.get("running_scaled_weight_baseline"),
    }
    # Approx KL: E[ratio * log(ratio) - ratio + 1] per token (Schulman)
    if "mean_log_importance_ratio" in record:
        r = float(record["mean_importance_ratio"])
        log_r = float(record["mean_log_importance_ratio"])
        extra["approx_kl"] = float(max(0.0, r * log_r - r + 1.0))
    extra.update(grpo_group_stats(outcome_ids, rewards, group_size=group_size))
    return extra


def learned_reverse_extra_metrics(
    record: dict[str, Any],
    *,
    log_w: torch.Tensor | None = None,
) -> dict[str, Any]:
    extra: dict[str, Any] = {
        "reverse_nll": record.get("reverse_loss"),
        "reverse_mle_steps": record.get("reverse_train_epochs"),
        "log_q_mean": record.get("reverse_log_probability_mean"),
        "log_pf_mean": record.get("forward_log_probability_mean"),
        "log_w_mean": record.get("log_importance_weight_mean"),
        "log_w_std": record.get("log_importance_weight_std"),
        "log_w_min": record.get("log_importance_weight_min"),
        "log_w_max": record.get("log_importance_weight_max"),
        "ips_ess": record.get("ips_ess"),
        "max_normalized_weight": record.get("max_normalized_weight"),
    }
    if log_w is not None:
        extra["log_w_hist_counts"] = log_w_histogram(log_w)
        extra["psis_khat"] = psis_khat(log_w)
    return extra
