"""Write run meta.json with git commit, config, hardware, param counts."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Sequence

from final.paths import REPO_ROOT


def git_commit() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def gpu_model() -> str | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        lines = [line.strip() for line in out.splitlines() if line.strip()]
        return lines[0] if lines else None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def count_params(state_dict: dict[str, Any] | None) -> int | None:
    if state_dict is None:
        return None
    total = 0
    for value in state_dict.values():
        if hasattr(value, "numel"):
            total += int(value.numel())
    return total


def write_run_meta(
    run_dir: Path,
    *,
    method: str,
    seed: int,
    log_w_bin_edges: Sequence[float],
    partial: bool,
    total_wall_clock_s: float | None = None,
    gpu_hours: float | None = None,
    forward_param_count: int | None = None,
    reverse_param_count: int | None = None,
) -> Path:
    run_dir = Path(run_dir)
    config_path = run_dir / "resolved_config.yaml"
    if not config_path.exists():
        config_path = run_dir / "config.yaml"
    experiment_path = run_dir / "experiment_config.json"

    payload: dict[str, Any] = {
        "method": method,
        "seed": seed,
        "git_commit": git_commit(),
        "gpu_model": gpu_model(),
        "config_path": str(config_path) if config_path.exists() else None,
        "experiment_config_path": str(experiment_path) if experiment_path.exists() else None,
        "log_w_bin_edges": list(log_w_bin_edges),
        "forward_param_count": forward_param_count,
        "reverse_param_count": reverse_param_count,
        "partial": partial,
    }
    if total_wall_clock_s is not None:
        payload["total_wall_clock_s"] = total_wall_clock_s
    if gpu_hours is not None:
        payload["gpu_hours"] = gpu_hours
    if config_path.exists():
        payload["config_yaml"] = config_path.read_text(encoding="utf-8")
    if experiment_path.exists():
        payload["experiment_config"] = json.loads(
            experiment_path.read_text(encoding="utf-8")
        )

    meta_path = run_dir / "meta.json"
    meta_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return meta_path
