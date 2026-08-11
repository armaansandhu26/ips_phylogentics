"""Timestamped experiment run directories and logging."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

DATA_DIR = Path(__file__).resolve().parent / "data"
RUNS_DIR = DATA_DIR / "runs"

CONFIG_NAME = "config.json"
TRAIN_LOG_NAME = "train.log"
HISTORY_NAME = "history.json"
CHECKPOINT_NAME = "checkpoint.pt"
SUMMARY_NAME = "summary.json"
TRAINING_PLOT_NAME = "training_curves.png"
SAMPLING_PLOT_NAME = "signature_sampling.png"
LOGQ_PLOT_NAME = "signature_qhat_vs_logreward.png"


def run_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def run_dir_name(*, reward_profile: str, propensity_mode: str, group_size: int, run_name: str | None = None) -> str:
    base = f"{run_stamp()}_{reward_profile}_{propensity_mode}_gs{group_size}"
    if run_name:
        base = f"{base}_{run_name}"
    return base


def new_run_dir(*, reward_profile: str, propensity_mode: str, group_size: int, run_name: str | None = None, base: Path | None = None) -> Path:
    root = base or RUNS_DIR
    run_dir = root / run_dir_name(
        reward_profile=reward_profile, propensity_mode=propensity_mode, group_size=group_size, run_name=run_name
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path

    @property
    def config(self) -> Path:
        return self.run_dir / CONFIG_NAME

    @property
    def train_log(self) -> Path:
        return self.run_dir / TRAIN_LOG_NAME

    @property
    def history(self) -> Path:
        return self.run_dir / HISTORY_NAME

    @property
    def checkpoint(self) -> Path:
        return self.run_dir / CHECKPOINT_NAME

    @property
    def summary(self) -> Path:
        return self.run_dir / SUMMARY_NAME

    @property
    def training_plot(self) -> Path:
        return self.run_dir / TRAINING_PLOT_NAME

    @property
    def sampling_plot(self) -> Path:
        return self.run_dir / SAMPLING_PLOT_NAME

    @property
    def logq_plot(self) -> Path:
        return self.run_dir / LOGQ_PLOT_NAME


class TeeStdout:
    """Mirror stdout to a log file for the duration of a run."""

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self._terminal: TextIO = sys.stdout
        self._log: TextIO | None = None

    def __enter__(self) -> "TeeStdout":
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = self.log_path.open("w", encoding="utf-8")
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        sys.stdout = self._terminal
        if self._log is not None:
            self._log.close()
            self._log = None

    def write(self, message: str) -> None:
        self._terminal.write(message)
        if self._log is not None:
            self._log.write(message)

    def flush(self) -> None:
        self._terminal.flush()
        if self._log is not None:
            self._log.flush()


def save_run_config(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def config_payload(train_config, *, reward_profile_spec: dict[str, object], catalog_summary: dict, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "train_config": asdict(train_config),
        "reward_profile_spec": reward_profile_spec,
        "catalog_summary": catalog_summary,
    }
    if extra:
        payload["run_args"] = extra
    return payload
