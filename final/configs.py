"""Load and validate paper comparison suite configs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from final.paths import CONFIGS_DIR, METHODS, REPO_ROOT, RUNS_DIR


@dataclass(frozen=True)
class SamplingConfig:
    num_trees: int
    batch_size: int


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    steps_per_epoch: int
    on_policy_batch_size: int
    replay_batch_size: int
    replay_buffer_size: int
    disable_replay: bool
    checkpoint_every: int
    seed: int


@dataclass(frozen=True)
class SuiteConfig:
    id: str
    taxa: int
    dataset: Path
    log_score_shift: float
    training: TrainingConfig
    sampling: SamplingConfig
    methods: dict[str, dict[str, Any]]
    source_path: Path

    def method_cfg(self, method: str) -> dict[str, Any]:
        if method not in self.methods:
            raise KeyError(f"method {method!r} not defined in suite {self.id}")
        return self.methods[method]

    def resolve_cfg_path(self, method: str) -> Path:
        rel = self.method_cfg(method)["cfg"]
        path = (REPO_ROOT / rel).resolve()
        if not path.exists():
            raise FileNotFoundError(f"missing cfg for {method}: {path}")
        return path

    def run_name(self, method: str) -> str:
        return f"{method}_{self.id}"

    def method_output_root(self, method: str) -> Path:
        return RUNS_DIR / self.id / method

    def suite_manifest_path(self) -> Path:
        return RUNS_DIR / self.id / "suite.json"


def _parse_training(raw: dict[str, Any]) -> TrainingConfig:
    return TrainingConfig(
        epochs=int(raw["epochs"]),
        steps_per_epoch=int(raw["steps_per_epoch"]),
        on_policy_batch_size=int(raw["on_policy_batch_size"]),
        replay_batch_size=int(raw.get("replay_batch_size", 0)),
        replay_buffer_size=int(raw.get("replay_buffer_size", 4096)),
        disable_replay=bool(raw.get("disable_replay", True)),
        checkpoint_every=int(raw.get("checkpoint_every", 500)),
        seed=int(raw.get("seed", 0)),
    )


def _parse_sampling(raw: dict[str, Any]) -> SamplingConfig:
    return SamplingConfig(
        num_trees=int(raw.get("num_trees", 1_000_000)),
        batch_size=int(raw.get("batch_size", 4096)),
    )


def load_suite(path: Path | str) -> SuiteConfig:
    path = Path(path)
    if path.suffix != ".json":
        candidate = CONFIGS_DIR / f"{path.name}.json"
        if candidate.exists():
            path = candidate
        elif (CONFIGS_DIR / path.name).with_suffix(".json").exists():
            path = (CONFIGS_DIR / path.name).with_suffix(".json")
    raw = json.loads(path.read_text(encoding="utf-8"))
    suite_id = raw.get("id") or path.stem
    missing = [m for m in METHODS if m not in raw.get("methods", {})]
    if missing:
        raise ValueError(f"suite {suite_id} missing methods: {missing}")
    dataset = (REPO_ROOT / raw["dataset"]).resolve()
    if not dataset.exists():
        raise FileNotFoundError(f"missing dataset: {dataset}")
    return SuiteConfig(
        id=suite_id,
        taxa=int(raw["taxa"]),
        dataset=dataset,
        log_score_shift=float(raw["log_score_shift"]),
        training=_parse_training(raw["training"]),
        sampling=_parse_sampling(raw["sampling"]),
        methods=raw["methods"],
        source_path=path.resolve(),
    )


def list_suites() -> list[Path]:
    return sorted(CONFIGS_DIR.glob("*.json"))


def read_log_score_shift(cfg_path: Path) -> float | None:
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    value = data.get("ENV", {}).get("LOG_SCORE_SHIFT")
    return float(value) if value is not None else None
