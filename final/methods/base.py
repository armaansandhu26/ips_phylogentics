"""Shared helpers for method runners."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from final.configs import SuiteConfig
from final.paths import PYTHON, REPO_ROOT


@dataclass(frozen=True)
class CommandSpec:
    argv: list[str]
    cwd: Path
    env: dict[str, str] | None = None


class MethodRunner(Protocol):
    name: str

    def output_root(self, suite: SuiteConfig) -> Path:
        return suite.method_output_root(self.name)

    def build_train_command(
        self,
        suite: SuiteConfig,
        *,
        output_root: Path,
        run_name: str,
        resume_from: Path | None = None,
        resume_checkpoint: str | None = None,
    ) -> CommandSpec: ...

    def build_sample_command(
        self,
        suite: SuiteConfig,
        run_dir: Path,
        *,
        num_trees: int,
        batch_size: int,
        device: str,
    ) -> CommandSpec: ...

    def build_training_curves_command(self, run_dir: Path) -> CommandSpec | None: ...

    def plot_method(self) -> str: ...

    def expected_checkpoint(self, run_dir: Path) -> Path: ...

    def comparison_metrics_path(self, run_dir: Path, num_trees: int) -> Path: ...

    def run_ready_marker(self, run_dir: Path) -> Path | None:
        return run_dir / "resolved_config.yaml"


def python_module(module: str, *args: str) -> list[str]:
    return [str(PYTHON), "-u", "-m", module, *args]


def repo_script(relative: str, *args: str) -> list[str]:
    return [str(PYTHON), "-u", str(REPO_ROOT / relative), *args]


def shared_ppo_train_args(suite: SuiteConfig, output_root: Path) -> list[str]:
    t = suite.training
    output_arg = (
        str(output_root.relative_to(REPO_ROOT))
        if output_root.is_relative_to(REPO_ROOT)
        else str(output_root)
    )
    return [
        "--dataset",
        str(suite.dataset.relative_to(REPO_ROOT)),
        "--output",
        output_arg,
        "--seed",
        str(t.seed),
        "--epochs",
        str(t.epochs),
        "--steps-per-epoch",
        str(t.steps_per_epoch),
        "--on-policy-batch-size",
        str(t.on_policy_batch_size),
        "--replay-batch-size",
        str(t.replay_batch_size),
        "--replay-buffer-size",
        str(t.replay_buffer_size),
        "--checkpoint-every",
        str(t.checkpoint_every),
        "--full-model",
        "--outcome-level",
        "signature",
        "--advantage-reward-mode",
        "log_reward",
        "--grpo-lr",
        "1e-4",
        "--grpo-entropy-coef",
        "0.0",
        "--grpo-num-iterations",
        "1",
        "--rollout-chunk-size",
        str(t.on_policy_batch_size),
        "--print-every",
        "25",
    ] + (["--disable-replay"] if t.disable_replay else [])


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
