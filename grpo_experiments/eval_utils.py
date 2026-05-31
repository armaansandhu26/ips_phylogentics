"""Shared helpers for evaluating grpo_experiments training runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.configs.defaults import get_cfg_defaults
from src.env import build_env
from src.gfn.build import build_gfn
from src.gfn.rollout_worker_phylo import RolloutWorker
from src.utils.utils import correct_cfg_data, load_sequences


@dataclass
class RunArtifacts:
    label: str
    root: Path
    checkpoint_path: Path
    cfg_path: Path
    dataset_path: Path
    method: str
    config: dict[str, Any]


def choose_device(device_arg: str | None) -> str:
    if device_arg is not None:
        return device_arg
    try:
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            torch.empty(1, device="cuda:0")
            return "cuda:0"
    except Exception as exc:
        print(f"warning: CUDA unavailable ({exc}); using cpu")
    return "cpu"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)


def load_metrics(run_dir: Path) -> list[dict]:
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        raise FileNotFoundError(f"missing metrics file: {metrics_path}")
    rows = []
    with metrics_path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"no metric rows found in: {metrics_path}")
    return rows


def load_experiment_config(run_dir: Path) -> dict[str, Any]:
    for name in ("experiment_config.json", "run_args.json"):
        path = run_dir / name
        if path.exists():
            return load_json(path)
    raise FileNotFoundError(
        f"missing experiment config in {run_dir} "
        "(expected experiment_config.json or run_args.json)"
    )


def resolve_dataset_path(config: dict[str, Any]) -> Path:
    raw = config.get("dataset_path") or config.get("dataset")
    if raw is None:
        raise KeyError("experiment config missing dataset_path / dataset")
    dataset_path = Path(raw)
    if not dataset_path.is_absolute():
        repo_root = Path(__file__).resolve().parents[1]
        candidate = repo_root / dataset_path
        if candidate.exists():
            dataset_path = candidate
    if not dataset_path.exists():
        raise FileNotFoundError(f"missing dataset: {dataset_path}")
    return dataset_path


def resolve_checkpoint(run_dir: Path, checkpoint_name: str | None = None) -> Path:
    if checkpoint_name:
        path = run_dir / checkpoint_name
        if not path.exists():
            raise FileNotFoundError(f"missing checkpoint: {path}")
        return path
    for name in ("final_checkpoint.pt", "generator_checkpoint.pt"):
        path = run_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"no checkpoint found in {run_dir}")


def resolve_run_artifacts(path_str: str, label: str | None = None) -> RunArtifacts:
    path = Path(path_str)
    run_dir = path.parent if path.is_file() else path
    config = load_experiment_config(run_dir)
    cfg_path = run_dir / "resolved_config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"missing resolved config: {cfg_path}")

    method = str(config.get("method", "unknown"))
    resolved_label = label or config.get("run_name") or method
    checkpoint_path = resolve_checkpoint(run_dir)
    dataset_path = resolve_dataset_path(config)

    return RunArtifacts(
        label=str(resolved_label),
        root=run_dir,
        checkpoint_path=checkpoint_path,
        cfg_path=cfg_path,
        dataset_path=dataset_path,
        method=method,
        config=config,
    )


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    if "runs" not in manifest:
        raise ValueError(f"manifest missing 'runs' list: {manifest_path}")
    return manifest


def manifest_run_entries(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    entries = []
    for row in manifest["runs"]:
        run_dir = Path(row["run_dir"])
        if not run_dir.exists():
            raise FileNotFoundError(f"manifest run_dir missing: {run_dir}")
        entries.append(row)
    return entries


def load_generator(artifacts: RunArtifacts, device: str):
    all_seqs = load_sequences(str(artifacts.dataset_path))
    cfg = get_cfg_defaults()
    cfg.merge_from_file(str(artifacts.cfg_path))
    cfg.AMP = False
    cfg = correct_cfg_data(all_seqs, 1, cfg)

    env = build_env(cfg, all_seqs)
    env.to(device)
    generator = build_gfn(cfg, env, device, ddp=False)
    generator.load(str(artifacts.checkpoint_path))
    generator.eval()
    return cfg, env, generator


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values[:]
    half = window // 2
    prefix = [0.0]
    for value in values:
        prefix.append(prefix[-1] + value)
    smoothed = []
    for idx in range(len(values)):
        start = max(0, idx - half)
        end = min(len(values), idx + half + 1)
        smoothed.append((prefix[end] - prefix[start]) / (end - start))
    return smoothed


def sample_series(
    rows: list[dict],
    key: str,
    smoothing_window: int,
    stride: int,
) -> tuple[list[int], list[float]]:
    if key not in rows[0]:
        raise KeyError(f"metric key not found in metrics.jsonl: {key}")
    steps = [int(row["global_step"]) for row in rows]
    values = [float(row[key]) for row in rows]
    smoothed = moving_average(values, smoothing_window)
    sampled_steps = steps[::stride]
    sampled_values = smoothed[::stride]
    if sampled_steps and sampled_steps[-1] != steps[-1]:
        sampled_steps.append(steps[-1])
        sampled_values.append(smoothed[-1])
    return sampled_steps, sampled_values


def entropy_from_counts(counts: dict[str, int]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    probs = np.asarray([count / total for count in counts.values()], dtype=np.float64)
    return float(-(probs * np.log(probs + 1e-12)).sum())
