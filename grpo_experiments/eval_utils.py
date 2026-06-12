"""Shared helpers for evaluating grpo_experiments training runs."""

from __future__ import annotations

import json
import math
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

    epoch_ckpts = sorted(run_dir.glob("checkpoint_epoch*.pt"))
    if epoch_ckpts:
        return epoch_ckpts[-1]

    round_ckpts = sorted(run_dir.glob("checkpoint_round*.pt"))
    if round_ckpts:
        return round_ckpts[-1]

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


def logmeanexp(values: torch.Tensor) -> torch.Tensor:
    """Stable log(mean(exp(values))) for a 1D tensor."""
    if values.ndim != 1:
        raise ValueError(f"logmeanexp expects shape (N,), got {tuple(values.shape)}")
    return torch.logsumexp(values, dim=0) - math.log(values.shape[0])


def check_finite_tensor(name: str, tensor: torch.Tensor) -> None:
    """Raise if a tensor contains nan/inf values."""
    if not torch.isfinite(tensor).all():
        bad = int((~torch.isfinite(tensor)).sum().item())
        raise ValueError(f"{name} contains {bad} non-finite values.")


def sample_trees_from_generator(
    rollout_worker: RolloutWorker,
    generator,
    *,
    sample_trees: int,
    batch_size: int,
) -> list[Any]:
    """Sample final trees from a trained generator."""
    trees: list[Any] = []
    generated = 0
    while generated < sample_trees:
        current_batch = min(batch_size, sample_trees - generated)
        _, trajectories = rollout_worker.rollout(
            generator,
            current_batch,
            generate_full_trajectories=True,
        )
        batch_trees = [traj.current_state.subtrees[0] for traj in trajectories]
        trees.extend(batch_trees)
        generated += len(batch_trees)
    return trees


def estimate_tree_logq(
    env,
    rollout_worker: RolloutWorker,
    generator,
    tree,
    *,
    n_backward_trajectories: int,
) -> dict[str, Any]:
    """Estimate tree-level log q(tree) via backward-sampled trajectories."""
    input_actions_set = []
    backward_lengths = []
    for _ in range(n_backward_trajectories):
        actions_list, _ = env.sample_backward_from_tree(tree)
        input_actions_set.append(actions_list)
        backward_lengths.append(len(actions_list))

    with torch.no_grad():
        data, _ = rollout_worker.rollout(
            generator,
            n_backward_trajectories,
            generate_full_trajectories=False,
            input_actions_set=input_actions_set,
        )

    log_paths_pf = data["log_paths_pf"]
    log_paths_pb = data["log_paths_pb"]
    log_rewards = data["log_rewards"]
    log_scores = data["log_scores"]

    check_finite_tensor("log_paths_pf", log_paths_pf)
    check_finite_tensor("log_paths_pb", log_paths_pb)
    check_finite_tensor("log_rewards", log_rewards)
    check_finite_tensor("log_scores", log_scores)

    log_pf = log_paths_pf.sum(dim=-1)
    log_pb = log_paths_pb.sum(dim=-1)
    check_finite_tensor("log_pf", log_pf)
    check_finite_tensor("log_pb", log_pb)

    importance_terms = log_pf - log_pb
    check_finite_tensor("log_pf - log_pb", importance_terms)

    log_q_tree = logmeanexp(importance_terms)
    check_finite_tensor("log_q_tree", log_q_tree.unsqueeze(0))

    return {
        "log_q_tree": float(log_q_tree.item()),
        "tree_log_score": float(tree.log_score),
        "tree_log_reward_replayed": float(log_rewards[0].item()),
        "importance_term_mean": float(importance_terms.mean().item()),
        "importance_term_std": float(importance_terms.std().item()),
        "importance_term_min": float(importance_terms.min().item()),
        "importance_term_max": float(importance_terms.max().item()),
        "log_pf_mean": float(log_pf.mean().item()),
        "log_pb_mean": float(log_pb.mean().item()),
        "num_backward_trajectories": int(n_backward_trajectories),
        "trajectory_length_mean": float(np.mean(backward_lengths)),
        "trajectory_length_min": int(min(backward_lengths)),
        "trajectory_length_max": int(max(backward_lengths)),
        "replayed_log_score_mean": float(log_scores.mean().item()),
        "replayed_log_score_std": float(log_scores.std().item()),
    }


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


def effective_stride(num_rows: int, stride: int | None, max_points: int) -> int:
    """Pick a stride so plotted points stay near max_points (always >= 1)."""
    if num_rows <= 0:
        return 1
    if stride is None or stride < 1:
        return max(1, (num_rows + max_points - 1) // max_points)
    min_stride = max(1, (num_rows + max_points - 1) // max_points)
    return max(stride, min_stride)


def sample_series(
    rows: list[dict],
    key: str,
    smoothing_window: int,
    stride: int | None,
    *,
    max_points: int = 2500,
) -> tuple[list[int], list[float]]:
    if key not in rows[0]:
        raise KeyError(f"metric key not found in metrics.jsonl: {key}")
    steps = [int(row["global_step"]) for row in rows]
    values = [float(row[key]) for row in rows]
    smoothed = moving_average(values, smoothing_window)
    step_stride = effective_stride(len(rows), stride, max_points)
    sampled_steps = steps[::step_stride]
    sampled_values = smoothed[::step_stride]
    if sampled_steps and sampled_steps[-1] != steps[-1]:
        sampled_steps.append(steps[-1])
        sampled_values.append(smoothed[-1])
    return sampled_steps, sampled_values


def raw_series(rows: list[dict], key: str) -> tuple[list[int], list[float]]:
    return (
        [int(row["global_step"]) for row in rows],
        [float(row[key]) for row in rows],
    )


def subsample_xy(
    steps: list[int],
    values: list[float],
    max_points: int,
) -> tuple[list[int], list[float]]:
    stride = effective_stride(len(steps), None, max_points)
    out_steps = steps[::stride]
    out_values = values[::stride]
    if out_steps and out_steps[-1] != steps[-1]:
        out_steps.append(steps[-1])
        out_values.append(values[-1])
    return out_steps, out_values


def sanitize_importance_ratio(value: float, cap: float) -> float:
    """Map inf/nan to nan (breaks lines) and cap huge ratios for display."""
    import math

    if not math.isfinite(value):
        return float("nan")
    return min(max(value, 0.0), cap)


def auto_smoothing_window(num_rows: int, requested: int) -> int:
    """Scale rolling window with run length (25 <= w <= 500)."""
    if requested > 0:
        return max(1, min(requested, num_rows))
    return max(25, min(500, num_rows // 400))


def rolling_quantiles(
    values: list[float],
    window: int,
    quantiles: tuple[float, ...] = (0.25, 0.5, 0.75),
) -> dict[float, list[float]]:
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    half = max(window // 2, 1)
    out: dict[float, list[float]] = {q: [] for q in quantiles}
    for idx in range(n):
        sl = arr[max(0, idx - half) : min(n, idx + half + 1)]
        finite = sl[np.isfinite(sl)]
        if finite.size == 0:
            for q in quantiles:
                out[q].append(float("nan"))
        else:
            qs = np.quantile(finite, quantiles)
            for q, val in zip(quantiles, qs):
                out[q].append(float(val))
    return out


def rolling_finite_mean(
    values: list[float],
    window: int,
    *,
    cap: float | None = None,
) -> list[float]:
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    half = max(window // 2, 1)
    out: list[float] = []
    for idx in range(n):
        sl = arr[max(0, idx - half) : min(n, idx + half + 1)]
        finite = sl[np.isfinite(sl)]
        if finite.size == 0:
            out.append(float("nan"))
            continue
        if cap is not None:
            finite = np.clip(finite, 0.0, cap)
        out.append(float(finite.mean()))
    return out


def rolling_nonfinite_fraction(values: list[float], window: int) -> list[float]:
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    half = max(window // 2, 1)
    out: list[float] = []
    for idx in range(n):
        sl = arr[max(0, idx - half) : min(n, idx + half + 1)]
        out.append(float(np.mean(~np.isfinite(sl))))
    return out


def percentile_ylim(
    values: list[float],
    *,
    lo: float = 5.0,
    hi: float = 95.0,
    pad_frac: float = 0.08,
    include_zero: bool = False,
) -> tuple[float, float] | None:
    finite = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if finite.size == 0:
        return None
    ymin = float(np.percentile(finite, lo))
    ymax = float(np.percentile(finite, hi))
    if include_zero:
        ymin = min(ymin, 0.0)
        ymax = max(ymax, 0.0)
    span = max(ymax - ymin, 1e-6)
    pad = span * pad_frac
    return ymin - pad, ymax + pad


def entropy_from_counts(counts: dict[str, int]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    probs = np.asarray([count / total for count in counts.values()], dtype=np.float64)
    return float(-(probs * np.log(probs + 1e-12)).sum())
