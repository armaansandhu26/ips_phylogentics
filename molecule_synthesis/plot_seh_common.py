"""Shared loaders and styling for seh_paper_medium figure scripts."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

METHOD_ORDER = ("grpo", "count_ips_grpo", "mips_grpo", "rgfn")
METHOD_LABELS = {
    "grpo": "GRPO",
    "count_ips_grpo": "IPS-GRPO",
    "mips_grpo": "MIPS-GRPO",
    "rgfn": "RGFN",
}
METHOD_COLORS = {
    "grpo": "#EF5350",
    "count_ips_grpo": "#4C78A8",
    "mips_grpo": "#2CA02C",
    "rgfn": "#FF7F0E",
}
METHOD_MARKERS = {
    "grpo": "o",
    "count_ips_grpo": "s",
    "mips_grpo": "^",
    "rgfn": "D",
}

METRIC_PREFIX = b"train/"
FORWARD_TRAJECTORIES = 100


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def resolve_run_dirs(suite_dir: Path, suite: dict) -> dict[str, Path]:
    runs: dict[str, Path] = {}
    for method, raw in suite.get("runs", {}).items():
        entries = {"legacy": raw} if isinstance(raw, str) else raw
        for seed, run_path in entries.items():
            path = Path(run_path)
            if not path.is_absolute():
                candidate = suite_dir.parent.parent.parent / path
                if candidate.is_dir():
                    path = candidate
                elif not path.is_dir():
                    path = Path.cwd() / path
            if path.is_dir():
                runs[method] = path.resolve()
                break
    return runs


def parse_metric_value(part: bytes) -> float | None:
    idx = part.find(b"\x82\x01")
    if idx == -1:
        return None
    chunk = part[idx + 3 : idx + 40]
    numeric = "".join(chr(byte) for byte in chunk if (48 <= byte <= 57) or byte in b".+-eE")
    if not numeric:
        ascii_chunk = "".join(chr(byte) for byte in chunk if 32 <= byte < 127)
        numeric = ascii_chunk.split("\\")[0].strip(": ")
    try:
        return float(numeric)
    except ValueError:
        return None


def parse_metrics_blob(blob: bytes) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for part in blob.split(METRIC_PREFIX)[1:]:
        name_bytes = part.split(b"\x82", 1)[0]
        name = name_bytes.decode("ascii", errors="ignore").strip("\x00\n\r ")
        value = parse_metric_value(part)
        if name and value is not None:
            metrics[f"train/{name}"] = value
    return metrics


def load_wandb_history(wandb_path: Path) -> list[dict[str, float]]:
    from wandb.sdk.internal import datastore

    rows: list[dict[str, float]] = []
    store = datastore.DataStore()
    store.open_for_scan(str(wandb_path))
    while True:
        try:
            record = store.scan_record()
        except AssertionError:
            break
        if record is None:
            break
        record_type, payload = record
        if record_type == 1 and b"train/proxy_mean" in payload:
            metrics = parse_metrics_blob(payload)
            if metrics:
                rows.append(metrics)
    return rows


def load_training_history(run_dir: Path) -> list[dict[str, float]]:
    wandb_root = run_dir / "logs" / "wandb"
    history: list[dict[str, float]] = []
    offset = 0
    for run_path in sorted(wandb_root.glob("offline-run-*")):
        wandb_files = list(run_path.glob("run-*.wandb"))
        if not wandb_files:
            continue
        try:
            segment = load_wandb_history(wandb_files[0])
        except (AssertionError, IndexError, OSError):
            continue
        if not segment:
            continue
        for index, row in enumerate(segment):
            point = dict(row)
            point["_iter"] = offset + index
            point["_oracle_calls"] = point["_iter"] * FORWARD_TRAJECTORIES
            history.append(point)
        offset += len(segment)
    return history


def load_mode_discovery_from_xlsx(run_dir: Path) -> list[tuple[int, int]]:
    try:
        import pandas as pd
    except ImportError:
        return []
    points: list[tuple[int, int]] = []
    for path in sorted((run_dir / "modes").glob("modes_*.xlsx")):
        match = re.search(r"modes_(\d+)\.xlsx$", path.name)
        if match is None:
            continue
        iteration = int(match.group(1))
        n_modes = len(pd.read_excel(path))
        points.append((iteration * FORWARD_TRAJECTORIES, n_modes))
    return sorted(points)


def load_sample_rows(run_dir: Path) -> list[dict]:
    sample_path = run_dir / "samples" / "samples.jsonl"
    if not sample_path.is_file():
        return []
    rows: list[dict] = []
    with sample_path.open(encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def discovery_curves(rows: list[dict]) -> tuple[list[int], list[int], list[float]]:
    seen: set[str] = set()
    unique_curve: list[int] = []
    mean_proxy_curve: list[float] = []
    running_total = 0.0
    for index, row in enumerate(rows, start=1):
        smiles = row.get("smiles")
        if smiles is not None:
            seen.add(smiles)
        running_total += float(row["proxy"])
        unique_curve.append(len(seen))
        mean_proxy_curve.append(running_total / index)
    return list(range(1, len(rows) + 1)), unique_curve, mean_proxy_curve


def unique_terminal_log_points(rows: list[dict]) -> tuple[list[float], list[float]]:
    proxies, log_rewards = unique_terminal_reward_points(rows)
    if not proxies:
        return [], []
    log_proxies = [float(__import__("math").log(max(proxy, 1e-6))) for proxy in proxies]
    return log_proxies, log_rewards


def unique_terminal_reward_points(rows: list[dict]) -> tuple[list[float], list[float]]:
    by_smiles: dict[str, tuple[float, float]] = {}
    for row in rows:
        smiles = row.get("smiles")
        if smiles is None:
            continue
        proxy = float(row["proxy"])
        log_reward = float(row["log_reward"])
        if smiles not in by_smiles:
            by_smiles[smiles] = (proxy, log_reward)
        else:
            old_proxy, old_log_reward = by_smiles[smiles]
            by_smiles[smiles] = (old_proxy, 0.5 * (old_log_reward + log_reward))
    if not by_smiles:
        return [], []
    proxies = [proxy for proxy, _ in by_smiles.values()]
    log_rewards = [log_reward for _, log_reward in by_smiles.values()]
    return proxies, log_rewards


def linear_fit_stats(
    x_values: list[float],
    y_values: list[float],
    *,
    oracle_slope: float | None = None,
) -> dict[str, float | int | None]:
    n = len(x_values)
    if n == 0:
        return {
            "n": 0,
            "pearson_r": None,
            "r2": None,
            "slope": None,
            "intercept": None,
            "rmse_oracle": None,
            "x_min": None,
            "x_max": None,
            "y_min": None,
            "y_max": None,
        }
    if n == 1:
        return {
            "n": 1,
            "pearson_r": None,
            "r2": None,
            "slope": None,
            "intercept": None,
            "rmse_oracle": None,
            "x_min": float(x_values[0]),
            "x_max": float(x_values[0]),
            "y_min": float(y_values[0]),
            "y_max": float(y_values[0]),
        }

    import numpy as np

    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    pearson_r = float(np.corrcoef(x, y)[0, 1])
    slope, intercept = (float(v) for v in np.polyfit(x, y, 1))
    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else None
    rmse_oracle = None
    if oracle_slope is not None:
        rmse_oracle = float(np.sqrt(np.mean((y - oracle_slope * x) ** 2)))

    return {
        "n": n,
        "pearson_r": pearson_r,
        "r2": r2,
        "slope": slope,
        "intercept": intercept,
        "rmse_oracle": rmse_oracle,
        "x_min": float(np.min(x)),
        "x_max": float(np.max(x)),
        "y_min": float(np.min(y)),
        "y_max": float(np.max(y)),
    }


def format_fit_annotation(
    stats: dict[str, float | int | None],
    *,
    oracle_slope: float | None = None,
) -> str:
    lines: list[str] = [f"n={stats['n']}"]
    if stats["pearson_r"] is not None:
        lines.append(f"r={stats['pearson_r']:.3f}")
    if stats["r2"] is not None:
        lines.append(f"R²={stats['r2']:.3f}")
    if stats["slope"] is not None:
        if oracle_slope is not None:
            lines.append(f"β̂={stats['slope']:.2f} (oracle {oracle_slope:g})")
        else:
            lines.append(f"slope={stats['slope']:.2f}")
    intercept = stats.get("intercept")
    if intercept is not None and abs(float(intercept)) > 1e-4:
        lines.append(f"b={float(intercept):.2f}")
    if stats.get("rmse_oracle") is not None:
        lines.append(f"RMSE={float(stats['rmse_oracle']):.2e}")
    if stats.get("x_min") is not None and stats.get("x_max") is not None:
        lines.append(
            f"x∈[{float(stats['x_min']):.2f}, {float(stats['x_max']):.2f}]"
        )
    return "\n".join(lines)


def save_figure(fig, output_dir: Path, stem: str, dpi: int) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"{stem}.png", output_dir / f"{stem}.pdf"]
    fig.savefig(paths[0], dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(paths[1], bbox_inches="tight", facecolor="white")
    return paths


def available_methods(run_dirs: dict[str, Path]) -> list[str]:
    return [method for method in METHOD_ORDER if method in run_dirs]
