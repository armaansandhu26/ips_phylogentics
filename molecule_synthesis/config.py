"""Small, dependency-free suite configuration loader."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .methods import normalize_method_name

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
SUITES_ROOT = PACKAGE_ROOT / "configs" / "suites"


@dataclass(frozen=True)
class Suite:
    suite_id: str
    description: str
    config: str
    methods: tuple[str, ...]
    training: dict[str, Any]
    sampling: dict[str, Any]
    enumeration: dict[str, Any] | None
    seeds: tuple[int, ...]
    method_overrides: dict[str, dict[str, Any]]
    evaluation: dict[str, Any]
    source_path: Path

    def resolve_config(self, rgfn_root: Path) -> Path:
        if self.config.startswith("@rgfn/"):
            path = rgfn_root / self.config.removeprefix("@rgfn/")
        else:
            path = REPO_ROOT / self.config
        return path.resolve()


def suite_path(name_or_path: str | Path) -> Path:
    candidate = Path(name_or_path)
    if candidate.is_file():
        return candidate.resolve()
    if candidate.suffix == ".json":
        path = SUITES_ROOT / candidate.name
    else:
        path = SUITES_ROOT / f"{candidate.name}.json"
    if not path.is_file():
        available = ", ".join(list_suites())
        raise FileNotFoundError(f"Suite {name_or_path!r} not found; available: {available}")
    return path.resolve()


def load_suite(name_or_path: str | Path) -> Suite:
    path = suite_path(name_or_path)
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)

    required = {"id", "description", "config", "methods", "training", "sampling"}
    missing = required.difference(raw)
    if missing:
        raise ValueError(f"{path} is missing fields: {', '.join(sorted(missing))}")

    methods = tuple(normalize_method_name(method) for method in raw["methods"])
    if not methods:
        raise ValueError(f"{path} must configure at least one method")
    if len(methods) != len(set(methods)):
        raise ValueError(f"{path} contains duplicate methods")

    training = dict(raw["training"])
    for key in ("iterations", "forward_trajectories", "replay_trajectories"):
        if key not in training:
            raise ValueError(f"{path}: training.{key} is required")

    raw_seeds = raw.get("seeds", [training.get("seed", 42)])
    seeds = tuple(int(seed) for seed in raw_seeds)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError(f"{path}: seeds must be a non-empty list of unique integers")

    method_overrides: dict[str, dict[str, Any]] = {}
    for raw_method, override in raw.get("method_overrides", {}).items():
        method = normalize_method_name(raw_method)
        if method not in methods:
            raise ValueError(f"{path}: override configured for disabled method {method!r}")
        method_overrides[method] = dict(override)

    return Suite(
        suite_id=str(raw["id"]),
        description=str(raw["description"]),
        config=str(raw["config"]),
        methods=methods,
        training=training,
        sampling=dict(raw["sampling"]),
        enumeration=dict(raw["enumeration"]) if raw.get("enumeration") else None,
        seeds=seeds,
        method_overrides=method_overrides,
        evaluation=dict(raw.get("evaluation", {})),
        source_path=path,
    )


def list_suites() -> list[str]:
    return sorted(path.stem for path in SUITES_ROOT.glob("*.json"))
