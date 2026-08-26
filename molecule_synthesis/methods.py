"""Method registry shared by the CLI, pipeline, and tests."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MethodSpec:
    name: str
    label: str
    objective: str


METHODS = {
    "rgfn": MethodSpec("rgfn", "RGFN (trajectory balance)", "trajectory_balance"),
    "grpo": MethodSpec("grpo", "GRPO", "grpo"),
    "count_ips_grpo": MethodSpec(
        "count_ips_grpo",
        "IPS-GRPO (original; count estimator)",
        "count_ips_grpo",
    ),
    "mips_grpo": MethodSpec(
        "mips_grpo",
        "MIPS-GRPO (learned reverse)",
        "mips_grpo",
    ),
}

METHOD_NAMES = tuple(METHODS)

_ALIASES = {
    "mips-grpo": "mips_grpo",
    "marginal_ips_grpo": "mips_grpo",
    "marginal-ips-grpo": "mips_grpo",
    "learned_reverse": "mips_grpo",
    "learned-reverse": "mips_grpo",
    "count_ips": "count_ips_grpo",
    "count-ips": "count_ips_grpo",
    "ips_grpo": "count_ips_grpo",
    "ips-grpo": "count_ips_grpo",
}


def normalize_method_name(name: str) -> str:
    normalized = name.strip().lower().replace("-", "_")
    normalized = _ALIASES.get(name.strip().lower(), normalized)
    if normalized not in METHODS:
        choices = ", ".join(METHOD_NAMES)
        raise ValueError(f"Unknown method {name!r}; choose one of: {choices}")
    return normalized
