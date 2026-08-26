"""Method-specific train / sample command builders."""

from __future__ import annotations

from final.methods.count_ips import CountIpsRunner
from final.methods.grpo import GrpoRunner
from final.methods.learned_reverse import LearnedReverseRunner
from final.methods.phylgfn import PhylgfnRunner

RUNNERS = {
    "grpo": GrpoRunner(),
    "count_ips": CountIpsRunner(),
    "learned_reverse": LearnedReverseRunner(),
    "phylgfn": PhylgfnRunner(),
}


def get_runner(method: str):
    try:
        return RUNNERS[method]
    except KeyError as exc:
        raise ValueError(f"unknown method {method!r}; choose from {list(RUNNERS)}") from exc
