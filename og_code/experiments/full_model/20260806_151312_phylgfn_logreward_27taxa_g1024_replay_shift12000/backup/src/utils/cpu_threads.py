"""Cap CPU thread usage for PyTorch and BLAS on shared servers."""

from __future__ import annotations

import os

import torch

DEFAULT_MAX_CPU_THREADS = 2
"""Default PyTorch/BLAS threads per training process when nothing else is configured."""


def resolve_cpu_threads(*, explicit: int | None = None, yaml_value: int | None = None) -> int:
    """Resolve the thread cap for this process.

    Priority: CLI ``explicit`` (>0) > ``PHYLOGFN_CPU_THREADS`` env > YAML > default (2).
    Set ``PHYLOGFN_CPU_THREADS=0`` to disable capping entirely.
    """
    if explicit is not None and explicit > 0:
        return int(explicit)
    if "PHYLOGFN_CPU_THREADS" in os.environ:
        return max(0, int(os.environ["PHYLOGFN_CPU_THREADS"]))
    if yaml_value is not None and yaml_value > 0:
        return int(yaml_value)
    return DEFAULT_MAX_CPU_THREADS


def configure_cpu_threads(num_threads: int) -> int:
    """Apply intra-op CPU thread limits for PyTorch and common BLAS backends."""
    if num_threads <= 0:
        return 0

    num_threads = max(1, int(num_threads))
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(num_threads)
    torch.set_num_threads(num_threads)
    interop = max(1, min(num_threads, 2))
    try:
        torch.set_num_interop_threads(interop)
    except RuntimeError:
        pass
    return num_threads


def apply_cpu_thread_limit(*, explicit: int | None = None, yaml_value: int | None = None) -> int:
    """Resolve and apply the CPU thread cap; returns 0 if disabled."""
    return configure_cpu_threads(resolve_cpu_threads(explicit=explicit, yaml_value=yaml_value))
