"""Locate and validate the pinned RGFN checkout."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from .config import PACKAGE_ROOT

RGFN_REPOSITORY = "https://github.com/koziarskilab/RGFN.git"
RGFN_COMMIT = "6ce59169f855ed18f34ba4e8279de93bee306e4f"


def configure_runtime_environment() -> None:
    """Set import/runtime defaults required by the pinned RGFN stack."""
    os.environ.setdefault("DGLBACKEND", "pytorch")
    os.environ.setdefault("RGFN_MINIMAL_PROXIES", "1")
    if sys.platform == "darwin":
        # PyTorch 2.3, DGL, and Accelerate/vecLib can load competing OpenMP
        # runtimes on Apple Silicon. The conflict surfaces while restoring the
        # public sEH proxy. One CPU thread is ample for the local smoke path.
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")


def resolve_rgfn_root(value: str | Path | None = None) -> Path:
    raw = value or os.environ.get("RGFN_ROOT") or PACKAGE_ROOT / "external" / "RGFN"
    return Path(raw).expanduser().resolve()


def validate_rgfn_root(root: Path, *, require_data: bool = True) -> None:
    required = [
        root / "pyproject.toml",
        root / "train.py",
        root / "rgfn" / "api" / "trajectories.py",
        root / "configs" / "rgfn_base.gin",
    ]
    if require_data:
        required.append(root / "data" / "chemistry.xlsx")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        message = "\n  - ".join(missing)
        raise FileNotFoundError(
            f"RGFN checkout is incomplete at {root}. Missing:\n  - {message}\n"
            "Run molecule_synthesis/scripts/setup_env.sh first or pass --rgfn-root."
        )


def get_rgfn_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()
