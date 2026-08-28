#!/usr/bin/env python3
"""Seed and launch the pristine upstream PhyloGFN trainer."""

from __future__ import annotations

import os
import random
import runpy
import sys
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    seed = int(os.environ.get("PHYLOGFN_SEED", "0"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    upstream_root = Path(__file__).resolve().parents[1] / "phylogfn_paper"
    sys.path.insert(0, str(upstream_root))
    runpy.run_path(str(upstream_root / "train.py"), run_name="__main__")


if __name__ == "__main__":
    main()
