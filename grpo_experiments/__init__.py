"""
Our GRPO / IPS-GRPO comparison code — separate from upstream PhyloGFN (src/).

Run from repo root:
    python -m grpo_experiments.train --method grpo ...
"""

from grpo_experiments.config import ExperimentConfig
from grpo_experiments.runner import run_experiment

__all__ = ["ExperimentConfig", "run_experiment"]
