"""IPS-GRPO experiments — separate from grpo.py / main runner."""

from grpo_experiments.ips_grpo.config import IPSExperimentConfig
from grpo_experiments.ips_grpo.runner import run_experiment

__all__ = ["IPSExperimentConfig", "run_experiment"]
