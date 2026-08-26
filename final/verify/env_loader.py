"""Load phylogfn env without a full ExperimentConfig."""

from __future__ import annotations

from pathlib import Path

from src.configs.defaults import get_cfg_defaults
from src.utils.utils import correct_cfg_data, load_sequences


def load_env_from_paths(cfg_path: str | Path, dataset_path: str | Path):
    cfg_path = Path(cfg_path)
    dataset_path = Path(dataset_path)
    sequences = load_sequences(str(dataset_path))
    cfg = get_cfg_defaults()
    cfg.merge_from_file(str(cfg_path))
    cfg.AMP = False
    cfg = correct_cfg_data(sequences, 1, cfg)
    cfg.LOGGING.ENABLE_TENSORBOARD = False
    from src.env import build_env

    return build_env(cfg, sequences), cfg, sequences
