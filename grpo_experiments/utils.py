"""Shared utilities for training experiments."""

from __future__ import annotations

import datetime
import json
import os
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

from grpo_experiments.core.log_score_discretization import (
    apply_experiment_log_score_discretization,
)
from src.configs.defaults import get_cfg_defaults
from src.utils.utils import correct_cfg_data, load_sequences, schedule

if TYPE_CHECKING:
    from grpo_experiments.config import ExperimentConfig


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_training_cpu_limits(exp_cfg, cfg) -> int:
    """Apply per-process CPU thread cap (default 4). Prints when active."""
    from src.utils.cpu_threads import apply_cpu_thread_limit

    explicit = getattr(exp_cfg, "cpu_threads", 0)
    applied = apply_cpu_thread_limit(
        explicit=explicit if explicit > 0 else None,
        yaml_value=int(getattr(cfg.GFN.TRAINING_DATA_LOADER, "MAX_CPU_THREADS", 2)),
    )
    if applied:
        print(f"cpu_threads={applied}")
    return applied


def choose_device(device_arg: str | None) -> str:
    if device_arg is not None:
        return device_arg
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def build_output_dir(output_root: str, method: str, run_name: str | None) -> str:
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    label = f"{stamp}_{method}"
    if run_name:
        label = f"{stamp}_{run_name}_{method}"
    path = os.path.join(output_root, label)
    os.makedirs(path, exist_ok=True)
    return path


def append_jsonl(path: str, record: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def scalar_metric(value) -> float:
    return float(np.asarray(value).reshape(-1)[0])


def infer_rollout_group_size(exp_cfg) -> int:
    """Training batch size used to auto-scale rollout / forward-replay chunks."""
    if hasattr(exp_cfg, "total_batch_size"):
        return int(exp_cfg.total_batch_size)
    if getattr(exp_cfg, "enable_policy_is", False):
        return int(getattr(exp_cfg, "effective_buffer_size", 64))
    if hasattr(exp_cfg, "grpo_group_size"):
        return int(exp_cfg.grpo_group_size)
    return int(getattr(exp_cfg, "on_policy_batch_size", 64))


def resolve_rollout_chunk_size(exp_cfg) -> int:
    """
    Micro-batch size for rollouts and forward replay.

  When rollout_chunk_size is smaller than the training batch, scale up so each
  step uses fewer, larger GPU forwards (better utilization on big batches).
  Set rollout_chunk_size=0 to match the full batch exactly.
    """
    raw = int(getattr(exp_cfg, "rollout_chunk_size", 2048))
    group = infer_rollout_group_size(exp_cfg)
    if raw <= 0:
        return max(group, 1)
    if raw < group:
        return group
    return raw


def get_generator_params(generator) -> list[nn.Parameter]:
    """Collect trainable parameters from TBGFlowNetGenerator."""
    params: list[nn.Parameter] = []
    seen: set[int] = set()
    for name in dir(generator):
        if name.startswith("_"):
            continue
        try:
            attr = getattr(generator, name)
        except Exception:
            continue
        if isinstance(attr, nn.Module):
            for p in attr.parameters():
                if id(p) not in seen and p.requires_grad:
                    params.append(p)
                    seen.add(id(p))
        elif isinstance(attr, nn.Parameter) and attr.requires_grad:
            if id(attr) not in seen:
                params.append(attr)
                seen.add(id(attr))
    if isinstance(generator, nn.Module):
        for p in generator.parameters():
            if id(p) not in seen and p.requires_grad:
                params.append(p)
                seen.add(id(p))
    return params


def load_phylogfn_cfg(exp_cfg: ExperimentConfig):
    """Load PhyloGFN YAML and apply experiment overrides."""
    all_seqs = load_sequences(exp_cfg.dataset_path)
    cfg = get_cfg_defaults()
    cfg.merge_from_file(exp_cfg.cfg_path)
    cfg.AMP = False
    cfg = correct_cfg_data(all_seqs, 1, cfg)

    tc = cfg.GFN.TRAINING_DATA_LOADER
    if getattr(exp_cfg, "enable_policy_is", False):
        tc.EPOCHS_NUM = exp_cfg.effective_resample_rounds
        tc.STEPS_PER_EPOCH = exp_cfg.effective_update_cycles
        tc.GFN_BATCH_SIZE = resolve_rollout_chunk_size(exp_cfg)
    else:
        tc.EPOCHS_NUM = exp_cfg.epochs
        tc.STEPS_PER_EPOCH = exp_cfg.steps_per_epoch
        tc.GFN_BATCH_SIZE = exp_cfg.on_policy_batch_size
    tc.BEST_STATE_BATCH_SIZE = exp_cfg.effective_replay_batch_size
    tc.BEST_TREES_BUFFER_SIZE = exp_cfg.replay_buffer_size
    tc.MINI_BATCH_SPLITS = exp_cfg.mini_batch_splits
    tc.NUM_WORKERS = 0
    edge_rep_grad_alpha = getattr(exp_cfg, "edge_rep_grad_alpha", None)
    if edge_rep_grad_alpha is not None:
        cfg.GFN.MODEL.EDGE_REP_GRAD_ALPHA = float(edge_rep_grad_alpha)
    only_train_tree_model = getattr(exp_cfg, "only_train_tree_model", None)
    if only_train_tree_model is not None:
        cfg.GFN.MODEL.ONLY_TRAIN_TREE_MODEL = bool(only_train_tree_model)
    cfg.LOGGING.ENABLE_TENSORBOARD = False
    return cfg, all_seqs


def generate_exploration_spec(exploration_cfg, epoch: int):
    if exploration_cfg.METHOD == "NONE":
        return None
    return {
        "exploration_method": exploration_cfg.METHOD,
        "start_value": schedule(
            exploration_cfg.START_VALUE, exploration_cfg.END_VALUE,
            exploration_cfg.T, epoch, type=exploration_cfg.ANNEAL_TYPE,
        ),
        "end_value": schedule(
            exploration_cfg.START_VALUE, exploration_cfg.END_VALUE,
            exploration_cfg.T, epoch + 1, type=exploration_cfg.ANNEAL_TYPE,
        ),
    }


def build_random_spec(exploration_specs: dict | None, steps_in_segment: int, step: int, anneal_type: str):
    """Turn exploration schedule into rollout_worker random_spec (matches TrainingDataLoader)."""
    if exploration_specs is None:
        return None
    value = schedule(
        exploration_specs["start_value"],
        exploration_specs["end_value"],
        steps_in_segment,
        step,
        type=anneal_type,
    )
    if exploration_specs["exploration_method"] == "EPS_ANNEALING":
        return {"random_action_prob": value}
    return {"T": value}


def reconstruct_trees(env, trajectories, log_scores):
    return env.batch_actions_to_trees([t.actions for t in trajectories], log_scores)
