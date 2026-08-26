from __future__ import annotations

import json
import os
import pickle
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal, Sequence, Union

import torch

from grpo_experiments.core.trainer import GRPOTrainer
from learned_reverse_ips.advantages import RunningLogWeightNormalizer
from learned_reverse_ips.mlp_policy import PhyloLearnedReverseConfig, PhyloLearnedReversePolicy
from learned_reverse_ips.reverse_policy import TabularTerminalReversePolicy

METHOD = "learned_reverse_ips_grpo"
ReversePolicyType = Literal["mlp", "uniform", "tabular"]
DEFAULT_REVERSE_POLICY_TYPE: ReversePolicyType = "mlp"


def paired_learned_reverse_state(checkpoint_path: Path) -> Path:
    root = checkpoint_path.parent
    name = checkpoint_path.name
    if name == "final_checkpoint.pt":
        path = root / "learned_reverse_state.pt"
    elif name.startswith("checkpoint_epoch") and name.endswith(".pt"):
        path = root / f"learned_reverse_epoch{name[len('checkpoint_epoch'):]}"
    else:
        raise ValueError(f"cannot infer learned-reverse state for checkpoint {name}")
    if not path.exists():
        raise FileNotFoundError(f"missing paired learned-reverse state: {path}")
    return path


def restore_running_normalizer(
    normalizer: RunningLogWeightNormalizer,
    state: dict[str, float | int | None],
) -> None:
    normalizer.log_first_moment = state["log_first_moment"]
    normalizer.log_second_moment = state["log_second_moment"]
    normalizer.updates = int(state["updates"])


def load_learned_reverse_state(
    path: Path,
    *,
    policy: Union[TabularTerminalReversePolicy, PhyloLearnedReversePolicy] | None,
    reverse_optimizer: torch.optim.Optimizer | None,
    forward_trainer: GRPOTrainer,
    normalizer: RunningLogWeightNormalizer | None,
    device: str,
) -> dict[str, Any]:
    state = torch.load(path, map_location=device, weights_only=False)
    if state.get("algorithm") != METHOD:
        raise ValueError(
            f"unexpected learned-reverse state algorithm: {state.get('algorithm')!r}"
        )
    reverse_policy_type = str(state.get("reverse_policy_type", DEFAULT_REVERSE_POLICY_TYPE))
    if reverse_policy_type != "uniform":
        if policy is None or reverse_optimizer is None:
            raise ValueError("policy and reverse_optimizer required for non-uniform reverse state")
        policy.load_state_dict(state["reverse_policy"])
        reverse_optimizer.load_state_dict(state["reverse_optimizer"])
    forward_trainer.load_state_dict(state["forward_trainer"])
    if normalizer is not None and state.get("running_normalizer") is not None:
        restore_running_normalizer(normalizer, state["running_normalizer"])
    return state


def save_learned_reverse_state(
    path: Path,
    *,
    policy: Union[TabularTerminalReversePolicy, PhyloLearnedReversePolicy] | None,
    optimizer: torch.optim.Optimizer | None,
    forward_trainer: GRPOTrainer,
    normalizer: RunningLogWeightNormalizer | None,
    update_step: int,
    reverse_config: PhyloLearnedReverseConfig | None = None,
    reverse_policy_type: ReversePolicyType = DEFAULT_REVERSE_POLICY_TYPE,
) -> None:
    payload: dict[str, Any] = {
        "algorithm": METHOD,
        "reverse_policy_type": reverse_policy_type,
        "forward_trainer": forward_trainer.state_dict(),
        "running_normalizer": (
            normalizer.state_dict() if normalizer is not None else None
        ),
        "update_step": update_step,
    }
    if reverse_policy_type == "uniform":
        payload["reverse_policy"] = None
        payload["reverse_optimizer"] = None
    else:
        if policy is None or optimizer is None:
            raise ValueError("policy and optimizer required for non-uniform reverse state")
        payload["reverse_policy"] = policy.state_dict()
        payload["reverse_optimizer"] = optimizer.state_dict()
    if policy is not None and isinstance(policy, TabularTerminalReversePolicy):
        payload["trajectories"] = policy.trajectories
        payload["terminal_ids"] = policy.terminal_ids
    if reverse_config is not None:
        payload["reverse_config"] = asdict(reverse_config)
    torch.save(payload, path)


def write_catalog(
    output_dir: Path,
    trajectories: Sequence[tuple[int, ...]] | None,
    terminal_ids: Sequence[str] | None,
    *,
    outcome_level: str,
    reverse_policy_type: ReversePolicyType = DEFAULT_REVERSE_POLICY_TYPE,
) -> None:
    payload: dict[str, Any] = {
        "reverse_policy_type": reverse_policy_type,
        "reverse_conditioning_level": "topology",
        "reported_outcome_level": outcome_level,
    }
    if trajectories is None or terminal_ids is None:
        payload["num_trajectories"] = None
        payload["num_structural_outcomes"] = None
        payload["entries"] = []
    else:
        multiplicities = Counter(terminal_ids)
        payload.update(
            {
                "num_trajectories": len(trajectories),
                "num_structural_outcomes": len(multiplicities),
                "multiplicity_histogram": {
                    str(multiplicity): count
                    for multiplicity, count in sorted(
                        Counter(multiplicities.values()).items()
                    )
                },
                "entries": [
                    {"trajectory": list(trajectory), "terminal_id": terminal_id}
                    for trajectory, terminal_id in zip(trajectories, terminal_ids)
                ],
            }
        )
    (output_dir / "reverse_catalog.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def buffered_best_trees(data_loader) -> list:
    return getattr(data_loader, "best_trees", [])


def save_best_trees(output_dir: Path, data_loader) -> None:
    if data_loader.best_state_batch_size <= 0:
        return
    trees = buffered_best_trees(data_loader)
    if not trees:
        return
    path = output_dir / "best_trees.pt"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(trees, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def replay_batch_metrics(
    batch: dict[str, torch.Tensor],
    replay_tree_count: int,
) -> dict[str, float]:
    if replay_tree_count <= 0:
        return {}
    log_scores = batch["log_scores"]
    if replay_tree_count >= log_scores.numel():
        return {
            "replay_tree_fraction": 1.0,
            "mean_log_score_replay": float(log_scores.mean().item()),
        }
    return {
        "replay_tree_fraction": replay_tree_count / log_scores.numel(),
        "mean_log_score_replay": float(log_scores[:replay_tree_count].mean().item()),
        "mean_log_score_on_policy": float(log_scores[replay_tree_count:].mean().item()),
    }
