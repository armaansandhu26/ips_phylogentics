"""Fixed-action forward replay for policy IS updates."""

from __future__ import annotations

import torch

from grpo_experiments.core.log_probs import step_log_paths_pf_at_sampling_temperature
from grpo_experiments.core.log_probs_split import step_log_paths_pf_split_at_sampling_temperature
from grpo_experiments.core.policy_entropy import step_entropy_from_forward
from src.gfn.action_tensors import TensorActionBatch


def forward_replay_fixed_actions(
    rollout_worker,
    generator,
    actions_set: list | TensorActionBatch,
    *,
    random_spec: dict | None,
    device: str,
    return_split: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Replay stored actions under the current policy.

    Returns (log_paths_pf, paths_entropy), or with return_split=True also
    (log_paths_pf_tree, log_paths_pf_edge).
    """
    env = rollout_worker.env
    episodes = len(actions_set)
    seq_arrays = env.seq_arrays

    all_log_paths_pf: list[torch.Tensor] = []
    all_log_paths_pf_tree: list[torch.Tensor] = []
    all_log_paths_pf_edge: list[torch.Tensor] = []
    all_paths_entropy: list[torch.Tensor] = []

    tree_features = seq_arrays.unsqueeze(0).repeat(episodes, 1, 1, 1)
    parsimony_problem = env.parsimony_problem
    tensor_actions = actions_set if isinstance(actions_set, TensorActionBatch) else None
    only_train_tree_model = bool(
        getattr(generator, "only_train_tree_model", False)
        or getattr(env, "only_train_tree_model", False)
    )
    edges_independent = False
    if not parsimony_problem and not only_train_tree_model and hasattr(generator, "edges_model"):
        edges_independent = bool(getattr(generator.edges_model, "edges_independent", False))

    step = 0
    while tree_features.shape[1] > 1:
        num_trees = tree_features.shape[1]
        input_actions = None
        if tensor_actions is None:
            input_actions = [trajectory_actions[step] for trajectory_actions in actions_set]
        input_dict = env.prepare_rollout_inputs(tree_features, input_actions, random_spec)
        if tensor_actions is not None:
            device_for_actions = tree_features.device
            input_dict["input_tree_actions"] = tensor_actions.step_tree_actions(step, device_for_actions)
            input_edge_actions = tensor_actions.step_edge_actions(step, device_for_actions)
            if input_edge_actions is not None:
                input_dict["input_edge_actions"] = input_edge_actions
        ret = generator(input_dict)
        trees_ret = ret["trees_ret"]

        tree_actions = trees_ret["tree_actions"]
        edge_actions = None if parsimony_problem else ret["edges_ret"]["edge_actions"]
        _, new_tree_features, _, _log_rewards = env.batch_apply_actions_tensors(
            tree_actions,
            edge_actions,
            tree_features,
            None,
            num_trees=num_trees,
        )

        batch_nb_seq = input_dict["batch_nb_seq"]
        if return_split:
            log_pf_tree, log_pf_edge = step_log_paths_pf_split_at_sampling_temperature(
                ret,
                parsimony_problem=parsimony_problem,
                batch_nb_seq=batch_nb_seq,
                edges_independent=edges_independent,
                random_spec=random_spec,
                only_train_tree_model=only_train_tree_model,
            )
            all_log_paths_pf_tree.append(log_pf_tree)
            all_log_paths_pf_edge.append(log_pf_edge)
            all_log_paths_pf.append(log_pf_tree + log_pf_edge)
        else:
            all_log_paths_pf.append(
                step_log_paths_pf_at_sampling_temperature(
                    ret,
                    parsimony_problem=parsimony_problem,
                    batch_nb_seq=batch_nb_seq,
                    edges_independent=edges_independent,
                    random_spec=random_spec,
                    only_train_tree_model=only_train_tree_model,
                )
            )
        all_paths_entropy.append(
            step_entropy_from_forward(
                ret,
                parsimony_problem=parsimony_problem,
                batch_nb_seq=batch_nb_seq,
                edges_independent=edges_independent,
                only_train_tree_model=only_train_tree_model,
            )
        )
        tree_features = new_tree_features
        step += 1

    log_paths_pf = torch.stack(all_log_paths_pf).T.to(device)
    paths_entropy = torch.stack(all_paths_entropy).T.to(device)
    if return_split:
        log_paths_pf_tree = torch.stack(all_log_paths_pf_tree).T.to(device)
        log_paths_pf_edge = torch.stack(all_log_paths_pf_edge).T.to(device)
        return log_paths_pf, paths_entropy, log_paths_pf_tree, log_paths_pf_edge
    return log_paths_pf, paths_entropy
