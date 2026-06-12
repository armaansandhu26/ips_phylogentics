"""Fixed-action forward replay for policy IS updates."""

from __future__ import annotations

import torch

from grpo_experiments.core.log_probs import step_log_paths_pf_at_sampling_temperature
from grpo_experiments.core.policy_entropy import step_entropy_from_forward


def forward_replay_fixed_actions(
    rollout_worker,
    generator,
    actions_set: list,
    *,
    random_spec: dict | None,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replay stored actions under the current policy; return (log_paths_pf, paths_entropy)."""
    env = rollout_worker.env
    episodes = len(actions_set)
    seq_arrays = env.seq_arrays
    n_taxa = seq_arrays.shape[0]

    all_log_paths_pf: list[torch.Tensor] = []
    all_paths_entropy: list[torch.Tensor] = []

    tree_features = seq_arrays.unsqueeze(0).repeat(episodes, 1, 1, 1)
    has_children = torch.zeros(episodes, n_taxa).bool()
    parsimony_problem = env.parsimony_problem
    edges_independent = False
    if not parsimony_problem and hasattr(generator, "edges_model"):
        edges_independent = bool(getattr(generator.edges_model, "edges_independent", False))

    step = 0
    while tree_features.shape[1] > 1:
        input_actions = [trajectory_actions[step] for trajectory_actions in actions_set]
        input_dict = env.prepare_rollout_inputs(tree_features, input_actions, random_spec)
        ret = generator(input_dict)
        trees_ret = ret["trees_ret"]

        tree_actions = trees_ret["tree_actions"].detach().cpu().numpy()
        actions = [{"tree_action": int(x)} for x in tree_actions]
        if not parsimony_problem:
            edge_actions = ret["edges_ret"]["edge_actions"].detach().cpu().numpy()
            for idx, action in enumerate(actions):
                action["edge_action"] = edge_actions[idx]

        _, new_tree_features, _, _log_rewards = env.batch_apply_actions(
            actions,
            tree_features,
            None,
        )

        b, n, _, _ = tree_features.shape
        left_trees_indices = [pair[0] for pair in trees_ret["tree_pairs"]]
        right_trees_indices = [pair[1] for pair in trees_ret["tree_pairs"]]
        new_states_inputs_indices = torch.ones(b, n).bool()
        new_states_inputs_indices[torch.arange(b), right_trees_indices] = False
        has_children = has_children[new_states_inputs_indices]
        has_children = has_children.reshape(b, -1)
        has_children[torch.arange(b), left_trees_indices] = 1

        batch_nb_seq = input_dict["batch_nb_seq"]
        all_log_paths_pf.append(
            step_log_paths_pf_at_sampling_temperature(
                ret,
                parsimony_problem=parsimony_problem,
                batch_nb_seq=batch_nb_seq,
                edges_independent=edges_independent,
                random_spec=random_spec,
            )
        )
        all_paths_entropy.append(
            step_entropy_from_forward(
                ret,
                parsimony_problem=parsimony_problem,
                batch_nb_seq=batch_nb_seq,
                edges_independent=edges_independent,
            )
        )
        tree_features = new_tree_features
        step += 1

    log_paths_pf = torch.stack(all_log_paths_pf).T.to(device)
    paths_entropy = torch.stack(all_paths_entropy).T.to(device)
    return log_paths_pf, paths_entropy
