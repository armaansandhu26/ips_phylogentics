"""Per-step tree/edge decomposition of forward log-probs for split-credit policy losses."""

from __future__ import annotations

import torch

from grpo_experiments.core.log_probs import _gather_log_prob, sampling_temperature


def step_log_pf_tree_at_sampling_temperature(
    ret: dict,
    *,
    random_spec: dict | None,
) -> torch.Tensor:
    """log pi_tree(a|s) at the temperature used for sampling."""
    temperature = sampling_temperature(random_spec)
    trees_ret = ret["trees_ret"]
    return _gather_log_prob(trees_ret["logits"], trees_ret["tree_actions"], temperature)


def step_log_pf_edge_at_sampling_temperature(
    ret: dict,
    *,
    parsimony_problem: bool,
    batch_nb_seq: torch.Tensor,
    edges_independent: bool,
    random_spec: dict | None,
    only_train_tree_model: bool = False,
) -> torch.Tensor:
    """log pi_edge(a|s) at sampling temperature; zero when edges are unused."""
    if parsimony_problem or only_train_tree_model:
        ref = ret["trees_ret"]["logits"]
        return torch.zeros(len(batch_nb_seq), device=ref.device, dtype=ref.dtype)

    temperature = sampling_temperature(random_spec)
    edges_ret = ret["edges_ret"]
    edge_actions = edges_ret["edge_actions"]
    root_edges_flag = batch_nb_seq == 2
    first_edges_flag = batch_nb_seq > 2

    ref = ret["trees_ret"]["logits"]
    log_pf_edge = torch.zeros(len(batch_nb_seq), device=ref.device, dtype=ref.dtype)

    if first_edges_flag.sum().item() > 0:
        first_edges_ret = edges_ret["first_edges_ret"]
        first_actions = edge_actions[first_edges_flag]
        if edges_independent:
            log_p_l = _gather_log_prob(
                first_edges_ret["l_logits"], first_actions[:, 0], temperature,
            )
            log_p_r = _gather_log_prob(
                first_edges_ret["r_logits"], first_actions[:, 1], temperature,
            )
            pf = log_p_l + log_p_r
        else:
            pf = _gather_log_prob(first_edges_ret["logits"], first_actions, temperature)
        log_pf_edge = log_pf_edge.clone()
        log_pf_edge[first_edges_flag] = pf

    if root_edges_flag.sum().item() > 0:
        root_actions = edge_actions[root_edges_flag]
        pf = _gather_log_prob(edges_ret["root_edges_ret"]["logits"], root_actions, temperature)
        log_pf_edge = log_pf_edge.clone()
        log_pf_edge[root_edges_flag] = pf

    return log_pf_edge


def step_log_paths_pf_split_at_sampling_temperature(
    ret: dict,
    *,
    parsimony_problem: bool,
    batch_nb_seq: torch.Tensor,
    edges_independent: bool,
    random_spec: dict | None,
    only_train_tree_model: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (log_pf_tree, log_pf_edge) with log_pf_tree + log_pf_edge = combined step log-prob."""
    log_pf_tree = step_log_pf_tree_at_sampling_temperature(ret, random_spec=random_spec)
    log_pf_edge = step_log_pf_edge_at_sampling_temperature(
        ret,
        parsimony_problem=parsimony_problem,
        batch_nb_seq=batch_nb_seq,
        edges_independent=edges_independent,
        random_spec=random_spec,
        only_train_tree_model=only_train_tree_model,
    )
    return log_pf_tree, log_pf_edge
