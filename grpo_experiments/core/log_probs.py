"""Temperature-aligned forward log-probs for GRPO (matches sampling distribution)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sampling_temperature(random_spec: dict | None) -> float:
    if random_spec is None or "T" not in random_spec:
        return 1.0
    return float(random_spec["T"])


def _gather_log_prob(logits: torch.Tensor, actions: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature != 1.0:
        logits = logits / temperature
    log_p = F.log_softmax(logits, dim=-1)
    if actions.ndim == 1:
        return log_p.gather(1, actions.long().unsqueeze(1)).squeeze(1)
    left = log_p.gather(1, actions[:, 0].long().unsqueeze(1)).squeeze(1)
    right = log_p.gather(1, actions[:, 1].long().unsqueeze(1)).squeeze(1)
    return left + right


def step_log_paths_pf_at_sampling_temperature(
    ret: dict,
    *,
    parsimony_problem: bool,
    batch_nb_seq: torch.Tensor,
    edges_independent: bool,
    random_spec: dict | None,
) -> torch.Tensor:
    """log pi(a|s) at the temperature used for sampling (random_spec['T'])."""
    temperature = sampling_temperature(random_spec)
    trees_ret = ret["trees_ret"]
    tree_actions = trees_ret["tree_actions"]
    log_paths_pf = _gather_log_prob(trees_ret["logits"], tree_actions, temperature)

    if parsimony_problem:
        return log_paths_pf

    edges_ret = ret["edges_ret"]
    edge_actions = edges_ret["edge_actions"]
    root_edges_flag = batch_nb_seq == 2
    first_edges_flag = batch_nb_seq > 2

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
        log_paths_pf = log_paths_pf.clone()
        log_paths_pf[first_edges_flag] = log_paths_pf[first_edges_flag] + pf

    if root_edges_flag.sum().item() > 0:
        root_actions = edge_actions[root_edges_flag]
        pf = _gather_log_prob(edges_ret["root_edges_ret"]["logits"], root_actions, temperature)
        log_paths_pf = log_paths_pf.clone()
        log_paths_pf[root_edges_flag] = log_paths_pf[root_edges_flag] + pf

    return log_paths_pf
