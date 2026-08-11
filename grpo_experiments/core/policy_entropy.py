"""Policy entropy helpers for GRPO (entropy regularization, not TRL KL)."""

from __future__ import annotations

import torch
from torch.distributions import Categorical


def tree_step_entropy(trees_ret: dict) -> torch.Tensor:
    return Categorical(logits=trees_ret["logits"]).entropy()


def edges_step_entropy_categorical(
    edges_ret: dict,
    batch_nb_seq: torch.Tensor,
    *,
    edges_independent: bool,
) -> torch.Tensor:
    root_edges_flag = batch_nb_seq == 2
    first_edges_flag = batch_nb_seq > 2

    if first_edges_flag.sum().item() > 0:
        ref = (
            edges_ret["first_edges_ret"]["l_logits"]
            if edges_independent
            else edges_ret["first_edges_ret"]["logits"]
        )
    else:
        ref = edges_ret["root_edges_ret"]["logits"]
    ent = torch.zeros(len(batch_nb_seq), device=ref.device, dtype=ref.dtype)

    if first_edges_flag.sum().item() > 0:
        if edges_independent:
            first_edges_ret = edges_ret["first_edges_ret"]
            ent[first_edges_flag] = (
                Categorical(logits=first_edges_ret["l_logits"]).entropy()
                + Categorical(logits=first_edges_ret["r_logits"]).entropy()
            )
        else:
            ent[first_edges_flag] = Categorical(
                logits=edges_ret["first_edges_ret"]["logits"]
            ).entropy()

    if root_edges_flag.sum().item() > 0:
        ent[root_edges_flag] = Categorical(
            logits=edges_ret["root_edges_ret"]["logits"]
        ).entropy()
    return ent


def edges_step_entropy_continuous(edges_ret: dict, batch_nb_seq: torch.Tensor) -> torch.Tensor:
    if batch_nb_seq[0].item() > 2:
        return edges_ret["first_edges_ret"]["dist"].entropy()
    return edges_ret["root_edges_ret"]["dist"].entropy()


def step_entropy_from_forward(
    ret: dict,
    *,
    parsimony_problem: bool,
    batch_nb_seq: torch.Tensor,
    edges_independent: bool,
    only_train_tree_model: bool = False,
) -> torch.Tensor:
    ent = tree_step_entropy(ret["trees_ret"])
    if parsimony_problem or only_train_tree_model:
        return ent

    edges_ret = ret["edges_ret"]
    if "first_edges_ret" in edges_ret and "dist" in edges_ret.get("first_edges_ret", {}):
        ent = ent + edges_step_entropy_continuous(edges_ret, batch_nb_seq)
    else:
        ent = ent + edges_step_entropy_categorical(
            edges_ret,
            batch_nb_seq,
            edges_independent=edges_independent,
        )
    return ent
