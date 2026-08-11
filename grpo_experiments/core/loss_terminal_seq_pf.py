"""Normalized terminal advantage × sequence log π_f (no importance ratio).

    advantage = normalize(log_score - log_p_hat)
    loss = -mean(advantage.detach() * seq_log_pf)
"""

from __future__ import annotations

import torch


def compute_terminal_seq_pf_loss(
    log_paths_pf: torch.Tensor,
    log_scores: torch.Tensor,
    log_p_hat: torch.Tensor,
    *,
    log_paths_pf_old: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    del log_paths_pf_old  # unused; kept for trainer API compatibility

    if log_paths_pf.ndim != 2:
        raise ValueError(f"log_paths_pf must be (B, T), got {tuple(log_paths_pf.shape)}.")
    if log_scores.ndim != 1:
        raise ValueError(f"log_scores must be (B,), got {tuple(log_scores.shape)}.")
    if log_p_hat.ndim != 1:
        raise ValueError(f"log_p_hat must be (B,), got {tuple(log_p_hat.shape)}.")
    if log_scores.shape[0] != log_paths_pf.shape[0]:
        raise ValueError(
            f"log_scores batch ({log_scores.shape[0]}) != log_paths_pf batch ({log_paths_pf.shape[0]})."
        )
    if log_p_hat.shape[0] != log_paths_pf.shape[0]:
        raise ValueError(
            f"log_p_hat batch ({log_p_hat.shape[0]}) != log_paths_pf batch ({log_paths_pf.shape[0]})."
        )

    if mask is None:
        mask = torch.ones_like(log_paths_pf, dtype=log_paths_pf.dtype)
    else:
        mask = mask.to(dtype=log_paths_pf.dtype)

    terminal = log_scores.detach() - log_p_hat.detach()
    seq_log_pf = (log_paths_pf * mask).sum(dim=-1)

    advantage = terminal
    advantage = advantage - advantage.mean()
    advantage = advantage / (advantage.std() + 1e-8)

    loss = -(advantage.detach() * seq_log_pf).mean()

    with torch.no_grad():
        metrics = {
            "policy_loss_mode": "terminal_seq_pf",
            "mean_log_ips_terminal": float(terminal.mean().item()),
            "std_log_ips_terminal": float(terminal.std().item()),
            "mean_seq_log_pf": float(seq_log_pf.mean().item()),
            "mean_advantage": float(advantage.mean().item()),
            "std_advantage": float(advantage.std().item()),
            "mean_log_objective": float((advantage.detach() * seq_log_pf).mean().item()),
        }

    return loss, metrics
