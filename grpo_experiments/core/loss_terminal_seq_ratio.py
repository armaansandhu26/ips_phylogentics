"""Normalized terminal × sequence log importance ratio.

    terminal = normalize(log_score - log_p_hat)
    seq_log_ratio = sum(log_pi_new) - sum(log_pi_old)
    loss = -mean(terminal.detach() * seq_log_ratio)
"""

from __future__ import annotations

import torch


def compute_terminal_seq_ratio_loss(
    log_paths_pf: torch.Tensor,
    log_scores: torch.Tensor,
    log_p_hat: torch.Tensor,
    *,
    log_paths_pf_old: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
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

    if log_paths_pf_old is None:
        old_per_token_logps = log_paths_pf.detach()
    else:
        if log_paths_pf_old.shape != log_paths_pf.shape:
            raise ValueError(
                f"log_paths_pf_old shape {tuple(log_paths_pf_old.shape)} != "
                f"log_paths_pf shape {tuple(log_paths_pf.shape)}."
            )
        old_per_token_logps = log_paths_pf_old.detach()

    seq_log_pf_new = (log_paths_pf * mask).sum(dim=-1)
    seq_log_pf_old = (old_per_token_logps * mask).sum(dim=-1)
    seq_log_ratio = seq_log_pf_new - seq_log_pf_old.detach()

    terminal = log_scores.detach() - log_p_hat.detach()
    terminal = terminal - terminal.mean()
    terminal = terminal / (terminal.std() + 1e-8)

    loss = -(terminal.detach() * seq_log_ratio).mean()

    with torch.no_grad():
        metrics = {
            "policy_loss_mode": "terminal_seq_ratio",
            "mean_log_ips_terminal": float((log_scores.detach() - log_p_hat.detach()).mean().item()),
            "std_log_ips_terminal": float((log_scores.detach() - log_p_hat.detach()).std().item()),
            "mean_log_importance_ratio_seq": float(seq_log_ratio.mean().item()),
            "mean_sequence_importance_ratio": float(seq_log_ratio.exp().mean().item()),
            "mean_advantage": float(terminal.mean().item()),
            "std_advantage": float(terminal.std().item()),
            "mean_log_objective": float((terminal.detach() * seq_log_ratio).mean().item()),
        }

    return loss, metrics
