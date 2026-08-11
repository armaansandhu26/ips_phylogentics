"""Broadcast terminal × per-token log importance ratio.

    terminal = (log_score - log_p_hat).unsqueeze(1)
    per_token = terminal * (log_pi_new - log_pi_old)
    loss = -masked_mean(per_token)
"""

from __future__ import annotations

import torch


def _masked_mean(per_token: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    token_counts = mask.sum(dim=-1).clamp(min=1.0)
    per_seq = (per_token * mask).sum(dim=-1) / token_counts
    return per_seq.mean()


def compute_terminal_token_ratio_loss(
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

    terminal = (log_scores.detach() - log_p_hat.detach()).unsqueeze(1)
    log_ratio = log_paths_pf - old_per_token_logps
    per_token = terminal * log_ratio
    loss = -_masked_mean(per_token, mask)

    with torch.no_grad():
        log_ratio_raw = log_ratio.detach()
        seq_log_ratio = (log_ratio_raw * mask).sum(dim=-1)
        terminal_scalar = log_scores.detach() - log_p_hat.detach()
        metrics = {
            "policy_loss_mode": "terminal_token_ratio",
            "mean_log_ips_terminal": float(terminal_scalar.mean().item()),
            "std_log_ips_terminal": float(terminal_scalar.std().item()),
            "mean_log_importance_ratio": float(
                (log_ratio_raw * mask).sum().item() / mask.sum().clamp(min=1.0).item()
            ),
            "mean_log_importance_ratio_seq": float(seq_log_ratio.mean().item()),
            "mean_sequence_importance_ratio": float(seq_log_ratio.exp().mean().item()),
            "mean_log_objective": float(_masked_mean(per_token, mask).item()),
        }

    return loss, metrics
