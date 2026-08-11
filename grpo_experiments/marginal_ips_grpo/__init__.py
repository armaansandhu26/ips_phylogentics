"""Marginal (backward-corrected) exact IPS-GRPO.

Self-contained experiment folder. Everything here is additive: it reuses the
existing IPS-GRPO trainer/loss/rollout machinery and only changes *which*
trajectory log-probability is fed into the exact inverse-propensity weight.

To revert: delete this folder. No files outside it are modified.
"""
