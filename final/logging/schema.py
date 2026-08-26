"""Metric key groups for the final paper logging spec."""

from __future__ import annotations

# Eval dump schedule: global_step -> True
EVAL_STEPS_LE_5K = {0} | {s for s in range(500, 5001, 500)}
EVAL_STEPS_GT_5K = {s for s in range(6000, 1_000_000, 1000)}


def eval_steps_through(max_step: int) -> set[int]:
    steps = {s for s in EVAL_STEPS_LE_5K if s <= max_step}
    steps |= {s for s in EVAL_STEPS_GT_5K if s <= max_step}
    return steps


def should_eval_dump(global_step: int, *, final_step: int | None = None) -> bool:
    if global_step in EVAL_STEPS_LE_5K or global_step in EVAL_STEPS_GT_5K:
        return True
    if final_step is not None and global_step == final_step:
        return True
    return False


COMMON_STEP_KEYS = (
    "step",
    "epoch",
    "global_step",
    "wall_clock_s",
    "gpu_seconds",
    "batch_size",
    "group_size",
    "log_R_mean",
    "log_R_std",
    "log_R_min",
    "log_R_max",
    "traj_len_mean",
    "traj_len_std",
    "unique_terminals_in_batch",
    "pf_entropy",
    "grad_norm",
    "lr",
    "loss",
)

PPO_EXTRA_KEYS = (
    "policy_loss",
    "clip_frac",
    "approx_kl",
    "ratio_mean",
    "ratio_std",
    "advantage_mean",
    "advantage_std",
    "ema_baseline",
    "group_reward_std",
    "distinct_terminals_per_group",
)

LEARNED_REVERSE_EXTRA_KEYS = (
    "reverse_nll",
    "reverse_nll_by_position",
    "reverse_mle_steps",
    "log_q_mean",
    "log_q_std",
    "log_pf_mean",
    "log_pf_std",
    "log_w_mean",
    "log_w_std",
    "log_w_min",
    "log_w_max",
    "log_w_hist_counts",
    "ips_ess",
    "max_normalized_weight",
    "psis_khat",
)

COUNT_IPS_EXTRA_KEYS = (
    "n_terminals_with_count_ge_2",
    "p_hat_mean",
    "p_hat_min",
    "p_hat_max",
)

GFN_EXTRA_KEYS = (
    "tb_loss",
    "log_Z",
    "log_pb_mean",
    "log_pb_std",
    "replay_size",
    "replay_hit_rate",
)

# Fixed histogram bin edges for log_w (importance weights in log space)
LOG_W_BIN_EDGES = [
    -50.0,
    -30.0,
    -20.0,
    -15.0,
    -10.0,
    -7.0,
    -5.0,
    -3.0,
    -2.0,
    -1.0,
    0.0,
    1.0,
    2.0,
    3.0,
    5.0,
    7.0,
    10.0,
    15.0,
    20.0,
    30.0,
    50.0,
]

EVAL_TRAJECTORY_FIELDS = (
    "signature_hash",
    "topology_id",
    "log_R",
    "log_pf_traj",
    "traj_len",
    "log_m_x",
    "log_q_traj",
    "log_w",
    "log_pb_traj",
    "log_Z_current",
    "p_hat_batch",
    "seed",
    "step",
)
