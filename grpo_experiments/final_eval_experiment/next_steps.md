
>>
code cleanup [P0] — done (softmax/gating removed)
comparison with ref [P1] — done
  - PPO clip default = 0.2 (TRL-aligned), entropy reg not KL (intentional)

CHECK: advantages / reward scale [P2]
  - Current: r = exp(log_reward - max(log_reward)), then A = (r - mean) / std
  - TRL reference: group-normalize raw reward values (not exp-transform)
  - Action: inspect advantage distribution in metrics; if skewed, try
    log_reward or linear reward directly instead of exp before group-norm

grpo_num_iterations (--grpo-num-iterations, TRL μ)
  - How many optimizer steps to reuse the SAME on-policy rollout before resampling
  - Implemented in `runner.py` via `core/on_policy_buffer.py` (on-policy only)
  - Hybrid: use `--update-cycles`; trainer `num_iterations` is always 1 there

reward to 2 decimal points -> reward signal + signature
(should reduce output size) [P3]





>> push to github

20k step runs


objective function -> topology + signature diversity balance
