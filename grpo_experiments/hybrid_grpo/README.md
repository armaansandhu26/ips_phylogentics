# Hybrid GRPO

Production path for best-tree replay + policy IS. Implements:

1. Sample a fixed batch under behavior policy `pi_old`.
2. Mix fresh rollouts (`--fresh-buffer-size`) and best-tree replay (`--replay-sample-size`).
3. Re-evaluate `pi_new` on stored actions for `--update-cycles` inner steps (frozen `log_paths_pf_old`, TRL PPO surrogate).
4. Resample and repeat (`--resample-rounds`).

Run from repo root:

```bash
python -m grpo_experiments.hybrid_grpo.train \
  --fresh-buffer-size 512 \
  --replay-sample-size 512 \
  --best-tree-buffer-size 2048 \
  --resample-rounds 100 \
  --update-cycles 1 \
  --grpo-clip-eps 0.2 \
  --entropy-coef 0.01
```

Inner reuse is controlled by `--update-cycles`, not `--grpo-num-iterations` (TRL μ applies to on-policy `runner.py` only).

Policy replay and temperature-aligned log-probs: `grpo_experiments/core/policy_replay.py`.
