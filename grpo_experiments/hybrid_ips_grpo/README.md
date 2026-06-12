# Hybrid IPS-GRPO

This runner implements the same hybrid sampling loop as [`hybrid_grpo`](../hybrid_grpo/README.md), but uses **IPS-GRPO** advantages (outcome-frequency scaling) plus policy importance weights:

1. Sample a fixed batch under current behavior policy `pi_old`.
2. Mix two sources in that batch:
   - fresh rollouts (`--fresh-buffer-size`)
   - best-tree replay (`--replay-sample-size`)
3. Compute IPS-scaled advantages from the mixed batch (`r_tilde = r / max(p_hat(o), eps)`).
4. Reuse that fixed mixed batch for multiple update cycles with policy importance
   weights `pi_new / pi_old` (`--update-cycles`).
5. Resample a new mixed batch and repeat (`--resample-rounds`).

Run from repo root:

```bash
python -m grpo_experiments.hybrid_ips_grpo.train \
  --fresh-buffer-size 512 \
  --replay-sample-size 512 \
  --best-tree-buffer-size 2048 \
  --resample-rounds 100 \
  --update-cycles 20 \
  --ips-prob-floor 0.01 \
  --is-ratio-clip 0.2
```
