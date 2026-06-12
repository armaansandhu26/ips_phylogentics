# IPS-GRPO

**Inverse Probability Scaling + GRPO** (Sinha et al., [arXiv:2601.21669](https://arxiv.org/pdf/2601.21669)).

Extends `grpo_experiments.core.trainer` with outcome-frequency reward scaling before group-normalized advantages.

Full GRPO + IPS flow diagrams and debugging map → [`../FLOWS.md`](../FLOWS.md).

## Quick start

```bash
# IPS-GRPO
python -m grpo_experiments.ips_grpo.train --on-policy-batch-size 64 --epochs 5 --steps-per-epoch 10

# Compare with
python -m grpo_experiments.train --method phylgfn --on-policy-batch-size 64 --epochs 5 --steps-per-epoch 10
python -m grpo_experiments.train --method grpo     --on-policy-batch-size 64 --epochs 5 --steps-per-epoch 10
```

Runs → `grpo_experiments/runs/<timestamp>_ips_grpo/`.

For production hybrid IPS + best-tree replay, use `python -m grpo_experiments.hybrid_ips_grpo.train`.

---

## How IPS differs from GRPO

Same rollout and same TRL-style PPO policy loss. **Only the reward signal changes** before advantages:

```
GRPO:     A_i from group-normalized rewards
IPS-GRPO: r_tilde_i = r_i / max(p_hat(o_i), eps)
          A_i from group-normalized r_tilde_i
```

where `p_hat(o) = count(o in batch) / G` and `r_i` comes from `exp(log_reward - max)` then IPS scaling (see `core/advantages.py`).

---

## Training modes

### On-policy (default)

Each step: fresh rollout → IPS advantages → GRPO update.

```bash
python -m grpo_experiments.ips_grpo.train --preset topology_sanity
```

### Policy importance sampling (`--enable-policy-is`)

Sample a fixed buffer under the behavior policy, then inner cycles replay stored actions with token-level `pi_new / pi_old` in the PPO surrogate. IPS outcome scaling and advantages are **frozen** per resample round.

```bash
python -m grpo_experiments.ips_grpo.train \
  --enable-policy-is \
  --resample-rounds 5 --update-cycles 10 --buffer-size 1000
```

---

## Training flow (one on-policy step)

```
ROLLOUT → outcome_ids
       → IPSGRPOTrainer.update
            p_hat(o) = count / G (floored)
            r_tilde = r / p_hat
            advantages = group-normalize(r_tilde)
            pg_loss = TRL PPO surrogate on token log-probs (core/loss.py)
            optional entropy bonus (not TRL KL)
```

Policy IS replay: `grpo_experiments/core/policy_replay.py`.

---

## Key files

| File | Role |
|------|------|
| `train.py` | CLI entry |
| `runner.py` | On-policy or policy-IS loop |
| `trainer.py` | IPS advantage scaling + `GRPOTrainer` |
| `config.py` | `--ips-prob-floor`, `--enable-policy-is`, presets |

---

## Hyperparameters

| Flag | Default | Meaning |
|------|---------|---------|
| `--ips-prob-floor` | `1e-6` | ε in `r / max(p_hat, ε)` |
| `--grpo-clip-eps` | `0.2` | PPO clip (TRL default) |
| `--grpo-entropy-coef` | `0` | Entropy bonus (we use this instead of TRL `beta`/KL) |
| `--grpo-num-iterations` | `1` | On-policy μ reuse before resampling |
| `--enable-policy-is` | off | Fixed buffer + inner update cycles |
| `--outcome-level` | `topology` | Outcome `o` for p_hat |
| `--preset` | — | `configs/ips_grpo_presets.json` |

Use **G ≥ 2** for meaningful advantages and p_hat estimates.

---

## Topology vs signature outcomes

| Level | Outcome ID |
|-------|------------|
| `topology` | tree shape only |
| `signature` | topology + discretized log_score |

```bash
python -m grpo_experiments.ips_grpo.train --preset topology_sanity
python -m grpo_experiments.ips_grpo.train --preset signature_sanity
```

Presets: `grpo_experiments/configs/ips_grpo_presets.json`.
