# IPS-GRPO

**Inverse Probability Scaling + GRPO** (Sinha et al., [arXiv:2601.21669](https://arxiv.org/pdf/2601.21669)).

Standalone experiment package — does **not** modify `grpo.py` or `runner.py`.
Use this alongside PhyloGFN and GRPO for three-way comparison.

## Quick start

```bash
# IPS-GRPO
python -m grpo_experiments.ips_grpo.train --on-policy-batch-size 64 --epochs 5 --steps-per-epoch 10

# Compare with (separate entry points)
python -m grpo_experiments.train --method phylgfn --on-policy-batch-size 64 --epochs 5 --steps-per-epoch 10
python -m grpo_experiments.train --method grpo     --on-policy-batch-size 64 --epochs 5 --steps-per-epoch 10
```

Runs → `grpo_experiments/runs/<timestamp>_ips_grpo/`.

---

## How IPS differs from GRPO

Same rollout and same policy-gradient step. **Only the reward signal changes** before advantages:

```
GRPO:     A_i from log_rewards_i
IPS-GRPO: r_tilde_i = log_rewards_i / max(p_hat(o_i), eps)
          A_i from r_tilde_i   (then same GRPO update)
```

where `p_hat(o) = count(o in batch) / G` (batch outcome frequency).

Rare outcomes get **upweighted**; duplicated outcomes get **downweighted** — reduces outcome-level mode collapse (paper §4.4, Algorithm 1).

---

## Training modes

### On-policy (default, presets unchanged)

Each step: fresh rollout → IPS advantages → GRPO update (`w = 1`).

```bash
python -m grpo_experiments.ips_grpo.train --preset topology_sanity
```

### Policy importance sampling (`--enable-policy-is`)

Same nested loop as [`is_grpo`](../is_grpo/README.md): sample buffer under behavior policy, then inner cycles replay stored actions with `w = pi_new / pi_old`. **IPS outcome scaling is frozen** from the buffer (p_hat and advantages computed once per round).

```bash
python -m grpo_experiments.ips_grpo.train \
  --enable-policy-is \
  --resample-rounds 5 --update-cycles 10 --buffer-size 1000
```

Omit `--resample-rounds` / `--update-cycles` / `--buffer-size` to reuse `--epochs`, `--steps-per-epoch`, and `--on-policy-batch-size` from a preset.

---

## Training flow (one on-policy step)

```
┌─────────────────────────────────────────────────────────────┐
│  ROLLOUT (shared — rollout_worker_phylo.py)                 │
│  same as GRPO → batch["log_paths_pf"] (B,T), log_rewards (B)│
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  reconstruct trees → outcome_ids (topology or signature)    │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  IPSGRPOTrainer.update (ips_grpo/trainer.py)                │
│    p_hat(o_g) = count in batch / G,  floor at ips_prob_floor│
│    r_tilde_g = log_rewards_g / p_hat(o_g)                   │
│    advantages = (r_tilde - mean) / (std + eps)   detached │
│    w = pi_new/pi_old  (if --enable-policy-is, else 1)       │
│    pg_loss = -mean(w * A * log_paths_pf / scale)            │
│    backward + Adam                                          │
└─────────────────────────────────────────────────────────────┘
```

Policy IS replay lives in `grpo_experiments/policy_replay.py` (shared with `is_grpo`).

---

## Key files

| File | Role |
|------|------|
| `train.py` | CLI entry |
| `runner.py` | Training loop → metrics.jsonl |
| `trainer.py` | Outcome IPS + optional policy IS weights |
| `runner.py` | On-policy loop or buffer + replay loop |
| `config.py` | `--ips-prob-floor`, `--enable-policy-is`, buffer schedule |

Shared with other methods: `grpo_experiments/utils.py`, `metrics.py`, rollout, generator.

---

## Hyperparameters

| Flag | Default | Meaning |
|------|---------|---------|
| `--ips-prob-floor` | `0.01` | ε in `r / max(p_hat, ε)` (paper §5.4) |
| `--enable-policy-is` | off | Fixed buffer + `w = pi_new/pi_old` replay |
| `--resample-rounds` | `epochs` | Outer loops when policy IS on |
| `--update-cycles` | `steps/epoch` | Inner IS updates per buffer |
| `--buffer-size` | `on-policy batch` | Trees per behavior rollout |
| `--outcome-level` | `topology` | What counts as outcome `o` for p_hat |
| `--on-policy-batch-size` | `64` | Group size G |
| `--preset` | — | Load matched settings from `configs/ips_grpo_presets.json` |
| `--list-presets` | — | Print available preset names |

Use **G ≥ 2** (same as GRPO) for meaningful batch advantages and p_hat estimates.

---

## Comparing topology vs signature outcomes

IPS `p_hat(o)` depends on how outcome `o` is defined:

| Level | Outcome ID | Use when |
|-------|------------|----------|
| `topology` (default) | `tree_topology_id` — unrooted shape only | Penalize resampling the same tree hypothesis |
| `signature` | `topology_id + "_" + log_score` (3 dp) | Finer duplicates (same shape + similar likelihood) |

**Quick paired sanity runs** (same seed, batch size, steps — only outcome level differs):

```bash
python -m grpo_experiments.ips_grpo.train --list-presets

python -m grpo_experiments.ips_grpo.train --preset topology_sanity
python -m grpo_experiments.ips_grpo.train --preset signature_sanity
```

**Longer DS1 runs** (G=512, 500 epochs):

```bash
python -m grpo_experiments.ips_grpo.train --preset topology_ds1
python -m grpo_experiments.ips_grpo.train --preset signature_ds1
```

Presets: `grpo_experiments/configs/ips_grpo_presets.json` (loadable) and
`grpo_experiments/configs/ips_grpo_outcomes.yaml` (reference).
CLI flags override presets, e.g. `--preset topology_sanity --seed 1`.

Each run logs `outcome_level` in `metrics.jsonl` and `experiment_config.json`.
Compare `batch_duplicate_fraction` (IPS outcome level) vs
`batch_duplicate_topology_fraction` (always topology) to see how outcome
granularity affects IPS scaling.
