# grpo_experiments

Compare **PhyloGFN**, **GRPO**, and **IPS-GRPO** on the same tree-building policy.

Three **entry points** (same rollout / model, different training objective):

| Method | Command |
|--------|---------|
| PhyloGFN (TB) | `python -m grpo_experiments.train --method phylgfn ...` |
| GRPO (on-policy or `--enable-policy-is`) | `python -m grpo_experiments.train --method grpo ...` |
| IPS-GRPO | `python -m grpo_experiments.ips_grpo.train ...` |

- **GRPO vs IPS-GRPO flows** (advantages, buffering, policy IS) → [`FLOWS.md`](FLOWS.md)
- IPS-GRPO (outcome-frequency scaling) → [`ips_grpo/README.md`](ips_grpo/README.md)
- Policy importance sampling (`w = π_new/π_old`) → `--enable-policy-is` on GRPO (formerly `is_grpo/`)

## Quick start (repo root)

```bash
# GRPO
python -m grpo_experiments.train --method grpo --on-policy-batch-size 64 --epochs 5 --steps-per-epoch 10

# PhyloGFN TB baseline
python -m grpo_experiments.train --method phylgfn --on-policy-batch-size 64 --epochs 5 --steps-per-epoch 10

# IPS-GRPO (separate package)
python -m grpo_experiments.ips_grpo.train --on-policy-batch-size 64 --epochs 5 --steps-per-epoch 10

# GRPO + policy importance sampling (fixed buffer, w = pi_new/pi_old)
python -m grpo_experiments.train --method grpo --enable-policy-is \
  --resample-rounds 5 --update-cycles 10 --buffer-size 1000
```

Runs → `grpo_experiments/runs/`. Full flags → `config.py`.

---

## Training flow (one step)

```
┌─────────────────────────────────────────────────────────────┐
│  runner.py: data_loader.generate_batch()                    │
│    → rollout_worker_phylo.rollout(generator, B, ...)        │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  ROLLOUT (rollout_worker_phylo.py)                          │
│  while forest has > 1 subtree (~T merge steps):             │
│    generator(input_dict)  ←── FORWARD (policy)              │
│      → sample tree pair + edge actions                      │
│      → log_paths_pf step vector  shape (B,)                 │
│         one log-prob per tree, for THIS merge step only     │
│    env.batch_apply_actions ←── score/reward (no policy grad)│
│  stack T step vectors → batch["log_paths_pf"]  shape (B, T) │
│    row i = tree i,  column t = merge step t                 │
│  last step only → batch["log_rewards"] shape (B,)           │
│              → batch["log_scores"]   shape (B,)             │
└──────────────────────────┬──────────────────────────────────┘
                           │
              method == grpo │
                           ▼
┌───────────────────────────────────────────────────────────────┐
│  GRPOTrainer.update (core/trainer.py)                         │
│    advantages = group-normalized rewards (detached)           │
│    pg_loss = TRL PPO surrogate on token log-probs (grpo_loss) │
│    + optional entropy regularization                          │
│    pg_loss.backward()              ←── BACKWARD               │
│    clip_grad_norm_ + Adam step on policy params only          │
└───────────────────────────────────────────────────────────────┘
```

PhyloGFN branch: same rollout batch → `generator.accumulate_loss()` (TB MSE) → `update_model()`. See `src/gfn/tb_gfn_phylo.py`.

---

## What we store after rollout

| Key | Shape | What it is |
|-----|-------|------------|
| `log_paths_pf` | `(B, T)` | **T log-probs per tree.** Entry `[i, t]` = log P_F of the action tree `i` took at merge step `t`. Each step’s tree + edge log-probs are summed in `TBGFlowNetGenerator.forward`. |
| `log_paths_pb` | `(B, T)` | Uniform backward log-prob per step (used by PhyloGFN only; GRPO ignores). |
| `log_rewards` | `(B,)` | Terminal reward only: `(C + log_score) / SCALE` after the full tree is built. |
| `log_scores` | `(B,)` | Terminal log-likelihood (no reward shaping); for logging / diversity. |

- **B** = batch size (`on_policy_batch_size` + replay if enabled).
- **T** ≈ number of taxa − 1 merge steps (e.g. ~26 for typical datasets).

Reward is **not** applied at each merge in the loss—only at the end. Intermediate merges still get a forward log-prob each step.

---

## Forward pass (where it runs)

Not inside the trainer. It runs **during rollout**, once per merge:

1. `env.prepare_rollout_inputs(tree_features, ...)` — current forest state.
2. `generator(input_dict)` → `tree_model` (pick pair) + `edges_model` (branch lengths).
3. `log_paths_pf` = log-prob of the **sampled** actions at that step.
4. Env applies actions, updates `tree_features` for the next step.

The update step reuses the tensor `log_paths_pf` from rollout (still connected to policy weights via autograd).

---

## GRPO loss (`core/loss.py` + `core/trainer.py`)

Group-relative advantages over the **batch** (group size G = B):

```
A_i = (reward_i - mean(reward)) / (std(reward) + eps)   # detached
```

Token-level PPO clipped surrogate (TRL-style), per sequence then batch mean:

```
r_t = exp(log π_new(a_t|s_t) - log π_old(a_t|s_t))     # in loss, not detached
pg_loss = - mean_seq( mean_t( min(r_t A, clip(r_t) A) ) )
```

Optional entropy bonus: `loss = pg_loss - entropy_coef * mean_entropy`.

- **One advantage per tree**, broadcast to every step `t`.
- **Per-sequence masked mean** over steps, then mean over batch (not flat mean over `(B, T)`).
- Step log-probs use sampling temperature from the env config (`core/log_probs.py`).
- On-policy μ reuse: `--grpo-num-iterations` + `core/on_policy_buffer.py`.
- Hybrid / policy-IS inner loop: `--update-cycles` (not `num_iterations`).
- `--grpo-clip-eps` / `--grpo-clip-eps-high` control asymmetric PPO clipping.

---

## Backward pass (what gets updated)

```
pg_loss.backward()
  → gradients on log_paths_pf[i, t]  for each tree i, step t
  → each entry backprops through that step's generator() forward
  → gradients accumulate on shared policy weights (tree + edge nets)
clip_grad_norm_(policy params)
optimizer.step()   # Adam in GRPOTrainer — does not train log Z
```

**Intuition**

- Tree **beat the batch** (A_i > 0) → increase log-prob of **every** action it took on the path.
- Tree **worse than batch** (A_i < 0) → decrease those action log-probs.
- Near-average tree (A_i ≈ 0) → little change.

**Per-step check:** `log_paths_pf` is `(B, T)` and loss is `mean` over both dims, so `log_paths_pf.grad` after `backward()` is shape `(B, T)`. Row `i` is constant across `t` (same A_i); different steps use different states from rollout.

**Not updated by GRPO:** partition `Z`, backward policy, reward / alignment computation.

---

## Key files

| File | Role |
|------|------|
| `train.py` | CLI entry (PhyloGFN / GRPO) |
| `runner.py` | On-policy + policy-IS loops |
| `core/trainer.py` | `GRPOTrainer`: advantages, PPO loss, entropy, Adam |
| `core/loss.py` | TRL PPO clipped surrogate |
| `core/advantages.py` | Reward transform + group normalization |
| `core/on_policy_buffer.py` | TRL μ buffering (on-policy only) |
| `core/policy_replay.py` | Fixed-trajectory replay for policy IS |
| `hybrid_grpo/` | Best-tree replay + policy IS (production eval) |
| `hybrid_ips_grpo/` | Hybrid + IPS outcome scaling |
| `ips_grpo/` | Standalone IPS-GRPO entry |
| `config.py` | Main experiment flags |

---

## PhyloGFN vs GRPO (one line each)

| | PhyloGFN | GRPO |
|---|----------|------|
| Loss | MSE( log Z + Σ log P_F − log R − Σ log P_B ) | −mean_seq( PPO_clip( r_t A ) ) + entropy |
| Uses terminal reward | Yes (balance target) | Yes (advantages only) |
| Batch comparison | No | Yes (group-relative A) |
| Trains Z | Yes | No |
