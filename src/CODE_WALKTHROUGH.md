# PhyloGFN Code Walkthrough

This guide helps you understand the forward sampling path and training objectives in this repository.

## Recommended Order

Follow this order once, end-to-end. It matches runtime flow.

1. `src/gfn/rollout_worker_phylo.py`
2. `src/env/binary_tree_env_one_step_likelihood.py`
3. `src/gfn/tb_gfn_phylo.py`
4. `src/model/tree_topologies_model/one_step_model.py`
5. `src/model/edges_model/categorical/categorical.py`
6. `grpo_experiments/runner.py`
7. `grpo_experiments/ips_grpo/runner.py`
8. `grpo_experiments/core/trainer.py`
9. `grpo_experiments/ips_grpo/trainer_log_ips.py`
10. `grpo_experiments/core/loss.py` and `grpo_experiments/core/loss_log_ips.py`

---

## 1) Sampling Loop Skeleton

### File
`src/gfn/rollout_worker_phylo.py`

### Methods to read
- `rollout(...)` --> [trajectory generator]
--> how we store an action:
`{
  "tree_action": 5,
  "edge_action": [0.12, 0.34]
}`

### What this helps you understand
- The outer loop over merge steps.
- Where `input_dict` is built and passed to the generator.
- How sampled actions are applied to the environment.
- What tensors are collected for training (`log_paths_pf`, `log_paths_pb`, `log_scores`, `log_rewards`).

---

## 2) Exact Model Inputs

### File
`src/env/binary_tree_env_one_step_likelihood.py`

### Methods to read
- `prepare_rollout_inputs(...)`
- `batch_apply_actions_tensors(...)`
- `batch_apply_actions(...)`
- `batch_actions_to_trees(...)`

### What this helps you understand
- Exact contents and shapes of `input_dict`.
- How current state features are flattened into `batch_input`.
- How chosen `tree_action` and `edge_action` change state.
- How terminal scores/rewards are produced and attached to trajectories.

---

## 3) Generator Wiring (Tree then Edge)

### File
`src/gfn/tb_gfn_phylo.py`

### Methods to read
- `TBGFlowNetGenerator.forward(...)`
- `get_loss_from_rollout_outputs(...)`
- `accumulate_loss(...)`
- `update_model(...)`

### What this helps you understand
- How tree head runs first.
- How selected pair indexes left/right subtree reps for edge head.
- How tree and edge forward log-probs are summed into `ret['log_paths_pf']`.
- Where TB-style loss reads rollout outputs.

---

## 4) Tree Head Internals

### File
`src/model/tree_topologies_model/one_step_model.py`

### Methods to read
- `forward(...)`
- `sample(...)`
- `compute_log_path_pf(...)`
- `get_head_token(...)`

### What this helps you understand
- How summary token + transformer produce `summary_reps` and `trees_reps`.
- How all pair candidates are constructed and scored (`logits`).
- How merge action is sampled and converted to log-prob.

### Forward pass (shape-first checklist)
Assume:
- `B` = batch size (number of episodes/states processed in parallel)
- `N` = current number of subtrees in each episode
- `P = N*(N-1)/2` = number of possible merge pairs
- `E` = embedding size (`SEQ_EMB.OUTPUT_SIZE`)

1. **Input tensors**
   - `batch_input`: `[B, N, m*c]` (flattened tree features from env)
   - `batch_nb_seq`: `[B]` (how many tree slots are valid per episode)

2. **Per-tree embedding**
   - `x = self.seq_emb(batch_input)`
   - Shape: `[B, N, m*c] -> [B, N, E]`

3. **Prepend summary token**
   - `summary_token` is expanded to `[B, 1, E]`
   - `x = torch.cat((summary_token, x), dim=1)`
   - Shape: `[B, N, E] -> [B, N+1, E]`
   - Note: one summary token per episode (not one for the whole batch)

4. **Build padding mask for attention**
   - Start with tree-only mask `[B, N]` where `True` means padded/invalid
   - Left-pad one `False` for summary token
   - Final `batch_padding_mask`: `[B, N+1]`

5. **Transformer encoding**
   - `x = self.encoder(x, batch_padding_mask)`
   - Shape stays `[B, N+1, E]`, but representations become contextual
   - Split:
     - `summary_token = x[:, :1]` -> `[B, 1, E]`
     - `trees_reps = x[:, 1:]` -> `[B, N, E]`

6. **Construct all candidate merge-pair reps**
   - Pairwise sums with broadcasting:
     - `tmp = trees_reps[:, :, None, :] + trees_reps[:, None, :, :]` -> `[B, N, N, E]`
   - Keep unique unordered pairs (`i < j`) via upper triangle indices
   - `x_pairs = tmp[:, row, col]` -> `[B, P, E]`

7. **Optionally append global context to each pair**
   - If `CONCATENATE_SUMMARY_TOKEN=True`:
     - expand summary to `[B, P, E]`
     - concatenate with pair reps on feature dim
     - Shape: `[B, P, E] -> [B, P, 2E]`

8. **Score every candidate pair**
   - `logits = self.logits_head(x_pairs).squeeze(-1)`
   - Shape: `[B, P, (E or 2E)] -> [B, P]`
   - Meaning: one unnormalized score per possible merge pair

9. **Optional state-flow head (non-TB losses)**
   - If `LOSS_TYPE != 'TB'`:
   - `log_flow = self.flow_head(summary_token).reshape(-1)` -> `[B]`

10. **Optional representation return**
    - If `return_tree_reps`:
      - `summary_reps = summary_token[:, 0]` -> `[B, E]`
      - `trees_reps` -> `[B, N, E]`

11. **Sample merge action**
    - `tree_actions = sample(logits, random_spec)` -> `[B]`
    - Default: categorical sampling from logits (stochastic, logits-biased)
    - Optional exploration: with `random_action_prob`, some actions are replaced by uniform random picks
    - Optional replay: `input_tree_actions` can override sampled actions

12. **Compute forward log-prob of chosen action**
    - `log_paths_pf = log_softmax(logits)[range(B), tree_actions]`
    - Shape: `[B]`

13. **Return dict**
    - Always: `logits`, `tree_actions`, `log_paths_pf`, `mask`
    - Optional: `summary_reps`, `trees_reps`, `log_flow`

---

## 5) Edge Head Internals

### File
`src/model/edges_model/categorical/categorical.py`

### Methods to read
- `forward(...)`
- `sample(...)`
- `compute_log_path_pf(...)`

### What this helps you understand
- For `N > 2`, how independent left/right logits (`l_logits`, `r_logits`) are produced.
- Root vs non-root edge prediction path.
- How edge action log-prob is computed and added to total step log-prob.

---

## 6) Experiment Entry Points

### Files
- `grpo_experiments/runner.py`
- `grpo_experiments/ips_grpo/runner.py`

### Methods to read
- `_run_grpo_on_policy(...)` in `runner.py`
- `_run_on_policy(...)` in `ips_grpo/runner.py`
- `_run_grpo_policy_is(...)` and `_run_policy_is(...)` if you use replay/policy-IS

### What this helps you understand
- How rollouts, outcomes, and trainer updates are orchestrated at experiment level.
- Where PhyloGFN vs IPS/Log-IPS training paths diverge.

---

## 7) Objective and Update Logic

### Files
- `grpo_experiments/core/trainer.py`
- `grpo_experiments/ips_grpo/trainer_log_ips.py`
- `grpo_experiments/core/loss.py`
- `grpo_experiments/core/loss_log_ips.py`

### Methods to read
- `GRPOTrainer.update(...)`
- `IPSLogLossTrainer.update(...)`
- `compute_grpo_policy_loss(...)`
- `compute_log_ips_policy_loss(...)`

### What this helps you understand
- Same policy outputs, different training objectives.
- How `log_paths_pf` is turned into policy loss.
- How Log-IPS injects outcome-frequency correction via `p_hat`.

---

## Quick Mental Model

At each merge step:

1. Build state input tensors.
2. Tree head predicts merge-pair logits and samples a pair.
3. Edge head predicts branch-length logits for that chosen pair and samples bins.
4. Environment applies action, updates state.
5. Collect log-probs and terminal scores for training loss.

---

## If You Want to Debug One Batch

Print these values in order:

1. `input_dict['batch_input'].shape`, `input_dict['batch_nb_seq'][0]`
2. `trees_ret['logits'].shape`, `trees_ret['tree_actions'][:5]`
3. `edges_ret` logits shape (`l_logits/r_logits` or root logits)
4. `ret['log_paths_pf'][:5]`
5. rollout `data['log_scores'][:5]`, `data['log_rewards'][:5]`

This gives a full trace from state to sampled action to learning signal.
