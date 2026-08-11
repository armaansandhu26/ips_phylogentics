species, N = 5
dna_seq_length, m = 100
bases/channels, c = 4

batch, B = 2 --> 2 complete trees
traj1, traj2

--> who do we merge?  <tree-model>
input_dict:
batch_input = [B, N, m*c]
active_trees = [B] -> [3,3]


1. embedding - NN: [B, N, m*c] -> [B, N, E]
2. summary token -> learnable parameter of size = [1,1,E]: add a placeholder summary token  (defines the global context per batch) #[B, N, E]--> [B, N+1, E] (adds a dummy row)
3. padding mask #[B, N, E] -> [B, N+1, E]
4. learnable transformer encoder: x = self.encoder(x, batch_padding_mask)
    we get summary_token and tree representations for this with context (attn)
    split them:
    summary_token = x[:, :1] -> [B,1,E]
    trees_reps = x[:, 1:] -> [B,N,E]
    ----------> all repr info collected -------------->[3 learnable networks so far]
5. converts tree repr to find P merge candidates: uses broadcasting addition in trees (tree1 embd+tree2 embd = merged tree embd) to find merged tree representation and triu_indices to get upper triange indices(undordered pairs)
tree_repr([B, N, E]) -> pairs([B, P, E])
6. concatenate summary token to each of the P tree representation: expand summary token to shape [1,P,E] -> add to P trees [B, P, E]: [B, P, E] -> [B, P, 2E]
--------------> potential pairs found ------------------->[3 learnable networks so far]
7. learnable: get logits for all P trees using MLP - logits = self.logits_head(x_pairs).squeeze(-1)
[B,P,E] -> [B,P] (one output score per candidate)
-------------->  logits found for all merge tree ---------->[4 learnable networks so far]
8. samples from logit dist -> [B] -> one merge choice for each batch
9. compute_log_path_pf -> takes log softmax on logits and stores this as forward log probab

------> who do we merge complete


---> connection between models
1. get the left and right tree idx for each batch by the mapping dictionary in env
2. separate left_idxs and right_idxs 

--> how do we merge? <edge-model>
input:
1. summary_tree per patch (from <tree-model>) [B,E]
2. left_idxs (from <tree-model> + some data modification) [B,E]
3. right_idxs (from <tree-model> + some data modification) [B,E]
4. input_dict (same as was input in <tree-model>) 

one per step = lr_model(learnable) OR root_edge_model(learnable)

----> [2 learnable networks]




=== ROLLOUT (repeat until 1 tree left) ===
step t:
  input_dict: batch_input [B, active_trees, m*c]
  │
  ├─ <tree-model> → logits [B,P], sample tree_actions [B]
  │                 log_tree_pf [B] = log_softmax(logits)[tree_actions]
  │
  ├─ <edge-model> → edge logits, sample edge_actions
  │                 log_edge_pf [B] = log_softmax(...)[edge_actions]
  │
  ├─ log_paths_pf_step = log_tree_pf + log_edge_pf   [B]
  │
  └─ env merge → next tree_features

after T steps:
  log_paths_pf  stacked → [B, T]
  log_paths_pb  from parent counts → [B, T]   (no grad)
  log_rewards   terminal only       → [B]     (no grad)

=== UPDATE ===

INPUT (from rollout):
  log_paths_pf  [B, T]   connected to tree+edge nets
  log_paths_pb  [B, T]   constants
  log_rewards   [B]      constants

1. SUM over steps
   log_pf[i] = Σ_t log_paths_pf[i,t]     → [B]
   log_pb[i] = Σ_t log_paths_pb[i,t]     → [B]

2. PARTITION FUNCTION
   log_Z = sum(_Z)                        → scalar, same for all i in batch

3. TB BALANCE
   forward[i]  = log_Z + log_pf[i]
   backward[i] = log_rewards[i] + log_pb[i]
   loss = mean_i (forward[i] - backward[i])²

4. BACKWARD
   loss.backward()
     → _Z.grad              (256-d, all entries get grad)
     → tree_model params    (grad from all B×T tree log-prob steps)
     → edge_model params    (grad from all B×T edge log-prob steps)

5. OPTIMIZER
   clip_grad_norm_(tree + edge only, max=GRAD_CLIP)
   Adam.step()              tree+edge @ LR_MODEL, _Z @ LR_Z
   zero_grad()
