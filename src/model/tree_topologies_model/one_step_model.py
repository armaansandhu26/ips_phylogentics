import torch
import torch.nn as nn
from src.model.mlp import MLP
import torch.nn.functional as F
from src.model.weight_init import trunc_normal_
from src.model.transformer import TransformerEncoder
from torch.distributions import Categorical


class PhyloTreeModelOneStep(nn.Module):

    def __init__(self, gfn_cfg):
        super().__init__()

        transformer_cfg = gfn_cfg.MODEL.TRANSFORMER
        self.compute_state_flow = (gfn_cfg.LOSS_TYPE != 'TB')
        self.concatenate_summary_token = transformer_cfg.PART1_HEAD.CONCATENATE_SUMMARY_TOKEN
        self.concatenate_candidate_tree = transformer_cfg.PART2_HEAD.CONCATENATE_CANDIDATE_TREE
        self.condition_on_scale = gfn_cfg.CONDITION_ON_SCALE
        self.encoder = TransformerEncoder(transformer_cfg)
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)

        self.seq_emb = MLP(transformer_cfg.SEQ_EMB)
        embedding_size = transformer_cfg.SEQ_EMB.OUTPUT_SIZE
        if self.condition_on_scale:
            scales_set = gfn_cfg.SCALES_SET
            self.scale_embeddings = torch.nn.Embedding(len(scales_set), embedding_size)
        else:
            self.summary_token = nn.Parameter(torch.zeros(1, 1, embedding_size), requires_grad=True)
            trunc_normal_(self.summary_token, std=0.1)

        self.logits_head = MLP(transformer_cfg.LOGITS_HEAD)
        if self.compute_state_flow:
            self.flow_head = MLP(transformer_cfg.FLOW_HEAD)

    def get_head_token(self, scale_key):
        if self.condition_on_scale:
            token = self.scale_embeddings(scale_key)
        else:
            token = self.summary_token
        return token

    def model_params(self):
        return list(self.parameters())

    def sample(self, ret, random_spec):

        logits = ret['logits']
        if random_spec is None:
            random_spec = {
                'random_action_prob': 0.0
            }
        if 'random_action_prob' in random_spec:
            random_p = random_spec['random_action_prob']
            distribution = Categorical(logits=logits)
            tree_actions = distribution.sample()
            if random_p > 0:
                batch_size = tree_actions.shape[0]
                B, N = logits.shape
                rand_flag = torch.rand(batch_size, device=tree_actions.device) <= random_p
                rand_actions = torch.randint(low=0, high=N, size=(batch_size,), device=tree_actions.device)
                tree_actions = torch.where(rand_flag, rand_actions, tree_actions)
        else:
            T = random_spec['T']
            distribution = Categorical(logits=logits / T)
            tree_actions = distribution.sample()
        return tree_actions

    def compute_log_path_pf(self, ret, tree_actions):

        logits = ret['logits']
        batch_size = logits.shape[0]
        log_p = self.logsoftmax(logits)
        log_paths_pf = log_p[torch.arange(batch_size, device=logits.device), tree_actions]
        return log_paths_pf

    def forward(self, **kwargs):
        """
        :param batch_input: input tensors of shape [batch_size, nb_seq, seq_len], each sample in the batch is a state
        :param batch_nb_seq: list of actual sequence length for each sample the batch
        """
        batch_input = kwargs['batch_input']
        batch_nb_seq = kwargs['batch_nb_seq']
        random_spec = kwargs.get('random_spec', None)
        input_tree_actions = kwargs.get('input_tree_actions', None)

        scale_key = kwargs.get('scale_key')
        return_tree_reps = kwargs.get('return_tree_reps', False)

        batch_size, max_nb_seq, _ = batch_input.shape

        # batch_size, max_nb_seq, emb_size
        x = self.seq_emb(batch_input)    # --> embed each subtree [B, N, m*c] -> [B, N, E]

        # add summary token  #[B, N, E]--> [B, N+1, E] (placeholder for a learnable summary token which tells all info about current subtrees)
        # after learning -> represent all N subtrees in that same episode
        B = x.shape[0]
        summary_token = self.get_head_token(scale_key)
        if self.condition_on_scale:
            traj_length = int(B / kwargs['batch_size'])
            summary_token = summary_token.unsqueeze(1).expand(-1, traj_length, -1).reshape(batch_size, 1, -1)
        else:
            summary_token = summary_token.expand(B, -1, -1)
        x = torch.cat((summary_token, x), dim=1)

        # padding mask #[B, N+1, E] + batch padding -> self attn -> [B, N+1, E] contextual info now
        # we start with fixed trees N(max possible = #leaves in starting) and then later on this might reduce -> so we use padding = True to ignore the subtrees that dont exist
        batch_padding_mask = torch.ones((batch_size, max_nb_seq)).to(x).cumsum(dim=1) > batch_nb_seq[:, None]
        batch_padding_mask = batch_padding_mask.bool()
        batch_padding_mask = F.pad(batch_padding_mask, (1, 0), "constant", False)

        x = self.encoder(x, batch_padding_mask)
        summary_token = x[:, :1]
        trees_reps = x[:, 1:]

        # add all pairs of embeddings  -> gives P merge candidates = N*(N-1)/2
        #  x[i, j]  + x[i, k] = C[i, j, k]
        #  B x N x E  =>     B x N x N x E
        tmp = (trees_reps[:, :, None, :] + trees_reps[:, None, :, :]) # [B, N, N, E]
        # get all distinct pairs
        row, col = torch.triu_indices(
            max_nb_seq,
            max_nb_seq,
            offset=1,
            device=trees_reps.device,
        ) #upper traingle
        x_pairs = tmp[:, row, col]  # [B, P, E]

        #[B, P, E] -> [B, P, 2E]
        if self.concatenate_summary_token:
            _, num_trees, _ = x_pairs.shape
            s = summary_token.expand(-1, num_trees, -1) #copy same summary for each tree
            x_pairs = torch.cat([x_pairs, s], dim=2) #For each possible merge pair, append global forest context to that pair’s local representation before scoring.

        #[B, P, 2E] -> #[B, P] (one output score per candidate)
        logits = self.logits_head(x_pairs).squeeze(-1)

        if self.compute_state_flow:
            log_state_flow = self.flow_head(summary_token).reshape(-1)
            ret = {
                'logits': logits,
                'log_flow': log_state_flow,
                'mask': batch_padding_mask
            }
        else:
            ret = {
                'logits': logits,
                'mask': batch_padding_mask
            }

        if return_tree_reps:
            ret['summary_reps'] = summary_token[:, 0] #[B,E]
            ret['trees_reps'] = trees_reps #[B,N,E]

        tree_actions = self.sample(ret, random_spec) #[B] - samples one merge index per batch item from ret['logits'].
        if input_tree_actions is not None:
            tree_actions[:len(input_tree_actions)] = input_tree_actions

        log_paths_pf = self.compute_log_path_pf(ret, tree_actions)#[B] - computes log-prob of chosen action under current policy
        ret['tree_actions'] = tree_actions
        ret['log_paths_pf'] = log_paths_pf

        return ret
