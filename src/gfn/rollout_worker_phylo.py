import torch
from src.env.trajectory import Trajectory, SimpleTrajectory
from src.gfn.action_tensors import TensorActionBatch


class RolloutWorker:

    def __init__(self, env):
        self.env = env

    # TODO SCALES
    def rollout(self, generator, episodes, scales=None, random_spec=None, generate_full_trajectories=False,
                input_actions_set=None):

        seq_arrays = self.env.seq_arrays
        n, m, c = seq_arrays.shape # # of species, # of nucleotides, # of bases (4)

        # store all parents number for log paths pbs
        all_num_parents = [] # possible states to reach this state -> need for backwrd probab scaling calculation

        # store all log paths pfs
        all_log_paths_pf = [] #[tensor([-0.7, ..])]
        all_tree_actions = []
        all_edge_actions = []

        # initial tree features
        tree_features = seq_arrays.unsqueeze(0)
        tree_features = tree_features.repeat(episodes, 1, 1, 1)  #[eps, n, m, c] --> eps means how many full trajectories you run in parallel in one rollout call
        # all the above are indexed by eps eg all_actions[i] for ith eps
        # all_log_paths_pf: [episodes, steps], all_num_parents: [episodes, steps]

        # store whether each tree of state has children
        has_children = torch.zeros(episodes, n).bool().to(tree_features.device) #[episodes, n]

        if generate_full_trajectories:
            states = [self.env.get_initial_state() for _ in range(episodes)]
            trajectories = [Trajectory(s) for s in states]
        else:
            states = None
            trajectories = [SimpleTrajectory() for _ in range(episodes)]

        step = 0
        with torch.inference_mode():
            while tree_features.shape[1] > 1:
                num_trees = tree_features.shape[1]

                # prepare input dict
                tensor_input_actions = isinstance(input_actions_set, TensorActionBatch)
                input_actions = None
                if input_actions_set is not None and not tensor_input_actions:
                    input_actions = [x[step] for x in input_actions_set]
                input_dict = self.env.prepare_rollout_inputs(tree_features, input_actions, random_spec)
                if tensor_input_actions:
                    device = tree_features.device
                    input_dict['input_tree_actions'] = input_actions_set.step_tree_actions(step, device)
                    input_edge_actions = input_actions_set.step_edge_actions(step, device)
                    if input_edge_actions is not None:
                        input_dict['input_edge_actions'] = input_edge_actions

                # forward inference
                ret = generator(input_dict)
                trees_ret = ret['trees_ret'] #tree-head output directory
                tree_actions = trees_ret['tree_actions']
                all_tree_actions.append(tree_actions.detach())
                # ret = {
                #     "trees_ret": {...},
                #     "edges_ret": {...},          # present in likelihood mode
                #     "log_paths_pf": tensor([...]) # per-episode total forward log-prob
                #     }

                # trees_ret = {
                #     "logits": ...,                  # pair-merge logits
                #     "mask": ...,
                #     "tree_actions": tensor([...]),  # chosen merge index per episode
                #     "log_paths_pf": tensor([...]),  # log-prob of chosen tree action
                #     "tree_pairs": tensor([...]),    # mapped (left_idx, right_idx) for chosen action
                #     "summary_reps": ...,            # only when requested
                #     "trees_reps": ...               # only when requested
                #     # sometimes "log_flow" if non-TB mode
                #     }

                # edges_ret = {
                #     "edge_actions": tensor([...]),  # chosen branch-length action(s)
                #     "log_paths_pf": tensor([...]),  # log-prob of chosen edge action
                #     ...                             # model-specific extra fields
                #     }

                edge_actions = None
                if not self.env.parsimony_problem:
                    edge_actions = ret['edges_ret']['edge_actions']
                    all_edge_actions.append(edge_actions.detach())

                if generate_full_trajectories:
                    tree_actions_cpu = tree_actions.detach().cpu().tolist()
                    actions = [{'tree_action': x} for x in tree_actions_cpu]
                    if edge_actions is not None:
                        edge_actions_cpu = edge_actions.detach().cpu().tolist()
                        for idx, action in enumerate(actions):
                            action['edge_action'] = edge_actions_cpu[idx]
                    states, new_tree_features, log_scores, log_rewards = self.env.batch_apply_actions(
                        actions,
                        tree_features,
                        states,
                    )
                    for a, s, r, traj in zip(actions, states, log_rewards, trajectories):
                        traj.update(s, a, r.item(), s.is_done)
                else:
                    _, new_tree_features, log_scores, log_rewards = self.env.batch_apply_actions_tensors(
                        tree_actions,
                        edge_actions,
                        tree_features,
                        None,
                        num_trees=num_trees,
                    )

                # collect num of possible parents to calculate pb
                b = tree_features.shape[0]
                pairs = self.env.retrieve_tree_pairs_tensor(num_trees, tree_actions)
                left_trees_indices = pairs[:, 0]
                right_trees_indices = pairs[:, 1]
                new_states_inputs_indices = torch.ones(b, num_trees, device=tree_features.device, dtype=torch.bool)
                new_states_inputs_indices[torch.arange(b, device=tree_features.device), right_trees_indices] = False
                has_children = has_children[new_states_inputs_indices]
                has_children = has_children.reshape(b, -1)
                has_children[torch.arange(b, device=tree_features.device), left_trees_indices] = 1
                num_parents = has_children.sum(-1)
                all_num_parents.append(num_parents)

                # add log paths pf
                all_log_paths_pf.append(ret['log_paths_pf'])
                tree_features = new_tree_features
                step += 1

        action_tensors = TensorActionBatch(
            tree_actions=tuple(all_tree_actions),
            edge_actions=tuple(all_edge_actions) if all_edge_actions else None,
        )
        if not generate_full_trajectories:
            log_rewards_cpu = log_rewards.detach().cpu().tolist()
            for reward, traj in zip(log_rewards_cpu, trajectories):
                traj.log_reward = reward

        all_log_paths_pf = torch.stack(all_log_paths_pf).T
        all_num_parents = torch.stack(all_num_parents).T
        all_num_parents[:, -1] = 2 * seq_arrays.shape[0] - 3
        log_paths_pb = -torch.log(all_num_parents).to(all_log_paths_pf)
        data = {
            'log_paths_pf': all_log_paths_pf,
            'log_paths_pb': log_paths_pb,
            'log_rewards': log_rewards,
            'log_scores': log_scores,
            'random_spec': random_spec,
            'scales': scales,
            'action_tensors': action_tensors,
        }
        return data, trajectories
