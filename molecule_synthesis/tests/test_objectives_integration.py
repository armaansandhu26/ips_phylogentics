"""Tiny adapter gradient tests, with API stubs when full RGFN is unavailable."""

from __future__ import annotations

import importlib.util
import os
import sys
import types
import unittest
from dataclasses import dataclass


HAS_TORCH = importlib.util.find_spec("torch") is not None
os.environ.setdefault("DGLBACKEND", "pytorch")
os.environ.setdefault("RGFN_MINIMAL_PROXIES", "1")


def _install_api_stubs() -> None:
    """Provide only the RGFN interfaces needed to import our objective adapter."""
    if importlib.util.find_spec("gin") is None:
        gin = types.ModuleType("gin")

        def configurable(*args, **kwargs):
            if args and callable(args[0]):
                return args[0]
            return lambda obj: obj

        gin.configurable = configurable
        sys.modules["gin"] = gin

    if importlib.util.find_spec("rgfn") is not None:
        return
    import torch

    rgfn = types.ModuleType("rgfn")
    api = types.ModuleType("rgfn.api")
    objective_base = types.ModuleType("rgfn.api.objective_base")
    policy_base = types.ModuleType("rgfn.api.policy_base")
    trajectories = types.ModuleType("rgfn.api.trajectories")

    class ObjectiveBase(torch.nn.Module):
        def __init__(self, forward_policy, backward_policy):
            super().__init__()
            self.forward_policy = forward_policy
            self.backward_policy = backward_policy

    @dataclass
    class ObjectiveOutput:
        loss: object
        metrics: dict

    class PolicyBase:
        pass

    class Trajectories:
        pass

    objective_base.ObjectiveBase = ObjectiveBase
    objective_base.ObjectiveOutput = ObjectiveOutput
    policy_base.PolicyBase = PolicyBase
    trajectories.Trajectories = Trajectories
    sys.modules.update(
        {
            "rgfn": rgfn,
            "rgfn.api": api,
            "rgfn.api.objective_base": objective_base,
            "rgfn.api.policy_base": policy_base,
            "rgfn.api.trajectories": trajectories,
        }
    )


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class ObjectiveGradientSmokeTest(unittest.TestCase):
    def test_all_policy_gradient_objectives_backpropagate(self):
        import torch

        _install_api_stubs()
        from molecule_synthesis.objectives import (
            CountIPSGRPOObjective,
            GRPOObjective,
            MIPSGRPOObjective,
        )

        class Policy(torch.nn.Module):
            def __init__(self, backward: bool = False):
                super().__init__()
                self.logit = torch.nn.Parameter(torch.tensor(-0.2 if backward else 0.2))

            def compute_action_log_probs(self, states, action_spaces, actions):
                logits = torch.stack((self.logit, -self.logit))
                log_probs = torch.log_softmax(logits, dim=0)
                return torch.stack([log_probs[int(action)] for action in actions])

        class RewardOutputs:
            log_reward = torch.tensor([-2.0, -1.0, 0.0, 1.0])

        class FakeTrajectories:
            actions = [0, 0, 0, 1]

            def __len__(self):
                return 4

            def get_actions_flat(self):
                return self.actions

            def get_non_last_states_flat(self):
                return ["s0", "s1", "s2", "s3"]

            def get_non_source_states_flat(self):
                return ["x0", "x1", "x2", "x3"]

            def get_forward_action_spaces_flat(self):
                return [None] * 4

            def get_backward_action_spaces_flat(self):
                return [None] * 4

            def get_index_flat(self):
                return torch.arange(4)

            def get_reward_outputs(self):
                return RewardOutputs()

            def get_last_states_flat(self):
                return ["Mol-A", "Mol-A", "Mol-B", "Mol-C"]

        trajectories = FakeTrajectories()
        objective_factories = (
            lambda: GRPOObjective(Policy(), Policy(backward=True)),
            lambda: CountIPSGRPOObjective(Policy(), Policy(backward=True)),
            lambda: MIPSGRPOObjective(Policy(), Policy(backward=True)),
        )
        for factory in objective_factories:
            objective = factory()
            output = objective.compute_objective_output(trajectories)
            self.assertTrue(torch.isfinite(output.loss))
            output.loss.backward()
            self.assertIsNotNone(objective.forward_policy.logit.grad)
            if isinstance(objective, MIPSGRPOObjective):
                self.assertIsNone(objective.backward_policy.logit.grad)
                objective.compute_reverse_loss().backward()
                self.assertIsNotNone(objective.backward_policy.logit.grad)

    def test_mips_running_normalizer_is_checkpointed(self):
        import torch

        _install_api_stubs()
        from molecule_synthesis.objectives import MIPSGRPOObjective

        class Policy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.logit = torch.nn.Parameter(torch.tensor(0.2))

            def compute_action_log_probs(self, states, action_spaces, actions):
                log_probs = torch.log_softmax(torch.stack((self.logit, -self.logit)), dim=0)
                return torch.stack([log_probs[int(action)] for action in actions])

        class RewardOutputs:
            log_reward = torch.tensor([-2.0, -1.0, 0.0, 1.0])

        class FakeTrajectories:
            def __len__(self):
                return 4

            def get_actions_flat(self):
                return [0, 0, 0, 1]

            def get_non_last_states_flat(self):
                return [None] * 4

            get_non_source_states_flat = get_non_last_states_flat

            def get_forward_action_spaces_flat(self):
                return [None] * 4

            get_backward_action_spaces_flat = get_forward_action_spaces_flat

            def get_index_flat(self):
                return torch.arange(4)

            def get_reward_outputs(self):
                return RewardOutputs()

        objective = MIPSGRPOObjective(Policy(), Policy())
        objective.compute_objective_output(FakeTrajectories())
        objective.compute_objective_output(FakeTrajectories())
        self.assertEqual(int(objective.running_scale_updates.item()), 2)
        state = objective.state_dict()
        self.assertIn("running_log_first_moment", state)
        self.assertIn("running_log_second_moment", state)

        updates_before_validation = int(objective.running_scale_updates.item())
        first_moment_before_validation = objective.running_log_first_moment.clone()
        with torch.no_grad():
            objective.compute_objective_output(FakeTrajectories())
        self.assertEqual(
            int(objective.running_scale_updates.item()), updates_before_validation
        )
        self.assertTrue(
            torch.equal(objective.running_log_first_moment, first_moment_before_validation)
        )

    def test_mips_weight_is_reward_times_reverse_over_forward(self):
        import math
        import torch

        _install_api_stubs()
        from molecule_synthesis.objectives import MIPSGRPOObjective

        class Policy(torch.nn.Module):
            def __init__(self, logit):
                super().__init__()
                self.logit = torch.nn.Parameter(torch.tensor(logit))

            def compute_action_log_probs(self, states, action_spaces, actions):
                log_probs = torch.log_softmax(torch.stack((self.logit, -self.logit)), dim=0)
                return torch.stack([log_probs[int(action)] for action in actions])

        class RewardOutputs:
            log_reward = torch.tensor([-1.0, 0.5])

        class FakeTrajectories:
            def __len__(self):
                return 2

            def get_actions_flat(self):
                return [0, 1, 0]

            def get_non_last_states_flat(self):
                return [None] * 3

            get_non_source_states_flat = get_non_last_states_flat

            def get_forward_action_spaces_flat(self):
                return [None] * 3

            get_backward_action_spaces_flat = get_forward_action_spaces_flat

            def get_index_flat(self):
                return torch.tensor([0, 0, 1])

            def get_reward_outputs(self):
                return RewardOutputs()

        forward = Policy(0.4)
        reverse = Policy(-0.3)
        trajectories = FakeTrajectories()
        output = MIPSGRPOObjective(forward, reverse).compute_objective_output(trajectories)

        forward_edge = forward.compute_action_log_probs(None, None, [0, 1, 0])
        reverse_edge = reverse.compute_action_log_probs(None, None, [0, 1, 0])
        index = trajectories.get_index_flat()
        log_pf = torch.zeros(2).scatter_add_(0, index, forward_edge.detach())
        log_q = torch.zeros(2).scatter_add_(0, index, reverse_edge.detach())
        expected = RewardOutputs.log_reward + log_q - log_pf
        self.assertTrue(
            math.isclose(
                output.metrics["log_importance_weight_mean"],
                float(expected.mean().item()),
                rel_tol=1e-6,
            )
        )

    def test_reverse_mle_averages_trajectory_log_probabilities(self):
        import math
        import torch

        _install_api_stubs()
        from molecule_synthesis.objectives import MIPSGRPOObjective

        class Policy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.logit = torch.nn.Parameter(torch.tensor(0.0))

            def compute_action_log_probs(self, states, action_spaces, actions):
                log_probs = torch.log_softmax(torch.stack((self.logit, -self.logit)), dim=0)
                return torch.stack([log_probs[int(action)] for action in actions])

        class FakeTrajectories:
            def __len__(self):
                return 2

            def get_actions_flat(self):
                return [0, 1, 0]

            def get_non_source_states_flat(self):
                return [None] * 3

            def get_backward_action_spaces_flat(self):
                return [None] * 3

            def get_index_flat(self):
                return torch.tensor([0, 0, 1])

        objective = MIPSGRPOObjective(Policy(), Policy())
        reverse_loss = objective.compute_reverse_loss(FakeTrajectories())
        # Route NLLs are 2*log(2) and log(2), so their mean is 1.5*log(2).
        self.assertTrue(
            math.isclose(float(reverse_loss.item()), 1.5 * math.log(2.0), rel_tol=1e-6)
        )

    def test_mips_uses_exact_exploration_mixture_probability(self):
        import math
        import torch

        _install_api_stubs()
        from molecule_synthesis.objectives import MIPSGRPOObjective

        class Policy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.logit = torch.nn.Parameter(torch.tensor(0.7))

            def compute_action_log_probs(self, states, action_spaces, actions):
                log_probs = torch.log_softmax(torch.stack((self.logit, -self.logit)), dim=0)
                return torch.stack([log_probs[int(action)] for action in actions])

        class ActionSpace:
            def __len__(self):
                return 4

        class RewardOutputs:
            log_reward = torch.tensor([0.0, 0.0])

        class FakeTrajectories:
            def __len__(self):
                return 2

            def get_actions_flat(self):
                return [0, 1]

            def get_non_last_states_flat(self):
                return [None, None]

            get_non_source_states_flat = get_non_last_states_flat

            def get_forward_action_spaces_flat(self):
                return [ActionSpace(), ActionSpace()]

            get_backward_action_spaces_flat = get_forward_action_spaces_flat

            def get_index_flat(self):
                return torch.arange(2)

            def get_reward_outputs(self):
                return RewardOutputs()

        objective = MIPSGRPOObjective(
            Policy(), Policy(), exploration_rate=0.05, advantage_normalization="batch"
        )
        output = objective.compute_objective_output(FakeTrajectories())
        forward_log_probs = torch.log_softmax(torch.tensor([0.7, -0.7]), dim=0)
        expected_behavior = torch.log(
            0.95 * forward_log_probs.exp() + torch.full((2,), 0.05 / 4.0)
        ).mean()
        self.assertTrue(
            math.isclose(
                output.metrics["behavior_log_probability_mean"],
                float(expected_behavior.item()),
                rel_tol=1e-6,
            )
        )
        self.assertEqual(output.metrics["exploration_rate"], 0.05)


if __name__ == "__main__":
    unittest.main()
