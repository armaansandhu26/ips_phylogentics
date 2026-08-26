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
            actions = [0, 1, 0, 1]

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
                self.assertIsNotNone(objective.backward_policy.logit.grad)


if __name__ == "__main__":
    unittest.main()
