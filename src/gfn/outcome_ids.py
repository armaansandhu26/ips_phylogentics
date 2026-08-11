"""Fast outcome / topology ID extraction without full tree reconstruction."""

from __future__ import annotations

from typing import Literal, Sequence

import torch

from src.gfn.action_tensors import TensorActionBatch

OutcomeLevel = Literal["signature", "topology"]

FAST_OUTCOME_ID_CACHE_TAG = "FAST_OUTCOME_ID_CACHE"


def extract_outcome_ids(trees, level: OutcomeLevel = "topology") -> tuple[list[str], list[str]]:
    """Return (outcome_ids, topology_ids) for a batch of PhylogeneticTree objects."""
    signatures = [t.signature for t in trees]
    topology_ids = [t.tree_topology_id for t in trees]
    if level == "topology":
        return topology_ids, topology_ids
    return signatures, topology_ids


def _tree_action_key(actions: Sequence[dict]) -> tuple[int, ...]:
    return tuple(int(action["tree_action"]) for action in actions)


class OutcomeIdCache:
    """Cache exact topology IDs by tree-action sequence."""

    def __init__(self, env) -> None:
        self.env = env
        self.topology_by_tree_actions: dict[tuple[int, ...], str] = {}
        self.hits = 0
        self.misses = 0

    def topology_id(self, actions: Sequence[dict], log_score: float) -> str:
        key = _tree_action_key(actions)
        cached = self.topology_by_tree_actions.get(key)
        if cached is not None:
            self.hits += 1
            return cached

        tree = self.env.build_tree_from_actions(actions, log_score)
        topology_id = tree.tree_topology_id
        self.topology_by_tree_actions[key] = topology_id
        self.misses += 1
        return topology_id

    def topology_id_from_key(self, key: tuple[int, ...], log_score: float) -> str:
        cached = self.topology_by_tree_actions.get(key)
        if cached is not None:
            self.hits += 1
            return cached

        actions = [{"tree_action": int(x)} for x in key]
        tree = self.env.build_tree_from_actions(actions, log_score)
        topology_id = tree.tree_topology_id
        self.topology_by_tree_actions[key] = topology_id
        self.misses += 1
        return topology_id

    def ids_from_actions(
        self,
        actions_set: Sequence[Sequence[dict]],
        log_scores: torch.Tensor,
        level: OutcomeLevel = "topology",
    ) -> tuple[list[str], list[str]]:
        scores = log_scores.detach().cpu().tolist()
        outcome_ids: list[str] = []
        topology_ids: list[str] = []
        for actions, score in zip(actions_set, scores):
            score = float(score)
            topology_id = self.topology_id(actions, score)
            topology_ids.append(topology_id)
            if level == "topology":
                outcome_ids.append(topology_id)
            else:
                outcome_ids.append(f"{topology_id}_{score:.3f}")
        return outcome_ids, topology_ids

    def ids_from_action_tensors(
        self,
        action_tensors: TensorActionBatch,
        log_scores: torch.Tensor,
        level: OutcomeLevel = "topology",
    ) -> tuple[list[str], list[str]]:
        keys = action_tensors.tree_action_keys()
        scores = log_scores.detach().cpu().tolist()
        outcome_ids: list[str] = []
        topology_ids: list[str] = []
        for i, (key, score) in enumerate(zip(keys, scores)):
            score = float(score)
            cached = self.topology_by_tree_actions.get(key)
            if cached is not None:
                self.hits += 1
                topology_id = cached
            else:
                actions = action_tensors.actions_for_index(i)
                tree = self.env.build_tree_from_actions(actions, score)
                topology_id = tree.tree_topology_id
                self.topology_by_tree_actions[key] = topology_id
                self.misses += 1
            topology_ids.append(topology_id)
            if level == "topology":
                outcome_ids.append(topology_id)
            else:
                outcome_ids.append(f"{topology_id}_{score:.3f}")
        return outcome_ids, topology_ids

    def ids_from_batch(
        self,
        batch: dict,
        *,
        level: OutcomeLevel = "topology",
    ) -> tuple[list[str], list[str]]:
        log_scores = batch["log_scores"]
        action_tensors = batch.get("action_tensors")
        if action_tensors is not None:
            return self.ids_from_action_tensors(action_tensors, log_scores, level)
        actions_set = batch.get("actions_set")
        if actions_set is not None:
            return self.ids_from_actions(actions_set, log_scores, level)
        raise ValueError("batch must contain action_tensors or actions_set")

    def ids_from_trajectories(
        self,
        trajectories,
        log_scores: torch.Tensor,
        level: OutcomeLevel = "topology",
    ) -> tuple[list[str], list[str]]:
        return self.ids_from_actions([traj.actions for traj in trajectories], log_scores, level)

    def assert_matches_slow_path(
        self,
        actions_set: Sequence[Sequence[dict]],
        log_scores: torch.Tensor,
        level: OutcomeLevel = "topology",
    ) -> None:
        trees = self.env.batch_actions_to_trees(actions_set, log_scores)
        slow_outcome_ids, slow_topology_ids = extract_outcome_ids(trees, level)
        fast_outcome_ids, fast_topology_ids = self.ids_from_actions(actions_set, log_scores, level)
        if slow_outcome_ids != fast_outcome_ids or slow_topology_ids != fast_topology_ids:
            raise AssertionError(
                f"{FAST_OUTCOME_ID_CACHE_TAG} parity check failed against ETE reconstruction."
            )

    def ids_from_rollout_batch(
        self,
        batch: dict,
        trajectories,
        *,
        level: OutcomeLevel = "topology",
        disable_fast_cache: bool = False,
        check_parity: bool = False,
    ) -> tuple[list[str], list[str]]:
        """Outcome IDs from a rollout batch (prefers action_tensors over traj.actions)."""
        log_scores = batch["log_scores"]
        action_tensors = batch.get("action_tensors")
        if disable_fast_cache:
            actions_set = (
                action_tensors.to_actions_set()
                if action_tensors is not None
                else [traj.actions for traj in trajectories]
            )
            trees = self.env.batch_actions_to_trees(actions_set, log_scores)
            return extract_outcome_ids(trees, level)
        if action_tensors is not None:
            if check_parity:
                self.assert_matches_slow_path(
                    action_tensors.to_actions_set(),
                    log_scores,
                    level,
                )
            return self.ids_from_action_tensors(action_tensors, log_scores, level)
        actions_set = [traj.actions for traj in trajectories]
        if check_parity:
            self.assert_matches_slow_path(actions_set, log_scores, level)
        return self.ids_from_actions(actions_set, log_scores, level)

    def ids_from_replay_buffer(
        self,
        buffer,
        *,
        level: OutcomeLevel = "topology",
        disable_fast_cache: bool = False,
        check_parity: bool = False,
    ) -> tuple[list[str], list[str]]:
        """Outcome IDs from a policy-IS replay buffer."""
        log_scores = buffer.log_scores
        action_tensors = getattr(buffer, "action_tensors", None)
        if disable_fast_cache:
            actions_set = (
                action_tensors.to_actions_set()
                if action_tensors is not None
                else buffer.actions_set
            )
            trees = self.env.batch_actions_to_trees(actions_set, log_scores)
            return extract_outcome_ids(trees, level)
        if action_tensors is not None:
            if check_parity:
                self.assert_matches_slow_path(
                    action_tensors.to_actions_set(),
                    log_scores,
                    level,
                )
            return self.ids_from_action_tensors(action_tensors, log_scores, level)
        if check_parity:
            self.assert_matches_slow_path(buffer.actions_set, log_scores, level)
        return self.ids_from_actions(buffer.actions_set, log_scores, level)
