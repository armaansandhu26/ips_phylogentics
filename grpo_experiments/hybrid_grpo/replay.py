"""Replay helpers for hybrid GRPO (fresh + best-tree replay)."""

from __future__ import annotations

import random
from dataclasses import dataclass
from heapq import heappush, heappop
from typing import Any

import torch

from grpo_experiments.core.policy_replay import clean_action, sample_replay_buffer
from src.gfn.action_tensors import TensorActionBatch, concat_tensor_action_batches


@dataclass
class HybridReplayBatch:
    """Fixed trajectories used for multi-cycle policy IS updates."""

    actions_set: list[list[dict]]
    trees: list[Any]
    log_paths_pf_old: torch.Tensor
    log_rewards: torch.Tensor
    log_scores: torch.Tensor
    random_spec: dict | None
    source_tags: list[str]  # "fresh" or "replay"
    action_tensors: TensorActionBatch | None = None

    @property
    def size(self) -> int:
        return len(self.actions_set)

    @property
    def log_pf_old(self) -> torch.Tensor:
        return self.log_paths_pf_old.sum(dim=-1)

    @property
    def fresh_count(self) -> int:
        return sum(1 for tag in self.source_tags if tag == "fresh")

    @property
    def replay_count(self) -> int:
        return sum(1 for tag in self.source_tags if tag == "replay")


class BestTreeReplayBuffer:
    """Top-scoring unique replay entries with stored behavior-policy probabilities."""

    @dataclass
    class ReplayEntry:
        key: str
        tree: Any
        actions: list[dict]
        log_path_pf_old: torch.Tensor  # (T,)
        log_reward: float
        log_score: float

    @dataclass
    class AddSamplesStats:
        inserted: int = 0
        found_in_buffer: int = 0
        replaced_existing: int = 0

    def __init__(self, capacity: int, *, topology_only: bool = False):
        if capacity <= 0:
            raise ValueError("BestTreeReplayBuffer capacity must be > 0.")
        self.capacity = int(capacity)
        self.topology_only = bool(topology_only)
        self.entries: dict[str, BestTreeReplayBuffer.ReplayEntry] = {}
        self._score_heap: list[tuple[float, str]] = []

    def __len__(self) -> int:
        return len(self.entries)

    def _key(self, tree: Any) -> str:
        return tree.tree_topology_id if self.topology_only else tree.signature

    def _min_live_entry(self) -> BestTreeReplayBuffer.ReplayEntry | None:
        while self._score_heap:
            score, key = self._score_heap[0]
            entry = self.entries.get(key)
            if entry is None or entry.log_score != score:
                heappop(self._score_heap)
                continue
            return entry
        return None

    def _insert_entry(self, entry: ReplayEntry) -> bool:
        prev = self.entries.get(entry.key)
        if prev is not None:
            if entry.log_score >= prev.log_score:
                self.entries[entry.key] = entry
                heappush(self._score_heap, (entry.log_score, entry.key))
                return True
            return False

        if len(self.entries) < self.capacity:
            self.entries[entry.key] = entry
            heappush(self._score_heap, (entry.log_score, entry.key))
            return True

        worst = self._min_live_entry()
        if worst is None or entry.log_score > worst.log_score:
            if worst is not None:
                del self.entries[worst.key]
                heappop(self._score_heap)
            self.entries[entry.key] = entry
            heappush(self._score_heap, (entry.log_score, entry.key))
            return True
        return False

    def add_samples_with_stats(
        self,
        trees: list[Any],
        actions_set: list[list[dict]],
        log_paths_pf_old: torch.Tensor,
        log_rewards: torch.Tensor,
        log_scores: torch.Tensor,
    ) -> AddSamplesStats:
        """Insert/refresh replay entries with stored trajectory probabilities from rollout."""
        n = len(trees)
        if len(actions_set) != n:
            raise ValueError("trees and actions_set length mismatch.")
        if log_paths_pf_old.shape[0] != n:
            raise ValueError("log_paths_pf_old batch size does not match samples.")
        if log_rewards.shape[0] != n or log_scores.shape[0] != n:
            raise ValueError("reward/score batch size does not match samples.")

        pf_cpu = log_paths_pf_old.detach().to("cpu")
        rew_cpu = log_rewards.detach().to("cpu")
        score_cpu = log_scores.detach().to("cpu")

        stats = BestTreeReplayBuffer.AddSamplesStats()
        for idx, tree in enumerate(trees):
            key = self._key(tree)
            was_present = key in self.entries
            actions = [clean_action(a) for a in actions_set[idx]]
            entry = BestTreeReplayBuffer.ReplayEntry(
                key=key,
                tree=tree,
                actions=actions,
                log_path_pf_old=pf_cpu[idx].clone(),
                log_reward=float(rew_cpu[idx].item()),
                log_score=float(score_cpu[idx].item()),
            )
            inserted = bool(self._insert_entry(entry))
            stats.inserted += int(inserted)
            stats.found_in_buffer += int(was_present)
            stats.replaced_existing += int(was_present and inserted)
        return stats

    def add_samples(
        self,
        trees: list[Any],
        actions_set: list[list[dict]],
        log_paths_pf_old: torch.Tensor,
        log_rewards: torch.Tensor,
        log_scores: torch.Tensor,
    ) -> int:
        """Backward-compatible helper returning only insertion count."""
        return self.add_samples_with_stats(
            trees=trees,
            actions_set=actions_set,
            log_paths_pf_old=log_paths_pf_old,
            log_rewards=log_rewards,
            log_scores=log_scores,
        ).inserted

    def warm_start_from_policy(
        self,
        rollout_worker,
        generator,
        env,
        *,
        num_samples: int,
        chunk_size: int,
        device: str,
    ) -> int:
        """Seed replay with trajectories sampled under current policy and stored pi_old."""
        if num_samples <= 0:
            return 0
        sampled = sample_replay_buffer(
            rollout_worker,
            generator,
            buffer_size=num_samples,
            chunk_size=chunk_size,
            random_spec=None,
            device=device,
        )
        trees = env.batch_actions_to_trees(
            [traj.actions for traj in sampled.trajectories],
            sampled.log_scores,
        )
        return self.add_samples(
            trees,
            sampled.actions_set,
            sampled.log_paths_pf_old,
            sampled.log_rewards,
            sampled.log_scores,
        )

    def sample_entries(self, sample_size: int) -> list[ReplayEntry]:
        if sample_size <= 0 or not self.entries:
            return []
        return random.choices(list(self.entries.values()), k=sample_size)


def sample_hybrid_replay_batch(
    rollout_worker,
    generator,
    env,
    *,
    replay_buffer: BestTreeReplayBuffer,
    fresh_buffer_size: int,
    replay_sample_size: int,
    chunk_size: int,
    random_spec: dict | None,
    device: str,
) -> HybridReplayBatch:
    """
    Sample one fixed hybrid batch:
      - fresh trajectories under current policy (behavior policy pi_old),
      - replay trajectories from stored entries (including stored pi_old probs).
    """
    if fresh_buffer_size < 0 or replay_sample_size < 0:
        raise ValueError("fresh_buffer_size and replay_sample_size must be >= 0.")
    if fresh_buffer_size + replay_sample_size <= 0:
        raise ValueError("At least one of fresh_buffer_size or replay_sample_size must be > 0.")

    actions_set: list[list[dict]] = []
    trees: list[Any] = []
    pf_parts: list[torch.Tensor] = []
    reward_parts: list[torch.Tensor] = []
    score_parts: list[torch.Tensor] = []
    action_tensor_parts: list[TensorActionBatch] = []
    source_tags: list[str] = []

    if fresh_buffer_size > 0:
        fresh = sample_replay_buffer(
            rollout_worker,
            generator,
            buffer_size=fresh_buffer_size,
            chunk_size=chunk_size,
            random_spec=random_spec,
            device=device,
        )
        fresh_trees = env.batch_actions_to_trees(
            [traj.actions for traj in fresh.trajectories],
            fresh.log_scores,
        )
        actions_set.extend(fresh.actions_set)
        trees.extend(fresh_trees)
        pf_parts.append(fresh.log_paths_pf_old)
        reward_parts.append(fresh.log_rewards)
        score_parts.append(fresh.log_scores)
        if fresh.action_tensors is not None:
            action_tensor_parts.append(fresh.action_tensors)
        source_tags.extend(["fresh"] * fresh.size)

    replay_entries = replay_buffer.sample_entries(replay_sample_size)
    if replay_entries:
        replay_actions_set = [entry.actions for entry in replay_entries]
        replay_pf = torch.stack([entry.log_path_pf_old for entry in replay_entries], dim=0).to(device)
        replay_rewards = torch.tensor(
            [entry.log_reward for entry in replay_entries],
            dtype=replay_pf.dtype,
            device=device,
        )
        replay_scores = torch.tensor(
            [entry.log_score for entry in replay_entries],
            dtype=replay_pf.dtype,
            device=device,
        )
        actions_set.extend(replay_actions_set)
        trees.extend([entry.tree for entry in replay_entries])
        pf_parts.append(replay_pf)
        reward_parts.append(replay_rewards)
        score_parts.append(replay_scores)
        action_tensor_parts.append(TensorActionBatch.from_actions_set(replay_actions_set, device=device))
        source_tags.extend(["replay"] * len(replay_entries))

    if not actions_set:
        raise RuntimeError(
            "Hybrid batch is empty. Increase fresh_buffer_size or seed replay buffer."
        )

    return HybridReplayBatch(
        actions_set=actions_set,
        trees=trees,
        log_paths_pf_old=torch.cat(pf_parts, dim=0),
        log_rewards=torch.cat(reward_parts, dim=0),
        log_scores=torch.cat(score_parts, dim=0),
        random_spec=random_spec,
        source_tags=source_tags,
        action_tensors=concat_tensor_action_batches(action_tensor_parts),
    )


def reevaluate_log_paths_pf_hybrid(
    rollout_worker,
    generator,
    batch: HybridReplayBatch,
    *,
    chunk_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward replay under pi_new; returns (log_paths_pf, paths_entropy) with shape (B, T)."""
    from grpo_experiments.core.forward_replay import forward_replay_fixed_actions

    pf_chunks: list[torch.Tensor] = []
    entropy_chunks: list[torch.Tensor] = []
    actions = batch.action_tensors if batch.action_tensors is not None else batch.actions_set
    for start in range(0, len(actions), chunk_size):
        if isinstance(actions, TensorActionBatch):
            chunk_actions = actions.slice(start, start + chunk_size)
        else:
            chunk_actions = actions[start : start + chunk_size]
        log_paths_pf, paths_entropy = forward_replay_fixed_actions(
            rollout_worker,
            generator,
            chunk_actions,
            random_spec=batch.random_spec,
            device=device,
        )
        pf_chunks.append(log_paths_pf)
        entropy_chunks.append(paths_entropy)

    return torch.cat(pf_chunks, dim=0), torch.cat(entropy_chunks, dim=0)


def hybrid_rollout_outputs_for_tb(
    rollout_worker,
    generator,
    batch: HybridReplayBatch,
    *,
    chunk_size: int,
    device: str,
) -> dict:
    """Replay fixed actions and return a rollout dict for PhyloGFN TB loss."""
    pf_chunks: list[torch.Tensor] = []
    pb_chunks: list[torch.Tensor] = []
    actions = batch.action_tensors if batch.action_tensors is not None else batch.actions_set
    for start in range(0, len(actions), chunk_size):
        if isinstance(actions, TensorActionBatch):
            chunk_actions = actions.slice(start, start + chunk_size)
        else:
            chunk_actions = actions[start : start + chunk_size]
        data, _ = rollout_worker.rollout(
            generator,
            len(chunk_actions),
            random_spec=batch.random_spec,
            generate_full_trajectories=False,
            input_actions_set=chunk_actions,
        )
        pf_chunks.append(data["log_paths_pf"])
        pb_chunks.append(data["log_paths_pb"])

    return {
        "log_paths_pf": torch.cat(pf_chunks, dim=0).to(device),
        "log_paths_pb": torch.cat(pb_chunks, dim=0).to(device),
        "log_rewards": batch.log_rewards,
        "log_scores": batch.log_scores,
    }
