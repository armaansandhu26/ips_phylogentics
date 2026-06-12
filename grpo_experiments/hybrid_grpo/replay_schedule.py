"""Per-round replay sizing for hybrid trainers (Panel G replay annealing)."""

from __future__ import annotations


def replay_anneal_sizes(
    round_idx: int,
    total_rounds: int,
    replay_start: int,
    replay_end: int,
    *,
    total_batch: int = 512,
    buffer_size: int = 2048,
) -> tuple[int, int, int]:
    """Linearly interpolate replay count; keep total batch size fixed."""
    if total_rounds <= 1:
        frac = 1.0
    else:
        frac = round_idx / (total_rounds - 1)
    replay = int(round(replay_start + frac * (replay_end - replay_start)))
    replay = max(0, min(replay, total_batch))
    fresh = total_batch - replay
    return fresh, replay, buffer_size


def effective_replay_sizes(
    round_idx: int,
    total_rounds: int,
    fresh_buffer_size: int,
    replay_sample_size: int,
    replay_anneal_start: int | None,
    replay_anneal_end: int | None,
    *,
    total_batch: int = 512,
    buffer_size: int = 2048,
) -> tuple[int, int, int]:
    """Return (fresh, replay, buffer) for a resample round."""
    if replay_anneal_start is not None and replay_anneal_end is not None:
        return replay_anneal_sizes(
            round_idx,
            total_rounds,
            replay_anneal_start,
            replay_anneal_end,
            total_batch=total_batch,
            buffer_size=buffer_size,
        )
    return fresh_buffer_size, replay_sample_size, buffer_size
