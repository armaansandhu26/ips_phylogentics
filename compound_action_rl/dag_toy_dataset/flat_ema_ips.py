"""Flat categorical control for EMA inverse-probability scaling.

This intentionally removes the DAG. One categorical action directly selects
one terminal outcome::

    action x  ->  terminal State(x, budget - x)  ->  reward R(x)

The policy therefore has one independent logit per terminal and the sampled
action probability is also the terminal-outcome probability. The EMA follows
the reference implementation exactly:

    q_0(o) = 1 / num_outcomes
    q_t(o) = (1 - alpha) * q_{t-1}(o) + alpha * batch_frequency_t(o)
    weight(o) = R(o) / max(q_t(o), tracker_eps)
    loss = -mean(weight(o) * log pi(o))

This is a diagnostic control, not a replacement for the sequential DAG model.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from config import default_terminal_rewards


@dataclass(frozen=True)
class FlatEMAConfig:
    budget: int = 128
    batch_size: int = 16
    num_updates: int = 2_000
    lr: float = 5e-3
    alpha: float = 0.005
    tracker_eps: float = 1e-6
    seed: int = 0
    log_every: int = 100
    final_samples: int = 10_000

    def validate(self) -> None:
        if self.budget < 1:
            raise ValueError("budget must be >= 1")
        if self.batch_size < 1 or self.num_updates < 1:
            raise ValueError("batch_size and num_updates must be >= 1")
        if self.lr <= 0.0:
            raise ValueError("lr must be > 0")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if self.tracker_eps <= 0.0:
            raise ValueError("tracker_eps must be > 0")
        if self.log_every < 1 or self.final_samples < 1:
            raise ValueError("log_every and final_samples must be >= 1")


def ema_update(
    tracker: torch.Tensor,
    actions: torch.Tensor,
    *,
    batch_size: int,
    alpha: float,
) -> torch.Tensor:
    """Update every tracker entry in place and return the batch frequencies."""
    counts = torch.bincount(actions, minlength=tracker.numel()).to(tracker.dtype)
    batch_frequency = counts / batch_size
    tracker.mul_(1.0 - alpha).add_(alpha * batch_frequency)
    return batch_frequency


def train_flat_ema(
    config: FlatEMAConfig,
    *,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, float]]]:
    """Train the flat policy and return policy, tracker, and history."""
    config.validate()
    torch.manual_seed(config.seed)
    target_device = torch.device(device)
    rewards = torch.tensor(
        default_terminal_rewards(config.budget),
        dtype=torch.float32,
        device=target_device,
    )
    num_outcomes = rewards.numel()
    logits = torch.nn.Parameter(torch.zeros(num_outcomes, device=target_device))
    optimizer = torch.optim.Adam([logits], lr=config.lr)
    tracker = torch.full(
        (num_outcomes,),
        1.0 / num_outcomes,
        dtype=torch.float32,
        device=target_device,
    )
    target = rewards / rewards.sum()
    history: list[dict[str, float]] = []

    for update in range(1, config.num_updates + 1):
        optimizer.zero_grad(set_to_none=True)
        policy = F.softmax(logits, dim=0)
        actions = torch.distributions.Categorical(policy).sample(
            (config.batch_size,)
        )
        log_prob = F.log_softmax(logits, dim=0)[actions]

        with torch.no_grad():
            ema_update(
                tracker,
                actions,
                batch_size=config.batch_size,
                alpha=config.alpha,
            )
            probability_estimate = tracker[actions].clamp(
                min=config.tracker_eps
            )
            weights = rewards[actions] / probability_estimate

        loss = -(weights * log_prob).mean()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            updated_policy = F.softmax(logits, dim=0)
            row = {
                "step": float(update),
                "loss": float(loss.item()),
                "mean_batch_reward": float(rewards[actions].mean().item()),
                "tracker_mass": float(tracker.sum().item()),
                "policy_tracker_tv": float(
                    0.5 * torch.abs(updated_policy - tracker).sum().item()
                ),
                "target_l1": float(torch.abs(updated_policy - target).sum().item()),
                "target_tv": float(
                    0.5 * torch.abs(updated_policy - target).sum().item()
                ),
                "policy_entropy": float(
                    -(updated_policy * updated_policy.clamp_min(1e-30).log())
                    .sum()
                    .item()
                ),
            }
        history.append(row)
        if update == 1 or update % config.log_every == 0:
            print(
                f"update {update:5d}  "
                f"target_L1={row['target_l1']:.4f}  "
                f"tracker_TV={row['policy_tracker_tv']:.4f}  "
                f"tracker_mass={row['tracker_mass']:.6f}"
            )

    return (
        F.softmax(logits, dim=0).detach().cpu(),
        tracker.detach().cpu(),
        history,
    )


def plot_flat_results(
    policy: torch.Tensor,
    sample_counts: torch.Tensor,
    rewards: torch.Tensor,
    *,
    final_samples: int,
    output: Path,
) -> None:
    target = rewards / rewards.sum()
    outcomes = np.arange(rewards.numel())
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(outcomes, rewards.numpy(), color="#0984e3")
    axes[0].set_title("Terminal reward R(o)")
    axes[0].set_xlabel("Directly selected terminal x")
    axes[0].set_ylabel("Reward")

    axes[1].plot(outcomes, target.numpy(), label="reward target", color="#0984e3")
    axes[1].plot(outcomes, policy.numpy(), label="learned policy", color="#e17055")
    axes[1].set_title("Exact policy probabilities")
    axes[1].set_xlabel("Directly selected terminal x")
    axes[1].set_ylabel("Probability")
    axes[1].legend()

    ideal_counts = target.numpy() * final_samples
    axes[2].plot(outcomes, ideal_counts, label="ideal expected count", color="#0984e3")
    axes[2].plot(
        outcomes,
        sample_counts.numpy(),
        label="sampled count",
        color="#e17055",
    )
    axes[2].set_title(f"Sampling ({final_samples:,} draws)")
    axes[2].set_xlabel("Directly selected terminal x")
    axes[2].set_ylabel("Count")
    axes[2].legend()

    for axis in axes:
        axis.grid(alpha=0.22)
    fig.suptitle("Flat action-equals-outcome EMA-IPS control")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    return requested


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-updates", type=int, default=2_000)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--alpha", type=float, default=0.005)
    parser.add_argument("--tracker-eps", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--final-samples", type=int, default=10_000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--run-dir", type=Path, default=None)
    args = parser.parse_args()

    config = FlatEMAConfig(
        budget=args.budget,
        batch_size=args.batch_size,
        num_updates=args.num_updates,
        lr=args.lr,
        alpha=args.alpha,
        tracker_eps=args.tracker_eps,
        seed=args.seed,
        log_every=args.log_every,
        final_samples=args.final_samples,
    )
    device = _resolve_device(args.device)
    run_dir = args.run_dir or (
        Path(__file__).resolve().parent
        / "data"
        / "flat_ema_ips_runs"
        / f"{datetime.now():%Y%m%d_%H%M%S}_b{config.budget}_seed{config.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    policy, tracker, history = train_flat_ema(config, device=device)
    rewards = torch.tensor(default_terminal_rewards(config.budget))
    target = rewards / rewards.sum()
    generator = torch.Generator().manual_seed(config.seed + 1)
    samples = torch.multinomial(
        policy,
        config.final_samples,
        replacement=True,
        generator=generator,
    )
    sample_counts = torch.bincount(samples, minlength=policy.numel())
    summary = {
        "config": asdict(config),
        "device": device,
        "meaning": "one action directly selects one terminal outcome",
        "target_l1": float(torch.abs(policy - target).sum().item()),
        "target_tv": float(0.5 * torch.abs(policy - target).sum().item()),
        "policy_tracker_tv": float(0.5 * torch.abs(policy - tracker).sum().item()),
        "tracker_mass": float(tracker.sum().item()),
        "sampled_unique_outcomes": int((sample_counts > 0).sum().item()),
        "policy": policy.tolist(),
        "tracker": tracker.tolist(),
        "target": target.tolist(),
        "sample_counts": sample_counts.tolist(),
    }
    torch.save(
        {"config": config, "policy": policy, "tracker": tracker},
        run_dir / "checkpoint.pt",
    )
    (run_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    plot_flat_results(
        policy,
        sample_counts,
        rewards,
        final_samples=config.final_samples,
        output=run_dir / "flat_ema_sampling.png",
    )
    print(json.dumps({key: value for key, value in summary.items() if key not in {
        "policy", "tracker", "target", "sample_counts"
    }}, indent=2))
    print(f"Artifacts: {run_dir}")


if __name__ == "__main__":
    main()
