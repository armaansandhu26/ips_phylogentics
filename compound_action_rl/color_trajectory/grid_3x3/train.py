"""Train hierarchical GRPO on the 3x3 color grid."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from grpo import GRPOAgent

from grid_environment import GridEnv


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GRPO on the 3x3 color grid.")
    parser.add_argument("--num-updates", type=int, default=200)
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--num-groups", type=int, default=4)
    parser.add_argument("--train-epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--save-path",
        type=str,
        default="checkpoints/grpo.pt",
        help="Where to save the trained policy checkpoint.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable, using CPU.")
        device = "cpu"

    env = GridEnv()
    agent = GRPOAgent(
        obs_dim=env.obs_dim,
        lr=args.lr,
        entropy_coef=args.entropy_coef,
        clip_ratio=args.clip_ratio,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        group_size=args.group_size,
        num_groups=args.num_groups,
        train_epochs=args.train_epochs,
        grad_clip_norm=args.grad_clip_norm,
        seed=args.seed,
        device=device,
    )

    print(
        f"3x3 GRPO config: max_episode_steps={env.max_episode_steps}, "
        f"group_size={args.group_size}, num_groups={args.num_groups}, "
        f"clip_ratio={args.clip_ratio}, train_epochs={args.train_epochs}, "
        f"hidden_size={args.hidden_size}, num_layers={args.num_layers}, "
        f"entropy_coef={args.entropy_coef}, device={device}"
    )
    print("Starting training...")
    agent.train(env, num_updates=args.num_updates, log_every=args.log_every)
    save_path = agent.save_checkpoint(args.save_path)
    print(f"Saved checkpoint to {save_path}")


if __name__ == "__main__":
    main()
