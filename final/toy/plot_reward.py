"""Plot a Hyper-Grid reward map for visual verification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from final.toy.build_dataset import DEFAULT_OUT_DIR


def plot_reward_map(
    rewards: np.ndarray,
    *,
    out_path: Path,
    title: str,
    vmax: float | None = None,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    im = ax.imshow(
        rewards.T,
        origin="lower",
        cmap="viridis",
        vmin=float(rewards.min()),
        vmax=vmax if vmax is not None else float(rewards.max()),
        aspect="equal",
    )
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot a Hyper-Grid reward heatmap.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output PNG path (default: <dataset>/reward_paper_indicator_4_modes.png)",
    )
    args = parser.parse_args(argv)

    dataset_dir = args.dataset.resolve()
    rewards_path = dataset_dir / "rewards.npy"
    meta_path = dataset_dir / "meta.json"
    if not rewards_path.exists():
        raise FileNotFoundError(f"missing rewards grid: {rewards_path}")

    rewards = np.load(rewards_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    H = int(meta.get("H", rewards.shape[0]))
    D = int(meta.get("D", rewards.ndim))
    title = f"reward_paper (indicator, {2**D} modes)"
    out_path = args.out or (dataset_dir / "reward_paper_indicator_4_modes.png")
    plot_reward_map(rewards, out_path=out_path, title=title)
    print(json.dumps({"plot": str(out_path), "shape": list(rewards.shape), "H": H, "D": D}, indent=2))


if __name__ == "__main__":
    main()
