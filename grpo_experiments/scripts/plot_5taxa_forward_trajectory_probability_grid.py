#!/usr/bin/env python3
"""Plot forward trajectory probability P_F(tau) against reward for 5 taxa."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import plot_5taxa_sampling_comparison_best_fit as best_fit  # noqa: E402
import plot_5taxa_sampling_comparison_grid as base  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
COMPARISON_DIR = REPO_ROOT / "grpo_experiments/comparisons/5taxa"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=COMPARISON_DIR / "forward_trajectory_probability_grid.png",
    )
    parser.add_argument(
        "--gflownet-samples",
        type=Path,
        default=REPO_ROOT
        / "og_code/experiments/full_model/20260703_172421_phylgfn_logreward_g4096_1m_full_replay_op3277_r819_rb4096/plots/forward_trajectory_eval_1000000/og_gflownet_reward_probability_samples.npz",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scatter-points", type=int, default=0)
    return parser.parse_args()


def load_forward_probability(
    method: best_fit.Method,
) -> tuple[np.ndarray, np.ndarray, int] | None:
    with np.load(method.samples) as payload:
        reward_key = "log_reward" if method.kind == "gflownet" else "log_score"
        reward = (
            np.exp(payload[reward_key].astype(np.float64))
            if method.kind == "gflownet"
            else payload[reward_key].astype(np.float64)
        )
        probability = np.exp(payload["log_pf"].astype(np.float64))
    spec = base.PanelSpec(
        title=method.title,
        samples=method.samples,
        estimator_label=r"$P_F(\tau)$",
        kind=method.kind,
    )
    _, _, unique_signatures = base.load_panel(spec)
    return reward, probability, unique_signatures


def main() -> None:
    args = parse_args()
    method_specs = [
        best_fit.Method(
            title=method.title,
            samples=args.gflownet_samples if method.kind == "gflownet" else method.samples,
            estimator_label=r"$P_F(\tau)$",
            kind=method.kind,
        )
        for method in best_fit.methods()
    ]
    loaded = [load_forward_probability(method) for method in method_specs]
    available_rewards = [item[0] for item in loaded if item is not None]
    reward_min = min(float(reward.min()) for reward in available_rewards)
    reward_max = max(float(reward.max()) for reward in available_rewards)
    reward_padding = 0.02 * (reward_max - reward_min)
    reward_limits = (reward_min - reward_padding, reward_max + reward_padding)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(18, 16),
        dpi=220,
        sharex=True,
        constrained_layout=True,
    )
    summary: dict = {
        "quantity": "Forward trajectory probability P_F(tau), without reverse correction.",
        "fit_definition": (
            "Independent ordinary least-squares fit in displayed linear space: "
            "P_F(tau) = slope * reward + intercept."
        ),
        "panels": {},
    }

    for ax, method, payload in zip(
        axes.ravel(),
        method_specs,
        loaded,
        strict=True,
    ):
        if payload is None:
            ax.set_title(method.title, fontsize=13)
            ax.text(
                0.5,
                0.5,
                (
                    r"$P_F(\tau)$ unavailable"
                    "\n\nThe saved GFlowNet evaluation contains only\n"
                    r"$\log P_F(\tau)-\log P_B(\tau)$."
                ),
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=12,
            )
            ax.set_xlim(*reward_limits)
            ax.grid(True, alpha=0.2)
            summary["panels"][method.title] = {
                "available": False,
                "reason": "The saved samples do not contain log_pf.",
                "samples_file": str(method.samples),
            }
            continue

        reward, probability, unique_signatures = payload
        fit = (
            best_fit.linear_fit(reward, probability)
            if unique_signatures >= 2
            else None
        )
        best_fit.plot_panel(
            ax,
            method=method,
            reward=reward,
            probability=probability,
            unique_signatures=unique_signatures,
            fit=fit,
            reward_limits=reward_limits,
            seed=args.seed,
            scatter_points=args.scatter_points,
        )
        summary["panels"][method.title] = {
            "available": True,
            "samples": int(len(reward)),
            "unique_signatures": unique_signatures,
            "fit": fit,
            "samples_file": str(method.samples),
        }

    fig.supxlabel(r"Terminal reward: $R(x)=3600+\log L(x)$", fontsize=14)
    fig.supylabel(r"Forward trajectory probability $P_F(\tau)$", fontsize=14)
    fig.suptitle(
        r"5-taxa forward trajectory probability $P_F(\tau)$ vs reward",
        fontsize=16,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)

    metrics_path = args.output.with_suffix(".json")
    metrics_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output.resolve()}")
    print(f"wrote {metrics_path.resolve()}")


if __name__ == "__main__":
    main()
