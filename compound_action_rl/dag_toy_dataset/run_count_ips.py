"""Run the minimal count-based IPS algorithm on the DAG toy."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from config import TrainConfig
from count_ips import CountIPSTrainer, _r2_against
from dag_env import State


def ideal_counts(rewards: tuple[float, ...], samples: int) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.asarray(rewards, dtype=np.float64)
    probabilities /= probabilities.sum()
    expected = probabilities * samples
    counts = np.floor(expected).astype(np.int64)
    remainder = samples - int(counts.sum())
    order = np.argsort(-(expected - counts), kind="stable")
    counts[order[:remainder]] += 1
    return probabilities, counts


def _render_final_counts_plot(
    rewards: np.ndarray,
    ideal: np.ndarray,
    actual: np.ndarray,
    *,
    samples: int,
    r2_reward_target: float,
    tv_reward_target: float,
    output: Path,
    suptitle: str = "Count-based IPS sampling vs ideal reward sampling",
) -> None:
    """Render an uncluttered comparison of ideal and sampled outcome counts."""
    order = np.argsort(rewards)
    sorted_rewards = rewards[order]
    sorted_ideal = ideal[order]
    sorted_actual = actual[order]
    outcome_count = len(rewards)
    ideal_unique = int(np.count_nonzero(ideal))
    actual_unique = int(np.count_nonzero(actual))

    fig, (ax_bar, ax_scatter) = plt.subplots(1, 2, figsize=(14, 5.4))
    positions = np.arange(outcome_count)
    width = 0.4
    ax_bar.bar(
        positions - width / 2,
        sorted_ideal,
        width,
        color="#0984e3",
        label="Ideal",
    )
    ax_bar.bar(
        positions + width / 2,
        sorted_actual,
        width,
        color="#e17055",
        label="Sampled",
    )
    ax_bar.set_xticks([])
    ax_bar.set_xlabel(f"Terminal outcomes sorted by reward ({outcome_count} total)")
    ax_bar.set_ylabel(f"Count (out of {samples:,})")
    ax_bar.set_title("Outcome counts")
    ax_bar.grid(axis="y", alpha=0.22)
    ax_bar.legend()

    ax_scatter.plot(
        sorted_rewards,
        sorted_ideal,
        "o--",
        color="#0984e3",
        linewidth=1.8,
        markersize=5,
        label="Ideal",
    )
    ax_scatter.scatter(
        sorted_rewards,
        sorted_actual,
        marker="X",
        s=60,
        color="#e17055",
        label="Sampled",
        zorder=3,
    )
    ax_scatter.set_xlabel("Terminal reward")
    ax_scatter.set_ylabel(f"Count (out of {samples:,})")
    ax_scatter.set_title("Ideal vs sampled counts")
    ax_scatter.grid(alpha=0.22)
    ax_scatter.legend()
    ax_scatter.text(
        0.03,
        0.97,
        (
            f"Unique outcomes\n"
            f"Ideal: {ideal_unique}/{outcome_count}\n"
            f"Sampled: {actual_unique}/{outcome_count}\n"
            f"$R^2$: {r2_reward_target:.3f}   TV: {tv_reward_target:.3f}"
        ),
        transform=ax_scatter.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.9},
    )

    fig.suptitle(suptitle)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_final_counts(
    trainer: CountIPSTrainer,
    evaluation: dict,
    *,
    samples: int,
    output: Path,
    suptitle: str = "Count-based IPS sampling vs ideal reward sampling",
) -> dict[str, object]:
    states = trainer.terminals
    rewards = tuple(trainer.reward_by_terminal[state] for state in states)
    probabilities, ideal = ideal_counts(rewards, samples)
    actual = np.array(
        [evaluation["eval_outcome_counts"][state.signature] for state in states]
    )
    actual_probabilities = actual / samples
    r2_reward_target = _r2_against(probabilities, actual_probabilities)
    l1 = float(np.abs(actual_probabilities - probabilities).sum())
    tv_reward_target = 0.5 * l1
    _render_final_counts_plot(
        np.asarray(rewards),
        ideal,
        actual,
        samples=samples,
        r2_reward_target=r2_reward_target,
        tv_reward_target=tv_reward_target,
        output=output,
        suptitle=suptitle,
    )
    expected_counts = probabilities * samples
    chi_square = float(
        np.sum(np.square(actual - expected_counts) / np.maximum(expected_counts, 1e-12))
    )
    return {
        "samples": samples,
        "outcomes": [state.signature for state in states],
        "rewards": list(rewards),
        "ideal_probabilities": probabilities.tolist(),
        "ideal_counts": ideal.tolist(),
        "actual_counts": actual.tolist(),
        "actual_probabilities": actual_probabilities.tolist(),
        "ideal_unique_outcomes": int(np.count_nonzero(ideal)),
        "actual_unique_outcomes": int(np.count_nonzero(actual)),
        "r2_reward_target": r2_reward_target,
        "l1_reward_target": l1,
        "tv_reward_target": tv_reward_target,
        "max_abs_prob_error": float(np.abs(actual_probabilities - probabilities).max()),
        "pearson_chi_square": chi_square,
    }


def _plot_trajectory_diagnostics(
    history: list[dict],
    trainer: CountIPSTrainer,
    evaluation: dict,
    *,
    samples: int,
    output: Path,
    subtitle: str = (
        "Uniform conditional allocation is a diagnostic reference, not a reward target"
    ),
) -> dict[str, object]:
    """Plot observed trajectory behavior without exhaustive path enumeration."""
    states = trainer.terminals
    colors = plt.colormaps["viridis"](
        np.linspace(0.08, 0.92, len(states))
    )
    state_by_signature = {state.signature: state for state in states}
    labels = [
        label
        for state in states
        for label, terminal in evaluation["eval_trajectory_terminal"].items()
        if terminal == state.signature
    ]
    terminals = [
        state_by_signature[evaluation["eval_trajectory_terminal"][label]]
        for label in labels
    ]
    counts = np.array([evaluation["eval_trajectory_counts"][label] for label in labels])
    conditional = np.array(
        [
            evaluation["eval_conditional_trajectory_probs"][state.signature][label]
            for label, state in zip(labels, terminals)
        ]
    )
    reference = np.array([
        1.0 / max(evaluation["eval_trajectory_coverage"][state.signature], 1)
        for state in terminals
    ])
    bar_colors = [colors[states.index(state)] for state in terminals]
    positions = np.arange(len(labels))
    reported_per_state = [
        sum(terminal == state for terminal in terminals) for state in states
    ]
    boundaries = np.cumsum(reported_per_state)[:-1]
    truncated = evaluation["eval_trajectory_details_truncated"]

    fig, axes = plt.subplots(2, 2, figsize=(18, 11))
    ax = axes[0, 0]
    if truncated:
        coverage = [evaluation["eval_trajectory_coverage"][s.signature] for s in states]
        ax.bar(np.arange(len(states)), coverage, color=colors)
        ax.set_xticks(np.arange(len(states)), [s.signature for s in states], rotation=65)
        ax.set_ylabel("Unique sampled trajectories")
        ax.set_title("Observed trajectory coverage by terminal")
    else:
        ax.bar(positions, counts, color=bar_colors)
        ax.set_ylabel(f"Count (out of {samples:,})")
        ax.set_title("Global sampling count for every observed trajectory")

    ax = axes[0, 1]
    if truncated:
        max_shares = [
            evaluation["eval_max_conditional_trajectory_share"][s.signature]
            for s in states
        ]
        ax.bar(np.arange(len(states)), max_shares, color=colors)
        ax.set_xticks(np.arange(len(states)), [s.signature for s in states], rotation=65)
        ax.set_ylabel("Largest P(trajectory | terminal)")
        ax.set_title("Observed path concentration by terminal")
    else:
        ax.bar(positions, conditional, color=bar_colors, label="learned conditional share")
        ax.scatter(
            positions,
            reference,
            marker="_",
            s=240,
            linewidth=2.5,
            color="#2d3436",
            label="uniform-over-observed-paths reference",
            zorder=3,
        )
        ax.set_ylim(0, max(0.55, float(conditional.max()) * 1.15))
        ax.set_ylabel("P(trajectory | terminal)")
        ax.set_title("Within-terminal observed-path allocation")
        ax.legend(fontsize=9)

    eval_rows = [row for row in history if "eval_trajectory_coverage" in row]
    ax = axes[1, 0]
    if eval_rows:
        eval_steps = [row["step"] for row in eval_rows]
        entropy_rows = np.asarray([
            [row["eval_normalized_trajectory_entropy"][state.signature] for state in states]
            for row in eval_rows
        ])
        ax.plot(eval_steps, entropy_rows.mean(axis=1), "o-", label="terminal mean")
        ax.fill_between(
            eval_steps,
            entropy_rows.min(axis=1),
            entropy_rows.max(axis=1),
            alpha=0.2,
            label="terminal min-max",
        )
    ax.axhline(1.0, color="#2d3436", linestyle="--", alpha=0.55)
    ax.set_ylim(-0.02, 1.05)
    ax.set_ylabel("Entropy / maximum entropy")
    ax.set_title("Conditional trajectory diversity during training")
    ax.legend()

    ax = axes[1, 1]
    if eval_rows:
        eval_steps = [row["step"] for row in eval_rows]
        share_rows = np.asarray([
            [row["eval_max_conditional_trajectory_share"][state.signature] for state in states]
            for row in eval_rows
        ])
        ax.plot(eval_steps, share_rows.mean(axis=1), "o-", label="terminal mean")
        ax.fill_between(
            eval_steps,
            share_rows.min(axis=1),
            share_rows.max(axis=1),
            alpha=0.2,
            label="terminal min-max",
        )
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Largest P(trajectory | terminal)")
    ax.set_title("Observed path concentration during training")
    ax.legend()

    if not truncated:
        for ax in axes[:1].flat:
            ax.set_xticks(positions, labels, rotation=65, ha="right", fontsize=8)
            for boundary in boundaries:
                ax.axvline(boundary - 0.5, color="#636e72", linewidth=1, alpha=0.5)
            ax.grid(axis="y", alpha=0.22)
    for ax in axes[1].flat:
        ax.set_xlabel("update")
        ax.grid(alpha=0.22)
    fig.suptitle(
        f"Trajectory diagnostics\n{subtitle}"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)

    return {
        "samples": samples,
        "unique_trajectories_hit": evaluation["eval_unique_trajectories"],
        "trajectory_order": labels,
        "trajectory_terminals": [state.signature for state in terminals],
        "trajectory_counts": counts.tolist(),
        "trajectory_probabilities": [
            evaluation["eval_trajectory_probs"][label] for label in labels
        ],
        "conditional_probabilities": conditional.tolist(),
        "uniform_over_observed_reference": reference.tolist(),
        "trajectory_details_truncated": truncated,
        "coverage_by_terminal": evaluation["eval_trajectory_coverage"],
        "normalized_entropy_by_terminal": evaluation["eval_normalized_trajectory_entropy"],
        "effective_trajectories_by_terminal": evaluation["eval_effective_trajectories"],
        "max_conditional_share_by_terminal": evaluation["eval_max_conditional_trajectory_share"],
    }


def _plot_training_curves(
    history: list[dict],
    trainer: CountIPSTrainer,
    *,
    output: Path,
    propensity_title: str = "Within-group count propensities",
    suptitle: str = "Count-IPS training diagnostics",
) -> None:
    steps = np.array([row["step"] for row in history])
    states = trainer.terminals
    target = trainer.target_reward()
    target_probabilities = np.array([target[state] for state in states])
    target_mean_reward = float(
        sum(target[state] * trainer.reward_by_terminal[state] for state in states)
    )
    colors = plt.colormaps["viridis"](
        np.linspace(0.08, 0.92, len(states))
    )

    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    ax = axes[0, 0]
    ax.plot(steps, [row["mean_reward"] for row in history], color="#0984e3")
    ax.axhline(target_mean_reward, color="#00b894", linestyle="--", label="ideal sampler")
    ax.set_title("Batch mean reward")
    ax.legend()

    ax = axes[0, 1]
    eval_rows = [row for row in history if "eval_outcome_probs" in row]
    if eval_rows:
        eval_steps = [row["step"] for row in eval_rows]
        for state, color in zip(states, colors):
            ax.plot(
                eval_steps,
                [row["eval_outcome_probs"][state.signature] for row in eval_rows],
                "o-",
                color=color,
                linewidth=1.0,
                markersize=2.5,
                alpha=0.7,
            )
            ax.axhline(
                target[state],
                color=color,
                linestyle="--",
                linewidth=0.8,
                alpha=0.4,
            )
        largest_probability = max(
            max(target.values()),
            max(
                probability
                for row in eval_rows
                for probability in row["eval_outcome_probs"].values()
            ),
        )
        ax.set_ylim(0.0, min(1.02, max(0.05, largest_probability * 1.15)))
    else:
        ax.set_ylim(0.0, 1.02)
    ax.set_title("Evaluation outcome probabilities")
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color="#0984e3",
                marker="o",
                linewidth=1.0,
                markersize=3,
                label="learned outcomes",
            ),
            Line2D(
                [0],
                [0],
                color="#636e72",
                linestyle="--",
                linewidth=1.0,
                label="ideal targets",
            ),
        ],
        fontsize=8,
    )

    ax = axes[0, 2]
    if eval_rows:
        eval_steps = [row["step"] for row in eval_rows]
        ax.plot(eval_steps, [row["tv_reward_target"] for row in eval_rows], "o-", label="TV distance")
        ax.plot(eval_steps, [row["max_abs_prob_error"] for row in eval_rows], "s--", label="max probability error")
    ax.set_ylim(bottom=0)
    ax.set_title("Distance from ideal sampling\n(lower is better)")
    ax.legend()

    ax = axes[1, 0]
    ax.plot(steps, [row["ips_prob_min"] for row in history], label="min p_hat")
    ax.plot(steps, [row["ips_prob_mean"] for row in history], label="mean p_hat")
    ax.plot(steps, [row["ips_prob_max"] for row in history], label="max p_hat")
    ax.set_title(propensity_title)
    ax.legend()

    ax = axes[1, 1]
    ax.plot(steps, [row["ips_ess_fraction"] for row in history], label="IPS ESS / G")
    ax.plot(
        steps,
        [row["ips_unique_outcomes"] / len(states) for row in history],
        label=f"outcome coverage / {len(states)}",
    )
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("IPS stability and coverage")
    ax.legend()

    ax = axes[1, 2]
    ax.plot(steps, [row["grad_norm"] for row in history], color="#d63031", label="gradient norm")
    entropy_ax = ax.twinx()
    entropy_ax.plot(steps, [row["entropy"] for row in history], color="#6c5ce7", label="policy entropy")
    ax.set_title("Optimization and exploration")
    ax.set_ylabel("gradient norm", color="#d63031")
    entropy_ax.set_ylabel("entropy", color="#6c5ce7")

    for ax in axes.flat:
        ax.set_xlabel("update")
        ax.grid(alpha=0.22)
    fig.suptitle(suptitle)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=32)
    parser.add_argument("--max-step", type=int, default=3)
    parser.add_argument("--num-updates", type=int, default=500)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--num-groups", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="auto uses CUDA when available, otherwise CPU",
    )
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=2_000)
    parser.add_argument("--final-samples", type=int, default=10_000)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=500,
        help="save an intermediate checkpoint every N updates",
    )
    parser.add_argument(
        "--terminal-rewards",
        type=float,
        nargs="+",
        default=None,
        metavar="R",
        help="budget + 1 rewards in increasing terminal x order",
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    args = parser.parse_args()

    config = TrainConfig(
        budget=args.budget,
        max_step=args.max_step,
        terminal_rewards=(
            tuple(args.terminal_rewards) if args.terminal_rewards is not None else None
        ),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        group_size=args.group_size,
        num_groups=args.num_groups,
        num_updates=args.num_updates,
        lr=args.lr,
        clip_ratio=args.clip_ratio,
        entropy_coef=args.entropy_coef,
        seed=args.seed,
        log_every=args.log_every,
    )
    run_dir = args.run_dir or (
        Path(__file__).resolve().parent
        / "data"
        / "count_ips_runs"
        / (
            f"{datetime.now():%Y%m%d_%H%M%S}_b{config.budget}"
            f"_gs{config.group_size}_seed{config.seed}"
        )
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is not available")
    trainer = CountIPSTrainer(config, device=device)
    print(f"Run directory: {run_dir}")
    print(f"Device: {trainer.device}")
    print(f"DAG: {trainer.environment_summary()}")
    print("Algorithm: normal reward / within-group terminal frequency -> PPO advantage")

    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "train_config": asdict(config),
                "device": str(trainer.device),
                "checkpoint_every": args.checkpoint_every,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    history = trainer.train(
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        checkpoint_every=args.checkpoint_every,
        checkpoint_dir=run_dir / "checkpoints",
    )
    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    trainer.save(run_dir / "checkpoint.pt", update_step=config.num_updates)
    training_plot = run_dir / "training_curves.png"
    _plot_training_curves(history, trainer, output=training_plot)
    final_evaluation = trainer.evaluate(args.final_samples)
    sampling = _plot_final_counts(
        trainer,
        final_evaluation,
        samples=args.final_samples,
        output=run_dir / "sampling_counts.png",
    )
    trajectory_plot = run_dir / "trajectory_sampling.png"
    trajectory_sampling = _plot_trajectory_diagnostics(
        history,
        trainer,
        final_evaluation,
        samples=args.final_samples,
        output=trajectory_plot,
    )
    eval_rows = [row for row in history if "tv_reward_target" in row]
    summary = {
        "environment": trainer.environment_summary(),
        "final_sampling": sampling,
        "trajectory_sampling": trajectory_sampling,
        "best_eval": (
            min(eval_rows, key=lambda row: row["tv_reward_target"])
            if eval_rows
            else None
        ),
        "plots": {
            "training_curves": training_plot.name,
            "sampling_counts": "sampling_counts.png",
            "trajectory_sampling": trajectory_plot.name,
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Final counts: {sampling['actual_counts']}")
    print(f"Ideal counts: {sampling['ideal_counts']}")
    print(f"Final ideal-line R^2: {sampling['r2_reward_target']:.4f}")
    print(f"Final TV distance: {sampling['tv_reward_target']:.4f}")
    print(
        "Trajectory coverage: "
        f"{trajectory_sampling['coverage_by_terminal']} "
        f"({trajectory_sampling['unique_trajectories_hit']} unique sampled paths)"
    )
    print(f"Training curves: {training_plot}")
    print(f"Sampling plots: {run_dir / 'sampling_counts.png'}")
    print(f"Trajectory plots: {trajectory_plot}")
    print(f"Artifacts: {run_dir}")


if __name__ == "__main__":
    main()


'''
CUDA_VISIBLE_DEVICES=7 python compound_action_rl/dag_toy_dataset/run_count_ips.py \
  --budget 32 \
  --max-step 3 \
  --group-size 512 \
  --num-updates 4000 \
  --eval-every 1000 \
  --eval-episodes 10000 \
  --final-samples 1000000 \
  --checkpoint-every 500 \
  --device cuda
  '''
