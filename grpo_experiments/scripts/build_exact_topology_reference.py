#!/usr/bin/env python3
"""Simulate a finite-sample reference from the exact five-taxon target."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grpo_experiments.learned_reverse_ips_grpo import (  # noqa: E402
    _edge_action,
    parse_config,
)
from grpo_experiments.utils import load_phylogfn_cfg  # noqa: E402
from src.env import build_env  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--reward-target",
        choices=("likelihood", "shifted_linear"),
        default=None,
        help="Defaults to experiment_config.json, or likelihood for older runs.",
    )
    parser.add_argument("--samples", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def exact_catalog(
    run_dir: Path,
) -> tuple[list[tuple[str, float]], float, float, float]:
    experiment = json.loads((run_dir / "experiment_config.json").read_text())
    config = parse_config(
        [
            "--cfg",
            str(run_dir / "resolved_config.yaml"),
            "--dataset",
            str(experiment["dataset_path"]),
        ]
    )
    cfg, sequences = load_phylogfn_cfg(config)
    env = build_env(cfg, sequences)
    num_taxa = len(sequences)
    action_ranges = [
        range(num_trees * (num_trees - 1) // 2)
        for num_trees in range(num_taxa, 1, -1)
    ]

    score_by_topology: dict[str, float] = {}
    for action_path in itertools.product(*action_ranges):
        actions = [
            {
                "tree_action": int(tree_action),
                "edge_action": _edge_action(step, len(action_path)),
            }
            for step, tree_action in enumerate(action_path)
        ]
        tree = env.actions_to_trajectory(actions).current_state.subtrees[0]
        topology_id = str(tree.tree_topology_id)
        score = float(tree.log_score)
        previous = score_by_topology.setdefault(topology_id, score)
        if not np.isclose(previous, score, rtol=0.0, atol=1e-5):
            raise RuntimeError(f"inconsistent score for topology {topology_id}")

    if num_taxa == 5 and len(score_by_topology) != 105:
        raise RuntimeError(f"expected 105 topologies, found {len(score_by_topology)}")
    return (
        list(score_by_topology.items()),
        float(getattr(env, "log_score_shift", 0.0)),
        float(cfg.ENV.REWARD.C),
        float(cfg.ENV.REWARD.SCALE),
    )


def plot_reference(rows: list[dict], output: Path, *, samples: int, seed: int) -> None:
    ranks = np.asarray([row["rank"] for row in rows])
    expected_frequency = np.asarray([row["expected_frequency"] for row in rows])
    empirical_frequency = np.asarray([row["empirical_frequency"] for row in rows])
    expected_count = np.asarray([row["expected_count"] for row in rows])
    sampled_count = np.asarray([row["sampled_count"] for row in rows])
    observed = sampled_count > 0

    fig, (top, bottom) = plt.subplots(
        2,
        1,
        figsize=(11, 8),
        dpi=220,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.0, 1.15]},
    )

    width = 0.4
    top_n = min(15, len(rows))
    top.bar(
        ranks[:top_n] - width / 2,
        expected_frequency[:top_n],
        width=width,
        color="#737373",
        alpha=0.75,
        label="exact target frequency",
    )
    top.bar(
        ranks[:top_n] + width / 2,
        empirical_frequency[:top_n],
        width=width,
        color="#1976d2",
        alpha=0.85,
        label=f"multinomial reference (seed {seed})",
    )
    top.set_ylabel("frequency")
    top.set_xlabel("topology reward rank")
    top.set_xticks(ranks[:top_n])
    top.set_title(f"Top 15 topology frequencies in {samples:,} reference samples")
    top.grid(axis="y", alpha=0.22)
    top.legend(frameon=False)

    bottom.plot(
        ranks,
        expected_count,
        color="#737373",
        linewidth=1.7,
        label="exact expected count",
    )
    bottom.scatter(
        ranks[observed],
        sampled_count[observed],
        s=24,
        color="#1976d2",
        zorder=3,
        label=f"observed count ({int(observed.sum())}/105 nonzero)",
    )
    bottom.scatter(
        ranks[~observed],
        np.full((~observed).sum(), 0.5),
        marker="x",
        s=16,
        color="#c62828",
        alpha=0.55,
        label="zero observations (shown at 0.5)",
    )
    bottom.axhline(1.0, color="#c62828", linestyle=":", linewidth=1.0)
    bottom.set_yscale("log")
    bottom.set_xlim(0.5, len(rows) + 0.5)
    bottom.set_ylim(0.35, max(expected_count.max(), sampled_count.max()) * 1.8)
    bottom.set_xlabel("topology reward rank (all 105)")
    bottom.set_ylabel("count (log scale)")
    bottom.set_title(
        "Full support: all topologies observed"
        if bool(np.all(observed))
        else "Full support: low-reward topologies remain unobserved"
    )
    bottom.grid(alpha=0.22, which="both")
    bottom.legend(frameon=False)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.samples <= 0:
        raise ValueError("--samples must be positive")
    run_dir = args.run_dir.resolve()
    experiment = json.loads((run_dir / "experiment_config.json").read_text())
    reward_target = (
        args.reward_target
        or str(experiment.get("reward_target", "likelihood"))
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else run_dir / "plots" / "reference"
    )

    catalog, log_score_shift, reward_c, reward_scale = exact_catalog(run_dir)
    topology_ids = [item[0] for item in catalog]
    model_log_scores = np.asarray([item[1] for item in catalog], dtype=np.float64)
    linear_scores = (reward_c + model_log_scores) / reward_scale
    if reward_target == "likelihood":
        target_log_rewards = linear_scores
        target_description = (
            "q*(x) proportional to exp((C + model_log_score(x)) / scale)"
        )
    elif reward_target == "shifted_linear":
        if np.any(linear_scores <= 0.0):
            raise ValueError("shifted_linear requires positive shifted scores")
        target_log_rewards = np.log(linear_scores)
        target_description = (
            "q*(x) proportional to (C + model_log_score(x)) / scale"
        )
    else:
        raise ValueError(f"unknown reward target: {reward_target!r}")
    probabilities = np.exp(target_log_rewards - target_log_rewards.max())
    probabilities /= probabilities.sum()
    counts = np.random.default_rng(args.seed).multinomial(args.samples, probabilities)

    order = np.argsort(-probabilities)
    rows = []
    for rank, idx in enumerate(order, start=1):
        rows.append(
            {
                "rank": rank,
                "topology_id": topology_ids[idx],
                "model_log_score": float(model_log_scores[idx]),
                "terminal_log_likelihood": float(
                    model_log_scores[idx] - log_score_shift
                ),
                # Backward-compatible alias used by older comparison helpers.
                "log_score": float(model_log_scores[idx]),
                "target_log_reward": float(target_log_rewards[idx]),
                "expected_frequency": float(probabilities[idx]),
                "expected_count": float(args.samples * probabilities[idx]),
                "sampled_count": int(counts[idx]),
                "empirical_frequency": float(counts[idx] / args.samples),
            }
        )

    stem = (
        f"exact_topology_reference_{reward_target}_"
        f"{args.samples}_seed{args.seed}"
    )
    json_path = output_dir / f"{stem}.json"
    plot_path = output_dir / f"{stem}.png"
    payload = {
        "metadata": {
            "run_dir": str(run_dir),
            "samples": args.samples,
            "seed": args.seed,
            "reward_target": reward_target,
            "target": target_description,
            "model_log_score_shift": log_score_shift,
            "simulation": "one multinomial draw from the exact 105-topology target",
        },
        "summary": {
            "num_topologies": len(rows),
            "observed_topologies": int(np.count_nonzero(counts)),
            "unobserved_topologies": int(len(rows) - np.count_nonzero(counts)),
        },
        "topologies": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot_reference(rows, plot_path, samples=args.samples, seed=args.seed)
    print(f"observed_topologies={payload['summary']['observed_topologies']}/105")
    print(f"wrote {json_path}")
    print(f"wrote {plot_path}")


if __name__ == "__main__":
    main()
